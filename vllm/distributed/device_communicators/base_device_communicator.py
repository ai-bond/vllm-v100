# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from weakref import WeakValueDictionary

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup


class Cache:
    def __init__(self):
        self._cache: WeakValueDictionary = WeakValueDictionary()
        self._lock = threading.RLock()

    def get_or_create(self, kwargs, func):
        key = tuple(sorted((k, v) for k, v in kwargs.items()))
        with self._lock:
            instance = self._cache.get(key)
            if instance is None:
                instance = func(**kwargs)
                self._cache[key] = instance
            return instance


class All2AllManagerBase:
    rank: int
    world_size: int

    def __init__(self, cpu_group, tcp_store_group=None):
        self.cpu_group = cpu_group
        self.tcp_store_group = tcp_store_group

        from vllm.distributed.parallel_state import (
            get_dp_group,
            get_tp_group,
            in_the_same_node_as,
        )

        self.dp_group = get_dp_group()
        self.tp_group = get_tp_group()

        self.dp_rank = self.dp_group.rank_in_group
        self.dp_world_size = self.dp_group.world_size
        self.rank = cpu_group.rank()
        self.world_size = cpu_group.size()

        if tcp_store_group is None:
            self.internode = not all(in_the_same_node_as(cpu_group, source_rank=0))
        else:
            self.internode = not all(
                in_the_same_node_as(tcp_store_group, source_rank=0)
            )

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        raise NotImplementedError

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        raise NotImplementedError

    def set_num_sms(self, num_sms: int):
        pass

    def max_sms_used(self) -> int | None:
        return None

    def combine(self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False):
        raise NotImplementedError

    def destroy(self):
        pass


class DeviceCommunicatorBase:
    """
    Base class for device-specific communicator.
    """

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device | None = None,
        device_group: ProcessGroup | None = None,
        unique_name: str = "",
        global_ranks: list[int] | None = None,
        global_world_size: int | None = None,
    ):
        self.device = device or torch.device("cpu")
        self.cpu_group = cpu_group
        self.device_group = device_group
        self.unique_name = unique_name

        from torch.distributed.distributed_c10d import _world

        is_stateless = _world.pg_map.get(cpu_group, None) is None

        if is_stateless:
            self.rank = cpu_group.rank()
            self.world_size = cpu_group.size()
            assert global_ranks is not None
            assert global_world_size is not None
            self.ranks = global_ranks
            self.global_rank = self.ranks[self.rank]
            self.global_world_size = global_world_size
            self.rank_in_group = self.rank
        else:
            self.rank = dist.get_rank(cpu_group)
            self.world_size = dist.get_world_size(cpu_group)
            self.ranks = dist.get_process_group_ranks(cpu_group)
            self.global_rank = dist.get_rank()
            self.global_world_size = dist.get_world_size()
            self.rank_in_group = dist.get_group_rank(self.cpu_group, self.global_rank)

        use_ep = False
        all2all_backend = None
        from vllm.config import get_current_vllm_config_or_none

        config = get_current_vllm_config_or_none()
        if config is not None:
            use_ep = config.parallel_config.data_parallel_size > 1
            all2all_backend = config.parallel_config.all2all_backend

        self.is_ep_communicator = unique_name.split(":")[0] == "ep"
        self.use_all2all = self.is_ep_communicator and use_ep
        self.all2all_backend = all2all_backend
        self.all2all_manager: All2AllManagerBase | None = None

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(input_, group=self.device_group)
        return input_

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if dim < 0:
            dim += input_.dim()
        input_size = input_.size()
        output_size = (input_size[0] * self.world_size,) + input_size[1:]
        output_tensor = torch.empty(
            output_size, dtype=input_.dtype, device=input_.device
        )
        dist.all_gather_into_tensor(output_tensor, input_, group=self.device_group)
        output_tensor = output_tensor.reshape((self.world_size,) + input_size)
        output_tensor = output_tensor.movedim(0, dim)
        output_tensor = output_tensor.reshape(
            input_size[:dim]
            + (self.world_size * input_size[dim],)
            + input_size[dim + 1 :]
        )
        return output_tensor

    def all_gatherv(
        self,
        input_: torch.Tensor | list[torch.Tensor],
        dim: int = 0,
        sizes: list[int] | None = None,
    ) -> torch.Tensor | list[torch.Tensor]:
        raise NotImplementedError

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        world_size = self.world_size
        if world_size == 1:
            return input_
        if dim < 0:
            dim += input_.dim()

        input_tensor = input_.movedim(0, dim).contiguous()
        assert input_tensor.shape[0] % world_size == 0
        chunk_size = input_tensor.shape[0] // world_size
        output_shape = (chunk_size,) + input_tensor.shape[1:]

        output_tensor = torch.empty(
            output_shape, dtype=input_tensor.dtype, device=input_tensor.device
        )

        torch.distributed.reduce_scatter_tensor(
            output_tensor, input_tensor, group=self.device_group
        )

        return output_tensor.movedim(0, dim).contiguous()

    def reduce_scatterv(
        self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None
    ) -> torch.Tensor:
        raise NotImplementedError

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        world_size = self.world_size
        if dim < 0:
            dim += input_.dim()

        if self.rank_in_group == dst:
            gather_list = [torch.empty_like(input_) for _ in range(world_size)]
        else:
            gather_list = None

        torch.distributed.gather(
            input_, gather_list, dst=self.ranks[dst], group=self.device_group
        )
        if self.rank_in_group == dst:
            return torch.cat(gather_list, dim=dim)
        return None

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        torch.distributed.send(tensor, self.ranks[dst], self.device_group)

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, self.ranks[src], self.device_group)
        return tensor

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        if self.world_size == 1:
            return tensor
        torch.distributed.broadcast(tensor, self.ranks[src], self.device_group)
        return tensor

    def destroy(self):
        pass

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module) -> None:
        if not self.is_ep_communicator:
            return

        moe_modules = [
            module
            for module in model.modules()
            if (
                module.__class__.__name__ == "FusedMoE"
                or module.__class__.__name__ == "SharedFusedMoE"
            )
        ]
        for module in moe_modules:
            module.maybe_init_modular_kernel()

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        if extra_tensors is not None:
            return hidden_states, router_logits, extra_tensors
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        if extra_tensors is not None:
            return hidden_states, topk_weights, topk_ids, extra_tensors
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        return hidden_states

    def batch_isend_irecv(self, p2p_ops: list):
        raise NotImplementedError