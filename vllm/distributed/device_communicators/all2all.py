# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import torch
import torch.distributed as dist

import vllm.envs as envs
from vllm.distributed import get_dp_group, get_ep_group
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.import_utils import has_deep_ep

from .base_device_communicator import All2AllManagerBase, Cache

logger = init_logger(__name__)


class NaiveAll2AllManager(All2AllManagerBase):
    """
    A naive implementation of all2all communication.
    Uses all-reduce under the hood. Mainly for testing and debugging.
    """

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def naive_multicast(
        self,
        x: torch.Tensor,
        cu_tokens_across_sp_cpu: torch.Tensor,
        is_sequence_parallel: bool,
    ) -> torch.Tensor:
        assert len(x.shape) == 2
        buffer = torch.empty(
            (cu_tokens_across_sp_cpu[-1], x.size(1)), device=x.device, dtype=x.dtype
        )

        rank = self.rank if is_sequence_parallel else self.dp_rank
        world_size = self.world_size if is_sequence_parallel else self.dp_world_size

        start = 0 if rank == 0 else cu_tokens_across_sp_cpu[rank - 1]
        end = cu_tokens_across_sp_cpu[rank]
        buffer[start:end, :].copy_(x)
        for idx in range(world_size):
            start = 0 if idx == 0 else cu_tokens_across_sp_cpu[idx - 1]
            end = cu_tokens_across_sp_cpu[idx]
            get_ep_group().broadcast(buffer[start:end, :], idx)

        return buffer

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        router_logits = self.naive_multicast(
            router_logits, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, router_logits

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if extra_tensors is not None:
            raise NotImplementedError(
                "extra_tensors is not supported for NaiveAll2AllManager"
            )
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        hidden_states = self.naive_multicast(
            hidden_states, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_weights = self.naive_multicast(
            topk_weights, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        topk_ids = self.naive_multicast(
            topk_ids, cu_tokens_across_sp_cpu, is_sequence_parallel
        )
        return hidden_states, topk_weights, topk_ids

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        ep_rank = self.rank if is_sequence_parallel else self.dp_rank
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sp_size = self.tp_group.world_size if is_sequence_parallel else 1
        cu_tokens_across_sp_cpu = dp_metadata.cu_tokens_across_sp(sp_size)

        start = 0 if ep_rank == 0 else cu_tokens_across_sp_cpu[ep_rank - 1]
        end = cu_tokens_across_sp_cpu[ep_rank]

        all_hidden_states = get_ep_group().all_reduce(hidden_states)
        hidden_states = all_hidden_states[start:end, :]
        return hidden_states

    def destroy(self):
        pass


class AgRsAll2AllManager(All2AllManagerBase):
    """All2All based on all-gather (dispatch) and reduce-scatter (combine)."""

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def dispatch_router_logits(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, router_logits]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather, dim=0, sizes=sizes
        )

        if extra_tensors is not None:
            return (gathered_tensors[0], gathered_tensors[1], gathered_tensors[2:])
        return gathered_tensors[0], gathered_tensors[1]

    def dispatch(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        is_sequence_parallel: bool = False,
        extra_tensors: list[torch.Tensor] | None = None,
    ):
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        assert sizes[dist_group.rank_in_group] == hidden_states.shape[0]

        tensors_to_gather = [hidden_states, topk_weights, topk_ids]
        if extra_tensors is not None:
            tensors_to_gather.extend(extra_tensors)

        gathered_tensors = dist_group.all_gatherv(
            tensors_to_gather, dim=0, sizes=sizes
        )

        hidden_states = gathered_tensors[0]
        topk_weights = gathered_tensors[1]
        topk_ids = gathered_tensors[2]

        if extra_tensors is None:
            return hidden_states, topk_weights, topk_ids
        return hidden_states, topk_weights, topk_ids, gathered_tensors[3:]

    def combine(
        self, hidden_states: torch.Tensor, is_sequence_parallel: bool = False
    ) -> torch.Tensor:
        dp_metadata = get_forward_context().dp_metadata
        assert dp_metadata is not None
        sizes = dp_metadata.get_chunk_sizes_across_dp_rank()
        assert sizes is not None
        dist_group = get_ep_group() if is_sequence_parallel else get_dp_group()
        hidden_states = dist_group.reduce_scatterv(hidden_states, dim=0, sizes=sizes)
        return hidden_states

    def destroy(self):
        pass


class DeepEPAll2AllManagerBase(All2AllManagerBase):
    """Base class for DeepEP-based All2All communication."""

    def __init__(self, cpu_group, tcp_store_group=None):
        assert has_deep_ep(), (
            "DeepEP kernels not found. Please install DeepEP kernels."
        )
        super().__init__(cpu_group, tcp_store_group)
        self.handle_cache = Cache()
        self.num_sms = 20

    def get_handle(self, kwargs):
        raise NotImplementedError

    def dispatch_router_logits(self, *args, **kwargs):
        raise NotImplementedError

    def dispatch(self, *args, **kwargs):
        raise NotImplementedError

    def combine(self, *args, **kwargs):
        raise NotImplementedError

    def destroy(self):
        with self.handle_cache._lock:
            for _, handle in self.handle_cache._cache.items():
                handle.destroy()
            self.handle_cache._cache.clear()


class DeepEPHTAll2AllManager(DeepEPAll2AllManagerBase):
    """All2All based on DeepEP High-Throughput kernels."""

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(self) -> dict[Any, Any]:
        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_rdma_bytes = None
        num_qps_per_rank = None

        if self.internode and not envs.VLLM_DEEPEP_HIGH_THROUGHPUT_FORCE_INTRA_NODE:
            num_rdma_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
            num_qps_per_rank = self.num_sms // 2
        else:
            num_rdma_bytes = 0
            num_qps_per_rank = 1

        assert num_rdma_bytes is not None
        assert num_qps_per_rank is not None

        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=False,
            num_qps_per_rank=num_qps_per_rank,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        assert len(kwargs) == 0
        import deep_ep

        buffer_kwargs = self._make_all2all_kwargs()
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def set_num_sms(self, num_sms: int):
        import deep_ep

        if num_sms > self.num_sms:
            num_sms = self.num_sms
        deep_ep.Buffer.set_num_sms(num_sms)


class DeepEPLLAll2AllManager(DeepEPAll2AllManagerBase):
    """All2All based on DeepEP Low-Latency kernels."""

    def __init__(self, cpu_group, tcp_store_group=None):
        super().__init__(cpu_group, tcp_store_group)

    def _make_all2all_kwargs(
        self,
        max_num_tokens_per_dp_rank: int,
        token_hidden_size: int,
        num_ep_ranks: int,
        num_global_experts: int,
        num_local_experts: int,
    ) -> dict[Any, Any]:
        import deep_ep

        num_nvl_bytes = envs.VLLM_DEEPEP_BUFFER_SIZE_MB * 1024 * 1024
        num_qps_per_rank = num_local_experts
        num_rdma_bytes = deep_ep.Buffer.get_low_latency_rdma_size_hint(
            num_max_dispatch_tokens_per_rank=max_num_tokens_per_dp_rank,
            hidden=token_hidden_size,
            num_ranks=num_ep_ranks,
            num_experts=num_global_experts,
        )
        assert num_rdma_bytes is not None

        kwargs = dict(
            group=self.cpu_group,
            num_nvl_bytes=num_nvl_bytes,
            num_rdma_bytes=num_rdma_bytes,
            low_latency_mode=True,
            num_qps_per_rank=num_qps_per_rank,
            allow_nvlink_for_low_latency_mode=True,
            allow_mnnvl=envs.VLLM_DEEPEP_LOW_LATENCY_USE_MNNVL,
            explicitly_destroy=True,
        )
        return kwargs

    def get_handle(self, kwargs):
        import deep_ep

        buffer_kwargs = self._make_all2all_kwargs(**kwargs)
        logger.debug("DeepEP all2all args %s", buffer_kwargs)
        handle: deep_ep.Buffer = self.handle_cache.get_or_create(
            buffer_kwargs, deep_ep.Buffer
        )
        return handle

    def max_sms_used(self) -> int | None:
        return 0