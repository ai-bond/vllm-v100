# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2023 The vLLM team.
# Adapted from
# https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/parallel_state.py
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

import contextlib
import gc
import pickle
import weakref
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from datetime import timedelta
from multiprocessing import shared_memory
from typing import TYPE_CHECKING, Any, Protocol
from unittest.mock import patch

import torch
import torch.distributed
import torch.distributed._functional_collectives as funcol
import torch.distributed._symmetric_memory
from torch.distributed import Backend, ProcessGroup, Store

import vllm.envs as envs
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.utils import (
    StatelessProcessGroup,
    get_cached_tcp_store_client,
)
from vllm.logger import init_logger
from vllm.utils.network_utils import get_distributed_init_method
from vllm.utils.system_utils import suppress_stdout
from vllm.utils.torch_utils import direct_register_custom_op

if TYPE_CHECKING:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator


@dataclass
class GraphCaptureContext:
    stream: torch.cuda.Stream


TensorMetadata = namedtuple("TensorMetadata", ["device", "dtype", "size"])


class Handle(Protocol):
    """Minimal async work handle used by P2P send/recv methods."""
    def is_completed(self) -> bool: ...
    def wait(self) -> None: ...


def _split_tensor_dict(
    tensor_dict: dict[str, torch.Tensor | Any],
) -> tuple[list[tuple[str, Any]], list[torch.Tensor]]:
    metadata_list: list[tuple[str, Any]] = []
    tensor_list: list[torch.Tensor] = []
    for key, value in tensor_dict.items():
        if isinstance(value, torch.Tensor):
            device = value.device.type
            metadata_list.append(
                (key, TensorMetadata(device, value.dtype, value.size()))
            )
            tensor_list.append(value)
        else:
            metadata_list.append((key, value))
    return metadata_list, tensor_list


_group_name_counter: dict[str, int] = {}

def _get_unique_name(name: str) -> str:
    if name not in _group_name_counter:
        _group_name_counter[name] = 0
    newname = f"{name}:{_group_name_counter[name]}"
    _group_name_counter[name] += 1
    return newname


_groups: dict[str, Callable[[], "GroupCoordinator | None"]] = {}

def _register_group(group: "GroupCoordinator") -> None:
    _groups[group.unique_name] = weakref.ref(group)


def all_reduce(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_reduce_out_place(tensor)

def all_reduce_fake(tensor: torch.Tensor, group_name: str) -> torch.Tensor:
    return torch.empty_like(tensor)

def reduce_scatter(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._reduce_scatter_out_place(tensor, dim)

def reduce_scatter_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] // world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)

def all_gather(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    assert group_name in _groups, f"Group {group_name} is not found."
    group = _groups[group_name]()
    if group is None:
        raise ValueError(f"Group {group_name} is destroyed.")
    return group._all_gather_out_place(tensor, dim)

def all_gather_fake(
    tensor: torch.Tensor, dim: int, world_size: int, group_name: str
) -> torch.Tensor:
    new_shape = list(tensor.shape)
    new_shape[dim] = tensor.shape[dim] * world_size
    return torch.empty(new_shape, dtype=tensor.dtype, device=tensor.device)


def patched_fused_scaled_matmul_reduce_scatter_fake(
    A: torch.Tensor, B: torch.Tensor, A_scale: torch.Tensor, B_scale: torch.Tensor,
    reduce_op: str, orig_scatter_dim: int, scatter_dim_after_maybe_reshape: int,
    group_name: str, output_shape: list[int], bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    if A_scale.numel() > 1:
        if A_scale.shape[:-1] != A.shape[:-1]:
            raise ValueError("A_scale leading dims must match A")
        A_scale = A_scale.flatten(0, -2).contiguous()
    elif A_scale.numel() != 1:
        raise ValueError("Invalid A_scale shape")

    C = torch._scaled_mm(
        A.flatten(0, -2).contiguous(), B, A_scale, B_scale,
        bias, result_scale, out_dtype, use_fast_accum,
    )
    C = C.view(*output_shape[:-1], B.shape[1])
    res = funcol.reduce_scatter_tensor(C, reduce_op, orig_scatter_dim, group_name)
    res = funcol.wait_tensor(res)
    return res

def patched_fused_scaled_matmul_reduce_scatter(
    A: torch.Tensor, B: torch.Tensor, A_scale: torch.Tensor, B_scale: torch.Tensor,
    reduce_op: str, orig_scatter_dim: int, scatter_dim_after_maybe_reshape: int,
    group_name: str, output_shape: list[int], bias: torch.Tensor | None = None,
    result_scale: torch.Tensor | None = None, out_dtype: torch.dtype | None = None,
    use_fast_accum: bool = False,
) -> torch.Tensor:
    return torch.ops.symm_mem.fused_scaled_matmul_reduce_scatter(
        A, B, A_scale, B_scale, reduce_op, orig_scatter_dim,
        scatter_dim_after_maybe_reshape, group_name, output_shape,
        bias, result_scale, out_dtype, use_fast_accum,
    )


direct_register_custom_op(op_name="all_reduce", op_func=all_reduce, fake_impl=all_reduce_fake)
direct_register_custom_op(op_name="reduce_scatter", op_func=reduce_scatter, fake_impl=reduce_scatter_fake)
direct_register_custom_op(op_name="all_gather", op_func=all_gather, fake_impl=all_gather_fake)
direct_register_custom_op(
    op_name="patched_fused_scaled_matmul_reduce_scatter",
    op_func=patched_fused_scaled_matmul_reduce_scatter,
    fake_impl=patched_fused_scaled_matmul_reduce_scatter_fake,
)


class GroupCoordinator:
    def __init__(
        self,
        group_ranks: list[list[int]],
        local_rank: int,
        torch_distributed_backend: str | Backend,
        use_device_communicator: bool,
        use_message_queue_broadcaster: bool = False,
        group_name: str | None = None,
    ):
        group_name = group_name or "anonymous"
        self.unique_name = _get_unique_name(group_name)
        _register_group(self)

        self.rank = torch.distributed.get_rank()
        self.local_rank = local_rank

        self_device_group = None
        self_cpu_group = None

        for ranks in group_ranks:
            device_group = torch.distributed.new_group(
                ranks, backend=torch_distributed_backend
            )
            with suppress_stdout():
                cpu_group = torch.distributed.new_group(ranks, backend="gloo")
            if self.rank in ranks:
                self.ranks = ranks
                self.world_size = len(ranks)
                self.rank_in_group = ranks.index(self.rank)
                self_device_group = device_group
                self_cpu_group = cpu_group

        assert self_cpu_group is not None
        assert self_device_group is not None

        self.cpu_group = self_cpu_group
        self.device_group = self_device_group

        # Hardcoded for CUDA (Volta)
        self.device = torch.device(f"cuda:{local_rank}")

        self.use_device_communicator = use_device_communicator
        self.device_communicator = None
        if use_device_communicator and self.world_size > 1:
            from vllm.distributed.device_communicators.cuda_communicator import (
                CudaCommunicator,
            )
            self.device_communicator = CudaCommunicator(
                cpu_group=self.cpu_group,
                device=self.device,
                device_group=self.device_group,
                unique_name=self.unique_name,
            )

        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue

        self.mq_broadcaster: MessageQueue | None = None
        if use_message_queue_broadcaster and self.world_size > 1:
            self.mq_broadcaster = MessageQueue.create_from_process_group(
                self.cpu_group, 1 << 22, 6
            )

        # CUDA uses custom ops for collectives
        self.use_custom_op_call = True
        self.use_cpu_custom_send_recv = False

    def create_mq_broadcaster(self, writer_rank=0, external_writer_handle=None, blocking=True):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        return MessageQueue.create_from_process_group(
            self.cpu_group, 1 << 22, 6, writer_rank=writer_rank,
            external_writer_handle=external_writer_handle, blocking=blocking,
        )

    def create_single_reader_mq_broadcasters(self, reader_rank_in_group=0, blocking=False):
        from vllm.distributed.device_communicators.shm_broadcast import MessageQueue
        return MessageQueue.create_from_process_group_single_reader(
            self.cpu_group, 1 << 22, 6,
            reader_rank=self.ranks[reader_rank_in_group], blocking=blocking,
        )

    @property
    def first_rank(self): return self.ranks[0]
    @property
    def last_rank(self): return self.ranks[-1]
    @property
    def is_first_rank(self): return self.rank == self.first_rank
    @property
    def is_last_rank(self): return self.rank == self.last_rank
    @property
    def next_rank(self): return self.ranks[(self.rank_in_group + 1) % self.world_size]
    @property
    def prev_rank(self): return self.ranks[(self.rank_in_group - 1) % self.world_size]

    @contextmanager
    def graph_capture(self, graph_capture_context: GraphCaptureContext | None = None):
        if graph_capture_context is None:
            stream = torch.cuda.Stream()
            graph_capture_context = GraphCaptureContext(stream)
        else:
            stream = graph_capture_context.stream

        maybe_ca_context = nullcontext()
        from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator

        if self.device_communicator is not None:
            assert isinstance(self.device_communicator, CudaCommunicator)
            ca_comm = self.device_communicator.ca_comm
            if ca_comm is not None:
                maybe_ca_context = ca_comm.capture()

        curr_stream = torch.cuda.current_stream()
        if curr_stream != stream:
            stream.wait_stream(curr_stream)

        with torch.cuda.stream(stream), maybe_ca_context:
            yield graph_capture_context

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1: return input_
        if self.use_custom_op_call:
            return torch.ops.vllm.all_reduce(input_, group_name=self.unique_name)
        return self._all_reduce_out_place(input_)

    def _all_reduce_out_place(self, input_: torch.Tensor) -> torch.Tensor:
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.all_reduce(input_)

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1: return input_
        if self.use_custom_op_call:
            return torch.ops.vllm.all_gather(input_, dim, self.world_size, group_name=self.unique_name)
        return self._all_gather_out_place(input_, dim)

    def _all_gather_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.all_gather(input_, dim)

    def all_gatherv(self, input_: torch.Tensor | list[torch.Tensor], dim: int = 0, sizes: list[int] | None = None):
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.all_gatherv(input_, dim, sizes)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        if self.world_size == 1: return input_
        if self.use_custom_op_call:
            return torch.ops.vllm.reduce_scatter(input_, dim, self.world_size, group_name=self.unique_name)
        return self._reduce_scatter_out_place(input_, dim)

    def reduce_scatterv(self, input_: torch.Tensor, dim: int = -1, sizes: list[int] | None = None) -> torch.Tensor:
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.reduce_scatterv(input_, dim, sizes)

    def _reduce_scatter_out_place(self, input_: torch.Tensor, dim: int) -> torch.Tensor:
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.reduce_scatter(input_, dim)

    def gather(self, input_: torch.Tensor, dst: int = 0, dim: int = -1) -> torch.Tensor | None:
        if self.world_size == 1: return input_
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.gather(input_, dst, dim)

    def broadcast(self, input_: torch.Tensor, src: int = 0):
        if self.world_size == 1: return input_
        torch.distributed.broadcast(input_, src=self.ranks[src], group=self.device_group)
        return input_

    def broadcast_object(self, obj: Any | None = None, src: int = 0):
        if self.world_size == 1: return obj
        if self.mq_broadcaster is not None:
            return self.mq_broadcaster.broadcast_object(obj)
        if self.rank_in_group == src:
            torch.distributed.broadcast_object_list([obj], src=self.ranks[src], group=self.cpu_group)
            return obj
        else:
            recv = [None]
            torch.distributed.broadcast_object_list(recv, src=self.ranks[src], group=self.cpu_group)
            return recv[0]

    def broadcast_object_list(self, obj_list: list[Any], src: int = 0, group: ProcessGroup | None = None):
        if self.world_size == 1: return obj_list
        torch.distributed.broadcast_object_list(obj_list, src=self.ranks[src], group=self.device_group)
        return obj_list

    def send_object(self, obj: Any, dst: int) -> None:
        object_tensor = torch.frombuffer(pickle.dumps(obj), dtype=torch.uint8)
        size_tensor = torch.tensor([object_tensor.numel()], dtype=torch.long, device="cpu")
        torch.distributed.send(size_tensor, dst=self.ranks[dst], group=self.cpu_group)
        torch.distributed.send(object_tensor, dst=self.ranks[dst], group=self.cpu_group)

    def recv_object(self, src: int) -> Any:
        size_tensor = torch.empty(1, dtype=torch.long, device="cpu")
        torch.distributed.recv(size_tensor, src=self.ranks[src], group=self.cpu_group)
        object_tensor = torch.empty(size_tensor.item(), dtype=torch.uint8, device="cpu")
        torch.distributed.recv(object_tensor, src=self.ranks[src], group=self.cpu_group)
        return pickle.loads(object_tensor.numpy().tobytes())

    def broadcast_tensor_dict(
        self, tensor_dict: dict[str, torch.Tensor | Any] | None = None, src: int = 0,
        group: ProcessGroup | None = None, metadata_group: ProcessGroup | None = None,
    ) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1: return tensor_dict
        group = self.device_group
        metadata_group = self.cpu_group

        if self.rank_in_group == src:
            metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
            self.broadcast_object(metadata_list, src=src)
            async_handles = []
            for tensor in tensor_list:
                if tensor.numel() == 0: continue
                handle = torch.distributed.broadcast(
                    tensor, src=self.ranks[src],
                    group=metadata_group if tensor.is_cpu else group, async_op=True
                )
                async_handles.append(handle)
            for h in async_handles: h.wait()
            return tensor_dict
        else:
            metadata_list = self.broadcast_object(None, src=src)
            tensor_dict = {}
            async_handles = []
            for key, value in metadata_list:
                if isinstance(value, TensorMetadata):
                    tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                    if tensor.numel() == 0:
                        tensor_dict[key] = tensor
                        continue
                    handle = torch.distributed.broadcast(
                        tensor, src=self.ranks[src],
                        group=metadata_group if tensor.is_cpu else group, async_op=True
                    )
                    async_handles.append(handle)
                    tensor_dict[key] = tensor
                else:
                    tensor_dict[key] = value
            for h in async_handles: h.wait()
            return tensor_dict

    def _should_use_all_gather(self, key: str, numel: int, all_gather_group: "GroupCoordinator | None", all_gather_tensors: dict[str, bool] | None) -> bool:
        if all_gather_group is None: return False
        use_all_gather = numel % all_gather_group.world_size == 0
        if all_gather_tensors is not None:
            use_all_gather = all_gather_tensors.get(key, use_all_gather)
        return use_all_gather

    def send_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any], dst: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1: return tensor_dict
        handles = self.isend_tensor_dict(tensor_dict, dst, all_gather_group, all_gather_tensors)
        for h in handles: h.wait()
        return None

    def isend_tensor_dict(self, tensor_dict: dict[str, torch.Tensor | Any], dst: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> list[Handle]:
        if self.world_size <= 1: return []
        if dst is None: dst = (self.rank_in_group + 1) % self.world_size
        
        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group

        metadata_list, tensor_list = _split_tensor_dict(tensor_dict)
        self.send_object(metadata_list, dst=dst)

        tensor_keys = [k for k, v in tensor_dict.items() if isinstance(v, torch.Tensor)]
        handles: list[Handle] = []
        for key, tensor in zip(tensor_keys, tensor_list):
            if tensor.numel() == 0: continue
            if self._should_use_all_gather(key, tensor.numel(), all_gather_group, all_gather_tensors):
                tensor = tensor.reshape(all_gather_size, -1)[all_gather_rank]
            comm_group = self.cpu_group if tensor.is_cpu else self.device_group
            handle = torch.distributed.isend(tensor, dst=self.ranks[dst], group=comm_group)
            if tensor.is_cuda: tensor.record_stream(torch.cuda.current_stream(tensor.device))
            handles.append(handle)
        return handles

    def recv_tensor_dict(self, src: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> dict[str, torch.Tensor | Any] | None:
        if not torch.distributed.is_initialized() or self.world_size == 1: return None
        tensor_dict, handles, postprocess = self.irecv_tensor_dict(src, all_gather_group, all_gather_tensors)
        for h in handles: h.wait()
        for fn in postprocess: fn()
        return tensor_dict

    def irecv_tensor_dict(self, src: int | None = None, all_gather_group: "GroupCoordinator | None" = None, all_gather_tensors: dict[str, bool] | None = None) -> tuple[dict[str, torch.Tensor | Any] | None, list[Handle], list[Callable[[], None]]]:
        if not torch.distributed.is_initialized() or self.world_size == 1: return None, [], []
        if src is None: src = (self.rank_in_group - 1) % self.world_size

        all_gather_size = 1 if all_gather_group is None else all_gather_group.world_size
        all_gather_rank = 0 if all_gather_group is None else all_gather_group.rank_in_group

        recv_metadata_list = self.recv_object(src=src)
        tensor_dict: dict[str, Any] = {}
        handles: list[Handle] = []
        postprocess: list[Callable[[], None]] = []

        for key, value in recv_metadata_list:
            if isinstance(value, TensorMetadata):
                full_tensor = torch.empty(value.size, dtype=value.dtype, device=value.device)
                if full_tensor.numel() == 0:
                    tensor_dict[key] = full_tensor
                    continue
                if self._should_use_all_gather(key, full_tensor.numel(), all_gather_group, all_gather_tensors):
                    slice_tensor = full_tensor.reshape(all_gather_size, -1)[all_gather_rank]
                    comm_group = self.cpu_group if slice_tensor.is_cpu else self.device_group
                    handle = torch.distributed.irecv(slice_tensor, src=self.ranks[src], group=comm_group)
                    handles.append(handle)
                    def _postprocess(key=key, slice_tensor=slice_tensor, orig_shape=tuple(full_tensor.shape), all_gather_group=all_gather_group):
                        tensor_dict[key] = all_gather_group.all_gather(slice_tensor, dim=0).reshape(orig_shape)
                    postprocess.append(_postprocess)
                    tensor_dict[key] = slice_tensor
                else:
                    comm_group = self.cpu_group if full_tensor.is_cpu else self.device_group
                    handle = torch.distributed.irecv(full_tensor, src=self.ranks[src], group=comm_group)
                    handles.append(handle)
                    tensor_dict[key] = full_tensor
            else:
                tensor_dict[key] = value
        return tensor_dict, handles, postprocess

    def barrier(self):
        torch.distributed.barrier(group=self.cpu_group)

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if self.device_communicator is None: raise ValueError("No device communicator")
        self.device_communicator.send(tensor, dst)

    def recv(self, size: torch.Size, dtype: torch.dtype, src: int | None = None) -> torch.Tensor:
        if self.device_communicator is None: raise ValueError("No device communicator")
        return self.device_communicator.recv(size, dtype, src)

    def destroy(self):
        if hasattr(self, "device_group"):
            torch.distributed.destroy_process_group(self.device_group)
            del self.device_group
        if hasattr(self, "cpu_group"):
            torch.distributed.destroy_process_group(self.cpu_group)
            del self.cpu_group
        if self.device_communicator is not None:
            self.device_communicator.destroy()
        if self.mq_broadcaster is not None:
            self.mq_broadcaster = None

    def prepare_communication_buffer_for_model(self, model: torch.nn.Module):
        if self.device_communicator is not None:
            self.device_communicator.prepare_communication_buffer_for_model(model)

    def dispatch_router_logits(self, hidden_states, router_logits, is_sequence_parallel=False, extra_tensors=None):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch_router_logits(hidden_states, router_logits, is_sequence_parallel, extra_tensors)
        return hidden_states, router_logits

    def dispatch(self, hidden_states, topk_weights, topk_ids, is_sequence_parallel=False, extra_tensors=None):
        if self.device_communicator is not None:
            return self.device_communicator.dispatch(hidden_states, topk_weights, topk_ids, is_sequence_parallel, extra_tensors)
        return hidden_states, topk_weights, topk_ids

    def combine(self, hidden_states, is_sequence_parallel=False):
        if self.device_communicator is not None:
            return self.device_communicator.combine(hidden_states, is_sequence_parallel)
        return hidden_states


_WORLD: GroupCoordinator | None = None
_INNER_DP_WORLD: GroupCoordinator | None = None
_NODE_COUNT: int | None = None

def get_world_group() -> GroupCoordinator:
    assert _WORLD is not None, "world group is not initialized"
    return _WORLD

def get_inner_dp_world_group() -> GroupCoordinator:
    assert _INNER_DP_WORLD is not None, "inner dp world group is not initialized"
    return _INNER_DP_WORLD

def init_world_group(ranks: list[int], local_rank: int, backend: str) -> GroupCoordinator:
    return GroupCoordinator(group_ranks=[ranks], local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=False, group_name="world")

def init_model_parallel_group(group_ranks: list[list[int]], local_rank: int, backend: str, use_message_queue_broadcaster: bool = False, group_name: str | None = None, use_device_communicator: bool = True) -> GroupCoordinator:
    return GroupCoordinator(group_ranks=group_ranks, local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=use_device_communicator, use_message_queue_broadcaster=use_message_queue_broadcaster, group_name=group_name)

def _init_stateless_group(group_ranks: list[list[int]], group_name: str, host: str, backend: str, coord_store: Store, use_device_communicator: bool = True) -> "StatelessGroupCoordinator":
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
    world = get_world_group()
    return StatelessGroupCoordinator(group_ranks=group_ranks, local_rank=world.local_rank, torch_distributed_backend=backend, use_device_communicator=use_device_communicator, group_name=group_name, host=host, coord_store=coord_store, global_rank=world.rank, global_world_size=world.world_size)

def _replace_active_groups(*, world, dp, ep, eplb, node_count) -> None:
    global _WORLD, _DP, _EP, _EPLB, _NODE_COUNT
    for group in (_DP, _EP, _WORLD, _EPLB):
        if group is not None: group.destroy()
    _WORLD = world
    _DP = dp
    _EP = ep
    _EPLB = eplb
    _NODE_COUNT = node_count

_TP: GroupCoordinator | None = None
def get_tp_group() -> GroupCoordinator:
    assert _TP is not None, "tensor model parallel group is not initialized"
    return _TP

_DCP: GroupCoordinator | None = None
def get_dcp_group() -> GroupCoordinator:
    assert _DCP is not None, "decode context model parallel group is not initialized"
    return _DCP
get_context_model_parallel_group = get_dcp_group

_PP: GroupCoordinator | None = None
def get_pp_group() -> GroupCoordinator:
    assert _PP is not None, "pipeline model parallel group is not initialized"
    return _PP

_DP: GroupCoordinator | None = None
def get_dp_group() -> GroupCoordinator:
    assert _DP is not None, "data parallel group is not initialized"
    return _DP

_EP: GroupCoordinator | None = None
def get_ep_group() -> GroupCoordinator:
    assert _EP is not None, "expert parallel group is not initialized."
    return _EP

_EPLB: GroupCoordinator | None = None
def get_eplb_group() -> GroupCoordinator:
    assert _EPLB is not None, "EPLB group is not initialized."
    return _EPLB

_PCP: GroupCoordinator | None = None
def get_pcp_group() -> GroupCoordinator:
    assert _PCP is not None, "prefill context parallel group is not initialized"
    return _PCP

@contextmanager
def graph_capture(device: torch.device):
    context = GraphCaptureContext(torch.cuda.Stream(device=device))
    with get_tp_group().graph_capture(context), get_pp_group().graph_capture(context):
        yield context

logger = init_logger(__name__)
_ENABLE_CUSTOM_ALL_REDUCE = True
def set_custom_all_reduce(enable: bool):
    global _ENABLE_CUSTOM_ALL_REDUCE
    _ENABLE_CUSTOM_ALL_REDUCE = enable

def _init_elastic_ep_world(config, local_rank: int, backend: str, rank: int, world_size: int) -> None:
    from vllm.distributed.stateless_coordinator import StatelessGroupCoordinator
    global _WORLD, _NODE_COUNT
    parallel_config = config.parallel_config
    global_rank = parallel_config.data_parallel_rank * world_size + rank
    global_world_size = parallel_config.world_size_across_dp
    all_ranks = list(range(global_world_size))
    group_ranks = [all_ranks[i : i + 1] for i in range(global_world_size)]
    if global_rank in all_ranks: group_ranks = [all_ranks]
    coord_store = get_cached_tcp_store_client(parallel_config.data_parallel_master_ip, parallel_config._coord_store_port)
    world = StatelessGroupCoordinator(group_ranks=group_ranks, local_rank=local_rank, torch_distributed_backend=backend, use_device_communicator=False, group_name="world", host=parallel_config.data_parallel_master_ip, coord_store=coord_store, global_rank=global_rank, global_world_size=global_world_size)
    _NODE_COUNT = _node_count(world.tcp_store_group)
    _WORLD = world

def init_distributed_environment(world_size: int = -1, rank: int = -1, distributed_init_method: str = "env://", local_rank: int = -1, backend: str = "nccl", timeout: timedelta | None = None):
    from vllm.config import get_current_vllm_config_or_none
    config = get_current_vllm_config_or_none()
    enable_elastic_ep = config is not None and config.parallel_config.enable_elastic_ep
    if config is not None and config.parallel_config.distributed_executor_backend != "external_launcher" and (config.parallel_config.nnodes > 1 or config.parallel_config.data_parallel_size > 1) and not enable_elastic_ep:
        parallel_config = config.parallel_config
        rank = parallel_config.data_parallel_rank * world_size + rank
        world_size = parallel_config.world_size_across_dp
        if parallel_config.nnodes > 1:
            ip, port = parallel_config.master_addr, parallel_config.master_port
        else:
            ip, port = parallel_config.data_parallel_master_ip, parallel_config.get_next_dp_init_port()
        distributed_init_method = get_distributed_init_method(ip, port)

    if not torch.distributed.is_initialized():
        if not torch.distributed.is_backend_available(backend):
            backend = "gloo"
        torch.distributed.init_process_group(backend=backend, init_method=distributed_init_method, world_size=world_size, rank=rank, timeout=timeout)
        if enable_elastic_ep:
            tp_pp_cpu_group = torch.distributed.new_group(backend="gloo", timeout=timeout)
            if _node_count(tp_pp_cpu_group) > 1:
                raise RuntimeError("Elastic EP is not yet supported with multi-node TP/PP")

    if local_rank == -1:
        local_rank = envs.LOCAL_RANK if distributed_init_method == "env://" else rank
        
    global _WORLD, _NODE_COUNT, _INNER_DP_WORLD
    if enable_elastic_ep:
        _init_elastic_ep_world(config, local_rank, backend, rank, world_size)
        return
        
    if _WORLD is None:
        ranks = list(range(torch.distributed.get_world_size()))
        _WORLD = init_world_group(ranks, local_rank, backend)
        _NODE_COUNT = config.parallel_config.nnodes if config is not None and config.parallel_config.nnodes > 1 else _node_count(_WORLD.cpu_group)
        
    if config is not None and config.parallel_config.nnodes_within_dp > 1:
        parallel_config = config.parallel_config
        if parallel_config.data_parallel_size > 1:
            world_size_inner_dp = parallel_config.world_size
            group_ranks = [[dp_rank * world_size_inner_dp + i for i in range(world_size_inner_dp)] for dp_rank in range(parallel_config.data_parallel_size)]
            _INNER_DP_WORLD = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="inner_dp_world", use_device_communicator=False)
        else:
            _INNER_DP_WORLD = _WORLD

def initialize_model_parallel(tensor_model_parallel_size: int = 1, pipeline_model_parallel_size: int = 1, prefill_context_model_parallel_size: int = 1, decode_context_model_parallel_size: int | None = 1, backend: str | None = None) -> None:
    assert torch.distributed.is_initialized()
    from vllm.config import get_current_vllm_config
    config = get_current_vllm_config()
    data_parallel_size = config.parallel_config.data_parallel_size
    enable_elastic_ep = config.parallel_config.enable_elastic_ep
    parallel_config = config.parallel_config
    coord_store: Store | None = None
    
    if enable_elastic_ep:
        coord_store = get_cached_tcp_store_client(parallel_config.data_parallel_master_ip, parallel_config._coord_store_port)
        world_size = get_world_group().world_size
        rank = get_world_group().rank
        backend = backend or "nccl"
        tp_pp_pcp_size = tensor_model_parallel_size * pipeline_model_parallel_size * prefill_context_model_parallel_size
        local_all_ranks = torch.arange(tp_pp_pcp_size).reshape(pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)
    else:
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
        backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    all_ranks = torch.arange(world_size).reshape(-1, data_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, tensor_model_parallel_size)

    global _TP
    group_ranks = all_ranks.view(-1, tensor_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.view(-1, tensor_model_parallel_size).unbind(0)]
    _TP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="tp")

    global _DCP
    group_ranks = all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.reshape(-1, decode_context_model_parallel_size).unbind(0)]
    _DCP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, use_message_queue_broadcaster=True, group_name="dcp")

    global _PCP
    group_ranks = all_ranks.transpose(3, 4).reshape(-1, prefill_context_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.transpose(1, 2).reshape(-1, prefill_context_model_parallel_size).unbind(0)]
    _PCP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="pcp")

    global _PP
    group_ranks = all_ranks.transpose(2, 4).reshape(-1, pipeline_model_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        group_ranks = [x.tolist() for x in local_all_ranks.transpose(0, 2).reshape(-1, pipeline_model_parallel_size).unbind(0)]
    _PP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="pp")

    global _DP
    group_ranks = all_ranks.transpose(1, 4).reshape(-1, data_parallel_size).unbind(0)
    group_ranks = [x.tolist() for x in group_ranks]
    if enable_elastic_ep:
        _DP = _init_stateless_group(group_ranks, "dp", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store)
    else:
        _DP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="dp")

    global _EP
    if config.model_config is None or config.model_config.is_moe:
        group_ranks = all_ranks.transpose(1, 2).reshape(-1, data_parallel_size * prefill_context_model_parallel_size * tensor_model_parallel_size).unbind(0)
        group_ranks = [x.tolist() for x in group_ranks]
        if enable_elastic_ep:
            _EP = _init_stateless_group(group_ranks, "ep", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store)
        else:
            _EP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="ep")

        global _EPLB
        if config is not None and config.parallel_config is not None and config.parallel_config.enable_eplb:
            if enable_elastic_ep:
                _EPLB = _init_stateless_group(group_ranks, "eplb", parallel_config.data_parallel_master_ip, backend, coord_store=coord_store)
            else:
                _EPLB = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend, group_name="eplb")

    logger.info_once("rank %s in world size %s is assigned as DP rank %s, PP rank %s, PCP rank %s, TP rank %s, EP rank %s, EPLB rank %s",
                     rank, world_size, _DP.rank_in_group, _PP.rank_in_group, _PCP.rank_in_group, _TP.rank_in_group,
                     _EP.rank_in_group if _EP is not None else "N/A", _EPLB.rank_in_group if _EPLB is not None else "N/A")

def ensure_model_parallel_initialized(tensor_model_parallel_size: int, pipeline_model_parallel_size: int, prefill_context_model_parallel_size: int = 1, decode_context_model_parallel_size: int | None = 1, backend: str | None = None) -> None:
    world_group = get_world_group()
    backend = backend or torch.distributed.get_backend(world_group.device_group)
    if not model_parallel_is_initialized():
        initialize_model_parallel(tensor_model_parallel_size, pipeline_model_parallel_size, prefill_context_model_parallel_size, decode_context_model_parallel_size, backend)
        return
    assert get_tensor_model_parallel_world_size() == tensor_model_parallel_size
    assert get_pp_group().world_size == pipeline_model_parallel_size
    assert get_pcp_group().world_size == prefill_context_model_parallel_size

def prepare_communication_buffer_for_model(model: torch.nn.Module):
    if _TP is not None: _TP.prepare_communication_buffer_for_model(model)
    if _PCP is not None: _PCP.prepare_communication_buffer_for_model(model)
    if _PP is not None: _PP.prepare_communication_buffer_for_model(model)
    if _DP is not None: _DP.prepare_communication_buffer_for_model(model)
    if _EP is not None: _EP.prepare_communication_buffer_for_model(model)
    if _EPLB is not None: _EPLB.prepare_communication_buffer_for_model(model)

def model_parallel_is_initialized():
    return _TP is not None and _PP is not None

_TP_STATE_PATCHED = False
@contextmanager
def patch_tensor_parallel_group(tp_group: GroupCoordinator):
    global _TP_STATE_PATCHED
    _TP_STATE_PATCHED = True
    old_tp_group = get_tp_group()
    global _TP
    _TP = tp_group
    try: yield
    finally:
        _TP_STATE_PATCHED = False
        _TP = old_tp_group

def get_tensor_model_parallel_world_size() -> int: return get_tp_group().world_size
def get_tensor_model_parallel_rank() -> int: return get_tp_group().rank_in_group
def get_decode_context_model_parallel_world_size() -> int: return get_dcp_group().world_size
def get_decode_context_model_parallel_rank() -> int: return get_dcp_group().rank_in_group
def get_node_count() -> int:
    assert _NODE_COUNT is not None
    return _NODE_COUNT

def destroy_model_parallel():
    global _TP, _DCP, _PCP, _PP, _DP, _EP, _EPLB
    for g in (_TP, _DCP, _PCP, _PP, _DP, _EP, _EPLB):
        if g: g.destroy()
    _TP = _DCP = _PCP = _PP = _DP = _EP = _EPLB = None

def destroy_distributed_environment():
    global _WORLD, _NODE_COUNT
    if _WORLD: _WORLD.destroy()
    _WORLD = None
    _NODE_COUNT = None
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()

def cleanup_dist_env_and_memory(shutdown_ray: bool = False):
    envs.disable_envs_cache()
    gc.unfreeze()
    destroy_model_parallel()
    destroy_distributed_environment()
    if shutdown_ray:
        import ray
        ray.shutdown()
    gc.collect()
    torch.cuda.empty_cache()
    try:
        torch._C._host_emptyCache()
    except AttributeError:
        pass

def in_the_same_node_as(pg: ProcessGroup | StatelessProcessGroup, source_rank: int = 0) -> list[bool]:
    if isinstance(pg, ProcessGroup):
        rank = torch.distributed.get_rank(group=pg)
        world_size = torch.distributed.get_world_size(group=pg)
        ranks = torch.distributed.get_process_group_ranks(pg)
    else:
        rank = pg.rank
        world_size = pg.world_size
        ranks = list(range(world_size))

    is_in_the_same_node = torch.tensor([0] * world_size, dtype=torch.int32, device="cpu")
    magic_message = b"magic_message"
    shm = None

    try:
        with contextlib.suppress(OSError):
            if rank == source_rank:
                shm = shared_memory.SharedMemory(create=True, size=128)
                shm.buf[: len(magic_message)] = magic_message
                if isinstance(pg, ProcessGroup):
                    torch.distributed.broadcast_object_list([shm.name], src=ranks[source_rank], group=pg)
                else:
                    pg.broadcast_obj(shm.name, src=source_rank)
                is_in_the_same_node[rank] = 1
            else:
                if isinstance(pg, ProcessGroup):
                    recv = [None]
                    torch.distributed.broadcast_object_list(recv, src=ranks[source_rank], group=pg)
                    name = recv[0]
                else:
                    name = pg.broadcast_obj(None, src=source_rank)
                with patch("multiprocessing.resource_tracker.register", lambda *args, **kwargs: None):
                    shm = shared_memory.SharedMemory(name=name)
                if shm.buf[: len(magic_message)] == magic_message:
                    is_in_the_same_node[rank] = 1
    except Exception as e:
        logger.error("Error ignored in is_in_the_same_node: %s", e)
    finally:
        if shm: shm.close()

    if isinstance(pg, ProcessGroup): torch.distributed.barrier(group=pg)
    else: pg.barrier()

    with contextlib.suppress(OSError):
        if rank == source_rank and shm: shm.unlink()

    if isinstance(pg, ProcessGroup):
        torch.distributed.all_reduce(is_in_the_same_node, group=pg)
        aggregated_data = is_in_the_same_node
    else:
        aggregated_data = torch.zeros_like(is_in_the_same_node)
        for i in range(world_size):
            rank_data = pg.broadcast_obj(is_in_the_same_node, src=i)
            aggregated_data += rank_data
    return [x == 1 for x in aggregated_data.tolist()]

def is_global_first_rank() -> bool:
    try:
        if _WORLD is not None: return _WORLD.is_first_rank
        if not torch.distributed.is_initialized(): return True
        return torch.distributed.get_rank() == 0
    except Exception: return True

def is_local_first_rank() -> bool:
    try:
        if _WORLD is not None: return _WORLD.local_rank == 0
        if not torch.distributed.is_initialized(): return True
        try: return int(envs.LOCAL_RANK) == 0
        except Exception: return torch.distributed.get_rank() == 0
    except Exception: return True

def _node_count(pg: ProcessGroup | StatelessProcessGroup) -> int:
    world_size = torch.distributed.get_world_size(group=pg) if isinstance(pg, ProcessGroup) else pg.world_size
    if world_size == 1: return 1
    node_assignment = [0] * world_size
    next_node_id = 0
    for current_rank in range(world_size):
        if node_assignment[current_rank] != 0: continue
        next_node_id += 1
        node_assignment[current_rank] = next_node_id
        same_node_flags = in_the_same_node_as(pg, current_rank)
        for other_rank, is_same_node in enumerate(same_node_flags):
            if is_same_node and node_assignment[other_rank] == 0:
                node_assignment[other_rank] = next_node_id
    return next_node_id
