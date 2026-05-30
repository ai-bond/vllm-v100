# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup, ReduceOp

from vllm.distributed.device_communicators.pynccl_wrapper import (
    NCCLLibrary,
    buffer_type,
    cudaStream_t,
    ncclComm_t,
    ncclDataTypeEnum,
    ncclRedOpTypeEnum,
    ncclUniqueId,
)
from vllm.distributed.utils import StatelessProcessGroup
from vllm.logger import init_logger
from vllm.utils.torch_utils import current_stream

logger = init_logger(__name__)


class PyNcclCommunicator:
    def __init__(
        self,
        group: ProcessGroup | StatelessProcessGroup,
        device: int | str | torch.device,
        library_path: str | None = None,
    ):
        """
        Args:
            group: the process group to work on (Gloo/CPU group or Stateless).
            device: the CUDA device to bind the PyNcclCommunicator to.
            library_path: the path to the NCCL library.
        """
        if not isinstance(group, StatelessProcessGroup):
            assert dist.is_initialized()
            assert dist.get_backend(group) != dist.Backend.NCCL, (
                "PyNcclCommunicator should be attached to a non-NCCL group."
            )
            self.rank = dist.get_rank(group)
            self.world_size = dist.get_world_size(group)
        else:
            self.rank = group.rank
            self.world_size = group.world_size

        self.group = group

        # if world_size == 1, no need to create communicator
        if self.world_size == 1:
            self.available = False
            self.disabled = True
            return

        try:
            self.nccl = NCCLLibrary(library_path)
        except Exception:
            # disable because of missing NCCL library
            self.available = False
            self.disabled = True
            return

        self.available = True
        self.disabled = False

        self.nccl_version = self.nccl.ncclGetRawVersion()
        if self.rank == 0:
            self.unique_id = self.nccl.ncclGetUniqueId()
            logger.info_once(
                "vLLM is using nccl==%s", self.nccl.ncclGetVersion(), scope="local"
            )
        else:
            self.unique_id = ncclUniqueId()

        # Broadcast the unique ID to all ranks
        if not isinstance(group, StatelessProcessGroup):
            tensor = torch.ByteTensor(list(self.unique_id.internal))
            ranks = dist.get_process_group_ranks(group)
            dist.broadcast(tensor, src=ranks[0], group=group)
            byte_list = tensor.tolist()
            for i, byte in enumerate(byte_list):
                self.unique_id.internal[i] = byte
        else:
            self.unique_id = group.broadcast_obj(self.unique_id, src=0)

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)

        assert isinstance(device, torch.device)
        self.device = device

        # nccl communicator and stream will use this device
        # FIXED: replaced non-existent torch.accelerator.device_index with torch.cuda.device
        with torch.cuda.device(self.device.index):
            self.comm: ncclComm_t = self.nccl.ncclCommInitRank(
                self.world_size, self.unique_id, self.rank
            )

            stream = current_stream()
            # A small all_reduce for warmup.
            data = torch.zeros(1, device=self.device)
            self.all_reduce(data)
            stream.synchronize()
            del data

    def destroy(self):
        if self.available and not self.disabled:
            with torch.cuda.device(self.device.index):
                self.nccl.ncclCommDestroy(self.comm)
            self.available = False
            self.disabled = True

    def _get_nccl_dtype(self, tensor: torch.Tensor):
        # Map FP8 types to uint8 for NCCL compatibility (useful for AWQ/quantized weights)
        if tensor.dtype in [
            torch.float8_e5m2,
            torch.float8_e4m3fn,
            torch.float8_e4m3fnuz,
            torch.float8_e5m2fnuz,
        ]:
            return ncclDataTypeEnum.from_torch(torch.uint8)
        return ncclDataTypeEnum.from_torch(tensor.dtype)

    def all_reduce(
        self,
        in_tensor: torch.Tensor,
        out_tensor: torch.Tensor | None = None,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ) -> torch.Tensor:
        if self.disabled:
            return None
        assert in_tensor.device == self.device, (
            f"NCCL device mismatch: expected {self.device}, got {in_tensor.device}"
        )

        if out_tensor is None:
            out_tensor = torch.empty_like(in_tensor)

        if stream is None:
            stream = current_stream()

        self.nccl.ncclAllReduce(
            buffer_type(in_tensor.data_ptr()),
            buffer_type(out_tensor.data_ptr()),
            in_tensor.numel(),
            ncclDataTypeEnum.from_torch(in_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )
        return out_tensor

    def all_gather(
        self, output_tensor: torch.Tensor, input_tensor: torch.Tensor, stream=None
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device
        if stream is None:
            stream = current_stream()

        self.nccl.ncclAllGather(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            input_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def all_gatherv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        stream=None,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device
        if stream is None:
            stream = current_stream()
        assert output_tensor.shape[0] == sum(sizes)

        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            dst_slice = output_tensor[split_offset : split_offset + split_size]
            self.nccl.ncclBroadcast(
                buffer_type(input_tensor.data_ptr()),
                buffer_type(dst_slice.data_ptr()),
                dst_slice.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def reduce_scatter(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device
        if stream is None:
            stream = current_stream()

        self.nccl.ncclReduceScatter(
            buffer_type(input_tensor.data_ptr()),
            buffer_type(output_tensor.data_ptr()),
            output_tensor.numel(),
            ncclDataTypeEnum.from_torch(input_tensor.dtype),
            ncclRedOpTypeEnum.from_torch(op),
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def reduce_scatterv(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        sizes: list[int],
        op: ReduceOp = ReduceOp.SUM,
        stream=None,
    ):
        if self.disabled:
            return
        assert input_tensor.device == self.device
        if stream is None:
            stream = current_stream()

        split_offset = 0
        self.nccl.ncclGroupStart()
        for root, split_size in enumerate(sizes):
            chunk = input_tensor[split_offset : split_offset + split_size, ...]
            self.nccl.ncclReduce(
                buffer_type(chunk.data_ptr()),
                buffer_type(output_tensor.data_ptr()),
                chunk.numel(),
                ncclDataTypeEnum.from_torch(input_tensor.dtype),
                ncclRedOpTypeEnum.from_torch(op),
                root,
                self.comm,
                cudaStream_t(stream.cuda_stream),
            )
            split_offset += split_size
        self.nccl.ncclGroupEnd()

    def send(self, tensor: torch.Tensor, dst: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device
        if stream is None:
            stream = current_stream()

        self.nccl.ncclSend(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            self._get_nccl_dtype(tensor),
            dst,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def recv(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device
        if stream is None:
            stream = current_stream()

        self.nccl.ncclRecv(
            buffer_type(tensor.data_ptr()),
            tensor.numel(),
            self._get_nccl_dtype(tensor),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def broadcast(self, tensor: torch.Tensor, src: int, stream=None):
        if self.disabled:
            return
        assert tensor.device == self.device
        if stream is None:
            stream = current_stream()

        if src == self.rank:
            sendbuff = buffer_type(tensor.data_ptr())
            recvbuff = buffer_type(tensor.data_ptr())
        else:
            sendbuff = buffer_type()
            recvbuff = buffer_type(tensor.data_ptr())

        self.nccl.ncclBroadcast(
            sendbuff,
            recvbuff,
            tensor.numel(),
            ncclDataTypeEnum.from_torch(tensor.dtype),
            src,
            self.comm,
            cudaStream_t(stream.cuda_stream),
        )

    def group_start(self):
        self.nccl.ncclGroupStart()

    def group_end(self):
        self.nccl.ncclGroupEnd()

    def batch_isend_irecv(self, p2p_ops: list, stream=None):
        if self.disabled:
            return
        if stream is None:
            stream = current_stream()

        self.group_start()
        for op in p2p_ops:
            if op.op is torch.distributed.isend:
                self.send(op.tensor, op.group_peer, stream)
            elif op.op is torch.distributed.irecv:
                self.recv(op.tensor, op.group_peer, stream)
        self.group_end()
