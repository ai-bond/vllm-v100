# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import atexit
import contextlib
from typing import Any

import torch

from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.logger import init_logger

logger = init_logger(__name__)

_mem_pool = None


def is_symmetric_memory_enabled() -> bool:
    """
    Disabled for Volta (sm_70) builds.
    """
    return False


def is_symmetric_memory_tensor(tensor: torch.Tensor) -> bool:
    return False


def set_graph_pool_id(graph_pool_id: Any) -> None:
    pass


def compile_nccl_allocator():
    pass


def get_nccl_mem_pool():
    return None


def _cleanup_nccl_mem_pool():
    global _mem_pool
    _mem_pool = None


atexit.register(_cleanup_nccl_mem_pool)


class nccl_symm_mem_context:
    def __init__(
        self,
        pynccl_comm: PyNcclCommunicator,
        disabled: bool = False,
    ):
        self.disabled = True
        self.pynccl_comm = None
        self._mem_pool_ctx: contextlib.AbstractContextManager[Any] = (
            contextlib.nullcontext()
        )
        self.is_graph_capture = None
        self.device = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass