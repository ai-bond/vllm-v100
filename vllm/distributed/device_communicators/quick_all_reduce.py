# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)


class QuickAllReduce:
    """
    QuickAllReduce is a ROCm MI300 specific feature (CDNA3 architecture).
    """

    def __init__(self, group: ProcessGroup, device: int | str | torch.device) -> None:
        self.disabled = True
        logger.debug("QuickAllReduce is disabled.")

    def should_quick_allreduce(self, inp: torch.Tensor) -> bool:
        return False

    def quick_all_reduce(self, inp: torch.Tensor, *, out: torch.Tensor = None):
        raise NotImplementedError("QuickAllReduce is not supported on CUDA.")

    def close(self):
        pass

    def __del__(self):
        self.close()