# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch.fx as fx

from vllm.config import VllmConfig
from vllm.logger import init_logger
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass

logger = init_logger(__name__)


class AllReduceFusionPass(VllmPatternMatcherPass):
    """
    AllReduce + RMSNorm/Quant Fusion Pass.
    
    NOTE: This feature relies on FlashInfer's fused allreduce kernels, which 
    require SM 9.0+ (Hopper/Blackwell). Since this fork targets Volta V100 (SM 7.0), 
    this pass is permanently disabled and acts as a no-op.
    """

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.disabled = True
        logger.debug_once(
            "AllReduceFusionPass is disabled on Volta V100 (SM 7.0). "
            "FlashInfer fused allreduce requires SM 9.0+."
        )

    def is_applicable_for_range(self, compile_range) -> bool:
        return False

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        # No-op for Volta V100
        return
