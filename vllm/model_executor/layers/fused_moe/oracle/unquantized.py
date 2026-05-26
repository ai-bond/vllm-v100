# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
from torch.nn import Module

import vllm.envs as envs
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    MoEPrepareAndFinalizeNoDPEPModular,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class UnquantizedMoeBackend(Enum):
    TRITON = "TRITON"
    OOT = "OOT"


# NOTE(zyongye): Unsupported backend means backend
# that is not conform with Modular kernel format.
# We will directly call the kernel for those backend
UNSUPPORTED_BACKEND = [
    UnquantizedMoeBackend.OOT,
]


def map_unquantized_backend(runner_backend: MoEBackend) -> UnquantizedMoeBackend:
    """Map user's MoEBackend to UnquantizedMoeBackend."""
    mapping = {
        "triton": UnquantizedMoeBackend.TRITON,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for unquantized MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_unquantized_moe_backend(
    moe_config: FusedMoEConfig,
    use_ep: bool,
    use_dp: bool,
) -> UnquantizedMoeBackend:

    def _make_log_backend(backend: UnquantizedMoeBackend):
        return f"Using {backend.value} backend for Unquantized MoE"

    # Handle explicit moe_backend from user.
    runner_backend = moe_config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_unquantized_backend(runner_backend)
        logger.info_once(_make_log_backend(requested_backend), scope="local")
        return requested_backend

    if current_platform.is_out_of_tree():
        backend = UnquantizedMoeBackend.OOT
    else:
        backend = UnquantizedMoeBackend.TRITON

    logger.info_once(_make_log_backend(backend), scope="local")
    return backend


def convert_to_unquantized_kernel_format(
    unquantized_backend: UnquantizedMoeBackend,
    layer: Module,
    w13_weight: torch.Tensor | None = None,
    w2_weight: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return w13_weight, w2_weight


def make_unquantized_moe_kernel(
    backend: UnquantizedMoeBackend,
    quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
) -> mk.FusedMoEKernel | None:
    if backend in UNSUPPORTED_BACKEND:
        return None

    if backend == UnquantizedMoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe import TritonExperts

        kernel = mk.FusedMoEKernel(
            MoEPrepareAndFinalizeNoDPEPModular(),
            TritonExperts(
                moe_config=moe_config,
                quant_config=quant_config,
            ),
            inplace=not moe_config.disable_inplace,
        )
        return kernel

    return None