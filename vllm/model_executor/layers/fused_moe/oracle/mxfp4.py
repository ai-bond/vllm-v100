# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.config.kernel import MoEBackend
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    mxfp4_mxfp8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.mxfp4_utils import _swizzle_mxfp4
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kMxfp4Static,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Mxfp4MoeBackend(Enum):
    NONE = "NONE"
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"

def _get_priority_backends(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> list[Mxfp4MoeBackend]:
    return [
        Mxfp4MoeBackend.TRITON,
        Mxfp4MoeBackend.BATCHED_TRITON,
    ]


def backend_to_kernel_cls(
    backend: Mxfp4MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Mxfp4MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.gpt_oss_triton_kernels_moe import (
            OAITritonMxfp4ExpertsMonolithic,
            OAITritonExperts,
        )

        # NOTE: prefer Monolithic > Modular, so return Monolithic first.
        return [OAITritonMxfp4ExpertsMonolithic, OAITritonExperts]

    elif backend == Mxfp4MoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )

        return [BatchedTritonExperts]

    else:
        raise ValueError(f"Unknown MXFP4 MoE backend: {backend.value}")


def map_mxfp4_backend(runner_backend: MoEBackend) -> Mxfp4MoeBackend:
    """Map user's MoEBackend to Mxfp4MoeBackend."""
    mapping = {
        "triton": Mxfp4MoeBackend.TRITON,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for MXFP4 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_mxfp4_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts] | None]:
    if config.is_lora_enabled:
        return Mxfp4MoeBackend.TRITON, backend_to_kernel_cls(Mxfp4MoeBackend.TRITON)[0]

    AVAILABLE_BACKENDS = _get_priority_backends(config, weight_key, activation_key)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Mxfp4MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Mxfp4 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Mxfp4MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"MXFP4 MoE backend '{backend.value}' does not support the "
                f"deployment configuration since {reason}."
            )
        return (
            f"MXFP4 MoE backend '{backend.value}' does not support the "
            "deployment configuration."
        )

    def _return_or_raise(
        backend: Mxfp4MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Mxfp4MoeBackend, type[mk.FusedMoEExperts]]:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls, config, weight_key, activation_key, activation_format
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
        raise ValueError(_make_log_unsupported(backend, reason))

    # Handle explicit moe_backend from user.
    runner_backend = config.moe_backend
    if runner_backend != "auto":
        requested_backend = map_mxfp4_backend(runner_backend)
        # For batched activation format, use BATCHED_TRITON if requested TRITON.
        if (
            activation_format == mk.FusedMoEActivationFormat.BatchedExperts
            and requested_backend == Mxfp4MoeBackend.TRITON
        ):
            requested_backend = Mxfp4MoeBackend.BATCHED_TRITON

        return _return_or_raise(
            requested_backend, config, weight_key, activation_key, activation_format
        )

    # Select kernels in order of backend priority.
    for backend in AVAILABLE_BACKENDS:
        for k_cls in backend_to_kernel_cls(backend):
            supported, reason = k_cls.is_supported_config(
                k_cls,
                config,
                weight_key,
                activation_key,
                activation_format,
            )
            if supported:
                logger.info_once(_make_log_backend(backend), scope="local")
                return backend, k_cls
            else:
                logger.debug_once(
                    _make_log_unsupported(backend, reason), scope="local"
                )

    # OOT plugin case — allow returning NONE for out-of-tree backends.
    if current_platform.is_cuda():
        raise NotImplementedError(
            "No MXFP4 MoE backend supports the deployment configuration."
        )

    return Mxfp4MoeBackend.NONE, None


def convert_to_mxfp4_moe_kernel_format(
    mxfp4_backend: Mxfp4MoeBackend,
    layer: torch.nn.Module,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_weight_scale: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w13_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    "PrecisionConfig",
    "PrecisionConfig",
    torch.Tensor | None,
    torch.Tensor | None,
]:
    # Triton / Batched Triton path: swizzle MXFP4 scales for OAI kernels.
    w13_weight, w13_flex, w13_scale = _swizzle_mxfp4(
        w13_weight,
        w13_weight_scale,
    )
    w2_weight, w2_flex, w2_scale = _swizzle_mxfp4(
        w2_weight,
        w2_weight_scale,
    )

    from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig

    w13_precision_config = PrecisionConfig(
        weight_scale=w13_scale, flex_ctx=FlexCtx(rhs_data=w13_flex)
    )
    w2_precision_config = PrecisionConfig(
        weight_scale=w2_scale, flex_ctx=FlexCtx(rhs_data=w2_flex)
    )

    del layer.w13_weight
    del layer.w2_weight

    return (
        w13_weight,
        w2_weight,
        w13_precision_config,
        w2_precision_config,
        w13_bias,
        w2_bias,
    )


def make_mxfp4_moe_quant_config(
    mxfp4_backend: Mxfp4MoeBackend,
    w1_scale,
    w2_scale,
    w1_bias: torch.Tensor | None = None,
    w2_bias: torch.Tensor | None = None,
) -> FusedMoEQuantConfig | None:
    # Both supported backends use the same MXFP4 W4A16 config.
    return mxfp4_w4a16_moe_quant_config(
        w1_bias=w1_bias,
        w2_bias=w2_bias,
        w1_scale=w1_scale,
        w2_scale=w2_scale,
    )


def make_mxfp4_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    mxfp4_backend: Mxfp4MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    """Create a FusedMoEKernel for the specified MXFP4 backend."""
    is_monolithic = issubclass(experts_cls, mk.FusedMoEExpertsMonolithic)

    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=is_monolithic,
    )
    assert prepare_finalize is not None

    logger.info_once("Using %s", prepare_finalize.__class__.__name__, scope="local")

    # Create Experts.
    if prepare_finalize.activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
        max_num_tokens = prepare_finalize.max_num_tokens_per_rank()
        assert max_num_tokens is not None
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
            max_num_tokens=max_num_tokens,
            num_dispatchers=prepare_finalize.num_dispatchers(),
        )
    else:
        experts = experts_cls(
            moe_config=moe_config,
            quant_config=moe_quant_config,
        )

    kernel = mk.FusedMoEKernel(
        prepare_finalize,
        experts,
        shared_experts=(
            shared_experts
            if moe_config.moe_parallel_config.use_deepep_ll_kernels
            else None
        ),
        moe_parallel_config=moe_config.moe_parallel_config,
        inplace=(not moe_config.disable_inplace),
    )

    return kernel