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
    fp8_w8a8_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
)
from vllm.platforms import current_platform

logger = init_logger(__name__)


class Fp8MoeBackend(Enum):
    TRITON = "TRITON"
    BATCHED_TRITON = "BATCHED_TRITON"
    VLLM_CUTLASS = "VLLM_CUTLASS"
    BATCHED_VLLM_CUTLASS = "BATCHED_VLLM_CUTLASS"


def _get_priority_backends(
    moe_config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
) -> list[Fp8MoeBackend]:
    return [
        Fp8MoeBackend.VLLM_CUTLASS,
        Fp8MoeBackend.TRITON,
        Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
        Fp8MoeBackend.BATCHED_TRITON,
    ]


def backend_to_kernel_cls(
    backend: Fp8MoeBackend,
) -> list[type[mk.FusedMoEExperts]]:
    if backend == Fp8MoeBackend.TRITON:
        from vllm.model_executor.layers.fused_moe.fused_moe import TritonExperts
        return [TritonExperts]

    elif backend == Fp8MoeBackend.BATCHED_TRITON:
        from vllm.model_executor.layers.fused_moe.fused_batched_moe import (
            BatchedTritonExperts,
        )
        return [BatchedTritonExperts]

    elif backend == Fp8MoeBackend.VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.triton_cutlass_moe import (
            TritonOrCutlassExperts,
        )
        return [TritonOrCutlassExperts]

    elif backend == Fp8MoeBackend.BATCHED_VLLM_CUTLASS:
        from vllm.model_executor.layers.fused_moe.cutlass_moe import (
            CutlassBatchedExpertsFp8,
        )
        return [CutlassBatchedExpertsFp8]

    else:
        raise ValueError(f"Unknown FP8 MoE backend: {backend.value}")


def map_fp8_backend(runner_backend: MoEBackend) -> Fp8MoeBackend:
    """Map user's MoEBackend to Fp8MoeBackend."""
    mapping = {
        "triton": Fp8MoeBackend.TRITON,
        "cutlass": Fp8MoeBackend.VLLM_CUTLASS,
    }
    if backend := mapping.get(runner_backend):
        return backend
    raise ValueError(
        f"moe_backend='{runner_backend}' is not supported for FP8 MoE. "
        f"Expected one of {list(mapping.keys())}."
    )


def select_fp8_moe_backend(
    config: FusedMoEConfig,
    weight_key: QuantKey | None,
    activation_key: QuantKey | None,
    allow_vllm_cutlass: bool = False,
) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts] | None]:
    if config.is_lora_enabled:
        return Fp8MoeBackend.TRITON, backend_to_kernel_cls(Fp8MoeBackend.TRITON)[0]

    AVAILABLE_BACKENDS = _get_priority_backends(config, weight_key, activation_key)

    activation_format = (
        mk.FusedMoEActivationFormat.BatchedExperts
        if config.moe_parallel_config.use_batched_activation_format
        else mk.FusedMoEActivationFormat.Standard
    )

    def _make_log_backend(backend: Fp8MoeBackend):
        available_backend_strs = [b.value for b in AVAILABLE_BACKENDS]
        return (
            f"Using {backend.value} Fp8 MoE backend out "
            f"of potential backends: {available_backend_strs}."
        )

    def _make_log_unsupported(backend: Fp8MoeBackend, reason: str | None) -> str:
        if reason:
            return (
                f"FP8 MoE backend {backend.value} does not support the "
                f"deployment configuration since {reason}."
            )
        else:
            return (
                f"FP8 MoE backend '{backend.value}' does not support the "
                "deployment configuration."
            )

    def _return_or_raise(
        backend: Fp8MoeBackend,
        config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[Fp8MoeBackend, type[mk.FusedMoEExperts]]:
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
        requested_backend = map_fp8_backend(runner_backend)
        # For batched activation format, use batched variants if available.
        if activation_format == mk.FusedMoEActivationFormat.BatchedExperts:
            if requested_backend == Fp8MoeBackend.TRITON:
                requested_backend = Fp8MoeBackend.BATCHED_TRITON
            elif requested_backend == Fp8MoeBackend.VLLM_CUTLASS:
                requested_backend = Fp8MoeBackend.BATCHED_VLLM_CUTLASS

        if (
            requested_backend
            in [
                Fp8MoeBackend.VLLM_CUTLASS,
                Fp8MoeBackend.BATCHED_VLLM_CUTLASS,
            ]
            and not allow_vllm_cutlass
        ):
            raise ValueError(
                "vLLM CUTLASS FP8 MoE backend is disabled for this configuration."
            )

        return _return_or_raise(
            requested_backend, config, weight_key, activation_key, activation_format
        )

    if not allow_vllm_cutlass:
        if Fp8MoeBackend.VLLM_CUTLASS in AVAILABLE_BACKENDS:
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.VLLM_CUTLASS)
        if Fp8MoeBackend.BATCHED_VLLM_CUTLASS in AVAILABLE_BACKENDS:
            AVAILABLE_BACKENDS.remove(Fp8MoeBackend.BATCHED_VLLM_CUTLASS)

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

    if current_platform.is_cuda():
        raise NotImplementedError(
            "No FP8 MoE backend supports the deployment configuration."
        )

    # OOT plugin case
    return None, None  # type: ignore[return-value]


def convert_to_fp8_moe_kernel_format(
    fp8_backend: Fp8MoeBackend,
    layer: torch.nn.Module,
    w13: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor | None,
    w2_input_scale: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return w13, w2, w13_scale, w2_scale


def make_fp8_moe_quant_config(
    fp8_backend: Fp8MoeBackend,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    a1_scale: torch.Tensor | None,
    a2_scale: torch.Tensor | None,
    block_shape: list[int] | None = None,
    per_act_token_quant: bool = False,
    per_out_ch_quant: bool = False,
) -> FusedMoEQuantConfig:
    return fp8_w8a8_moe_quant_config(
        w1_scale=w1_scale,
        w2_scale=w2_scale,
        a1_scale=a1_scale,
        a2_scale=a2_scale,
        block_shape=block_shape,
        per_act_token_quant=per_act_token_quant,
        per_out_ch_quant=per_out_ch_quant,
    )


def make_fp8_moe_kernel(
    moe_quant_config: FusedMoEQuantConfig,
    moe_config: FusedMoEConfig,
    experts_cls: type[mk.FusedMoEExperts],
    fp8_backend: Fp8MoeBackend,
    routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    shared_experts: torch.nn.Module | None = None,
) -> mk.FusedMoEKernel:
    # Create Prepare/Finalize.
    prepare_finalize = maybe_make_prepare_finalize(
        moe=moe_config,
        quant_config=moe_quant_config,
        routing_tables=routing_tables,
        allow_new_interface=True,
        use_monolithic=False,
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

    # NOTE(rob): we only want the mk to control the shared_expert
    # if using all2all (for SBO).
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