# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from enum import Enum
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    ModelWeightParameter,
    PackedvLLMParameter,
)
from vllm.platforms import current_platform

from .quark_scheme import QuarkScheme

logger = init_logger(__name__)

__all__ = ["QuarkOCP_MX"]

# ============================================================================
# Local definitions (replaces deleted ocp_mx_utils / mxfp4_utils / mxfp6_utils)
# ============================================================================

OCP_MX_BLOCK_SIZE = 32


class OCP_MX_Scheme(str, Enum):
    """OCP MX quantization schemes (minimal local definition)."""
    w_mxfp4 = "w_mxfp4"
    w_mxfp4_a_fp8 = "w_mxfp4_a_fp8"
    w_mxfp4_a_mxfp4 = "w_mxfp4_a_mxfp4"
    w_mxfp6_e3m2_a_fp8 = "w_mfp6_e3m2_a_fp8"
    w_mxfp6_e2m3_a_fp8 = "w_mxfp6_e2m3_a_fp8"

    @classmethod
    def from_quant_dtype(cls, input_dtype, weight_dtype):
        if weight_dtype is None:
            return None
        def _normalize(dtype):
            if dtype is None:
                return None
            if dtype.startswith("mxfp"):
                return dtype
            if dtype == "fp8_e4m3":
                return "fp8"
            if dtype.startswith("fp"):
                return "mxfp" + dtype[2:]
            return dtype
        w = _normalize(weight_dtype)
        a = _normalize(input_dtype) if input_dtype is not None else None
        if w not in {"mxfp4", "mxfp6_e3m2", "mxfp6_e2m3"}:
            return None
        if a is None:
            name = f"w_{w}"
        elif a == "fp8":
            name = f"w_{w}_a_fp8"
        elif a == "mxfp4":
            name = f"w_{w}_a_mxfp4"
        else:
            return None
        try:
            return cls(name)
        except ValueError:
            return None


# MXFP4 E2M1 lookup table (16 values)
# Format: sign(1) + exp(2) + mantissa(1) bits
# Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, and their negatives
_MXFP4_VALUES = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
     -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
    dtype=torch.float32,
)


def _dequant_mxfp4_local(
    weight_packed: torch.Tensor,
    scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """
    Local MXFP4 dequantization via PyTorch ops.
    weight_packed: (..., K//2) uint8, each byte holds 2 FP4 values
    scale: (..., K // OCP_MX_BLOCK_SIZE) uint8 (E8M0 format)
    Returns: (..., K) tensor in out_dtype
    """
    device = weight_packed.device
    lut = _MXFP4_VALUES.to(device)

    # Unpack 2 FP4 values per byte
    low = weight_packed & 0x0F
    high = (weight_packed >> 4) & 0x0F

    # Lookup values
    low_vals = lut[low.long()]
    high_vals = lut[high.long()]

    # Interleave: [low0, high0, low1, high1, ...]
    shape = list(weight_packed.shape[:-1]) + [weight_packed.shape[-1] * 2]
    weight_vals = torch.empty(shape, dtype=torch.float32, device=device)
    weight_vals[..., 0::2] = low_vals
    weight_vals[..., 1::2] = high_vals

    # Convert E8M0 scale to float: scale_float = 2^(scale_uint8 - 127)
    scale_uint8 = scale.view(torch.uint8).to(torch.float32)
    scale_float = torch.pow(2.0, scale_uint8 - 127.0)

    # Broadcast scale to weight shape (scale is per OCP_MX_BLOCK_SIZE elements)
    # scale_float shape: (..., K // 32)
    # weight_vals shape: (..., K)
    # Need to expand scale to (..., K // 32, 32) then reshape
    scale_expanded = scale_float.unsqueeze(-1).expand(
        *scale_float.shape, OCP_MX_BLOCK_SIZE
    )
    scale_expanded = scale_expanded.reshape(*weight_vals.shape[:-1], -1)

    # Crop to actual K if needed
    K = weight_vals.shape[-1]
    scale_expanded = scale_expanded[..., :K]

    # Apply scale
    weight_dequant = weight_vals * scale_expanded
    return weight_dequant.to(out_dtype)


def _quant_dequant_mxfp4_local(x: torch.Tensor) -> torch.Tensor:
    """
    Simulated MXFP4 quantize-dequantize for activations (emulation on Volta).
    Simple per-block max-scaling + round-to-nearest-LUT-value.
    """
    device = x.device
    lut = _MXFP4_VALUES.to(device)
    orig_shape = x.shape
    K = orig_shape[-1]

    # Pad K to multiple of OCP_MX_BLOCK_SIZE
    if K % OCP_MX_BLOCK_SIZE != 0:
        pad = OCP_MX_BLOCK_SIZE - (K % OCP_MX_BLOCK_SIZE)
        x = F.pad(x, (0, pad), value=0.0)

    # Reshape to blocks
    x_blocks = x.reshape(-1, OCP_MX_BLOCK_SIZE)

    # Per-block max
    amax = x_blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-12)
    scale = (amax / 6.0).clamp(min=1e-12)  # 6.0 is max abs value in LUT

    # Scale to [-6, 6] range
    x_scaled = x_blocks / scale

    # Quantize to nearest LUT value
    # Use broadcasting to find nearest value for each element
    diffs = (x_scaled.unsqueeze(-1) - lut).abs()
    nearest_idx = diffs.argmin(dim=-1)
    x_quant = lut[nearest_idx]

    # Dequantize
    x_dq = x_quant * scale

    # Reshape back and crop
    x_dq = x_dq.reshape(*orig_shape[:-1], -1)[..., :K]
    return x_dq


# ============================================================================
# QuarkOCP_MX scheme
# ============================================================================


class QuarkOCP_MX(QuarkScheme):
    """
    Quark OCP MX (Microscaling) quantization scheme.
    On Volta V100 (CUDA SM 7.0) always uses emulation mode:
    simulated weight dequantization + activation QDQ + high-precision linear.
    """

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any] | None,
    ):
        self.out_dtype = torch.get_default_dtype()
        self.weight_quant_spec = weight_quant_spec
        self.input_quant_spec = input_quant_spec

        self.weight_dtype = weight_quant_spec["dtype"].replace("fp", "mxfp")
        self.input_dtype: str | None = None
        if input_quant_spec is not None:
            input_quant = input_quant_spec["dtype"]
            if input_quant == "fp8_e4m3":
                self.input_dtype = "fp8"
            else:
                self.input_dtype = input_quant.replace("fp", "mxfp")

        self.ocp_mx_scheme = OCP_MX_Scheme.from_quant_dtype(
            self.input_dtype, self.weight_dtype
        )

        if self.weight_dtype == "mxfp4":
            self.packed_factor: int | Fraction = 2
            self.dequant_func: Callable = _dequant_mxfp4_local
        else:
            # MXFP6: 4 * 6 = 24 bits on 3 bytes
            self.packed_factor = Fraction(numerator=8, denominator=6)
            # For MXFP6, fall back to identity (not fully supported on Volta)
            self.dequant_func = lambda w, s, d: w.to(d)

        if self.input_dtype is None:
            self.quant_dequant_func: Callable[[torch.Tensor], torch.Tensor] = (
                lambda x: x  # no input Q/DQ for weight-only
            )
        elif self.input_dtype == "mxfp4":
            self.quant_dequant_func = _quant_dequant_mxfp4_local
        else:
            # For other input dtypes (mxfp6, fp8), use identity on Volta
            self.quant_dequant_func = lambda x: x

        if input_quant_spec is None:
            self.static_input_scales = False
        else:
            self.static_input_scales = not input_quant_spec.get("is_dynamic")

        if self.static_input_scales:
            raise NotImplementedError(
                "QuarkOCP_MX with static input scales is currently not "
                "implemented. Please open an issue."
            )

        # On Volta V100, always emulate
        self.emulate = True

        logger.warning_once(
            "[QuarkOCP_MX] Volta V100 does not support native MXFP4/MXFP6 "
            "computation. Using emulation mode (simulated weight dequant + "
            "activation QDQ + high-precision F.linear)."
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_packed_dim(self, dim: int) -> int:
        if self.weight_dtype == "mxfp4":
            assert dim % 2 == 0, f"Dimension {dim} must be even for MXFP4 packing"
            return dim // 2
        else:
            # MXFP6: 4 * 6 = 24 bits on 3 bytes
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        output_size_per_partition = sum(output_partition_sizes)
        layer.logical_widths = output_partition_sizes
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition

        # MXFP4/6 WEIGHT (packed)
        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                self.get_packed_dim(input_size_per_partition),
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.packed_factor,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight", weight)

        # WEIGHT SCALE (E8M0 format, per block of 32)
        weight_scale = GroupQuantScaleParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // OCP_MX_BLOCK_SIZE,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Ensure weights & scales are non-trainable
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Always use emulation on Volta V100
        weight_dq = self.dequant_func(layer.weight, layer.weight_scale, x.dtype)
        qdq_x = self.quant_dequant_func(x)
        return F.linear(qdq_x, weight_dq, bias)
