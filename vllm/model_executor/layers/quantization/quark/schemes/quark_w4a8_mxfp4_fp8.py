# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from fractions import Fraction
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.model_executor.parameter import (
    GroupQuantScaleParameter,
    PackedvLLMParameter,
    PerTensorScaleParameter,
)
from vllm.platforms import current_platform
from .quark_scheme import QuarkScheme

logger = init_logger(__name__)

__all__ = ["QuarkW4A8_MXFP4_FP8"]

OCP_MX_BLOCK_SIZE = 32

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
    device = weight_packed.device
    lut = _MXFP4_VALUES.to(device)

    low = weight_packed & 0x0F
    high = (weight_packed >> 4) & 0x0F

    low_vals = lut[low.long()]
    high_vals = lut[high.long()]

    shape = list(weight_packed.shape[:-1]) + [weight_packed.shape[-1] * 2]
    weight_vals = torch.empty(shape, dtype=torch.float32, device=device)
    weight_vals[..., 0::2] = low_vals
    weight_vals[..., 1::2] = high_vals

    scale_uint8 = scale.view(torch.uint8).to(torch.float32)
    scale_float = torch.pow(2.0, scale_uint8 - 127.0)

    scale_expanded = scale_float.unsqueeze(-1).expand(*scale_float.shape, 32)
    scale_expanded = scale_expanded.reshape(*weight_vals.shape[:-1], -1)

    K = weight_vals.shape[-1]
    scale_expanded = scale_expanded[..., :K]

    weight_dequant = weight_vals * scale_expanded
    return weight_dequant.to(out_dtype)


class QuarkW4A8_MXFP4_FP8(QuarkScheme):

    def __init__(
        self,
        weight_quant_spec: dict[str, Any],
        input_quant_spec: dict[str, Any],
    ):
        self.out_dtype = None

        self.weight_dtype = "mxfp4"
        self.packed_factor: Fraction = Fraction(2, 1)
        self.weight_block_size = OCP_MX_BLOCK_SIZE

        self.is_static_input_scheme = not input_quant_spec.get("is_dynamic")
        self.input_qscheme = input_quant_spec.get("qscheme")

        self.fp8_min, self.fp8_max = get_fp8_min_max()
        self.fp8_dtype = current_platform.fp8_dtype()

        if not self.is_static_input_scheme:
            raise NotImplementedError(
                "Dynamic FP8 activation quantization is not yet supported "
                "for W4A8. The current implementation expects static per-tensor "
                "FP8 scales stored in the checkpoint."
            )

        # AITER не поддерживается на Volta V100
        self.use_aiter_kernel = False

        logger.warning_once(
            "[W4A8 MXFP4+FP8] Aiter Triton kernel not supported on Volta V100. "
            "Using emulation mode."
        )

    @classmethod
    def get_min_capability(cls) -> int:
        return 70

    def get_packed_dim(self, dim: int) -> int:
        assert dim % 2 == 0, f"Dimension {dim} must be even for MXFP4 packing"
        return dim // 2

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

        # MXFP4 WEIGHT (packed, 2 values per byte)
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
                input_size_per_partition // self.weight_block_size,
                dtype=torch.uint8,
            ),
            input_dim=1,
            output_dim=0,
            weight_loader=weight_loader,
        )
        layer.register_parameter("weight_scale", weight_scale)

        # INPUT SCALE (FP8 per-tensor static scale)
        if self.is_static_input_scheme:
            input_scale = PerTensorScaleParameter(
                data=torch.empty(
                    len(output_partition_sizes),
                    dtype=torch.float32,
                ),
                weight_loader=weight_loader,
            )
            input_scale[:] = torch.finfo(torch.float32).min
            layer.register_parameter("input_scale", input_scale)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        layer.weight = torch.nn.Parameter(layer.weight.data, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(
            layer.weight_scale.data, requires_grad=False
        )

        if self.is_static_input_scheme:
            input_scale = layer.input_scale.data
            if input_scale.numel() != 1:
                input_scale = input_scale.max()

            layer.input_scale = torch.nn.Parameter(
                torch.tensor(input_scale, dtype=torch.float32),
                requires_grad=False,
            )

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._apply_emulation(layer, x, bias)

    def _apply_emulation(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        weight_dq = _dequant_mxfp4_local(
            layer.weight,
            layer.weight_scale,
            x.dtype,
        )

        input_scale = layer.input_scale
        x_fp8 = (x / input_scale).clamp(self.fp8_min, self.fp8_max).to(self.fp8_dtype)
        x_dq = (x_fp8.to(x.dtype) * input_scale).to(x.dtype)

        return F.linear(x_dq, weight_dq, bias)