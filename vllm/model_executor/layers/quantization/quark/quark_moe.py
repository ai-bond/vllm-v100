# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEConfig,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
)
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEQuantConfig,
    fp8_w8a8_moe_quant_config,
    mxfp4_w4a8_moe_quant_config,
    mxfp4_w4a16_moe_quant_config,
    ocp_mx_moe_quant_config,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    all_close_1d,
    normalize_e4m3fn_to_e4m3fnuz,
    per_tensor_dequantize,
)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform

logger = init_logger(__name__)

__all__ = [
    "QuarkMoEMethod",
    "QuarkOCP_MX_MoEMethod",
]


class QuarkMoEMethod(FusedMoEMethodBase):
    def __init__(self, moe: FusedMoEConfig):
        super().__init__(moe)
        self.has_bias = self.moe.has_bias

    @staticmethod
    def get_moe_method(
        quant_config: "QuarkConfig",  # type: ignore
        module: torch.nn.Module,
        layer_name: str,
    ) -> "QuarkMoEMethod":
        layer_quant_config = quant_config._find_matched_config(layer_name, module)

        if layer_quant_config.get("output_tensors") or layer_quant_config.get("bias"):
            raise NotImplementedError(
                "Currently, Quark models with output tensors and bias "
                "quantized are not supported"
            )

        weight_config = layer_quant_config.get("weight")
        input_config = layer_quant_config.get("input_tensors")

        if quant_config._is_fp8_w4a8(weight_config, input_config):
            raise NotImplementedError(
                "Quark W4A8 FP8 MoE requires ROCm AITER, which is not supported on Volta V100."
            )
        elif quant_config._is_fp8_w8a8(weight_config, input_config):
            return QuarkW8A8Fp8MoEMethod(weight_config, input_config, module.moe_config)
        elif quant_config._is_w_ocp_mx_a_x(weight_config, input_config):
            return QuarkOCP_MX_MoEMethod(weight_config, input_config, module.moe_config)
        else:
            raise RuntimeError("Unsupported FusedMoe scheme")


class QuarkW8A8Fp8MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any],
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        self.weight_qscheme = self.weight_quant.get("qscheme")
        self.input_qscheme = self.input_quant.get("qscheme")
        self.weight_dtype = self.weight_quant.get("dtype", "").replace("fp8_e4m3", "fp8")
        self.input_dtype = self.input_quant.get("dtype", "").replace("fp8_e4m3", "fp8")
        
        per_tensor = (self.weight_qscheme == "per_tensor" and self.input_qscheme == "per_tensor")
        per_channel = (self.weight_qscheme == "per_channel" and self.input_qscheme == "per_channel")
        
        self.act_quant_group_shape = (
            GroupShape.PER_TOKEN if per_channel else GroupShape.PER_TENSOR
        )
        if not (per_tensor or per_channel):
            raise ValueError(
                "For FP8 Fused MoE layers, only per-tensor and per-channel "
                "scales for weights and activations are supported. Found "
                f"{self.weight_qscheme}, {self.input_qscheme}"
            )

        self.static_input_scales = not self.input_quant.get("is_dynamic")
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization."
            )

        self.model_type = getattr(
            get_current_vllm_config().model_config.hf_config, "model_type", None
        )

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        layer.num_experts = num_experts
        layer.orig_dtype = params_dtype
        layer.weight_block_size = None
        params_dtype = torch.float8_e4m3fn

        w13_weight = torch.nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size_per_partition, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, intermediate_size_per_partition, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        if self.weight_qscheme == "per_tensor":
            if self.model_type != "gpt_oss":
                w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts, 2, dtype=torch.float32), requires_grad=False)
            else:
                w13_weight_scale = torch.nn.Parameter(torch.ones(num_experts, 1, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.TENSOR.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        elif self.weight_qscheme == "per_channel":
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

        if self.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.static_input_scales:
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError("QuantConfig has static quantization, but found activation scales are None.")
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(layer.w2_input_scale):
                logger.warning_once("Found input_scales that are not equal for fp8 MoE layer. Using the maximum across experts for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(layer.w2_input_scale.max(), requires_grad=False)

        if current_platform.is_fp8_fnuz():
            w13_weight, w13_weight_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w13_weight, layer.w13_weight_scale, layer.w13_input_scale
            )
            w2_weight, w2_weight_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                layer.w2_weight, layer.w2_weight_scale, layer.w2_input_scale
            )
            layer.w13_weight = torch.nn.Parameter(w13_weight, requires_grad=False)
            layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
            if w13_input_scale is not None:
                layer.w13_input_scale = torch.nn.Parameter(w13_input_scale, requires_grad=False)
            layer.w2_weight = torch.nn.Parameter(w2_weight, requires_grad=False)
            layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)
            if w2_input_scale is not None:
                layer.w2_input_scale = torch.nn.Parameter(w2_input_scale, requires_grad=False)

        if self.weight_qscheme == "per_tensor":
            assert layer.w13_weight_scale is not None
            shard_size = layer.intermediate_size_per_partition
            max_w13_scales = layer.w13_weight_scale.max(dim=1).values

            if self.model_type == "gpt_oss":
                for expert_id in range(layer.local_num_experts):
                    dq_weight = per_tensor_dequantize(layer.w13_weight[expert_id], layer.w13_weight_scale[expert_id][0])
                    layer.w13_weight[expert_id], _ = ops.scaled_fp8_quant(dq_weight, max_w13_scales[expert_id])
            else:
                for expert_id in range(layer.local_num_experts):
                    start = 0
                    for shard_id in range(2):
                        dq_weight = per_tensor_dequantize(
                            layer.w13_weight[expert_id][start : start + shard_size, :],
                            layer.w13_weight_scale[expert_id][shard_id],
                        )
                        layer.w13_weight[expert_id][start : start + shard_size, :], _ = ops.scaled_fp8_quant(
                            dq_weight, max_w13_scales[expert_id]
                        )
                        start += shard_size
            layer.w13_weight_scale = torch.nn.Parameter(max_w13_scales, requires_grad=False)

        elif self.weight_qscheme == "per_channel":
            if self.act_quant_group_shape == GroupShape.PER_TOKEN:
                w13_weight_scale = layer.w13_weight_scale.unsqueeze(-1)
                layer.w13_weight_scale = torch.nn.Parameter(w13_weight_scale, requires_grad=False)
                w2_weight_scale = layer.w2_weight_scale.unsqueeze(-1)
                layer.w2_weight_scale = torch.nn.Parameter(w2_weight_scale, requires_grad=False)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        return fp8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            per_act_token_quant=self.input_qscheme == "per_channel",
            per_out_ch_quant=self.weight_qscheme == "per_channel",
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts
        return fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )


class QuarkOCP_MX_MoEMethod(QuarkMoEMethod):
    def __init__(
        self,
        weight_config: dict[str, Any],
        input_config: dict[str, Any] | None,
        moe: FusedMoEConfig,
    ):
        super().__init__(moe)
        self.weight_quant = weight_config
        self.input_quant = input_config

        weight_qscheme = self.weight_quant.get("qscheme")
        if not weight_qscheme == "per_group":
            raise ValueError(f"For MX(FP4) Fused MoE layers, only per-group scales for weights are supported. Found {weight_qscheme}.")

        self.weight_dtype = self.weight_quant["dtype"].replace("fp", "mxfp")
        if self.input_quant is not None:
            input_quant = self.input_quant["dtype"]
            if input_quant in ["fp4", "fp6_e3m2", "fp6_e2m3"]:
                self.input_dtype = input_quant.replace("fp", "mxfp")
            elif input_quant == "fp8_e4m3":
                self.input_dtype = input_quant.replace("fp8_e4m3", "fp8")
            else:
                raise NotImplementedError(f"Current input dtype {input_quant} is not compatible with OCP MX (weight) MoE quantization.")
        else:
            self.input_dtype = None

        self.ocp_mx_scheme = None # Simplified for Volta
        self.mxfp4_backend = None # oracle/mxfp4 removed
        
        if self.input_quant is not None:
            self.static_input_scales = not self.input_quant.get("is_dynamic")
        else:
            self.static_input_scales = False

        self.model_type = getattr(get_current_vllm_config().model_config.hf_config, "model_type", None)
        
        # Always emulate on Volta V100
        self.emulate = True
        logger.warning_once(
            "The current mode (Volta V100) does not support native MXFP4/MXFP6 computation. "
            "Simulated weight dequantization and activation QDQ will be used."
        )

    def maybe_roundup_sizes(
        self, hidden_size: int, intermediate_size_per_partition: int, act_dtype: torch.dtype, moe_parallel_config
    ) -> tuple[int, int]:
        return super().maybe_roundup_sizes(hidden_size, intermediate_size_per_partition, act_dtype, moe_parallel_config)

    def get_packed_dim(self, dim: int, quant_dtype: str):
        if quant_dtype == "mxfp4":
            assert dim % 2 == 0
            return dim // 2
        else:
            assert (dim * 3) % 4 == 0
            return (dim * 3) // 4

    def create_weights(
        self, layer: torch.nn.Module, num_experts: int, hidden_size: int,
        intermediate_size_per_partition: int, params_dtype: torch.dtype, **extra_weight_attrs,
    ):
        extra_weight_attrs.update({"quant_method": FusedMoeWeightScaleSupported.BLOCK.value})
        params_dtype = torch.uint8

        w13_weight = torch.nn.Parameter(
            torch.zeros(num_experts, 2 * intermediate_size_per_partition, self.get_packed_dim(hidden_size, self.weight_dtype), dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.zeros(num_experts, hidden_size, self.get_packed_dim(intermediate_size_per_partition, self.weight_dtype), dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        from vllm.model_executor.layers.quantization.utils.ocp_mx_utils import OCP_MX_BLOCK_SIZE
        
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, 2 * intermediate_size_per_partition, hidden_size // OCP_MX_BLOCK_SIZE, dtype=params_dtype),
            requires_grad=False,
        )
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, intermediate_size_per_partition // OCP_MX_BLOCK_SIZE, dtype=params_dtype),
            requires_grad=False,
        )
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        layer.register_parameter("w2_weight_scale", w2_weight_scale)

        if self.has_bias:
            w13_bias = torch.nn.Parameter(torch.zeros(num_experts, 2 * intermediate_size_per_partition, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_bias", w13_bias)
            set_weight_attrs(w13_bias, extra_weight_attrs)
            w2_bias = torch.nn.Parameter(torch.zeros(num_experts, hidden_size, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w2_bias", w2_bias)
            set_weight_attrs(w2_bias, extra_weight_attrs)
        else:
            layer.w13_bias, layer.w2_bias = None, None

        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)
            w2_input_scale = torch.nn.Parameter(torch.ones(num_experts, dtype=torch.float32), requires_grad=False)
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer):
        if self.static_input_scales and self.input_dtype == "fp8":
            if layer.w13_input_scale is None or layer.w2_input_scale is None:
                raise ValueError("QuantConfig has static quantization, but found activation scales are None.")
            if not all_close_1d(layer.w13_input_scale) or not all_close_1d(layer.w2_input_scale):
                logger.warning_once("Found input_scales that are not equal for fp8 MoE layer. Using the maximum across experts for each layer.")
            layer.w13_input_scale = torch.nn.Parameter(layer.w13_input_scale.max(), requires_grad=False)
            layer.w2_input_scale = torch.nn.Parameter(layer.w2_input_scale.max(), requires_grad=False)

            if current_platform.is_fp8_fnuz():
                _, _, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w13_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(layer.w13_weight_scale, dtype=layer.w13_weight_scale.dtype),
                    layer.w13_input_scale,
                )
                _, _, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                    torch.empty_like(layer.w2_weight, dtype=torch.float8_e4m3fn),
                    torch.empty_like(layer.w2_weight_scale, dtype=layer.w13_weight_scale.dtype),
                    layer.w2_input_scale,
                )
                if w13_input_scale is not None:
                    layer.w13_input_scale = torch.nn.Parameter(w13_input_scale, requires_grad=False)
                if w2_input_scale is not None:
                    layer.w2_input_scale = torch.nn.Parameter(w2_input_scale, requires_grad=False)

        # Emulation path: nothing to do, weights will be dequantized on the fly
        torch.accelerator.empty_cache()
        return

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig | None:
        # Fallback to generic ocp_mx_moe_quant_config for emulation
        return ocp_mx_moe_quant_config(
            quant_dtype=self.input_dtype if self.input_dtype else "mxfp4",
            weight_dtype=self.weight_dtype,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_bias=layer.w13_bias,
            w2_bias=layer.w2_bias,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            block_shape=None,
        )

    def apply(
        self, layer: FusedMoE, x: torch.Tensor, topk_weights: torch.Tensor,
        topk_ids: torch.Tensor, shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.fused_moe import fused_experts
        return fused_experts(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )