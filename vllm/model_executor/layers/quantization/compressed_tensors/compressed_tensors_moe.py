# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import enum
from enum import Enum

import torch

from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (
    ActivationOrdering,
    QuantizationArgs,
    QuantizationStrategy,
)

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    FusedMoEActivationFormat,
    FusedMoEExpertsModular,
    FusedMoEMethodBase,
    FusedMoeWeightScaleSupported,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    int4_w4a16_moe_quant_config,
    int4_w4afp8_moe_quant_config,
    int8_w8a8_moe_quant_config,
    int8_w8a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
    convert_to_fp8_moe_kernel_format,
    make_fp8_moe_kernel,
    make_fp8_moe_quant_config,
    select_fp8_moe_backend,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (  # noqa
    WNA16_SUPPORTED_BITS,
    WNA16_SUPPORTED_TYPES_MAP,
)
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    process_fp8_input_tensor_strategy_moe,
    process_fp8_weight_tensor_strategy_moe,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    convert_bf16_scales_to_fp8,
    convert_packed_uint4b8_to_signed_int4_inplace,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8Static128BlockSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    kNvfp4Static,
)
from vllm.model_executor.layers.quantization.utils.w8a8_utils import (
    normalize_e4m3fn_to_e4m3fnuz,
)
from vllm.model_executor.utils import replace_parameter, set_weight_attrs
from vllm.platforms import CpuArchEnum, current_platform

logger = init_logger(__name__)


class GPTQMarlinState(Enum):
    REPACK = enum.auto()
    READY = enum.auto()


__all__ = [
    "CompressedTensorsMoEMethod",
    "CompressedTensorsW8A8Fp8MoEMethod",
    "CompressedTensorsW8A8Int8MoEMethod",
    "CompressedTensorsWNA16MarlinMoEMethod",
    "CompressedTensorsWNA16MoEMethod",
    "CompressedTensorsW4A4Nvfp4MoEMethod",
    "CompressedTensorsW4A8Int8MoEMethod",
]


class CompressedTensorsMoEMethod(FusedMoEMethodBase):
    @staticmethod
    def get_moe_method(
        quant_config: "CompressedTensorsConfig",  # type: ignore # noqa E501
        layer: torch.nn.Module,
        layer_name: str,
    ) -> FusedMoEMethodBase:
        # FusedMoE was made by combining multiple Linears so need to
        # make sure quantization config for Linear can target it
        quant_config._add_fused_moe_to_target_scheme_map()
        unfused_names = [
            layer_name + proj_name
            for proj_name in [".0.gate_proj", ".0.up_proj", ".0.down_proj"]
        ]
        # TODO: refactor this to use expert_mapping and check all layer numbers
        all_scheme_dicts = [
            quant_config.get_scheme_dict(layer, name) for name in unfused_names
        ]
        scheme_dict = all_scheme_dicts.pop()

        # multiple schemes found
        if not all([cur_dict == scheme_dict for cur_dict in all_scheme_dicts]):
            raise ValueError(
                "All MoE projections need to have same "
                "quantization scheme but found multiple "
            )

        if scheme_dict is None:  # ignored layer
            return UnquantizedFusedMoEMethod(layer.moe_config)

        # TODO: @dsikka: refactor this to use schemes as other kernels
        # are supported + check if the layer is being ignored.
        weight_quant = scheme_dict.get("weights")
        input_quant = scheme_dict.get("input_activations")
        format = scheme_dict.get("format")

        if quant_config._is_mxfp4(weight_quant):
            # MXFP4 removed - not supported on Volta V100
            raise NotImplementedError(
                "MXFP4 MoE is not supported on Volta V100. "
                "Please use a different quantization method."
            )

        if quant_config._is_wNa16_group_channel(weight_quant, input_quant):
            # group_size=None means channelwise
            group_size = weight_quant.group_size or -1

            valid_format_and_bits = (
                weight_quant.num_bits in WNA16_SUPPORTED_BITS
                and format == CompressionFormat.pack_quantized.value
            )

            if not valid_format_and_bits:
                raise ValueError(
                    "For Fused MoE layers, only format: ",
                    f"{CompressionFormat.pack_quantized.value} ",
                    f" and bits: {WNA16_SUPPORTED_BITS} is supported ",
                    f"but got format: {CompressionFormat.pack_quantized.value} "
                    f" and bits: {weight_quant.num_bits}",
                )

            # Marlin removed - not supported on Volta V100
            # Always use WNA16MoEMethod
            if (
                weight_quant.strategy == QuantizationStrategy.GROUP
                and weight_quant.actorder
                in (ActivationOrdering.GROUP, ActivationOrdering.DYNAMIC)
            ):
                raise ValueError(
                    "WNA16MoE is not supported with actorder=group/dynamic. "
                )
            logger.info_once("Using CompressedTensorsWNA16MoEMethod")
            return CompressedTensorsWNA16MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )

        elif quant_config._is_nvfp4_format(weight_quant):
            # NVFP4 removed - not supported on Volta V100
            raise NotImplementedError(
                "NVFP4 MoE is not supported on Volta V100. "
                "Please use a different quantization method."
            )

        elif (
            quant_config._is_fp8_w8a8_sm90(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8_sm100(weight_quant, input_quant)
            or quant_config._is_fp8_w8a8(weight_quant, input_quant)
        ):
            return CompressedTensorsW8A8Fp8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_dynamic_token_w8a8(weight_quant, input_quant):
            return CompressedTensorsW8A8Int8MoEMethod(
                weight_quant, input_quant, layer.moe_config
            )
        elif quant_config._is_fp8_w4a8_sm90(weight_quant, input_quant):
            # W4A8 FP8 removed - requires SM90+
            raise NotImplementedError(
                "W4A8 FP8 MoE is not supported on Volta V100. "
                "Please use a different quantization method."
            )
        elif quant_config._is_dynamic_token_w4a8_int(weight_quant, input_quant):
            # W4A8 Int8 removed - CPU-only
            raise NotImplementedError(
                "W4A8 Int8 MoE is not supported on Volta V100. "
                "Please use a different quantization method."
            )
        else:
            raise RuntimeError(
                f"Unsupported FusedMoe scheme: {weight_quant}, {input_quant}"
            )


class CompressedTensorsW8A8Fp8MoEMethod(CompressedTensorsMoEMethod):
    """W8A8 FP8 MoE quantization using compressed tensors."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant

        per_tensor = (
            self.weight_quant.strategy == QuantizationStrategy.TENSOR
            and self.input_quant.strategy == QuantizationStrategy.TENSOR
        )
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not (per_tensor or per_channel):
            assert self.weight_quant.strategy == QuantizationStrategy.BLOCK
            self.weight_block_size = self.weight_quant.block_structure
            assert self.weight_quant.dynamic is not None
        else:
            self.weight_block_size = None
        self.block_quant = self.weight_block_size is not None

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales and per_channel:
            raise ValueError(
                "For FP8 Fused MoE layer, we require either per tensor or "
                "channelwise, dynamic per token quantization. "
            )

        ct2vllm_weight = {
            QuantizationStrategy.CHANNEL: kFp8StaticChannelSym,
            QuantizationStrategy.TENSOR: kFp8StaticTensorSym,
            QuantizationStrategy.BLOCK: kFp8Static128BlockSym,
        }
        ct2vllm_act = {
            QuantizationStrategy.TOKEN: kFp8DynamicTokenSym,
            QuantizationStrategy.TENSOR: (
                kFp8StaticTensorSym if self.static_input_scales else kFp8Dynamic128Sym
            ),
        }
        weight_key = ct2vllm_weight[self.weight_quant.strategy]
        if weight_key == kFp8Static128BlockSym:
            activation_key = kFp8Dynamic128Sym
        else:
            activation_key = ct2vllm_act[self.input_quant.strategy]

        # Select Fp8 MoE backend
        self.fp8_backend, self.experts_cls = select_fp8_moe_backend(
            config=self.moe,
            weight_key=weight_key,
            activation_key=activation_key,
            allow_vllm_cutlass=True,
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
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        if self.block_quant:
            assert self.weight_block_size is not None
            layer.weight_block_size = self.weight_block_size
            tp_size = get_tensor_model_parallel_world_size()
            block_n, block_k = (
                self.weight_block_size[0],
                self.weight_block_size[1],
            )
            # NOTE: To ensure proper alignment of the block-wise quantization
            # scales, the output_size of the weights for both the gate and up
            # layers must be divisible by block_n.
            # Required by column parallel or enabling merged weights
            if intermediate_size_per_partition % block_n != 0:
                raise ValueError(
                    f"The output_size of gate's and up's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_n = {block_n}."
                )
            if tp_size > 1 and intermediate_size_per_partition % block_k != 0:
                # Required by row parallel
                raise ValueError(
                    f"The input_size of down's weight = "
                    f"{intermediate_size_per_partition} is not divisible by "
                    f"weight quantization block_k = {block_k}."
                )

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            # For gated MoE, allocate 2 scales for w1 and w3 respectively.
            # They will be combined to a single scale after weight loading.
            # For non-gated MoE, allocate 1 scale for w13.
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, w13_num_shards, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-TENSOR quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.TENSOR.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.CHANNEL:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards * intermediate_size_per_partition,
                    1,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        elif self.weight_quant.strategy == QuantizationStrategy.BLOCK:
            w13_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    w13_num_shards
                    * ((intermediate_size_per_partition + block_n - 1) // block_n),
                    (hidden_size + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w13_weight_scale", w13_weight_scale)
            w2_weight_scale = torch.nn.Parameter(
                torch.ones(
                    num_experts,
                    (hidden_size + block_n - 1) // block_n,
                    (intermediate_size_per_partition + block_k - 1) // block_k,
                    dtype=torch.float32,
                ),
                requires_grad=False,
            )
            layer.register_parameter("w2_weight_scale", w2_weight_scale)
            # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
            extra_weight_attrs.update(
                {"quant_method": FusedMoeWeightScaleSupported.BLOCK.value}
            )
            set_weight_attrs(w13_weight_scale, extra_weight_attrs)
            set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        if self.static_input_scales:
            w13_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w13_input_scale", w13_input_scale)
            set_weight_attrs(w13_input_scale, extra_weight_attrs)

            w2_input_scale = torch.nn.Parameter(
                torch.ones(num_experts, dtype=torch.float32), requires_grad=False
            )
            layer.register_parameter("w2_input_scale", w2_input_scale)
            set_weight_attrs(w2_input_scale, extra_weight_attrs)
        else:
            layer.w13_input_scale = None
            layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: FusedMoE) -> None:
        # Allow for accessing weights and scales in standard way.
        w13 = layer.w13_weight
        w2 = layer.w2_weight
        w13_scale = layer.w13_weight_scale
        w2_scale = layer.w2_weight_scale
        w13_input_scale = layer.w13_input_scale
        w2_input_scale = layer.w2_input_scale

        # MI300x and MI325x use FNUZ format for FP8. Convert if needed.
        if current_platform.is_fp8_fnuz():
            w13, w13_scale, w13_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w13, w13_scale, w13_input_scale
            )
            w2, w2_scale, w2_input_scale = normalize_e4m3fn_to_e4m3fnuz(
                w2, w2_scale, w2_input_scale
            )

        # Per tensor kernels require single activation scale. Use the max.
        if self.static_input_scales:
            assert self.input_quant.strategy == QuantizationStrategy.TENSOR
            assert w13_input_scale is not None and w2_input_scale is not None
            w13_input_scale, w2_input_scale = process_fp8_input_tensor_strategy_moe(
                w13_input_scale, w2_input_scale
            )
            replace_parameter(layer, "w13_input_scale", w13_input_scale)
            replace_parameter(layer, "w2_input_scale", w2_input_scale)

        # Per-tensor kernels use a single scale, for W13, but on disk there
        # is a separate scale for W1 and W3. Requantize with the max scale.
        if self.weight_quant.strategy == QuantizationStrategy.TENSOR:
            w13, w13_scale = process_fp8_weight_tensor_strategy_moe(
                w13,
                w13_scale,
                shard_size=layer.intermediate_size_per_partition,
                num_experts=layer.local_num_experts,
                is_act_and_mul=self.moe.is_act_and_mul,
            )

        w13, w2, w13_scale, w2_scale = convert_to_fp8_moe_kernel_format(
            fp8_backend=self.fp8_backend,
            layer=layer,
            w13=w13,
            w2=w2,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
            w13_input_scale=w13_input_scale,
            w2_input_scale=w2_input_scale,
        )

        # Replace parameters with updated versions. Note that this helper
        # function ensures the replacement is compatible with RL weight reloads.
        replace_parameter(layer, "w13_weight", w13)
        replace_parameter(layer, "w2_weight", w2)
        replace_parameter(layer, "w13_weight_scale", w13_scale)
        replace_parameter(layer, "w2_weight_scale", w2_scale)

        # Setup modular kernel for TP case and naive DP/EP case.
        # In non-naive DP/EP case, we will create a ModularKernelMethod.
        # TODO(rob): unify these so FP8MoEMethod owns the ModularKernel
        # in both cases.
        self.moe_quant_config = self.get_fused_moe_quant_config(layer)
        if self.moe_quant_config:
            assert self.experts_cls is not None
            self.moe_kernel = make_fp8_moe_kernel(
                moe_quant_config=self.moe_quant_config,
                moe_config=self.moe,
                fp8_backend=self.fp8_backend,
                experts_cls=self.experts_cls,
                routing_tables=layer._maybe_init_expert_routing_tables(),
                shared_experts=layer.shared_experts,
            )

    def maybe_make_prepare_finalize(
        self,
        routing_tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None,
    ) -> mk.FusedMoEPrepareAndFinalizeModular | None:
        raise ValueError(
            f"{self.__class__.__name__} uses the new modular kernel initialization "
            "logic. This function should not be called."
        )

    def get_fused_moe_quant_config(self, layer: torch.nn.Module) -> FusedMoEQuantConfig:
        is_per_token = self.input_quant.strategy == QuantizationStrategy.TOKEN
        return make_fp8_moe_quant_config(
            fp8_backend=self.fp8_backend,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=is_per_token,
            per_out_ch_quant=is_per_token,
            block_shape=self.weight_block_size,
        )

    def apply_monolithic(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert self.moe_kernel is not None
        return self.moe_kernel.apply_monolithic(
            x,
            layer.w13_weight,
            layer.w2_weight,
            router_logits,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            num_expert_group=layer.num_expert_group,
            topk_group=layer.topk_group,
            e_score_correction_bias=layer.e_score_correction_bias,
            routed_scaling_factor=layer.routed_scaling_factor,
        )

    def apply(
        self,
        layer: FusedMoE,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        assert not self.is_monolithic
        assert self.moe_kernel is not None
        return self.moe_kernel.apply(
            x,
            layer.w13_weight,
            layer.w2_weight,
            topk_weights,
            topk_ids,
            activation=layer.activation,
            global_num_experts=layer.global_num_experts,
            # TODO(rob): investigate the disable_expert_map introduced by:
            # https://github.com/vllm-project/vllm/commit/84166fee9770e6fba71a96978b3e7d149392fb28 # noqa: E501
            expert_map=layer.expert_map,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            shared_experts_input=shared_experts_input,
        )

    @property
    def supports_eplb(self) -> bool:
        return True


class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        per_channel = (
            self.weight_quant.strategy == QuantizationStrategy.CHANNEL
            and self.input_quant.strategy == QuantizationStrategy.TOKEN
        )
        if not per_channel:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found "
                f"{self.weight_quant}, {self.input_quant}"
            )

        self.static_input_scales = not self.input_quant.dynamic
        if self.static_input_scales:
            raise ValueError(
                "For INT8 Fused MoE layers, we require channelwise, "
                "dynamic per token quantization. Found static input scales."
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
        params_dtype = torch.int8
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1

        # WEIGHTS
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                hidden_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        # WEIGHT_SCALES
        assert self.weight_quant.strategy == QuantizationStrategy.CHANNEL
        w13_weight_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                w13_num_shards * intermediate_size_per_partition,
                1,
                dtype=torch.float32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_weight_scale)
        w2_weight_scale = torch.nn.Parameter(
            torch.ones(num_experts, hidden_size, 1, dtype=torch.float32),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_weight_scale)
        # Add PER-CHANNEL quantization for FusedMoE.weight_loader.
        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value}
        )
        set_weight_attrs(w13_weight_scale, extra_weight_attrs)
        set_weight_attrs(w2_weight_scale, extra_weight_attrs)

        # INPUT_SCALES
        assert not self.static_input_scales
        layer.w13_input_scale = None
        layer.w2_input_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        pass

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        return int8_w8a8_moe_quant_config(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            a1_scale=layer.w13_input_scale,
            a2_scale=layer.w2_input_scale,
            per_act_token_quant=True,
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


class CompressedTensorsWNA16MarlinMoEMethod(CompressedTensorsMoEMethod):
    """Marlin MoE removed - not supported on Volta V100."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CompressedTensorsWNA16MarlinMoEMethod is not supported on Volta V100. "
            "Please use CompressedTensorsWNA16MoEMethod instead."
        )


class CompressedTensorsWNA16MoEMethod(CompressedTensorsMoEMethod):
    def __init__(
        self,
        weight_quant: QuantizationArgs,
        input_quant: QuantizationArgs | None,
        moe: FusedMoEConfig,
        layer_name: str | None = None,
    ):
        super().__init__(moe)
        self.weight_quant = weight_quant
        self.input_quant = input_quant
        # Extract properties from weight_quant
        self.num_bits = weight_quant.num_bits
        self.packed_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        # channelwise is not supported by this kernel
        assert weight_quant.strategy == "group"
        self.group_size = weight_quant.group_size
        # grouped actorder isn't supported by this kernel
        assert weight_quant.actorder != "group"
        assert weight_quant.symmetric, (
            "Only symmetric quantization is supported for MoE"
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
        # Will transpose the loaded weight along the
        # intermediate and hidden dim sizes. Will
        # shard for TP along the transposed dims
        extra_weight_attrs.update(
            {"is_transposed": True, "quant_method": self.strategy}
        )
        w13_num_shards = 2 if self.moe.is_act_and_mul else 1
        w13_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size // self.packed_factor,
                w13_num_shards * intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_packed", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        w2_weight = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition // self.packed_factor,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_packed", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)

        w2_scales_size = intermediate_size_per_partition

        if self.strategy == "channel":
            num_groups_w2 = num_groups_w13 = 1
            self.group_size = -1
        else:
            num_groups_w2 = w2_scales_size // self.group_size
            num_groups_w13 = hidden_size // self.group_size

        w13_scale = torch.nn.Parameter(
            torch.ones(
                num_experts,
                num_groups_w13,
                w13_num_shards * intermediate_size_per_partition,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_scale", w13_scale)
        set_weight_attrs(w13_scale, extra_weight_attrs)

        w2_scale = torch.nn.Parameter(
            torch.ones(num_experts, num_groups_w2, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_scale", w2_scale)
        set_weight_attrs(w2_scale, extra_weight_attrs)
        set_weight_attrs(w2_scale, {"load_full_w2": False})

        w2_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )
        layer.register_parameter("w2_weight_shape", w2_weight_shape)
        set_weight_attrs(w2_weight_shape, extra_weight_attrs)
        w13_weight_shape = torch.nn.Parameter(
            torch.empty(num_experts, 2), requires_grad=False
        )

        layer.register_parameter("w13_weight_shape", w13_weight_shape)
        set_weight_attrs(w13_weight_shape, extra_weight_attrs)

        w13_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_weight_g_idx", w13_g_idx)
        set_weight_attrs(w13_g_idx, extra_weight_attrs)

        w2_g_idx = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_weight_g_idx", w2_g_idx)
        set_weight_attrs(w2_g_idx, extra_weight_attrs)

        w13_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                hidden_size,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w13_g_idx_sort_indices", w13_g_idx_sort_indices)
        set_weight_attrs(w13_g_idx_sort_indices, extra_weight_attrs)

        w2_g_idx_sort_indices = torch.nn.Parameter(
            torch.empty(
                num_experts,
                intermediate_size_per_partition,
                dtype=torch.int32,
            ),
            requires_grad=False,
        )
        layer.register_parameter("w2_g_idx_sort_indices", w2_g_idx_sort_indices)
        set_weight_attrs(w2_g_idx_sort_indices, extra_weight_attrs)

        layer.a13_scale = None
        layer.a2_scale = None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # Reconfigure packed weights and scales to match moe_wna16 format
        layer.w13_weight_packed = torch.nn.Parameter(
            layer.w13_weight_packed.transpose(1, 2).contiguous().view(torch.uint8),
            requires_grad=False,
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            layer.w2_weight_packed.transpose(1, 2).contiguous().view(torch.uint8),
            requires_grad=False,
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            layer.w13_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            layer.w2_weight_scale.transpose(1, 2).contiguous(), requires_grad=False
        )

    def get_fused_moe_quant_config(
        self, layer: torch.nn.Module
    ) -> FusedMoEQuantConfig | None:
        assert self.num_bits == 4 or self.num_bits == 8
        config_builder = (
            int4_w4a16_moe_quant_config
            if self.num_bits == 4
            else int8_w8a16_moe_quant_config
        )

        return config_builder(
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            w1_zp=None,
            w2_zp=None,
            block_shape=[0, self.group_size],
        )

    def select_gemm_impl(
        self,
        prepare_finalize: mk.FusedMoEPrepareAndFinalizeModular,
        layer: torch.nn.Module,
    ) -> mk.FusedMoEExpertsModular:
        if self.moe.is_lora_enabled:
            assert self.moe_quant_config is not None
            from vllm.triton_utils import HAS_TRITON

            if HAS_TRITON:
                from vllm.model_executor.layers.fused_moe import TritonWNA16Experts

                layer.w13_weight = layer.w13_weight_packed
                layer.w2_weight = layer.w2_weight_packed
                return TritonWNA16Experts(
                    moe_config=self.moe, quant_config=self.moe_quant_config
                )
            else:
                raise NotImplementedError(
                    "TritonExperts requires Triton. "
                    "Install triton or disable LoRA for MoE."
                )

        raise NotImplementedError

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
            x,
            layer.w13_weight_packed,
            layer.w2_weight_packed,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            inplace=not self.moe.disable_inplace,
            activation=layer.activation,
            apply_router_weight_on_input=layer.apply_router_weight_on_input,
            global_num_experts=layer.global_num_experts,
            expert_map=layer.expert_map,
            quant_config=self.moe_quant_config,
        )

    @property
    def supports_eplb(self) -> bool:
        return True


class CompressedTensorsW4A8Int8MoEMethod(CompressedTensorsMoEMethod):
    """
    CPU-only MoE method - not supported on Volta V100.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CompressedTensorsW4A8Int8MoEMethod is CPU-only and not supported on Volta V100. "
            "Please use a different quantization method."
        )


class CompressedTensorsW4A4Nvfp4MoEMethod(CompressedTensorsMoEMethod):
    """NVFP4 MoE removed - not supported on Volta V100."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "CompressedTensorsW4A4Nvfp4MoEMethod is not supported on Volta V100. "
            "Please use a different quantization method."
        )
