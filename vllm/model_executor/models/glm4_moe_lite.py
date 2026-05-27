# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The ZhipuAI Team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only GLM-4.7-Flash model compatible with HuggingFace weights."""
import typing
from collections.abc import Callable, Iterable
from itertools import islice
from typing import TYPE_CHECKING

import torch
from torch import nn
from transformers import PretrainedConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.fused_moe import FusedMoE, SharedFusedMoE
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    PPMissingLayer,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

if TYPE_CHECKING:
    from transformers import Glm4MoeLiteConfig


class Glm4MoeLiteMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = nn.Linear(
            hidden_size,
            2 * intermediate_size,
            bias=False,
        )
        self.down_proj = nn.Linear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        x = self.act_fn(gate) * up
        x = self.down_proj(x)
        return x


class Glm4MoeLite(nn.Module):
    def __init__(
        self,
        config: "Glm4MoeLiteConfig",
        parallel_config: "ParallelConfig",
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        self.tp_size = 1 # Simplified for Volta fork
        
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)
        self.n_routed_experts: int = config.n_routed_experts
        self.n_shared_experts: int = config.n_shared_experts

        self.gate = nn.Linear(
            config.hidden_size,
            config.n_routed_experts,
            bias=False,
        )

        if config.n_shared_experts is None:
            self.shared_experts = None
        else:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.shared_experts",
            )

        self.experts = SharedFusedMoE(
            shared_experts=self.shared_experts,
            gate=self.gate,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            reduce_results=False,
            renormalize=config.norm_topk_prob,
            quant_config=quant_config,
            use_grouped_topk=True,
            num_expert_group=getattr(config, "n_group", 1),
            topk_group=getattr(config, "topk_group", 1),
            prefix=f"{prefix}.experts",
            scoring_func=getattr(config, "scoring_func", "softmax"),
            routed_scaling_factor=1.0,
            e_score_correction_bias=None,
            enable_eplb=False,
            num_redundant_experts=0,
            is_sequence_parallel=False,
            n_shared_experts=config.n_shared_experts,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        router_logits = self.gate(hidden_states)
        fused_moe_out = self.experts(
            hidden_states=hidden_states, router_logits=router_logits
        )

        shared_output, final_hidden_states = fused_moe_out
        if self.shared_experts is None:
            assert shared_output is None

        if hidden_states.dtype != torch.float16:
            final_hidden_states *= self.routed_scaling_factor
        elif self.shared_experts is not None:
            assert shared_output is not None
            shared_output *= 1.0 / self.routed_scaling_factor

        if self.shared_experts is not None:
            assert shared_output is not None
            final_hidden_states += shared_output

        return final_hidden_states.view(num_tokens, hidden_dim)


class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        prefix: str,
        config: "Glm4MoeLiteConfig | None" = None,
        topk_indices_buffer: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = vllm_config.model_config.hf_config
            
        self.hidden_size = config.hidden_size
        
        layer_idx = int(prefix.split(sep=".")[-1])
        self.layer_idx = layer_idx

        self.self_attn = nn.Identity() # Placeholder for attention

        moe_layer_freq = getattr(config, "moe_layer_freq", 1)
        if (
            config.n_routed_experts is not None
            and layer_idx >= config.first_k_dense_replace
            and layer_idx % moe_layer_freq == 0
        ):
            self.mlp = Glm4MoeLite(
                config=config,
                parallel_config=vllm_config.parallel_config,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Glm4MoeLiteMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=vllm_config.quant_config,
                prefix=f"{prefix}.mlp",
            )
            
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.routed_scaling_factor = getattr(config, "routed_scaling_factor", 1.0)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        llama_4_scaling: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states.clone()
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(hidden_states)

        if hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor
            if self.layer_idx == 0:
                residual *= 1.0 / self.routed_scaling_factor

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)

        if isinstance(self.mlp, Glm4MoeLiteMLP) and hidden_states.dtype == torch.float16:
            hidden_states *= 1.0 / self.routed_scaling_factor

        return hidden_states, residual


@support_torch_compile
class Glm4MoeLiteModel(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        self.device = current_platform.device_type

        self.vocab_size = config.vocab_size
        
        if get_pp_group().is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Glm4MoeLiteDecoderLayer(
                vllm_config, prefix, topk_indices_buffer=None
            ),
            prefix=f"{prefix}.layers",
        )

        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()
            
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(positions, hidden_states, residual)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Glm4MixtureOfExperts:
    moe_mlp_layers: list[Glm4MoeLite]

    def extract_moe_parameters(self, example_moe: Glm4MoeLite | None):
        if example_moe is None:
            self.num_moe_layers = 0
            self.num_expert_groups = 0
            self.num_logical_experts = 0
            self.num_physical_experts = 0
            self.num_local_physical_experts = 0
            self.num_routed_experts = 0
            self.num_shared_experts = 0
            self.num_redundant_experts = 0
        else:
            self.num_logical_experts = example_moe.n_routed_experts
            self.num_physical_experts = example_moe.n_routed_experts
            self.num_local_physical_experts = example_moe.n_routed_experts
            self.num_routed_experts = example_moe.n_routed_experts
            self.num_shared_experts = example_moe.n_shared_experts
            self.num_redundant_experts = 0


class Glm4MoeLiteForCausalLM(nn.Module, SupportsPP, Glm4MixtureOfExperts, SupportsLoRA):
    packed_modules_mapping = {
        "gate_up_proj": ["gate_proj", "up_proj"],
    }
    model_cls = Glm4MoeLiteModel

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        self.config = config
        
        self.model = self.model_cls(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )
        
        if get_pp_group().is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
            )
        else:
            self.lm_head = PPMissingLayer()
            
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )
        
        self.num_moe_layers = (
            self.config.num_hidden_layers - self.config.first_k_dense_replace
        )
        self.set_moe_parameters()

    def set_moe_parameters(self):
        self.expert_weights = []
        self.num_expert_groups = getattr(self.config, "n_group", 1)
        self.moe_layers = []
        self.moe_mlp_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, PPMissingLayer):
                continue
            assert isinstance(layer, Glm4MoeLiteDecoderLayer)
            if isinstance(layer.mlp, Glm4MoeLite):
                example_moe = layer.mlp
                self.moe_mlp_layers.append(layer.mlp)
                self.moe_layers.append(layer.mlp.experts)
        self.extract_moe_parameters(example_moe)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        logits = self.logits_processor(self.lm_head, hidden_states)
        return logits

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts,
            num_redundant_experts=0,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        rocm_aiter_moe_shared_expert_enabled = False
        
        stacked_params_mapping = [
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = SharedFusedMoE.make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.n_routed_experts
            + (
                self.config.n_shared_experts
                if rocm_aiter_moe_shared_expert_enabled
                else 0
            ),
        )

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            is_fusion_moe_shared_experts_layer = (
                rocm_aiter_moe_shared_expert_enabled and ("mlp.shared_experts" in name)
            )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                if ("mlp.experts." in name) and name not in params_dict:
                    continue
                if is_fusion_moe_shared_experts_layer:
                    continue
                name_mapped = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                is_expert_weight = False
                num_chunks = 1
                if is_fusion_moe_shared_experts_layer:
                    num_chunks = getattr(self.config, "n_shared_experts", 1) or 1
                    split_dim = 1 if "down_proj.weight" in name else 0
                    total = loaded_weight.shape[split_dim]
                    assert total % num_chunks == 0, (
                        f"Shared expert weight dim {total} "
                        f"not divisible by num_chunks {num_chunks}"
                    )
                    chunk_size = total // num_chunks

                for j in range(num_chunks):
                    chunk_name = name
                    weight_to_load = loaded_weight

                    if is_fusion_moe_shared_experts_layer:
                        chunk_slice = slice(j * chunk_size, (j + 1) * chunk_size)
                        if loaded_weight.ndim == 1:
                            weight_to_load = loaded_weight[chunk_slice]
                        elif split_dim == 0:
                            weight_to_load = loaded_weight[chunk_slice, :]
                        else:
                            weight_to_load = loaded_weight[:, chunk_slice]
                        chunk_name = name.replace(
                            "mlp.shared_experts",
                            f"mlp.experts.{self.config.n_routed_experts + j}",
                        )

                    for mapping in expert_params_mapping:
                        param_name, weight_name, expert_id, shard_id = mapping
                        if weight_name not in chunk_name:
                            continue

                        is_expert_weight = True
                        name_mapped = chunk_name.replace(weight_name, param_name)

                        if is_pp_missing_parameter(name_mapped, self):
                            continue

                        param = params_dict[name_mapped]
                        weight_loader = typing.cast(
                            Callable[..., bool], param.weight_loader
                        )
                        success = weight_loader(
                            param,
                            weight_to_load,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                            return_success=True,
                        )
                        if success:
                            if not is_fusion_moe_shared_experts_layer:
                                name = name_mapped
                            else:
                                loaded_params.add(name_mapped)
                            break
                    else:
                        if is_expert_weight:
                            continue

                        if name.endswith(".bias") and name not in params_dict:
                            continue

                        name = maybe_remap_kv_scale_name(name, params_dict)
                        if name is None:
                            continue

                        if is_pp_missing_parameter(name, self):
                            continue

                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
            if not is_fusion_moe_shared_experts_layer:
                loaded_params.add(name)

        return loaded_params


def get_spec_layer_idx_from_weight_name(
    config: "Glm4MoeLiteConfig", weight_name: str
) -> int | None:
    if hasattr(config, "num_nextn_predict_layers") and (
        config.num_nextn_predict_layers > 0
    ):
        layer_idx = config.num_hidden_layers
        for i in range(config.num_nextn_predict_layers):
            if weight_name.startswith(f"model.layers.{layer_idx + i}."):
                return layer_idx + i
    return None
