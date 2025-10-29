# Copyright (c) 2024-present, BAAI. All Rights Reserved.
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
# ------------------------------------------------------------------------
"""Simple implementation of Phi model."""

import torch
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.phi.configuration_phi import PhiConfig

from diffnext.models.flash_attention import apply_rotary_emb


def maybe_apply_ckpt(module, name, x) -> torch.Tensor:
    """Apply gradient checkpointing if possible."""
    if module.gradient_checkpointing and x.requires_grad:
        return torch.utils.checkpoint.checkpoint(getattr(module, name), x, use_reentrant=False)
    return getattr(module, name)(x)


class PhiRotaryEmbedding(nn.Module):
    """Rotary embedding layer."""

    class PEFunc(object):
        """Apply RoPE weight to Q/K tensor."""

        def __init__(self, weight):
            self.cos, self.sin = weight

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            self.cos, self.sin = self.cos.to(x), self.sin.to(x)
            return apply_rotary_emb(x, self.cos, self.sin, inplace=True)

    @staticmethod
    def from_config(config):
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = int(config.partial_rotary_factor * head_dim)
        return PhiRotaryEmbedding(rotary_dim, config.max_position_embeddings, config.rope_theta)

    def __init__(self, dim, max_position_embeddings=2048, base=10000):
        super().__init__()
        self.dim, self.base = dim, base
        self.max_position_embeddings = max_position_embeddings
        freq = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim)
        self.register_buffer("inv_freq", freq.reciprocal_(), persistent=False)
        self.set_cos_sin_cache(max_position_embeddings, dtype=torch.get_default_dtype())

    def set_cos_sin_cache(self, seqlen, dtype):
        self.max_seqlen_cached, device = seqlen, self.inv_freq.device
        t = torch.arange(self.max_seqlen_cached, device=device, dtype=torch.int64)
        freq = torch.outer(t.float(), self.inv_freq.float())
        emb = torch.cat((freq, freq), dim=-1)
        self.register_buffer("cos", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin", emb.sin().to(dtype), persistent=False)

    def get_func(self, pos=0, seqlen=1) -> PEFunc:
        return self.PEFunc(_[pos : pos + seqlen].chunk(2, -1)[0] for _ in (self.cos, self.sin))


class PhiMLP(nn.Module):
    """Phi MLP."""

    def __init__(self, config: PhiConfig):
        super().__init__()
        self.activation = ACT2FN[config.hidden_act]
        self.config, self.hidden_size = config, config.hidden_size
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class PhiAttention(nn.Module):
    """Phi attention."""

    def __init__(self, config: PhiConfig, layer_idx=None):
        super().__init__()
        self.layer_idx, hidden_size = layer_idx, config.hidden_size
        self.config, self.is_causal, self.gradient_checkpointing = config, True, False
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.q_proj = nn.Linear(hidden_size, config.num_attention_heads * self.head_dim)
        self.k_proj = nn.Linear(hidden_size, config.num_key_value_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, config.num_key_value_heads * self.head_dim)
        self.dense = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size)
        self.attn_mask = self.past_key_value = self.pe_func = self.flex_attn = None

    def forward_qkv(self, x) -> torch.Tensor:
        q, k, v = [m(x) for m in (self.q_proj, self.k_proj, self.v_proj)]
        return [_.unflatten(-1, (-1, self.head_dim)) for _ in (q, k, v)]

    def repeat_kv(self, x) -> torch.Tensor:
        return x.unsqueeze(2).expand(-1, -1, self.num_key_value_groups, -1, -1).flatten(1, 2)


class PhiSdpaAttention(PhiAttention):
    """Phi SDPA attention."""

    def forward(self, x) -> torch.Tensor:
        q, k, v = maybe_apply_ckpt(self, "forward_qkv", x)
        q, k = [self.pe_func(_) for _ in (q, k)]
        q, k, v = [_.transpose(1, 2) for _ in (q, k, v)]
        if self.past_key_value is not None and getattr(self.past_key_value, "is_frozen", False):
            k, v = [torch.cat(_, -2) for _ in zip(self.past_key_value[self.layer_idx], (k, v))]
        elif self.past_key_value is not None:  # Fallback to legacy NTP caching.
            k, v = self.past_key_value.update(k, v, self.layer_idx)
        self.past_key_value = None  # Release cache reference.
        if self.flex_attn and self.flex_attn.offsets:
            return self.dense(self.flex_attn(q, k, v).transpose(1, 2).flatten(2))
        is_causal = self.is_causal and self.attn_mask is None and x.size(1) > 1
        sdpa_args = {"is_causal": is_causal, "enable_gqa": True}
        o = nn.functional.scaled_dot_product_attention(q, k, v, self.attn_mask, **sdpa_args)
        return self.dense(o.transpose(1, 2).flatten(2))


class PhiDecoderLayer(nn.Module):
    """Phi decoder layer."""

    def __init__(self, config: PhiConfig, layer_idx: int):
        super().__init__()
        self.self_attn = PhiSdpaAttention(config, layer_idx)
        self.mlp, self.gradient_checkpointing = PhiMLP(config), False
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.resid_pdrop, inplace=True)
        self.mlp_checkpointing = False

    def forward(self, x) -> torch.Tensor:
        shortcut, x = x, self.input_layernorm(x)
        return self.self_attn(x).add_(maybe_apply_ckpt(self, "mlp", x)).add_(shortcut)


class PhiPreTrainedModel(PreTrainedModel):
    """Phi pre-trained model."""

    config_class = PhiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["PhiDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class PhiModel(PhiPreTrainedModel):
    """Phi transformer model."""

    def __init__(self, config: PhiConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = [PhiDecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        self.layers = nn.ModuleList(self.layers)
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.rotary_emb = PhiRotaryEmbedding.from_config(config)
        self.gradient_checkpointing, _ = False, self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        past_key_values: torch.Tensor = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        x = inputs_embeds if input_ids is None else self.embed_tokens(input_ids)
        pe_pos = kwargs.get("rope_pos", past_key_values.get_seq_length() if past_key_values else 0)
        pe_embedder = self.flex_rope if isinstance(pe_pos, torch.Tensor) else self.rotary_emb
        pe_func = pe_embedder.get_func(pe_pos, x.size(1))
        for layer in self.layers:
            layer.self_attn.pe_func = pe_func
            layer.self_attn.attn_mask = attention_mask
            layer.self_attn.past_key_value = past_key_values
            if self.gradient_checkpointing and self.training:
                x = self._gradient_checkpointing_func(layer.__call__, x)
            else:
                x = layer(x)
        x = self.final_layernorm(x)
        return BaseModelOutputWithPast(last_hidden_state=x, past_key_values=past_key_values)


class PhiEncoderModel(PhiPreTrainedModel):
    """Phi encoder model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = PhiModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.vocab_size, _ = config.vocab_size, self.post_init()

    def forward(self, input_ids, attention_mask=None, **kwargs) -> BaseModelOutputWithPast:
        return self.model(input_ids, attention_mask, **kwargs)


class PhiForCausalLM(PhiPreTrainedModel, GenerationMixin):
    """Phi causal language model."""

    def __init__(self, config):
        super().__init__(config)
        self.model = PhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        self.lm_shift, _ = 0, self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self) -> PhiModel:
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor = None,
        inputs_embeds: torch.Tensor = None,
        logits_to_keep=None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        outputs = self.model(input_ids, attention_mask, inputs_embeds, **kwargs)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        head_w = self.lm_head.weight[self.lm_shift :] if self.lm_shift else self.lm_head.weight
        logits = nn.functional.linear(outputs[0] if keep is None else outputs[0][:, keep], head_w)
        return CausalLMOutputWithPast(logits=logits, past_key_values=outputs.past_key_values)

    def prepare_inputs_for_generation(self, input_ids, inputs_embeds=None, **kwargs):
        past_key_values, _ = kwargs.get("past_key_values", None), kwargs.pop("attention_mask", None)
        past_pos = past_key_values.get_seq_length() if past_key_values else 0
        inputs = {"input_ids": input_ids[:, past_pos:] if past_pos else input_ids, **kwargs}
        if inputs_embeds is not None and not past_pos:
            inputs["inputs_embeds"] = inputs_embeds
        return inputs
