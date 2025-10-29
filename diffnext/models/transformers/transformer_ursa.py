# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""3D transformer model for URSA."""

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.modeling_outputs import Transformer2DModelOutput

from diffnext.models.embeddings import FlexRotaryEmbedding
from diffnext.models.flash_attention import cross_entropy_loss
from diffnext.models.flex_attention import FlexAttentionCausal2D
from diffnext.models.text_encoders.qwen3 import Qwen3Config, Qwen3Model


class URSATransformer3DModel(ModelMixin, ConfigMixin):
    """3D transformer model for URSA."""

    @register_to_config
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=6144,
        num_attention_heads=16,
        num_key_value_heads=8,
        num_hidden_layers=28,
        max_window_layers=28,
        rope_theta=1000000,
        vocab_size=215669,
        lm_vocab_size=151669,
        lm_head_size=64000,
        bov_token_id=151652,
        **kwargs,
    ):
        super().__init__()
        self.model = Qwen3Model(Qwen3Config.from_dict(self._internal_dict))
        self.model.flex_attn = FlexAttentionCausal2D()
        self.model.flex_rope = FlexRotaryEmbedding.from_config(self.model.config)
        [setattr(layer.self_attn, "is_causal", False) for layer in self.model.layers]
        [setattr(layer.self_attn, "flex_attn", self.model.flex_attn) for layer in self.model.layers]
        self.lm_head = nn.Linear(hidden_size, lm_head_size, bias=False)

    def forward(
        self,
        input_ids,
        inputs_embeds=None,
        labels=None,
        logits_to_keep=None,
        lm_head_shift=0,
        **kwargs,
    ) -> Transformer2DModelOutput:
        outputs = self.model(input_ids, inputs_embeds=inputs_embeds, **kwargs)
        keep = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        head_w = self.lm_head.weight[lm_head_shift:] if lm_head_shift else self.lm_head.weight
        logits = nn.functional.linear(outputs[0] if keep is None else outputs[0][:, keep], head_w)

        def flash_loss(logits, labels):
            if cross_entropy_loss:
                return cross_entropy_loss(logits.flatten(0, 1), labels, inplace_backward=True)[0]
            return nn.functional.cross_entropy(logits.flatten(0, 1), labels, reduction="none")

        if labels is not None:
            lbls = torch.nn.functional.pad(labels[:, 1:], (0, 1), value=-100)
            loss = flash_loss(logits.float(), lbls.flatten()).view(lbls.shape)
            acc1, mask = logits.data.argmax(-1).eq(lbls), lbls.ne(-100)
            return loss.sum().div(mask.sum()), acc1[mask].float().mean()

        return Transformer2DModelOutput(sample=logits)
