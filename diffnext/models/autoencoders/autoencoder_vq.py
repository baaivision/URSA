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
"""Simple implementation of AutoEncoderVQ."""

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.autoencoder_kl import Attention, Decoder, Encoder
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, IdentityDistribution
from diffnext.models.autoencoders import quantizers


class AutoencoderVQ(ModelMixin, ConfigMixin):
    """AutoEncoder VQ."""

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",) * 4,
        up_block_types=("UpDecoderBlock2D",) * 4,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=32,
        sample_size=1024,
        num_vq_embeddings=16384,
        vq_embed_dim=8,
        attn_down_block=False,
        attn_up_block=False,
        force_upcast=False,
        temporal_stride=1,
        spatial_stride=16,
        decoder_dtype=None,
        _quantizer_name="VQuantizer",
    ):
        super(AutoencoderVQ, self).__init__()
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, latent_channels, channels, layers)
        self.decoder = Decoder(latent_channels, out_channels, channels, layers)
        self.quant_conv = nn.Conv2d(latent_channels, vq_embed_dim, 1)
        self.post_quant_conv = nn.Conv2d(vq_embed_dim, latent_channels, 1)
        if attn_down_block:
            attentions = [Attention(block_out_channels[-1]) for _ in range(layers_per_block)]
            self.encoder.down_blocks[-1].attentions += attentions
        if attn_up_block:
            attentions = [Attention(block_out_channels[-1]) for _ in range(layers_per_block + 1)]
            self.decoder.up_blocks[0].attentions += attentions
        self.quantizer = getattr(quantizers, _quantizer_name)(num_vq_embeddings, vq_embed_dim)
        self.latent_dist = IdentityDistribution

    def to(self, *args, **kwargs):
        """Convert to given device and dtype."""
        super().to(*args, **kwargs)
        if self.config.decoder_dtype:
            self.decoder.to(dtype=getattr(torch, self.config.decoder_dtype))
        return self

    def scale_(self, x) -> torch.Tensor:
        """Scale the input latents."""
        return x

    def unscale_(self, x) -> torch.Tensor:
        """Unscale the input latents."""
        return x

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        z = self.encoder(self.forward(x))
        z = self.quant_conv(z)
        posterior = self.latent_dist(self.quantizer.quantize(z))
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, ids) -> DecoderOutput:
        """Decode the input indices."""
        z = self.quantizer.dequantize(ids)
        t = z.size(2) if z.dim() == 5 else 1
        z = z.transpose(1, 2).flatten(0, 1) if t > 1 else z
        z = z.squeeze_(2) if z.dim() == 5 else z
        x = self.post_quant_conv(self.forward(z))
        x = self.decoder(x.to(self.decoder.conv_in.weight))
        x = x.view(-1, t, *x.shape[1:]).transpose(1, 2) if t > 1 else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
