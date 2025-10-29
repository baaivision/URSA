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
"""Simple implementation of AutoEncoderKL."""

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import DecoderOutput
from diffnext.models.autoencoders.modeling_utils import DiagonalGaussianDistribution
from diffnext.models.autoencoders.modeling_utils import IdentityDistribution


class Attention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, dim, num_heads=1):
        super(Attention, self).__init__()
        self.num_heads = num_heads or dim // 64
        self.head_dim = dim // self.num_heads
        self.group_norm = nn.GroupNorm(32, dim, eps=1e-6)
        self.to_q, self.to_k, self.to_v = [nn.Linear(dim, dim) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])
        self._from_deprecated_attn_block = True  # Fix for diffusers>=0.15.0

    def forward(self, x) -> torch.Tensor:
        x, shape = self.group_norm(x), (-1,) + x.shape[1:]
        x = x.flatten(2).transpose(1, 2).contiguous()
        qkv_shape = (-1, x.size(1), self.num_heads, self.head_dim)
        q, k, v = [f(x).view(qkv_shape).transpose(1, 2) for f in (self.to_q, self.to_k, self.to_v)]
        o = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        return self.to_out[0](o.flatten(2)).transpose(1, 2).reshape(shape)


class Resize(nn.Module):
    """Resize layer."""

    def __init__(self, dim, downsample=1):
        super(Resize, self).__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 0) if downsample else None
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1) if not downsample else self.conv
        self.downsample, self.upsample = downsample, int(not downsample)

    def forward(self, x) -> torch.Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1)) if self.downsample else x
        return self.conv(nn.functional.interpolate(x, None, (2, 2)) if self.upsample else x)


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim, out_dim):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, dim, eps=1e-6)
        self.conv1 = nn.Conv2d(dim, out_dim, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_dim, eps=1e-6)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.conv_shortcut = nn.Conv2d(dim, out_dim, 1) if out_dim != dim else None
        self.nonlinearity = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut else x
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        return self.conv2(self.nonlinearity(self.norm2(x))).add_(shortcut)


class UNetResBlock(nn.Module):
    """UNet resnet block."""

    def __init__(self, dim, out_dim, depth=2, downsample=0, upsample=0):
        super(UNetResBlock, self).__init__()
        block_dims = [(out_dim, out_dim) if i > 0 else (dim, out_dim) for i in range(depth)]
        self.resnets = nn.ModuleList(ResBlock(*dims) for dims in block_dims)
        self.attentions = nn.ModuleList()  # Legacy AttnBlock.
        self.downsamplers = nn.ModuleList([Resize(out_dim, 1)]) if downsample else []
        self.upsamplers = nn.ModuleList([Resize(out_dim, 0)]) if upsample else []

    def forward(self, x) -> torch.Tensor:
        for i, resnet in enumerate(self.resnets):
            x = resnet(x)
            x = self.attentions[i](x).add_(x) if i < len(self.attentions) else x
        x = self.downsamplers[0](x) if self.downsamplers else x
        return self.upsamplers[0](x) if self.upsamplers else x


class UNetMidBlock(nn.Module):
    """UNet mid block."""

    def __init__(self, dim, num_heads=1, depth=1):
        super(UNetMidBlock, self).__init__()
        self.resnets = nn.ModuleList(ResBlock(dim, dim) for _ in range(depth + 1))
        self.attentions = nn.ModuleList(Attention(dim, num_heads) for _ in range(depth))

    def forward(self, x) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x).add_(x))
        return x


class Encoder(nn.Module):
    """VAE encoder."""

    def __init__(self, dim, out_dim, block_dims, block_depth=2):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(dim, block_dims[0], 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        for i, block_dim in enumerate(block_dims):
            downsample = 1 if i < (len(block_dims) - 1) else 0
            args = (block_dims[max(i - 1, 0)], block_dim, block_depth)
            self.down_blocks += [UNetResBlock(*args, downsample=downsample)]
        self.mid_block = UNetMidBlock(block_dims[-1])
        self.conv_act = nn.SiLU()
        self.conv_norm_out = nn.GroupNorm(32, block_dims[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(block_dims[-1], out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """VAE decoder."""

    def __init__(self, dim, out_dim, block_dims, block_depth=2):
        super(Decoder, self).__init__()
        block_dims = list(reversed(block_dims))
        self.up_blocks = nn.ModuleList()
        for i, block_dim in enumerate(block_dims):
            upsample = 1 if i < (len(block_dims) - 1) else 0
            args = (block_dims[max(i - 1, 0)], block_dim, block_depth + 1)
            self.up_blocks += [UNetResBlock(*args, upsample=upsample)]
        self.conv_in = nn.Conv2d(dim, block_dims[0], 3, 1, 1)
        self.mid_block = UNetMidBlock(block_dims[0])
        self.conv_act = nn.SiLU()
        self.conv_norm_out = nn.GroupNorm(32, block_dims[-1], eps=1e-6)
        self.conv_out = nn.Conv2d(block_dims[-1], out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class AutoencoderKL(ModelMixin, ConfigMixin):
    """AutoEncoder KL."""

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
        scaling_factor=0.18215,
        shift_factor=None,
        latents_mean=None,
        latents_std=None,
        force_upcast=True,
        double_z=True,
        use_quant_conv=True,
        use_post_quant_conv=True,
    ):
        super(AutoencoderKL, self).__init__()
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, (1 + double_z) * latent_channels, channels, layers)
        self.decoder = Decoder(latent_channels, out_channels, channels, layers)
        quant_conv_type = type(self.decoder.conv_in) if use_quant_conv else nn.Identity
        post_quant_conv_type = type(self.decoder.conv_in) if use_post_quant_conv else nn.Identity
        self.quant_conv = quant_conv_type(*([(1 + double_z) * latent_channels] * 2 + [1]))
        self.post_quant_conv = post_quant_conv_type(latent_channels, latent_channels, 1)
        self.latent_dist = DiagonalGaussianDistribution if double_z else IdentityDistribution

    def scale_(self, x) -> torch.Tensor:
        """Scale the input latents."""
        x.add_(-self.config.shift_factor) if self.config.shift_factor else None
        return x.mul_(self.config.scaling_factor)

    def unscale_(self, x) -> torch.Tensor:
        """Unscale the input latents."""
        x.mul_(1 / self.config.scaling_factor)
        return x.add_(self.config.shift_factor) if self.config.shift_factor else x

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        z = self.quant_conv(self.encoder(self.forward(x)))
        posterior = self.latent_dist(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z) -> DecoderOutput:
        """Decode the input latents."""
        t = z.size(2) if z.dim() == 5 else 1
        z = z.transpose(1, 2).flatten(0, 1) if t > 1 else z
        z = z.squeeze_(2) if z.dim() == 5 else z
        x = self.decoder(self.post_quant_conv(self.forward(z)))
        x = x.view(-1, t, *x.shape[1:]).transpose(1, 2) if t > 1 else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
