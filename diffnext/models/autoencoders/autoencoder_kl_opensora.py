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
##############################################################################
"""Simple implementation of AutoEncoderKL for OpenSoraPlan."""

from functools import partial

import torch
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import DiagonalGaussianDistribution
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, TilingMixin


class Conv3d(nn.Conv3d):
    """3D convolution."""

    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(*args, **kwargs)
        self.padding = (0,) + self.padding[1:]
        self.pad = nn.ReplicationPad3d((0,) * 4 + (self.kernel_size[0] - 1, 0))
        self.pad = nn.Identity() if self.kernel_size[0] == 1 else self.pad

    def forward(self, x) -> torch.Tensor:
        return super(Conv3d, self).forward(self.pad(x))


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
        num_windows = 1 if x.dim() == 4 else x.size(2)
        x, x_shape = self.group_norm(x), (-1,) + x.shape[1:]
        if num_windows == 1:
            x = x.flatten(2).transpose(1, 2).contiguous()
        else:  # i.e., Frame windows.
            x = x.permute(0, 2, 3, 4, 1).flatten(0, 1).flatten(1, 2).contiguous()
        qkv_shape = (-1, x.size(1), self.num_heads, self.head_dim)
        q, k, v = [f(x).view(qkv_shape).transpose(1, 2) for f in (self.to_q, self.to_k, self.to_v)]
        o = nn.functional.scaled_dot_product_attention(q, k, v).transpose(1, 2)
        x = self.to_out[0](o.flatten(2)).transpose(1, 2)
        x = x.view((-1, num_windows) + x.shape[1:]).transpose(1, 2) if num_windows > 1 else x
        return x.reshape(x_shape)


class Resize(nn.Module):
    """Resize layer."""

    def __init__(self, dim, conv_type, downsample=1):
        super(Resize, self).__init__()
        self.conv = conv_type(dim, dim, 3, 2, 0) if downsample else None
        self.conv = conv_type(dim, dim, stride=1, padding=1) if not downsample else self.conv
        self.downsample, self.upsample, self.t = downsample, int(not downsample), 1
        self.upsample = 0 if downsample else (2 if isinstance(self.conv, Conv3d) else 1)
        self.upsample = 1 if self.conv.kernel_size[0] == 1 else self.upsample

    def forward(self, x) -> torch.Tensor:
        if self.upsample == 2:
            x1, x2 = (x[:, :, :1], x[:, :, 1:]) if x.size(2) > 1 else (x, None)
            x1 = nn.functional.interpolate(x1, None, (1, 2, 2), "trilinear")
            x2 = x2 if x2 is None else nn.functional.interpolate(x2, None, (2, 2, 2), "trilinear")
            x = torch.cat([x1, x2], dim=2) if x2 is not None else x1
        elif self.downsample:
            padding = (0, 1, 0, 1) + ((0, 0) if isinstance(self.conv, Conv3d) else ())
            if x.dim() == 4 and len(padding) == 6:  # 2D->3D
                x = x.view((-1, self.t) + x.shape[1:]).transpose(1, 2)
            x = nn.functional.pad(x, padding)
        elif self.upsample:
            x = x.repeat_interleave(2, 3).repeat_interleave(2, 4)
        return self.conv(x)


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim, out_dim, conv_type=nn.Conv2d):
        super(ResBlock, self).__init__()
        self.norm1 = nn.GroupNorm(32, dim, eps=1e-6)
        self.conv1 = conv_type(dim, out_dim, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, out_dim, eps=1e-6)
        self.conv2 = conv_type(out_dim, out_dim, 3, 1, 1)
        self.conv_shortcut = conv_type(dim, out_dim, 1) if out_dim != dim else None
        self.nonlinearity = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut else x
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        return self.conv2(self.nonlinearity(self.norm2(x))).add_(shortcut)


class UNetResBlock(nn.Module):
    """UNet resnet block."""

    def __init__(self, dim, out_dim, conv_type, depth=2, downsample=False, upsample=False):
        super(UNetResBlock, self).__init__()
        block_dims = [(out_dim, out_dim) if i > 0 else (dim, out_dim) for i in range(depth)]
        self.resnets = nn.ModuleList(ResBlock(*dims, conv_type=conv_type) for dims in block_dims)
        self.downsamplers = nn.ModuleList([Resize(out_dim, downsample)]) if downsample else []
        self.upsamplers = nn.ModuleList([Resize(out_dim, upsample, 0)]) if upsample else []

    def forward(self, x) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        x = self.downsamplers[0](x) if self.downsamplers else x
        return self.upsamplers[0](x) if self.upsamplers else x


class UNetMidBlock(nn.Module):
    """UNet mid block."""

    def __init__(self, dim, conv_type, num_heads=1, depth=1):
        super(UNetMidBlock, self).__init__()
        self.resnets = nn.ModuleList(ResBlock(dim, dim, conv_type) for _ in range(depth + 1))
        self.attentions = nn.ModuleList(Attention(dim, num_heads) for _ in range(depth))

    def forward(self, x) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x).add_(x))
        return x


class Encoder(nn.Module):
    """VAE encoder."""

    def __init__(self, dim, out_dim, block_types, block_dims, block_depth=2):
        super(Encoder, self).__init__()
        self.conv_in = nn.Conv2d(dim, block_dims[0], 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        for i, (block_type, block_dim) in enumerate(zip(block_types, block_dims)):
            conv_type, conv_down = nn.Conv2d if "Block2D" in block_type else Conv3d, None
            if i < len(block_dims) - 1:
                conv_down = nn.Conv2d if "Block2D" in block_types[i + 1] else Conv3d
            args = (block_dims[max(i - 1, 0)], block_dim, conv_type, block_depth)
            self.down_blocks += [UNetResBlock(*args, downsample=conv_down)]
        self.mid_block = UNetMidBlock(block_dims[-1], conv_type)
        self.conv_act = nn.SiLU()
        self.conv_norm_out = nn.GroupNorm(32, block_dims[-1], eps=1e-6)
        self.conv_out = conv_type(block_dims[-1], 2 * out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        t = x.size(2) if x.dim() == 5 else 1
        x = x.transpose(1, 2).flatten(0, 1) if x.dim() == 5 else x
        x = self.conv_in(x)
        for blk in self.down_blocks:
            [setattr(m, "t", t) for m in blk.downsamplers]
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """VAE decoder."""

    def __init__(self, dim, out_dim, block_types, block_dims, block_depth=2):
        super(Decoder, self).__init__()
        block_dims = list(reversed(block_dims))
        self.up_blocks = nn.ModuleList()
        for i, (block_type, block_dim) in enumerate(zip(block_types, block_dims)):
            conv_type, conv_up = nn.Conv2d if "Block2D" in block_type else Conv3d, None
            if i < len(block_dims) - 1:
                kernel_size = 3 if i < len(block_dims) - 2 or conv_type is nn.Conv2d else (1, 3, 3)
                conv_up = partial(conv_type, kernel_size=kernel_size)
            args = (block_dims[max(i - 1, 0)], block_dim, conv_type, block_depth + 1)
            self.up_blocks += [UNetResBlock(*args, upsample=conv_up)]
        self.conv_in = conv_type(dim, block_dims[0], 3, 1, 1)
        self.mid_block = UNetMidBlock(block_dims[0], conv_type)
        self.conv_act = nn.SiLU()
        self.conv_norm_out = nn.GroupNorm(32, block_dims[-1], eps=1e-6)
        self.conv_out = conv_type(block_dims[-1], out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class AutoencoderKLOpenSora(ModelMixin, ConfigMixin, TilingMixin):
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
        sample_size=256,
        scaling_factor=0.18215,
        shift_factor=None,
        latents_mean=None,
        latents_std=None,
        force_upcast=True,
        use_quant_conv=True,
        use_post_quant_conv=True,
    ):
        super(AutoencoderKLOpenSora, self).__init__()
        TilingMixin.__init__(self, sample_min_t=17, latent_min_t=5, sample_ovr_t=1, latent_ovr_t=1)
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, latent_channels, down_block_types, channels, layers)
        self.decoder = Decoder(latent_channels, out_channels, up_block_types, channels, layers)
        quant_conv_type = type(self.decoder.conv_in) if use_quant_conv else nn.Identity
        post_quant_conv_type = type(self.decoder.conv_in) if use_post_quant_conv else nn.Identity
        self.quant_conv = quant_conv_type(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = post_quant_conv_type(latent_channels, latent_channels, 1)
        self.latent_dist = DiagonalGaussianDistribution

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
        extra_dim = 2 if isinstance(self.quant_conv, Conv3d) and x.dim() == 4 else None
        z = self.tiled_encoder(self.forward(x))
        z = self.quant_conv(z)
        z = z.squeeze_(extra_dim) if extra_dim is not None else z
        posterior = DiagonalGaussianDistribution(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z) -> DecoderOutput:
        """Decode the input latents."""
        extra_dim = 2 if isinstance(self.quant_conv, Conv3d) and z.dim() == 4 else None
        z = z.unsqueeze_(extra_dim) if extra_dim is not None else z
        z = self.post_quant_conv(self.forward(z))
        x = self.tiled_decoder(z)
        x = x.squeeze_(extra_dim) if extra_dim is not None else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
