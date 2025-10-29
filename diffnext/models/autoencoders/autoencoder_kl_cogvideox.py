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
"""Simple implementation of AutoEncoderKL for CogVideoX."""

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import DiagonalGaussianDistribution
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, TilingMixin


class Conv3d(nn.Conv3d):
    """3D convolution layer."""

    def __init__(self, *args, **kwargs):
        super(Conv3d, self).__init__(*args, **kwargs)
        self.padding = (0,) + self.padding[1:]
        self.pad = nn.ReplicationPad3d((0,) * 4 + (self.kernel_size[0] - 1, 0))
        self.pad = nn.Identity() if self.kernel_size[0] == 1 else self.pad

    def forward(self, x) -> torch.Tensor:
        x = self.pad(x)
        num_splits = x.numel() // 1073741824 + 1
        if num_splits == 1 or x.size(2) <= 3:
            return super().forward(x)
        if self.kernel_size[0] == 1:
            return torch.cat([super(Conv3d, self).forward(x) for x in x.chunk(num_splits, 2)], 2)
        x, ks = list(x.chunk(num_splits, 2)), self.kernel_size[0]
        for i in range(num_splits - 1, -1, -1):
            x[i] = super().forward(torch.cat((x[i - 1][:, :, -ks + 1 :], x[i]), 2) if i else x[i])
        return torch.cat(x, 2)


class AdaGroupNorm(nn.GroupNorm):
    """Adaptive group normalization layer."""

    def __init__(self, dim, z_dim=None, num_groups=32, eps=1e-6):
        super(AdaGroupNorm, self).__init__(num_groups, dim, eps=eps)
        self.scale = Conv3d(z_dim, dim, 1) if z_dim else None
        self.shift = Conv3d(z_dim, dim, 1) if z_dim else None

    def forward(self, x, z=None) -> torch.Tensor:
        if not self.scale or z is None:
            return super().forward(x)
        t, h, w = x.shape[2:]
        if t > 1 and t % 2 == 1:
            _ = nn.functional.interpolate(z[:, :, :1], (1, h, w))
            z = torch.cat([_, nn.functional.interpolate(z[:, :, 1:], (t - 1, h, w))], 2)
        else:
            z = nn.functional.interpolate(z, (t, h, w))
        return super().forward(x).mul_(self.scale(z)).add_(self.shift(z))


class Resize(nn.Module):
    """Resize layer."""

    def __init__(self, dim, downsample=1, upsample=0):
        super(Resize, self).__init__()
        self.downsample, self.upsample = downsample, upsample
        self.conv = nn.Conv2d(dim, dim, 3, 2, 0) if downsample else None
        self.conv = nn.Conv2d(dim, dim, 3, 1, 1) if upsample else self.conv

    def forward(self, x) -> torch.Tensor:
        c, t, h, w = x.shape[1:]
        if self.downsample == 2 and t > 1:
            x = x.permute(0, 3, 4, 1, 2).reshape((-1, c, t))
            x = torch.cat([x[..., :1], nn.functional.avg_pool1d(x[..., 1:], 2, 2)], dim=-1)
            x = x.view(-1, h, w, c, x.size(-1)).permute(0, 4, 3, 1, 2)
        elif self.upsample == 2 and t > 1:
            x1 = x[:, :, :1].repeat_interleave(2, 3).repeat_interleave(2, 4)
            x2 = x[:, :, 1:].repeat_interleave(2, 2).repeat_interleave(2, 3).repeat_interleave(2, 4)
            x = torch.cat([x1, x2], dim=2) if x1 is not None else x2
        elif self.downsample:
            x = x.permute(0, 2, 1, 3, 4)
        elif self.upsample:
            x = x.repeat_interleave(2, 3).repeat_interleave(2, 4)
        if self.downsample:
            t, c, h, w = x.shape[1:]
            x = self.conv(nn.functional.pad(x.flatten(0, 1), (0, 1, 0, 1)))
        elif self.upsample:
            c, t, h, w = x.shape[1:]
            x = self.conv(x.permute(0, 2, 1, 3, 4).flatten(0, 1))
        return x.view(*((-1, t, c) + x.shape[-2:])).transpose(1, 2)


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim, out_dim, z_dim=None):
        super(ResBlock, self).__init__()
        self.norm1 = AdaGroupNorm(dim, z_dim)
        self.norm2 = AdaGroupNorm(out_dim, z_dim)
        self.conv1 = Conv3d(dim, out_dim, 3, 1, 1)
        self.conv2 = Conv3d(out_dim, out_dim, 3, 1, 1)
        self.conv_shortcut = Conv3d(dim, out_dim, 1) if out_dim != dim else None
        self.nonlinearity, self.dropout = nn.SiLU(), nn.Dropout(0, inplace=True)

    def forward(self, x, z=None) -> torch.Tensor:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut else x
        x = self.norm1(x, z) if z is not None else self.norm1(x)
        x = self.conv1(self.nonlinearity(x))
        x = self.norm2(x, z) if z is not None else self.norm2(x)
        return self.conv2(self.dropout(self.nonlinearity(x))).add_(shortcut)


class UNetMidBlock(nn.Module):
    """UNet mid block."""

    def __init__(self, dim, z_dim=None, depth=2):
        super(UNetMidBlock, self).__init__()
        self.resnets = nn.ModuleList(ResBlock(dim, dim, z_dim) for _ in range(depth))

    def forward(self, x, z=None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, z)
        return x


class UNetResBlock(nn.Module):
    """UNet resnet block."""

    def __init__(self, dim, out_dim, depth, z_dim=None, downsample=0, upsample=0):
        super(UNetResBlock, self).__init__()
        block_dims = [(out_dim, out_dim) if i > 0 else (dim, out_dim) for i in range(depth)]
        self.resnets = nn.ModuleList(ResBlock(*(dims + (z_dim,))) for dims in block_dims)
        self.downsamplers = nn.ModuleList([Resize(out_dim, downsample)]) if downsample else []
        self.upsamplers = nn.ModuleList([Resize(out_dim, 0, upsample)]) if upsample else []

    def forward(self, x, z=None) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x, z)
        x = self.downsamplers[0](x) if self.downsamplers else x
        return self.upsamplers[0](x) if self.upsamplers else x


class Encoder(nn.Module):
    """VAE encoder."""

    def __init__(self, dim, out_dim, block_dims, block_depth):
        super(Encoder, self).__init__()
        self.conv_in = Conv3d(dim, block_dims[0], 3, 1, 1)
        self.down_blocks = nn.ModuleList()
        for i, block_dim in enumerate(block_dims):
            downsample = 2 if i < 2 else (1 if i < (len(block_dims) - 1) else 0)
            args = (block_dims[max(i - 1, 0)], block_dim, block_depth)
            self.down_blocks += [UNetResBlock(*args, downsample=downsample)]
        self.mid_block = UNetMidBlock(block_dims[-1])
        self.conv_norm_out = AdaGroupNorm(block_dims[-1])
        self.conv_act = nn.SiLU()
        self.conv_out = Conv3d(block_dims[-1], 2 * out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """VAE decoder."""

    def __init__(self, dim, out_dim, block_dims, block_depth):
        super(Decoder, self).__init__()
        block_dims = list(reversed(block_dims))
        self.up_blocks = nn.ModuleList()
        for i, block_dim in enumerate(block_dims):
            upsample = 2 if i < 2 else (1 if i < (len(block_dims) - 1) else 0)
            args = (block_dims[max(i - 1, 0)], block_dim, block_depth + 1, dim)
            self.up_blocks += [UNetResBlock(*args, upsample=upsample)]
        self.conv_in = Conv3d(dim, block_dims[0], 3, 1, 1)
        self.mid_block = UNetMidBlock(block_dims[0], dim)
        self.conv_act = nn.SiLU()
        self.conv_norm_out = AdaGroupNorm(block_dims[-1], dim)
        self.conv_out = Conv3d(block_dims[-1], out_dim, 3, 1, 1)

    def forward(self, x) -> torch.Tensor:
        x, z = self.conv_in(x), x
        x = self.mid_block(x, z)
        for blk in self.up_blocks:
            x = blk(x, z)
        return self.conv_out(self.conv_act(self.conv_norm_out(x, z)))


class AutoencoderKLCogVideoX(ModelMixin, ConfigMixin, TilingMixin):
    """AutoEncoder KL."""

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("CogVideoXDownBlock3D",) * 4,
        up_block_types=("CogVideoXUpBlock3D",) * 4,
        block_out_channels=(128, 256, 256, 512),
        layers_per_block=3,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=32,
        sample_size=480,
        scaling_factor=0.7,
        shift_factor=None,
        latents_mean=None,
        latents_std=None,
        force_upcast=True,
        use_quant_conv=False,
        use_post_quant_conv=False,
    ):
        super(AutoencoderKLCogVideoX, self).__init__()
        TilingMixin.__init__(self, sample_min_t=17, latent_min_t=5, sample_ovr_t=1)
        self.encoder = Encoder(in_channels, latent_channels, block_out_channels, layers_per_block)
        self.decoder = Decoder(latent_channels, out_channels, block_out_channels, layers_per_block)
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
        z = self.tiled_encoder(self.forward(x))
        z = self.quant_conv(z)
        posterior = DiagonalGaussianDistribution(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z) -> DecoderOutput:
        """Decode the input latents."""
        extra_dim = 2 if z.dim() == 4 else None
        z = z.unsqueeze_(extra_dim) if extra_dim is not None else z
        z = self.post_quant_conv(self.forward(z))
        x = self.tiled_decoder(z)
        x = x.squeeze_(extra_dim) if extra_dim is not None else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
