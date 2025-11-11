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
"""Simple implementation of AutoEncoderVQ for Cosmos3D."""

import math

import torch
from einops import rearrange
from torch import nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import IdentityDistribution
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, TilingMixin
from diffnext.models.autoencoders.quantizers import FSQuantizer
from diffnext.models.autoencoders.wavelets_utils import Patcher3D


class GroupNorm2D(nn.GroupNorm):
    """2D group normalization."""

    def forward(self, x) -> torch.Tensor:
        x, bsz = super().forward(x.transpose(1, 2).flatten(0, 1)), x.size(0)
        return rearrange(x, "(b t) c h w -> b c t h w", b=bsz)


class Conv3d(nn.Conv3d):
    """3D convolution."""

    def __init__(self, *args, **kwargs):
        stride_t = kwargs.pop("time_stride", None)
        super(Conv3d, self).__init__(*args, **kwargs)
        pad_t = (self.kernel_size[0] - 1) + (1 - (stride_t or self.stride[0]))
        self.stride = (stride_t or self.stride[0],) + self.stride[1:]
        self.padding = (0,) + self.padding[1:]
        self.pad = nn.ReplicationPad3d((0,) * 4 + (pad_t, 0))
        self.pad = nn.Identity() if self.kernel_size[0] == 1 else self.pad

    @classmethod
    def new_factorized(cls, dim, out_dim):
        return nn.Sequential(cls(dim, out_dim, (1, 3, 3), 1, 1), cls(out_dim, out_dim, (3, 1, 1)))

    def forward(self, x) -> torch.Tensor:
        return super(Conv3d, self).forward(self.pad(x))


class Attention(nn.Module):
    """Multi-headed attention."""

    def __init__(self, dim, perm="(b t) 1 (h w) c"):
        super(Attention, self).__init__()
        self.group_norm, self.perm = GroupNorm2D(1, dim, eps=1e-6), perm
        self.to_q, self.to_k, self.to_v = [nn.Linear(dim, dim) for _ in range(3)]
        self.to_out = nn.ModuleList([nn.Linear(dim, dim)])

    @classmethod
    def new_factorized(cls, dim) -> nn.Sequential:
        return nn.Sequential(cls(dim, "(b t) 1 (h w) c"), cls(dim, "(b h w) 1 t c"))

    def forward(self, x) -> torch.Tensor:
        shortcut, x, (bsz, _, _, h, w) = x, self.group_norm(x), x.size()
        x = rearrange(x, "b c t h w -> %s" % self.perm)
        q, k, v = [f(x) for f in (self.to_q, self.to_k, self.to_v)]
        o = self.to_out[0](nn.functional.scaled_dot_product_attention(q, k, v))
        return rearrange(o, "%s -> b c t h w" % self.perm, b=bsz, h=h, w=w).add_(shortcut)


class Resize(nn.Module):
    """Downsample layer."""

    def __init__(self, dim, spatial=1, temporal=1):
        super(Resize, self).__init__()
        self.spatial, self.temporal = spatial, temporal
        self.conv1, self.conv2 = nn.Identity(), nn.Identity()
        if spatial == 1 or temporal == 1:  # Down.
            self.conv1 = Conv3d(dim, dim, (1, 3, 3), 2, time_stride=1)
            self.conv2 = Conv3d(dim, dim, (3, 1, 1), 1, time_stride=2) if temporal else self.conv2
        elif spatial == 2 or temporal == 2:  # Up.
            self.conv1 = Conv3d(dim, dim, (3, 1, 1), 1, 0) if temporal else self.conv1
            self.conv2 = Conv3d(dim, dim, (1, 3, 3), 1, 1)
        self.conv3 = Conv3d(dim, dim, 1) if spatial or temporal else nn.Identity()

    def forward(self, x) -> torch.Tensor:
        if self.spatial == 1:
            _ = nn.functional.avg_pool3d(x, (1, 2, 2), (1, 2, 2))
            x = self.conv1(nn.functional.pad(x, (0, 1, 0, 1, 0, 0))).add_(_)
        if self.temporal == 1:
            x = nn.functional.pad(x, (0, 0, 0, 0, 1, 0), "replicate")
            x = self.conv2(x).add_(nn.functional.avg_pool3d(x, (2, 1, 1), (2, 1, 1)))
        if self.temporal == 2:
            x = x.repeat_interleave(2, dim=2)[:, :, 1:]
            x = self.conv1(x).add_(x)
        if self.spatial == 2:
            x = x.repeat_interleave(2, dim=3).repeat_interleave(2, dim=4)
            x = self.conv2(x).add_(x)
        return self.conv3(x)


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim, out_dim):
        super(ResBlock, self).__init__()
        self.norm1 = GroupNorm2D(1, dim, eps=1e-6)
        self.conv1 = Conv3d.new_factorized(dim, out_dim)
        self.norm2 = GroupNorm2D(1, out_dim, eps=1e-6)
        self.conv2 = Conv3d.new_factorized(out_dim, out_dim)
        self.conv_shortcut = Conv3d(dim, out_dim, 1) if out_dim != dim else None
        self.nonlinearity, self.dropout = nn.SiLU(), nn.Dropout(0)

    def forward(self, x) -> torch.Tensor:
        shortcut = self.conv_shortcut(x) if self.conv_shortcut else x
        x = self.conv1(self.nonlinearity(self.norm1(x)))
        return self.conv2(self.nonlinearity(self.norm2(x))).add_(shortcut)


class UNetResBlock(nn.Module):
    """UNet resnet block."""

    def __init__(self, dim, out_dim, depth=2, downsample=None, upsample=None):
        super(UNetResBlock, self).__init__()
        block_dims = [(out_dim, out_dim) if i > 0 else (dim, out_dim) for i in range(depth)]
        self.resnets = nn.ModuleList(ResBlock(*dims) for dims in block_dims)
        self.downsamplers = nn.ModuleList([Resize(out_dim, *downsample)]) if downsample else []
        self.upsamplers = nn.ModuleList([Resize(out_dim, *upsample)]) if upsample else []

    def forward(self, x) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        x = self.downsamplers[0](x) if self.downsamplers else x
        return self.upsamplers[0](x) if self.upsamplers else x


class UNetMidBlock(nn.Module):
    """UNet mid block."""

    def __init__(self, dim, depth=1):
        super(UNetMidBlock, self).__init__()
        self.resnets = nn.ModuleList(ResBlock(dim, dim) for _ in range(depth + 1))
        self.attentions = nn.ModuleList(Attention.new_factorized(dim) for _ in range(depth))

    def forward(self, x) -> torch.Tensor:
        x = self.resnets[0](x)
        for attn, resnet in zip(self.attentions, self.resnets[1:]):
            x = resnet(attn(x))
        return x


class Encoder(nn.Module):
    """AE encoder."""

    def __init__(
        self,
        dim,
        out_dim,
        block_dims,
        block_depth,
        patch_size=4,
        temporal_stride=8,
        spatial_stride=8,
    ):
        super(Encoder, self).__init__()
        spatial_downs = int(math.log2(spatial_stride)) - int(math.log2(patch_size))
        temporal_downs = int(math.log2(temporal_stride)) - int(math.log2(patch_size))
        self.patcher = Patcher3D(patch_size)
        self.conv_in = Conv3d.new_factorized(dim * patch_size**3, block_dims[0])
        self.down_blocks = nn.ModuleList()
        for i, dim in enumerate(block_dims[:-1]):
            downsample, block_dim = None, block_dims[i + 1]
            if i < len(block_dims) - 2:
                downsample = int(i < spatial_downs), int(i < temporal_downs)
            args = (dim, block_dim, block_depth)
            self.down_blocks += [UNetResBlock(*args, downsample=downsample)]
        self.mid_block = UNetMidBlock(block_dim)
        self.conv_norm_out, self.conv_act = GroupNorm2D(1, block_dim, eps=1e-6), nn.SiLU()
        self.conv_out = Conv3d.new_factorized(block_dim, out_dim)

    def forward(self, x) -> torch.Tensor:
        x = torch.cat([x[:, :, :1].repeat_interleave(self.patcher.patch_size, 2), x[:, :, 1:]], 2)
        for _ in range(self.patcher.num_strides):
            x = self.patcher.dwt(x)
        x = self.conv_in(x)
        for blk in self.down_blocks:
            x = blk(x)
        x = self.mid_block(x)
        return self.conv_out(self.conv_act(self.conv_norm_out(x)))


class Decoder(nn.Module):
    """AE decoder."""

    def __init__(
        self,
        dim,
        out_dim,
        block_dims,
        block_depth,
        patch_size=4,
        temporal_stride=8,
        spatial_stride=8,
    ):
        super(Decoder, self).__init__()
        block_dims = list(reversed(block_dims))
        spatial_ups = int(math.log2(spatial_stride)) - int(math.log2(patch_size))
        temporal_ups = int(math.log2(temporal_stride)) - int(math.log2(patch_size))
        self.patcher = Patcher3D(patch_size)
        self.conv_in = Conv3d.new_factorized(dim, block_dims[0])
        self.mid_block = UNetMidBlock(block_dims[0])
        self.up_blocks = nn.ModuleList()
        for i, block_dim in enumerate(block_dims[:-1]):
            upsample, dim = None, block_dims[max(i - 1, 0)]
            if i < len(block_dims) - 2:
                temporal = 0 < i < temporal_ups + 1
                spatial = temporal or (i < spatial_ups and spatial_ups > temporal_ups)
                upsample = (2 if spatial else 0, 2 if temporal else 0)
            args = (dim, block_dim, block_depth + 1)
            self.up_blocks += [UNetResBlock(*args, upsample=upsample)]
        self.conv_norm_out, self.conv_act = GroupNorm2D(1, block_dim, eps=1e-6), nn.SiLU()
        self.conv_out = Conv3d.new_factorized(block_dim, out_dim * patch_size**3)

    def forward(self, x) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.mid_block(x)
        for blk in self.up_blocks:
            x = blk(x)
        x = self.conv_out(self.conv_act(self.conv_norm_out(x)))
        for _ in range(self.patcher.num_strides):
            x = self.patcher.idwt(x)
        return x[:, :, self.patcher.patch_size - 1 :]


class AutoencoderVQCosmos3D(ModelMixin, ConfigMixin, TilingMixin):
    """AutoEncoder VQ."""

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock3D",) * 3,
        up_block_types=("UpDecoderBlock3D",) * 3,
        block_out_channels=(128, 256, 512, 512),
        layers_per_block=2,
        act_fn="silu",
        latent_channels=16,
        norm_num_groups=1,
        sample_size=1024,
        sample_frames=17,
        num_vq_embeddings=64000,
        vq_embed_dim=6,
        force_upcast=False,
        patch_size=4,
        temporal_stride=4,
        spatial_stride=8,
        _quantizer_name="FSQuantizer",
    ):
        super(AutoencoderVQCosmos3D, self).__init__()
        latent_min_t = (sample_frames - 1) // temporal_stride + 1
        TilingMixin.__init__(self, sample_frames, latent_min_t=latent_min_t, sample_ovr_t=1)
        extra_args = {"patch_size": patch_size}
        extra_args.update({"temporal_stride": temporal_stride, "spatial_stride": spatial_stride})
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, latent_channels, channels, layers, **extra_args)
        self.decoder = Decoder(latent_channels, out_channels, channels, layers, **extra_args)
        self.quant_conv = Conv3d(latent_channels, vq_embed_dim, 1)
        self.post_quant_conv = Conv3d(vq_embed_dim, latent_channels, 1)
        self.quantizer, self.latent_dist = FSQuantizer(), IdentityDistribution

    def scale_(self, x) -> torch.Tensor:
        """Scale the input latents."""
        return x

    def unscale_(self, x) -> torch.Tensor:
        """Unscale the input latents."""
        return x

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        z = self.tiled_encoder(self.forward(x))
        z = self.quant_conv(z)
        posterior = self.latent_dist(self.quantizer.quantize(z))
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, ids) -> DecoderOutput:
        """Decode the input indices."""
        z = self.quantizer.dequantize(ids)
        extra_dim = 2 if z.dim() == 4 else None
        z = z.unsqueeze_(extra_dim) if extra_dim is not None else z
        z = self.post_quant_conv(self.forward(z))
        x = self.tiled_decoder(z)
        x = x.squeeze_(extra_dim) if extra_dim is not None else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
