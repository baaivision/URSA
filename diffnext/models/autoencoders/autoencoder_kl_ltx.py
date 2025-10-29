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
"""Simple implementation of AutoEncoderKL for LTX v0.95."""

from einops import rearrange
import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_outputs import AutoencoderKLOutput
from diffusers.models.modeling_utils import ModelMixin

from diffnext.models.autoencoders.modeling_utils import DiagonalGaussianDistribution
from diffnext.models.autoencoders.modeling_utils import DecoderOutput, TilingMixin


class Conv3d(nn.Conv3d):
    """3D convolution."""

    def __init__(self, *args, **kwargs):
        self.causal = kwargs.pop("causal", True)
        super(Conv3d, self).__init__(*args, **kwargs)
        self.padding = (0,) + tuple((_ // 2 for _ in self.kernel_size[1:]))
        self.pad1 = nn.ReplicationPad3d((0,) * 4 + (self.kernel_size[0] - 1, 0))
        self.pad2 = nn.ReplicationPad3d((0,) * 4 + (self.pad1.padding[-2] // 2,) * 2)
        self.pad1 = nn.Identity() if self.kernel_size[0] == 1 else self.pad1
        self.pad2 = nn.Identity() if self.kernel_size[0] == 1 else self.pad2

    def forward(self, x):
        return super().forward(self.pad1(x) if self.causal else self.pad2(x))


class RMSNorm(nn.Module):
    """RMS normalization."""

    def forward(self, x):
        # Enforce high precision RMS to avoid float16 underflow.
        return x.mul(x.float().square().mean(-1, True).add_(1e-8).rsqrt().to(x.dtype))


class TimeEmbed(nn.Module):
    """Time embedding layer."""

    def __init__(self, embed_dim, freq_dim=256):
        super(TimeEmbed, self).__init__()
        self.timestep_proj = nn.Module()
        self.timestep_proj.fc1 = nn.Linear(freq_dim, embed_dim)
        self.timestep_proj.fc2 = nn.Linear(embed_dim, embed_dim)
        self.freq_dim, self.time_freq = freq_dim, None

    def get_freq_embed(self, timestep) -> torch.Tensor:
        if self.time_freq is None:
            dim, log_theta = self.freq_dim // 2, 9.210340371976184  # math.log(10000)
            freq = torch.arange(dim, dtype=torch.float32, device=timestep.device)
            self.time_freq = freq.mul(-log_theta / dim).exp().unsqueeze_(0)
        emb = timestep.unsqueeze(-1).float() * self.time_freq
        return torch.cat([emb.cos(), emb.sin()], dim=-1).to(dtype=timestep.dtype)

    def forward(self, temb) -> torch.Tensor:
        x = self.get_freq_embed(temb) if temb.dim() == 1 else temb
        return self.timestep_proj.fc2(nn.functional.silu(self.timestep_proj.fc1(x)))


class ResBlock(nn.Module):
    """Resnet block."""

    def __init__(self, dim, out_dim, causal=True):
        super(ResBlock, self).__init__()
        self.norm1, self.norm2 = RMSNorm(), RMSNorm()
        self.conv1 = Conv3d(dim, out_dim, 3, causal=causal)
        self.conv2 = Conv3d(out_dim, out_dim, 3, causal=causal)
        self.nonlinearity, self.dropout = nn.SiLU(), nn.Dropout(0, inplace=True)
        self.scale_shift_table = None if causal else nn.Parameter(torch.randn(4, dim) / dim**0.5)

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        shortcut, stats = x, []
        if self.scale_shift_table is not None:
            stats = temb.add(self.scale_shift_table.view(1, -1))[..., None, None, None].chunk(4, 1)
        x = self.norm1(x.movedim(1, -1)).movedim(-1, 1)
        x = x.mul(1 + stats[1]).add_(stats[0]) if stats else x
        x = self.conv1(self.nonlinearity(x))
        x = self.norm2(x.movedim(1, -1)).movedim(-1, 1)
        x = x.mul(1 + stats[3]).add_(stats[2]) if stats else x
        return self.conv2(self.dropout(self.nonlinearity(x))).add_(shortcut)


class MidBlock(nn.Module):
    """UNet mid block."""

    def __init__(self, dim, depth=1, causal=True):
        super(MidBlock, self).__init__()
        self.time_embed = None if causal else TimeEmbed(dim * 4)
        self.resnets = nn.ModuleList(ResBlock(dim, dim, causal=causal) for _ in range(depth))

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        temb = self.time_embed(temb) if self.time_embed else None
        for resnet in self.resnets:
            x = resnet(x, temb)
        return x


class Downsample(nn.Module):
    """Residual downsample layer."""

    def __init__(self, dim, out_dim, stride, causal=True):
        super(Downsample, self).__init__()
        self.stride = stride = stride if isinstance(stride, (tuple, list)) else (stride,) * 3
        self.group_size = (dim * torch.Size(stride).numel()) // out_dim
        self.pad_t, conv_dim = stride[0] - 1, out_dim // torch.Size(stride).numel()
        self.conv = Conv3d(dim, conv_dim, 3, 1, causal=causal)
        self.patch_args = {"r": stride[0], "p": stride[1], "q": stride[2]}
        self.patch_args["pattern"] = "b c (t r) (h p) (w q) -> b (c r p q) t h w"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.pad(x, (0,) * 4 + (self.pad_t, 0), "replicate") if self.pad_t else x
        shortcut = rearrange(x, **self.patch_args).unflatten(1, (-1, self.group_size)).mean(dim=2)
        return rearrange(self.conv(x), **self.patch_args).add_(shortcut)


class Upsample(nn.Module):
    """Residual upsample layer."""

    def __init__(self, dim, out_dim, stride, causal=False):
        super(Upsample, self).__init__()
        self.stride = stride = stride if isinstance(stride, (tuple, list)) else (stride,) * 3
        self.repeats = (out_dim * torch.Size(stride).numel()) // dim
        self.slice_t, conv_dim = stride[0] - 1, out_dim * torch.Size(stride).numel()
        self.conv = Conv3d(dim, conv_dim, 3, 1, causal=causal)
        self.patch_args = {"r": stride[0], "p": stride[1], "q": stride[2]}
        self.patch_args["pattern"] = "b (c r p q) t h w -> b c (t r) (h p) (w q)"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = rearrange(x, **self.patch_args).repeat(1, self.repeats, 1, 1, 1)
        x = rearrange(self.conv(x), **self.patch_args)
        x = x[:, :, self.slice_t :] if self.slice_t else x
        return x.add_(shortcut[:, :, self.slice_t :] if self.slice_t else shortcut)


class DownBlock(nn.Module):
    """Downsample block."""

    def __init__(self, dim, out_dim, depth=1, causal=True, downsample=""):
        super(DownBlock, self).__init__()
        self.resnets, self.downsamplers = nn.ModuleList(), nn.ModuleList()
        for _ in range(depth):
            self.resnets.append(ResBlock(dim, dim, causal=causal))
        for _ in range(1 if downsample else 0):
            stride = {"spatial": (1, 2, 2), "temporal": (2, 1, 1), "spatiotemporal": 2}[downsample]
            self.downsamplers.append(Downsample(dim, out_dim, stride, causal=causal))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for resnet in self.resnets:
            x = resnet(x)
        for downsampler in self.downsamplers:
            x = downsampler(x)
        return x


class UpBlock(nn.Module):
    """Upsample block."""

    def __init__(self, dim, out_dim, depth=1, causal=False, upscale_factor=2):
        super(UpBlock, self).__init__()
        self.time_embed = TimeEmbed(out_dim * 4)
        self.resnets, self.upsamplers = nn.ModuleList(), nn.ModuleList()
        for _ in range(1 if upscale_factor > 1 else 0):
            self.upsamplers.append(Upsample(dim, out_dim, 2, causal=causal))
        for _ in range(depth):
            self.resnets.append(ResBlock(out_dim, out_dim, causal=causal))

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        for upsampler in self.upsamplers:
            x = upsampler(x)
        temb = self.time_embed(temb)
        for resnet in self.resnets:
            x = resnet(x, temb)
        return x


class Encoder(nn.Module):
    """VAE encoder."""

    def __init__(self, dim, out_dim, block_dims, block_depths, patch_size=4):
        super(Encoder, self).__init__()
        self.patch_args = {"p": patch_size, "q": patch_size}
        downsample_type = ["spatial", "temporal", "spatiotemporal", "spatiotemporal"]
        self.conv_in = Conv3d(dim * patch_size**2, block_dims[0], 3, 1)
        self.down_blocks = nn.ModuleList()
        for i, (in_dim, depth, down) in enumerate(zip(block_dims, block_depths, downsample_type)):
            blk = DownBlock(in_dim, block_dims[i + 1], depth, downsample=down)
            self.down_blocks.append(blk)
        self.mid_block = MidBlock(block_dims[-1], block_depths[-1])
        self.norm_out, self.conv_act = RMSNorm(), nn.SiLU()
        self.conv_out = Conv3d(block_dims[-1], out_dim + 1, 3, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b c t (h p) (w q) -> b (c q p) t h w", **self.patch_args)
        x = self.conv_in(x)
        for down_block in self.down_blocks:
            x = down_block(x)
        x = self.mid_block(x)
        x = self.norm_out(x.movedim(1, -1)).movedim(-1, 1)
        return self.conv_out(self.conv_act(x))


class Decoder(nn.Module):
    """VAE decoder."""

    def __init__(self, dim, out_dim, block_dims, block_depths, patch_size=4):
        super(Decoder, self).__init__()
        block_dims = tuple(reversed(block_dims))
        self.patch_args = {"p": patch_size, "q": patch_size}
        self.conv_in = Conv3d(dim, block_dims[0], 3, 1, causal=False)
        self.mid_block = MidBlock(block_dims[0], block_depths[-1], causal=False)
        self.up_blocks = nn.ModuleList([])
        for in_dim, depth in zip(block_dims, block_depths[:-1]):
            self.up_blocks.append(UpBlock(in_dim, in_dim // 2, depth, upscale_factor=2))
        self.norm_out, self.conv_act = RMSNorm(), nn.SiLU()
        self.conv_out = Conv3d(block_dims[-1], out_dim * patch_size**2, 3, 1, causal=False)
        self.time_embed = TimeEmbed(block_dims[-1] * 2)
        self.scale_shift_table = nn.Parameter(torch.randn(2, block_dims[-1]))
        self.timestep_scale = nn.Parameter(torch.tensor(1000, dtype=torch.float32))

    def forward(self, x: torch.Tensor, temb: torch.Tensor = None) -> torch.Tensor:
        x = self.conv_in(x)
        temb = self.time_embed.get_freq_embed(temb * self.timestep_scale)
        x = self.mid_block(x, temb)
        for up_block in self.up_blocks:
            x = up_block(x, temb)
        x = self.norm_out(x.movedim(1, -1)).movedim(-1, 1)
        temb = self.time_embed(temb)
        stats = temb.add(self.scale_shift_table.view(1, -1))[..., None, None, None].chunk(2, 1)
        x = x.mul(1 + stats[1]).add_(stats[0])
        x = self.conv_out(self.conv_act(x))
        return rearrange(x, "b (c q p) t h w -> b c t (h p) (w q)", **self.patch_args)


class AutoencoderKLLTXVideo(ModelMixin, ConfigMixin, TilingMixin):
    """AutoEncoder KL."""

    @register_to_config
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        down_block_types=("LTXVideoDownBlock3D",) * 4,
        block_out_channels=(128, 256, 512, 1024, 2048),
        layers_per_block=(4, 6, 6, 2, 2),
        decoder_block_out_channels=(128, 256, 512, 1024),
        decoder_layers_per_block=(5, 5, 5, 5),
        act_fn="silu",
        latent_channels=128,
        sample_size=1024,
        scaling_factor=1.0,
        shift_factor=None,
        latents_mean=None,
        latents_std=None,
        patch_size=4,
    ):
        super(AutoencoderKLLTXVideo, self).__init__()
        TilingMixin.__init__(self, sample_min_t=249, latent_min_t=32, sample_ovr_t=1)
        channels, layers = block_out_channels, layers_per_block
        self.encoder = Encoder(in_channels, latent_channels, channels, layers)
        channels, layers = decoder_block_out_channels, decoder_layers_per_block
        self.decoder = Decoder(latent_channels, out_channels, channels, layers)
        self.register_buffer("shift_factors", torch.zeros(latents_mean) if latents_mean else None)
        self.register_buffer("scaling_factors", torch.ones(latents_std) if latents_std else None)
        self.latent_dist = DiagonalGaussianDistribution

    def scale_(self, x) -> torch.Tensor:
        """Scale the input latents."""
        if self.shift_factors is not None:
            return x.sub_(self.shift_factors).mul_(self.scaling_factors)
        x.add_(-self.config.shift_factor) if self.config.shift_factor else None
        return x.mul_(self.config.scaling_factor)

    def unscale_(self, x) -> torch.Tensor:
        """Unscale the input latents."""
        if self.shift_factors is not None:
            return x.div_(self.scaling_factors).add_(self.shift_factors)
        x.mul_(1 / self.config.scaling_factor)
        return x.add_(self.config.shift_factor) if self.config.shift_factor else x

    def encode(self, x) -> AutoencoderKLOutput:
        """Encode the input samples."""
        z = self.tiled_encoder(self.forward(x))
        posterior = self.latent_dist(z)
        return AutoencoderKLOutput(latent_dist=posterior)

    def decode(self, z, temb: torch.Tensor = None) -> DecoderOutput:
        """Decode the input latents."""
        if temb is None:
            temb = torch.tensor([0] * z.size(0), dtype=z.dtype, device=z.device)
        extra_dim = 2 if z.dim() == 4 else None
        z = z.unsqueeze_(extra_dim) if extra_dim is not None else z
        x = self.tiled_decoder(self.forward(z), temb=temb)
        x = x.squeeze_(extra_dim) if extra_dim is not None else x
        return DecoderOutput(sample=x)

    def forward(self, x):  # NOOP.
        return x
