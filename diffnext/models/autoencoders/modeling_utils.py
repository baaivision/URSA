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
"""AutoEncoder utilities."""

from diffusers.models.modeling_outputs import BaseOutput
import torch


class DecoderOutput(BaseOutput):
    """Output of decoding method."""

    sample: torch.Tensor


class IdentityDistribution(object):
    """IdentityGaussianDistribution."""

    def __init__(self, z):
        self.parameters = z

    def sample(self, generator=None):
        return self.parameters


class DiagonalGaussianDistribution(object):
    """DiagonalGaussianDistribution."""

    def __init__(self, z):
        self.parameters = z
        self.device, self.dtype = z.device, z.dtype
        if z.size(1) % 2:
            z = torch.cat([z, z[:, -1:].expand((-1, z.shape[1] - 2) + (-1,) * (z.dim() - 2))], 1)
        self.mean, self.logvar = z.float().chunk(2, dim=1)
        self.logvar = self.logvar.clamp(-30.0, 20.0)
        self.std, self.var = self.logvar.mul(0.5).exp_(), self.logvar.exp()

    def sample(self, generator=None):
        device, dtype = self.mean.device, self.mean.dtype
        norm_dist = torch.randn(self.mean.shape, generator=generator, device=device, dtype=dtype)
        return norm_dist.mul_(self.std).add_(self.mean).to(device=self.device, dtype=self.dtype)


class TilingMixin(object):
    """Base class for input tiling.

    Shape hints:

    print(torch.Size((1, 256, 17, 480, 768)).numel() < 2147483647, "Supported")
    print(torch.Size((1, 256, 17, 576, 1024)).numel() > 2147483647, "Unsupported")

    """

    def __init__(self, sample_min_t=17, latent_min_t=5, sample_ovr_t=1, latent_ovr_t=0):
        self.sample_min_t, self.latent_min_t = sample_min_t, latent_min_t
        self.sample_ovr_t, self.latent_ovr_t = sample_ovr_t, latent_ovr_t

    def tiled_encoder(self, x):
        if x.dim() == 4 or x.size(2) <= self.sample_min_t:
            return self.encoder(x)
        t = x.shape[2]
        t_start = [i for i in range(0, t, self.sample_min_t - self.sample_ovr_t)]
        t_slice = [slice(i, i + self.sample_min_t) for i in t_start]
        t_tiles = [self.encoder(x[:, :, s]) for s in t_slice if s.stop <= t]
        t_tiles = [x[:, :, self.latent_ovr_t :] if i else x for i, x in enumerate(t_tiles)]
        return torch.cat(t_tiles, dim=2)

    def tiled_decoder(self, x, **kwargs):
        if x.dim() == 4 or x.size(2) <= self.latent_min_t:
            return self.decoder(x, **kwargs)
        t = x.shape[2]
        t_start = [i for i in range(0, t, self.latent_min_t - self.latent_ovr_t)]
        t_slice = [slice(i, i + self.latent_min_t) for i in t_start]
        t_tiles = [self.decoder(x[:, :, s], **kwargs) for s in t_slice if s.stop <= t]
        t_tiles = [x[:, :, self.sample_ovr_t :] if i else x for i, x in enumerate(t_tiles)]
        return torch.cat(t_tiles, dim=2)


class HybridMixin(object):
    """Base class for hybrid module."""

    def forward(self, x):
        return self.forward_image(x) if x.dim() == 4 else self.forward_video(x)
