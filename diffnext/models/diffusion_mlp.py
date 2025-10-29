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
"""Diffusion MLP."""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as apply_ckpt

from diffnext.models.embeddings import PatchEmbed
from diffnext.models.normalization import AdaLayerNormZero


class Projector(nn.Module):
    """MLP Projector layer."""

    def __init__(self, dim, mlp_dim=None, out_dim=None):
        super(Projector, self).__init__()
        self.fc1 = nn.Linear(dim, mlp_dim or dim)
        self.fc2 = nn.Linear(mlp_dim or dim, out_dim or dim)
        self.activation = nn.SiLU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class DiffusionBlock(nn.Module):
    """Diffusion block."""

    def __init__(self, dim):
        super(DiffusionBlock, self).__init__()
        self.dim, self.mlp_checkpointing = dim, False
        self.norm1 = AdaLayerNormZero(dim, num_stats=3, eps=1e-6)
        self.proj, self.norm2 = Projector(dim, dim, dim), nn.LayerNorm(dim)

    def forward(self, x, z) -> torch.Tensor:
        if self.mlp_checkpointing and x.requires_grad:
            h, (gate,) = apply_ckpt(self.norm1, x, z, use_reentrant=False)
            return self.norm2(apply_ckpt(self.proj, h, use_reentrant=False)).mul(gate).add_(x)
        h, (gate,) = self.norm1(x, z)
        return self.norm2(self.proj(h)).mul(gate).add_(x)


class TimeCondEmbed(nn.Module):
    """Time-Condition embedding layer."""

    def __init__(self, cond_dim, embed_dim, freq_dim=256):
        super(TimeCondEmbed, self).__init__()
        self.timestep_proj = Projector(freq_dim, embed_dim, embed_dim)
        self.condition_proj = Projector(cond_dim, embed_dim, embed_dim)
        self.freq_dim, self.time_freq = freq_dim, None

    def get_freq_embed(self, timestep, dtype) -> torch.Tensor:
        if self.time_freq is None:
            dim, log_theta = self.freq_dim // 2, 9.210340371976184  # math.log(10000)
            freq = torch.arange(dim, dtype=torch.float32, device=timestep.device)
            self.time_freq = freq.mul(-log_theta / dim).exp().unsqueeze(0)
        emb = timestep.unsqueeze(-1).float() * self.time_freq
        return torch.cat([emb.cos(), emb.sin()], dim=-1).to(dtype=dtype)

    def forward(self, timestep, z) -> torch.Tensor:
        t = self.timestep_proj(self.get_freq_embed(timestep, z.dtype))
        return self.condition_proj(z).add_(t.unsqueeze_(1) if t.dim() == 2 else t)


class DiffusionMLP(nn.Module):
    """Diffusion MLP model."""

    def __init__(self, depth, embed_dim, cond_dim, patch_size=2, image_dim=4):
        super(DiffusionMLP, self).__init__()
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.time_cond_embed = TimeCondEmbed(cond_dim, embed_dim)
        self.blocks = nn.ModuleList(DiffusionBlock(embed_dim) for _ in range(depth))
        self.norm = AdaLayerNormZero(embed_dim, num_stats=2, eps=1e-6)
        self.head = nn.Linear(embed_dim, patch_size**2 * image_dim)

    def forward(self, x, timestep, z, pred_ids=None) -> torch.Tensor:
        x, o = self.patch_embed(x), None if pred_ids is None else x
        o = None if pred_ids is None else self.patch_embed.patchify(o)
        x = x if pred_ids is None else x.gather(1, pred_ids.expand(-1, -1, x.size(-1)))
        z = z if pred_ids is None else z.gather(1, pred_ids.expand(-1, -1, z.size(-1)))
        z = self.time_cond_embed(timestep, z)
        for blk in self.blocks:
            x = blk(x, z)
        x = self.norm(x, z)[0]
        x = self.head(x)
        return x if pred_ids is None else o.scatter(1, pred_ids.expand(-1, -1, x.size(-1)), x)
