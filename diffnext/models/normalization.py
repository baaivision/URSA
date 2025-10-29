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
"""Normalization Layers."""

from typing import Tuple

import torch
from torch import nn


class AdaLayerNormZero(nn.Module):
    """Adaptive LayerNorm with residual stats."""

    def __init__(self, dim, rank=None, num_stats=2, eps=1e-6):
        super(AdaLayerNormZero, self).__init__()
        self.lora = nn.Linear(dim, rank, bias=False) if rank else nn.Identity()
        self.proj = nn.Linear(rank if rank else dim, num_stats * dim)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False) if eps else nn.Identity()
        self.activation, self.num_stats = nn.SiLU(), num_stats

    def forward(self, x, z) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        stats = self.proj(self.lora(self.activation(z))).chunk(self.num_stats, dim=-1)
        return self.norm(x).mul(1 + stats[0]).add_(stats[1]), stats[2:]


class AdaLayerNorm(AdaLayerNormZero):
    """Adaptive LayerNorm."""

    def __init__(self, dim, rank=None, eps=1e-6):
        super(AdaLayerNorm, self).__init__(dim, rank, num_stats=2, eps=eps)

    def forward(self, x, z) -> torch.Tensor:
        return super().forward(x, z)[0]


class AdaLayerNormSingle(nn.Module):
    """Adaptive LayerNorm with shared residual stats."""

    def __init__(self, dim, num_stats=2, eps=1e-6):
        super(AdaLayerNormSingle, self).__init__()
        self.bias = nn.Parameter(torch.randn(num_stats, dim) / dim**0.5)
        self.norm = nn.LayerNorm(dim, eps, elementwise_affine=False) if eps else nn.Identity()
        self.num_stats = num_stats

    def forward(self, x, z) -> Tuple[torch.Tensor, Tuple[torch.Tensor]]:
        axis = -2 if z.size(-1) == self.bias.size(-1) else -1
        bias = self.bias.flatten(-1 if z.size(-1) == self.bias.size(-1) else 0)
        stats = z.add(bias).chunk(self.num_stats, dim=axis)
        return self.norm(x).mul(1 + stats[0]).add_(stats[1]), stats[2:]
