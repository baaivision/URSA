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
"""Discrete quantizers."""

import math
import torch
from torch import nn


class VQuantizer(nn.Identity):
    """Vector Quantizer."""

    def __init__(self, n_e, vq_embed_dim):
        super(VQuantizer, self).__init__()
        self.n_e, self.vq_embed_dim = n_e, vq_embed_dim
        self.embedding = nn.Embedding(n_e, vq_embed_dim)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to indices."""
        ids = nn.functional.linear(z.transpose(1, -1), self.embedding.weight).argmax(-1).int()
        return ids.permute(0, 2, 3, 1) if ids.dim() > 3 else ids.permute(0, 2, 1)

    def dequantize(self, ids) -> torch.Tensor:
        """Dequantize indices to z."""
        z = self.embedding(ids)
        return z.permute(0, 4, 1, 2, 3) if z.dim() > 4 else z.permute(0, 3, 1, 2)


class LFQuantizer(nn.Identity):
    """Lookup-Free Quantizer."""

    def __init__(self, n_e, vq_embed_dim):
        super(LFQuantizer, self).__init__()
        self.groups = vq_embed_dim // int(math.log2(n_e))
        self.n_e, self.vq_embed_dim = n_e, vq_embed_dim // self.groups
        self.embedding = torch.nn.Embedding(n_e, self.vq_embed_dim)
        del self.embedding.weight
        basis = 2 ** torch.arange(self.vq_embed_dim, dtype=torch.int32)
        weight = 2 * torch.arange(n_e).unsqueeze(-1).bitwise_and(basis).ne(0).float() - 1
        self.register_buffer("basis", basis, persistent=False)
        self.embedding.register_buffer("weight", weight, persistent=False)

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to indices."""
        ids = z.transpose(1, -1).unflatten(-1, (self.groups, -1)).gt(0).int().mul(self.basis).sum(-1)  # noqa
        ids = ids.permute(0, 2, 3, 1, 4) if ids.dim() > 4 else ids.permute(0, 2, 1, 3)
        return ids if self.groups > 1 else ids.squeeze(-1)

    def dequantize(self, ids) -> torch.Tensor:
        """Dequantize indices to z."""
        z = self.embedding(ids).flatten(-2 if self.groups > 1 else -1)
        return z.permute(0, 4, 1, 2, 3) if z.dim() > 4 else z.permute(0, 3, 1, 2)


class FSQuantizer(nn.Identity):
    """Finite Scalar Quantizer."""

    def __init__(self, levels=(8, 8, 8, 5, 5, 5)):
        super(FSQuantizer, self).__init__()
        self.n_e, self.vq_embed_dim = torch.Size(levels).numel(), len(levels)
        basis = torch.cumprod(torch.tensor([1] + list(levels[:-1])), dim=0, dtype=torch.int32)
        self.register_buffer("scalar", torch.zeros(0), persistent=False)  # Dummy dtype indicator.
        self.register_buffer("levels", torch.tensor(levels, dtype=torch.int32), persistent=False)
        self.register_buffer("half_width", self.levels // 2, persistent=False)  # For normalization.
        self.register_buffer("basis", basis, persistent=False)  # Quantization basis.

    def bound(self, z: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        """Bound z."""
        half_l = (self.levels - 1) * (1 + eps) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = (offset / half_l).atanh()
        return (z + shift).tanh() * half_l - offset

    def quantize(self, z: torch.Tensor) -> torch.Tensor:
        """Quantize z to indices."""
        z_q = self.bound(z.transpose(1, -1)).round()
        ids = (z_q + self.half_width).mul(self.basis).sum(-1).int()
        return ids.permute(0, 2, 3, 1) if ids.dim() > 3 else ids.permute(0, 2, 1)

    def dequantize(self, ids) -> torch.Tensor:
        """Dequantize indices to z."""
        z_q = ids.unsqueeze(-1).floor_divide(self.basis).fmod(self.levels) - self.half_width
        z = z_q.div(self.half_width).to(self.scalar.dtype)
        return z.permute(0, 4, 1, 2, 3) if z.dim() > 4 else z.permute(0, 3, 1, 2)
