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
"""Wavelets utilities.

References:

[Cosmos](https://github.com/nvidia-cosmos/cosmos-predict1/blob/main/cosmos_predict1/tokenizer/modules/patching.py)
"""

import math

import torch
from torch import nn


class Patcher2D(nn.Module):
    """2D discrete wavelet transform."""

    def __init__(self, patch_size=4):
        super(Patcher2D, self).__init__()
        self.rescale_factor = 2
        self.patch_size, self.num_strides = patch_size, int(math.log2(patch_size))
        wavelets1 = torch.tensor([0.7071067811865476] * 2)
        wavelets2 = wavelets1 * ((-1) ** torch.arange(2))
        self.register_buffer("wavelets1", wavelets1, persistent=False)
        self.register_buffer("wavelets2", wavelets2, persistent=False)

    def dwt(self, x) -> torch.Tensor:
        g = x.size(1)
        hl = self.wavelets1.flip(0).view(1, 1, -1).repeat(g, 1, 1)
        hh = self.wavelets2.view(1, 1, -1).repeat(g, 1, 1)
        x1, out = nn.functional.pad(x, (0, 1, 0, 1), "reflect"), []
        for w1 in (hl, hh):
            x2 = nn.functional.conv2d(x1, w1[:, :, None, :], stride=(1, 2), groups=g)
            for w2 in (hl, hh):
                out.append(nn.functional.conv2d(x2, w2[:, :, :, None], stride=(2, 1), groups=g))
        return torch.cat(out, dim=1).mul_(1 / self.rescale_factor)

    def idwt(self, x) -> torch.Tensor:
        g = x.size(1) // 4
        hl = self.wavelets1.flip([0]).view(1, 1, -1).repeat([g, 1, 1])
        hh = self.wavelets2.view(1, 1, -1).repeat(g, 1, 1)
        out = list(torch.chunk(x, 4, dim=1))
        for i in range(2):
            for j, w in enumerate((hl, hh)):
                x, w = out[i * 2 + j], w[:, :, :, None]
                out.append(nn.functional.conv_transpose2d(x, w, stride=(2, 1), groups=g))
        out = [out[i] + out[i + 1] for i in range(4, 8, 2)]
        for j, w in enumerate((hl, hh)):
            x, w = out[j], w[:, :, None, :]
            out.append(nn.functional.conv_transpose2d(x, w, stride=(1, 2), groups=g))
        return out[2].add(out[3]).mul_(self.rescale_factor)

    def forward(self, x) -> torch.Tensor:
        for _ in range(self.num_strides):
            x = self.dwt(x)
        return x


class Patcher3D(Patcher2D):
    """3D discrete wavelet transform."""

    def __init__(self, patch_size=4):
        super(Patcher3D, self).__init__(patch_size)
        self.rescale_factor = 2 * 2**0.5

    def dwt(self, x) -> torch.Tensor:
        g = x.size(1)
        hl = self.wavelets1.flip(0).view(1, 1, -1).repeat(g, 1, 1)
        hh = self.wavelets2.view(1, 1, -1).repeat(g, 1, 1)
        x1, out = nn.functional.pad(x, (0, 1, 0, 1, 0, 1), "reflect"), []
        for w1 in (hl, hh):
            x2 = nn.functional.conv3d(x1, w1[:, :, :, None, None], stride=(2, 1, 1), groups=g)
            for w2 in (hl, hh):
                x3 = nn.functional.conv3d(x2, w2[:, :, None, :, None], stride=(1, 2, 1), groups=g)
                for w3 in (hl, hh):
                    w3 = w3[:, :, None, None, :]
                    out.append(nn.functional.conv3d(x3, w3, stride=(1, 1, 2), groups=g))
        return torch.cat(out, dim=1).mul_(1.0 / self.rescale_factor)

    def idwt(self, x) -> torch.Tensor:
        g = x.size(1) // 8
        hl = self.wavelets1.flip([0]).view(1, 1, -1).repeat([g, 1, 1])
        hh = self.wavelets2.view(1, 1, -1).repeat(g, 1, 1)
        out = list(torch.chunk(x, 8, dim=1))
        for i in range(4):
            for j, w in enumerate((hl, hh)):
                x, w = out[i * 2 + j], w[:, :, None, None, :]
                out.append(nn.functional.conv_transpose3d(x, w, stride=(1, 1, 2), groups=g))
        out = [out[i] + out[i + 1] for i in range(8, 16, 2)]
        for i in range(2):
            for j, w in enumerate((hl, hh)):
                x, w = out[i * 2 + j], w[:, :, None, :, None]
                out.append(nn.functional.conv_transpose3d(x, w, stride=(1, 2, 1), groups=g))
        out = [out[i] + out[i + 1] for i in range(4, 8, 2)]
        for j, w in enumerate((hl, hh)):
            x, w = out[j], w[:, :, :, None, None]
            out.append(nn.functional.conv_transpose3d(x, w, stride=(2, 1, 1), groups=g))
        return out[2].add(out[3]).mul_(self.rescale_factor)

    def forward(self, x) -> torch.Tensor:
        x = torch.cat([x[:, :, :1].repeat_interleave(self.patch_size, 2), x[:, :, 1:]], 2)
        for _ in range(self.num_strides):
            x = self.dwt(x)
        return x
