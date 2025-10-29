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
"""Embedding layers."""

import sys
from typing import List, Tuple, Union

import numpy as np
import scipy.stats as stats
import torch
from torch import nn


class FlexRotaryEmbedding(nn.Identity):
    """Flexible rotary position embedding layer."""

    class PEFunc(object):
        """Apply RoPE weight to Q/K tensor."""

        def __init__(self, weight: torch.Tensor):
            self.weight = weight

        @torch.compile(fullgraph=True, disable=sys.platform != "linux")
        def interleaved_impl(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return w[..., 0].mul(x[..., 0]).add_(w[..., 1] * x[..., 1]).flatten(3)

        @torch.compile(fullgraph=True, disable=sys.platform != "linux")
        def partitioned_impl(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return w[..., 0].mul(x[:, :, :, 0]).add_(w[..., 1] * x[:, :, :, 1]).flatten(3)

        def __call__(self, x: torch.Tensor, interleaved=False) -> torch.Tensor:
            w = self.weight = self.weight.to(dtype=x.dtype)
            x = x.unflatten(-1, (-1, 1, 2) if interleaved else (2, -1, 1))
            return (self.interleaved_impl if interleaved else self.partitioned_impl)(x, w)

    @staticmethod
    def from_config(config):
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        return FlexRotaryEmbedding(head_dim, base=config.rope_theta)

    def __init__(self, dim=128, base=10000.0):
        super(FlexRotaryEmbedding, self).__init__()
        self.dim, self.base = dim, base
        self.rep1, self.rep2 = dim // 8, (dim // 2 - dim // 8 * 3) // 2
        self.register_buffer("scale", torch.arange(0, dim, 2).float() / dim, persistent=False)

    def get_pos(self, input_shape, shift=0, has_bov=True) -> torch.Tensor:
        num_blocks = 1 if len(input_shape) < 4 else input_shape[-4]
        block_size = 1 if len(input_shape) < 3 else input_shape[-3]
        grid_shape = [num_blocks * block_size] + list(input_shape[-2:])
        pos = torch.zeros(grid_shape + [3], dtype=torch.int32, device=self.scale.device)
        grid = [torch.arange(_, device=pos.device) for _ in grid_shape]
        [pos[..., i].add_(grid[i].view([-1 if i == j else 1 for j in range(3)])) for i in range(3)]
        pos, device = pos.unflatten(0, (-1, block_size)).flatten(1, 3), pos.device
        bov_pos = torch.arange(num_blocks, device=device).view(-1, 1, 1).repeat(1, 1, 3)
        pos[..., 0] += torch.arange(num_blocks, device=device).view(-1, 1).add_(shift + has_bov)
        return torch.cat([bov_pos.mul(block_size + 1).add(shift), pos], 1) if has_bov else pos

    def get_func(self, pos: torch.Tensor, *args, **kwargs) -> PEFunc:
        t = torch.cat([pos.repeat(1, 1, self.rep1), pos[..., 1:].repeat(1, 1, self.rep2)], -1)
        freq = t * torch.pow(self.base, self.scale.float()).reciprocal_().unsqueeze(0)
        freq = torch.stack([freq.cos(), -freq.sin(), freq.sin(), freq.cos()], dim=-1)
        return self.PEFunc(freq.view(freq.shape[:-1] + (2, 2)).unsqueeze(2))


class RotaryEmbed3D(nn.Identity):
    """3D rotary position embedding layer."""

    class PEFunc(object):
        """Apply RoPE weight to Q/K tensor."""

        def __init__(self, weight: torch.Tensor):
            self.weight = weight

        @torch.compile(fullgraph=True, disable=sys.platform != "linux")
        def call_impl(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
            return w[..., 0].mul(x[..., 0]).add_(w[..., 1] * x[..., 1]).flatten(3)

        def __call__(self, x: torch.Tensor) -> torch.Tensor:
            x = x.view(*x.shape[:-1], -1, 1, 2)
            w = self.weight = self.weight.to(dtype=x.dtype)
            return self.call_impl(x, w)

    def __init__(self, dim=64, base_size=(16, 16), theta=10000.0):
        super(RotaryEmbed3D, self).__init__()
        self.dim, self.base_size, self.theta = dim, base_size, theta
        for i, rotary_dim in enumerate(([dim // 8] + [(dim - dim // 8) // 2] * 2)):
            scale = torch.arange(0, rotary_dim, 2).float().div_(rotary_dim)
            self.register_buffer("scale%d" % i, scale, persistent=False)

    def get_pos(self, t=1, bs=1, hw=None) -> torch.Tensor:
        thw = [t] + list(hw or self.base_size)
        pos = torch.zeros(thw + [3], device=self.scale1.device)
        grid = [torch.arange(_, device=self.scale1.device) for _ in thw]
        [pos[..., i].add_(grid[i].view([-1 if i == j else 1 for j in range(3)])) for i in range(3)]
        return pos.view(1, -1, 3).expand(bs, -1, -1)

    def get_func(self, pos: torch.Tensor, pad=0, ids: torch.Tensor = None) -> PEFunc:
        pos, weight = pos.gather(1, ids) if ids is not None else pos, []
        pos = nn.functional.pad(pos, (0, 0, pad, 0), value=0) if pad else pos
        for i, grid in enumerate(pos.split(1, dim=-1)):
            freq = torch.pow(self.theta, getattr(self, "scale%d" % i).float())
            freq = grid * freq.reciprocal().unsqueeze(0)
            freq = torch.stack([freq.cos(), -freq.sin(), freq.sin(), freq.cos()], dim=-1)
            weight += [freq.view(freq.shape[:-1] + (2, 2))]
        return self.PEFunc(torch.cat(weight, dim=-3).unsqueeze(1))


class PosEmbed(nn.Module):
    """Position embedding layer."""

    def __init__(self, dim, base_size=(16, 16)):
        super(PosEmbed, self).__init__()
        (self.base_h, self.base_w), self.space_embed = base_size, None
        self.freq_hw = 1 / (10000 ** (torch.arange(dim // 4, dtype=torch.float32) / (dim // 4)))

    def get_space_embed(self, device=None, dtype=None) -> torch.Tensor:
        h, w = self.base_h, self.base_w
        if self.space_embed is not None and self.space_embed.size(0) == h * w:
            return self.space_embed
        grid_h = torch.arange(h, dtype=torch.float32) * (self.base_h / h)
        grid_w = torch.arange(w, dtype=torch.float32) * (self.base_w / w)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing="xy")
        freq_w, freq_h = [_.reshape(-1, 1) * self.freq_hw.unsqueeze(0) for _ in (grid_w, grid_h)]
        embed = torch.cat([freq_w.sin(), freq_w.cos(), freq_h.sin(), freq_h.cos()], dim=-1)
        self.space_embed = embed.to(device=device, dtype=dtype)
        return self.space_embed

    def forward(self, x) -> torch.Tensor:
        return x.add_(self.get_space_embed(x.device, x.dtype))


class VideoPosEmbed(PosEmbed):
    """Video position embedding layer."""

    def __init__(self, dim, base_size):
        super(VideoPosEmbed, self).__init__(dim, base_size=base_size[1:])
        self.base_t, self.time_embed, self.norm = base_size[0], None, nn.LayerNorm(dim)
        self.time_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_t = 1 / (10000 ** (torch.arange(128, dtype=torch.float32).unsqueeze(0) / 128))

    def get_time_embed(self, t) -> torch.Tensor:
        if self.time_embed is not None and t == self.time_embed.size(0):
            return self.norm(self.time_proj(self.time_embed))
        device, dtype = self.time_proj[0].weight.device, self.time_proj[0].weight.dtype
        grid = torch.arange(t, dtype=torch.float32) / (t / self.base_t)
        freq_t = grid.view(-1, 1, 1).mul(self.freq_t)
        sincos = torch.cat([freq_t.sin(), freq_t.cos()], dim=-1)
        self.time_embed = sincos.to(device=device, dtype=dtype)
        return self.norm(self.time_proj(self.time_embed))

    def forward(self, x) -> torch.Tensor:
        x = x.add_(self.get_time_embed(x.size(-3))) if x.dim() == 4 else x
        return x.add_(self.get_space_embed(x.device, x.dtype))


class MotionEmbed(nn.Module):
    """Motion embedding layer."""

    def __init__(self, dim, base_flow=5, base_fps=12):
        super(MotionEmbed, self).__init__()
        self.base_flow, self.base_fps = base_flow, base_fps
        self.flow_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.fps_proj = nn.Sequential(nn.Linear(256, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.freq_m = 1 / (10000 ** (torch.arange(128, dtype=torch.float32).unsqueeze(0) / 128))

    def get_embed(self, c, x, k) -> torch.Tensor:
        x = [getattr(self, f"base_{k}")] * c.size(0) if x is None else x
        freq_m = torch.as_tensor(x).view(-1, 1, 1).float().mul(self.freq_m)
        sincos = torch.cat([freq_m.sin(), freq_m.cos()], dim=-1)
        return getattr(self, f"{k}_proj")(sincos.to(device=c.device, dtype=c.dtype))

    def forward(self, c, flow=None, fps=None) -> torch.Tensor:
        outputs = [self.get_embed(c, x, k) for k, x in [("flow", flow), ("fps", fps)]]
        return torch.cat(outputs, dim=1) if len(outputs) > 1 else outputs[0]


class PatchEmbed(nn.Module):
    """Patch embedding layer."""

    def __init__(self, image_dim, embed_dim, patch_size):
        super(PatchEmbed, self).__init__()
        self.patch_size = patch_size
        self.image_dim, self.height, self.width = image_dim, None, None
        self.proj = nn.Conv2d(image_dim, embed_dim, patch_size, patch_size)

    @property
    def hw(self) -> Tuple[int, int]:
        return self.height, self.width

    def patchify(self, x) -> torch.Tensor:
        x = x.view(-1, self.image_dim, self.height, self.patch_size, self.width, self.patch_size)
        return x.permute(0, 2, 4, 3, 5, 1).flatten(1, 2).flatten(2, 4).contiguous()

    def unpatchify(self, x) -> torch.Tensor:
        x = x.view(-1, self.height, self.width, self.patch_size, self.patch_size, self.image_dim)
        return x.permute(0, 5, 1, 3, 2, 4).flatten(2, 3).flatten(3, 4).contiguous()

    def forward(self, x) -> torch.Tensor:
        flat_shape = (x.size(0), x.size(2)) if x.dim() == 5 else None
        x = x.transpose(1, 2).flatten(0, 1) if x.dim() == 5 else x
        self.width = x.size(-1) // self.patch_size if x.dim() == 4 else self.width
        self.height = x.size(-2) // self.patch_size if x.dim() == 4 else self.height
        x = self.proj(x).flatten(2).transpose(1, 2) if x.dim() == 4 else x
        return x.view(flat_shape + x.shape[1:]) if flat_shape else x


class TextEmbed(nn.Module):
    """Encode text tokens into embeddings."""

    def __init__(self, token_dim, embed_dim, num_tokens=256, dropout=0.1):
        super(TextEmbed, self).__init__()
        self.token_dim, self.num_tokens, self.encoders = token_dim, num_tokens, []
        self.proj, self.norm = nn.Linear(token_dim, embed_dim), nn.LayerNorm(embed_dim)
        self.register_buffer("weight", torch.zeros(512, token_dim))  # Maximum positions.
        _, self.dropout, self.mask = nn.init.normal_(self.weight, std=0.02), dropout, []

    @torch.no_grad()
    def encode_prompts(self, prompts, prompt_size=None) -> torch.Tensor:
        device, dtype = self.weight.device, self.weight.dtype
        x = self.weight[: self.num_tokens].expand(len(prompts), -1, -1).clone()
        for i, p in enumerate(prompts if not isinstance(prompts[0], str) else []):
            if self.training and self.dropout > 0 and np.random.rand() < self.dropout:
                continue
            x[i, : p.shape[0]] = torch.as_tensor(p, device=device).to(dtype)
        if not isinstance(prompts[0], str):
            return x
        tokenizer, encoder = self.encoders
        trunc_args = {"max_length": self.num_tokens, "truncation": True}
        pad_args = {"padding": "max_length", **trunc_args}
        tokens = [tokenizer(p, **pad_args).input_ids for p in prompts]
        maxlens = [len(tokenizer(p, **trunc_args).input_ids) for p in prompts]
        tokens = torch.as_tensor(tokens, device=encoder.device)
        embeds, x = encoder(tokens).last_hidden_state.to(dtype), x.to(encoder.device)
        self.mask = [0] * (x.size(0) // prompt_size if prompt_size else 0)
        for i, maxlen in enumerate([] if prompt_size else maxlens):
            if self.training and self.dropout and np.random.rand() < self.dropout:
                continue
            x[i, :maxlen] = embeds[i, :maxlen]
        for k in range(x.size(0) // prompt_size if prompt_size else 0):
            if np.random.rand() < self.dropout:
                self.mask[k] = 1
                continue
            for j in range(prompt_size):
                if j and np.random.rand() < self.dropout:
                    continue
                i, maxlen = k * prompt_size + j, maxlens[k * prompt_size + j]
                x[i, :maxlen] = embeds[i, :maxlen]
        return x

    def apply_mask(self, x, mask_token=0) -> torch.Tensor:
        """Apply the current mask to input."""
        if len(self.mask) == 0:
            return x
        mask = torch.as_tensor(self.mask, device=x.device, dtype=x.dtype)
        mask = mask.view([-1] + [1] * (x.dim() - 1))
        return x.mul(1 - mask).add_(mask_token * mask)

    def forward(self, x, prompt_size=None) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            return self.norm(self.proj(self.encode_prompts(x, prompt_size)))
        return self.norm(self.proj(x))


class LabelEmbed(nn.Module):
    """Encode class labels into embeddings."""

    def __init__(self, embed_dim, num_classes=1000, dropout=0.1):
        super(LabelEmbed, self).__init__()
        self.dropout, self.num_classes = dropout, num_classes
        self.weight = nn.Parameter(torch.zeros(num_classes + (dropout > 0), embed_dim))
        _, self.norm = nn.init.normal_(self.weight, std=0.02), nn.LayerNorm(embed_dim)

    def forward(self, input_ids):
        input_ids = input_ids.unsqueeze(-1) if input_ids.dim() == 1 else input_ids
        if self.training and self.dropout > 0:
            keep = torch.rand(input_ids.size(), device=input_ids.device).gt(self.dropout)
            input_ids = input_ids.where(keep, self.num_classes)
        return self.norm(self.weight[input_ids])


class MaskEmbed(nn.Module):
    """Apply mask positions to input embeddings."""

    def __init__(self, embed_dim, mask_ratios=(0.7, 1.0)):
        super(MaskEmbed, self).__init__()
        self.mask_ratios = list(mask_ratios) + ([0.25] if len(mask_ratios) == 2 else [])
        self.bos_token = nn.Parameter(torch.zeros(1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        [nn.init.normal_(_, std=0.02) for _ in (self.bos_token, self.mask_token)]
        self.mask, self.attn_mask = None, None
        self.pred_ids, self.pred_pos, self.generator = None, 0, None

    def get_attn_lens(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor]], c: torch.Tensor = None
    ) -> List[int]:
        """Return the attention length according to inputs."""
        lens = [_.shape[1:3].numel() for _ in x] if isinstance(x, (tuple, list)) else []
        lens += [x.size(2)] * x.size(1) if not isinstance(x, (tuple, list)) else []
        lens[0] += c.size(1) if c is not None else 0
        return lens

    def get_attn_mask(
        self, x: Union[torch.Tensor, Tuple[torch.Tensor]], c: torch.Tensor = None, persistent=True
    ) -> torch.Tensor:
        """Return the attention mask according to inputs."""
        if self.attn_mask is not None and persistent:
            return self.attn_mask
        if isinstance(x, (tuple, list)):
            d = torch.cat([torch.full(_.shape[1:3], t) for t, _ in enumerate(x)]).flatten()
        else:
            d = torch.cat([torch.full([x.size(2)], i) for i in range(x.size(1))])
        d = torch.cat([torch.full([c.size(1)], 0), d]) if c is not None else d
        attn_mask = torch.where(d.unsqueeze(1).ge(d.unsqueeze(0)), 0, -float("inf"))
        self.attn_mask = attn_mask.to(device=self.bos_token.device, dtype=self.bos_token.dtype)
        return self.attn_mask

    def get_pred_mask(self, num_preds) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the current mask for next prediction."""
        if self.pred_ids is None:
            u_dist = torch.empty_like(self.mask).uniform_(generator=self.generator)
            self.pred_ids = u_dist.argsort(dim=1)
        pred_ids = self.pred_ids[:, self.pred_pos : self.pred_pos + num_preds]
        pred_mask = torch.zeros_like(self.mask).scatter_(1, pred_ids, 1)
        self.pred_pos, self.mask = self.pred_pos + num_preds, self.mask.mul_(1 - pred_mask)
        return pred_mask, pred_ids

    def apply_mask(self, x) -> torch.Tensor:
        """Apply the current mask to input."""
        return x.mul(1 - self.mask).add_(self.mask_token * self.mask)

    def forward(self, x) -> torch.Tensor:
        if self.training:
            u_dist = torch.rand(x.shape[:-1] + (1,), device=x.device)
            a, b = [(v - 1) / self.mask_ratios[2] for v in self.mask_ratios[:2]]
            mask_ratio = stats.truncnorm(a, b, loc=1, scale=self.mask_ratios[2]).rvs(1)[0]
            prev_ids = u_dist.argsort(1)[:, : int(np.round((1 - mask_ratio) * u_dist.size(1)))]
            self.mask = x.new_ones(u_dist.shape).scatter_(1, prev_ids, 0)
            return self.apply_mask(x), prev_ids
        if self.mask is None:
            self.mask, self.pred_pos = x.new_ones(x.shape[:-1] + (1,)), 0
        return self.apply_mask(x)
