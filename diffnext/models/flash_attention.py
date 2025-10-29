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
"""Flash attention layers. Copied from https://github.com/Dao-AILab/flash-attention"""

import torch

# RoPE (Triton)
try:
    from flash_attn.layers.rotary import apply_rotary_emb
except ImportError:
    from einops import rearrange, repeat

    def rotate_half(x, interleaved=False) -> torch.Tensor:
        if not interleaved:
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

    def apply_rotary_emb(x, cos, sin, interleaved=False, inplace=False) -> torch.Tensor:
        ro_dim = cos.shape[-1] * 2
        cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
        return torch.cat(
            [
                x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin,
                x[..., ro_dim:],
            ],
            -1,
        )


# SwiGLU (TorchJIT)
swiglu_fwd_codestring = """
template <typename T> T swiglu_fwd(T x, T y) {
    return float(x) * float(y) / (1.0f + ::exp(-float(x)));
}
"""
swiglu_bwd_codestring = """
template <typename T> void swiglu_bwd(T x, T y, T g, T& dx, T& dy) {
    float x_sigmoid = 1.0f / (1.0f + ::exp(-float(x)));
    dx = x_sigmoid * (1 + float(x) * (1.0f - x_sigmoid)) * float(g) * float(y);
    dy = float(x) * x_sigmoid * float(g);
}
"""
swiglu_fwd = torch.cuda.jiterator._create_jit_fn(swiglu_fwd_codestring)
swiglu_bwd = torch.cuda.jiterator._create_multi_output_jit_fn(swiglu_bwd_codestring, num_outputs=2)


class SwiGLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_bwd(x, y, dout)


swiglu = SwiGLUFunction.apply

# RMSNorm (Triton)
try:
    from flash_attn.ops.triton.layer_norm import RMSNorm
except ImportError:

    class RMSNorm(torch.nn.Module):

        def __init__(self, hidden_size, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.mul(x.float().square().mean(-1, True).add_(self.eps).rsqrt().to(x.dtype))
            return x * self.weight


# CrossEntropy (Triton)
try:
    from flash_attn.ops.triton.cross_entropy import cross_entropy_loss
except ImportError:
    cross_entropy_loss = None
