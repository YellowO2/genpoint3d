"""The transformer blocks `PointDiT.py` is built from.

Adapted from `tesfaldet/genpt` (HDiT): RMSNorm, AdaRMSNorm (the conditioning
scales each channel), attention with QK-norm and RoPE, GEGLU feed-forward, and
residual branches that start at zero."""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


# ------------------------------------------------------------ normalisation

def zero_init(layer: nn.Linear, almost: bool = True) -> nn.Linear:
    """Start a residual branch at (almost) zero so the block begins as identity."""
    if almost:
        nn.init.normal_(layer.weight, std=1e-4)
    else:
        nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    return layer


def rms_norm(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMSNorm with the mean taken in fp32 but no fp32 copy of `x`, which
    under autocast was the step's largest memory cost."""
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True, dtype=torch.float32)
    inv = torch.rsqrt(mean_sq + eps).to(x.dtype)     # (..., 1), cheap
    return x * inv * scale.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.scale, self.eps)


class AdaRMSNorm(nn.Module):
    """RMSNorm whose scale comes from the conditioning (AdaLN), starting at 1.
    It only multiplies: a token of exact zeros stays zero."""

    def __init__(self, dim: int, cond_dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.linear = zero_init(nn.Linear(cond_dim, dim, bias=False))

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return rms_norm(x, self.linear(cond) + 1.0, self.eps)


class GEGLU(nn.Linear):
    """Linear -> split in half -> gate one half by GELU of the other."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__(in_dim, out_dim * 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = super().forward(x).chunk(2, dim=-1)
        return value * F.gelu(gate)


# ---------------------------------------------------------------------- RoPE

def apply_rope(x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
    """Rotate the leading `2*d` channels of `x` by angles `theta`.

    `x`: (..., H, N, Dh), `theta`: (..., H, N, d) with `2*d <= Dh`.
    Channels beyond `2*d` pass through untouched.
    """
    d = theta.shape[-1]
    x1, x2, rest = x[..., :d], x[..., d : 2 * d], x[..., 2 * d :]
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1)


class RoPE(nn.Module):
    """Rotary position embedding over `n_axes` coordinates (1: time, 3: xyz,
    channels split evenly), log-spaced from pi to `max_freq` pi per unit and
    different per head: built for positions in about [-1, 1]."""

    def __init__(self, head_dim: int, num_heads: int, n_axes: int = 1, max_freq: float = 10.0) -> None:
        super().__init__()
        per_axis = (head_dim // 2) // n_axes
        if per_axis == 0:
            raise ValueError(f"head_dim {head_dim} too small for {n_axes} rotary axes")
        self.n_axes, self.per_axis = n_axes, per_axis

        n = num_heads * per_axis
        freqs = torch.linspace(math.log(math.pi), math.log(max_freq * math.pi), n + 1)[:-1].exp()
        # (n_axes, H, per_axis) -- each axis gets its own copy of the ladder
        self.register_buffer("freqs", freqs.view(per_axis, num_heads).T.contiguous()[None].repeat(n_axes, 1, 1))

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """`pos`: (..., N, n_axes) -> theta (..., H, N, n_axes*per_axis)."""
        # fp32: bf16 cannot hold an angle of up to 10 pi per unit.
        with torch.autocast(device_type=pos.device.type, enabled=False):
            theta = pos.float()[..., None, None] * self.freqs  # (..., N, n_axes, H, per_axis)
        theta = theta.movedim(-2, -3)                            # (..., N, H, n_axes, per_axis)
        theta = theta.flatten(-2)                                # (..., N, H, d)
        return theta.transpose(-3, -2)                            # (..., H, N, d)


# ----------------------------------------------------------------- attention

class Attention(nn.Module):
    """Self- or cross-attention (with `context`), q and k L2-normalised with a
    learned temperature per head, and optional RoPE."""

    def __init__(self, dim: int, num_heads: int, context_dim: Optional[int] = None) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.scale = nn.Parameter(torch.full([num_heads], 10.0))
        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_kv = nn.Linear(context_dim if context_dim else dim, dim * 2, bias=False)
        self.to_out = zero_init(nn.Linear(dim, dim, bias=False))

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, Dh)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        theta: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._split(self.to_q(x))
        k, v = (self._split(t) for t in self.to_kv(context if context is not None else x).chunk(2, dim=-1))

        s = self.scale[None, :, None, None]
        q = F.normalize(q, dim=-1) * s.sqrt()
        k = F.normalize(k, dim=-1) * s.sqrt()

        if theta is not None:
            q, k = apply_rope(q, theta), apply_rope(k, theta)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, scale=1.0)
        return self.to_out(out.transpose(1, 2).flatten(2))


class Block(nn.Module):
    """Attention then feed-forward, both residual and AdaLN-conditioned; the
    temporal and spatial blocks differ only in how `PointDiT` lays out tokens."""

    def __init__(self, dim: int, num_heads: int, mlp_mult: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_attn = AdaRMSNorm(dim, cond_dim)
        self.attn = Attention(dim, num_heads)
        self.norm_ff = AdaRMSNorm(dim, cond_dim)
        self.ff_up = GEGLU(dim, dim * mlp_mult)
        self.ff_down = zero_init(nn.Linear(dim * mlp_mult, dim, bias=False))

    def forward(self, x, cond, theta=None, attn_mask=None):
        x = x + self.attn(self.norm_attn(x, cond), theta=theta, attn_mask=attn_mask)
        x = x + self.ff_down(self.ff_up(self.norm_ff(x, cond)))
        return x


class CrossBlock(nn.Module):
    """Each point reads the patches of its frame: the same shape as `Block`,
    keys and values from `context`."""

    # The bias that switches a patch off; finite, so it survives bf16.
    FAR = 100.0

    def __init__(self, dim: int, num_heads: int, mlp_mult: int, cond_dim: int) -> None:
        super().__init__()
        self.norm_q = AdaRMSNorm(dim, cond_dim)
        self.norm_kv = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, context_dim=dim)
        self.norm_ff = AdaRMSNorm(dim, cond_dim)
        self.ff_up = GEGLU(dim, dim * mlp_mult)
        self.ff_down = zero_init(nn.Linear(dim * mlp_mult, dim, bias=False))

    def forward(self, x, cond, context, window):
        """`window` (B, N, P): 0 on the patches a point may read, -FAR on the rest."""
        x = x + self.attn(self.norm_q(x, cond), context=self.norm_kv(context),
                          attn_mask=window[:, None].to(context.dtype))
        x = x + self.ff_down(self.ff_up(self.norm_ff(x, cond)))
        return x


# --------------------------------------------------------------- embeddings

class FourierEmbedding(nn.Module):
    """A low-dimensional value as sines and cosines of random frequencies, so
    nearby values are told apart."""

    def __init__(self, in_dim: int, out_dim: int, std: float = 16.0) -> None:
        super().__init__()
        if out_dim % 2:
            raise ValueError("out_dim must be even")
        self.register_buffer("freqs", torch.randn(in_dim, out_dim // 2) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 in and out: bf16 rounds `x` enough to scramble the phase.
        with torch.autocast(device_type=x.device.type, enabled=False):
            proj = 2 * math.pi * (x.float() @ self.freqs)
            return torch.cat((proj.sin(), proj.cos()), dim=-1)
