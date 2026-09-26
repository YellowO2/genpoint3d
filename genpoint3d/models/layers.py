"""
The machinery. You rarely need to open this file.

These are the standard transformer building blocks, adapted from
`tesfaldet/genpt` (whose transformer descends from HDiT, Crowson et al.,
ICML 2024). `model.py` assembles them into our architecture -- read that one
to see the design; read this one only to check the maths.

What each piece is for:

  RMSNorm      cheaper LayerNorm, no mean subtraction. The 2026 default.
  AdaRMSNorm   how conditioning enters the network: it predicts a per-channel
               scale for the norm. This is what "AdaLN" means.
  Attention    with QK-norm -- q and k are L2-normalised and given a learnable
               per-head temperature, which stops attention logits exploding.
  RoPE         relative position baked into attention rather than added to the
               token. 1 axis = time, 3 axes = a point's xyz.
  GEGLU        gated feed-forward. Slightly better than plain GELU per param.
  zero_init    every residual branch starts as a no-op, so an untrained block
               is the identity and depth costs nothing at initialisation.

Shapes: B batch, N tokens, D model dim, H heads, Dh head dim.
"""

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
    """RMSNorm that does not undo autocast.

    The obvious spelling upcasts `x` itself -- `x.float() * scale.float() *
    rsqrt(...)` -- which under bf16 autocast materialises three full-size fp32
    copies of the input. On the context tensor, 226 MB per batch of 16, that ran
    once per cross block and dominated the step's memory.

    `mean(dtype=torch.float32)` accumulates the reduction in fp32 without
    materialising an fp32 copy, which is the part that actually needs the
    precision. The reciprocal is a single scalar per row, so scaling stays in
    the input dtype.
    """
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
    """RMSNorm whose scale is predicted from the conditioning vector (= AdaLN).

    Zero-init means it starts at `scale = 1`, i.e. an ordinary RMSNorm, and the
    conditioning only takes effect as training moves the weights off zero.

    Note this conditioning is purely *multiplicative*. `rms_norm(0) = 0`, so a
    token stream of exact zeros stays zero through the whole network and no
    gradient flows -- never feed zeros as tokens.
    """

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
    """Rotary position embedding over `n_axes` continuous coordinates.

    `n_axes=1` with the frame index gives standard temporal RoPE. `n_axes=3`
    with a point's xyz gives the axial variant the paper uses for spatial
    attention -- the head's rotary channels are split evenly across x, y, z.

    Frequencies are log-spaced and differ per head, so heads attend at
    different wavelengths.
    """

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
        theta = pos[..., None, None] * self.freqs.to(pos.dtype)  # (..., N, n_axes, H, per_axis)
        theta = theta.movedim(-2, -3)                            # (..., N, H, n_axes, per_axis)
        theta = theta.flatten(-2)                                # (..., N, H, d)
        return theta.transpose(-3, -2)                            # (..., H, N, d)


# ----------------------------------------------------------------- attention

class Attention(nn.Module):
    """Self- or cross-attention with QK-norm and optional RoPE.

    The learnable per-head `scale` replaces the usual `1/sqrt(d)`: q and k are
    L2-normalised first, so logits are bounded cosine similarities and the
    model learns how sharp it wants each head to be.

    Passing `context` makes this cross-attention -- that is the hook step 3
    uses to let point tokens query the DINOv3 feature map.
    """

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
    """One attention + one feed-forward, both residual and both AdaLN-conditioned.

    The same class serves temporal and spatial attention -- only the tensor
    reshape and the RoPE positions differ, and `model.py` handles those.
    """

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
    """Cross-attention + feed-forward. Point tokens read the visual feature map.

    Same shape as `Block`, but keys and values come from `context` instead of
    from the tokens themselves. This is stage [4c], and it is the whole
    tracking/forecasting switch: pass real features to track, pass the learned
    null embedding to forecast. No branching anywhere else in the model.
    """

    def __init__(self, dim: int, num_heads: int, mlp_mult: int, cond_dim: int,
                 locality: bool = True) -> None:
        super().__init__()
        self.norm_q = AdaRMSNorm(dim, cond_dim)
        self.norm_kv = RMSNorm(dim)
        self.attn = Attention(dim, num_heads, context_dim=dim)
        self.norm_ff = AdaRMSNorm(dim, cond_dim)
        self.ff_up = GEGLU(dim, dim * mlp_mult)
        self.ff_down = zero_init(nn.Linear(dim * mlp_mult, dim, bias=False))
        # How sharply to prefer patches near the point's current position. One
        # scalar per block, so early blocks may look broadly and later ones
        # narrowly. Softplus keeps it positive: a NEGATIVE value would prefer
        # patches far away, which is never what is wanted.
        # Created only when enabled, so a checkpoint from before this existed
        # still loads under `--locality 0`.
        self.locality = nn.Parameter(torch.tensor(1.0)) if locality else None

    def forward(self, x, cond, context, dist2=None):
        """`dist2` (B, N, P): squared distance from each point's current position
        estimate to each patch, in the shared normalised space.

        Added to the attention logits as `-dist2 * locality`, which turns the
        search "which of 576 patches is mine?" into the arithmetic "which are
        near me?". Geometry answers it; the model only has to compare
        appearances among the survivors -- what a correlation volume does in
        CoTracker and TAPIP3D, expressed as an attention prior.
        """
        bias = None
        if dist2 is not None and self.locality is not None:
            # (B, 1, N, P) broadcasts over heads: a per-head bias would cost
            # num_heads times the memory for the same prior.
            bias = (-dist2 * F.softplus(self.locality))[:, None].to(context.dtype)
        x = x + self.attn(self.norm_q(x, cond), context=self.norm_kv(context),
                          attn_mask=bias)
        x = x + self.ff_down(self.ff_up(self.norm_ff(x, cond)))
        return x


# --------------------------------------------------------------- embeddings

class FourierEmbedding(nn.Module):
    """Map a low-dimensional continuous value to a high-dimensional vector.

    A raw scalar like `k = 0.31` is a terrible network input -- nearby values
    look nearly identical. Projecting onto random sinusoids of many frequencies
    makes small differences immediately distinguishable.
    """

    def __init__(self, in_dim: int, out_dim: int, std: float = 16.0) -> None:
        super().__init__()
        if out_dim % 2:
            raise ValueError("out_dim must be even")
        self.register_buffer("freqs", torch.randn(in_dim, out_dim // 2) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = 2 * math.pi * (x @ self.freqs)
        return torch.cat((proj.sin(), proj.cos()), dim=-1)
