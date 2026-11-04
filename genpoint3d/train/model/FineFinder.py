"""Find, to the pixel, where a point went near its current guess.

A small CNN on the RGB frames (RAFT's encoder as CoTracker ships it), one
128-d cell per 4 px, average-pooled into 4 levels. At each level the K cells
nearest the guess in 3D are compared with the K cells nearest the point in its
query frame (the template), and with the K nearest its guess in the frame
before, by TAPIP3D's `NeighborTransformer`, whose trained weights load as
they are.

Cell j of the stride-4 map is centred on pixel index 4j, so `cell = u * w - 1/8`
for `u` in [0, 1] across the frame; level l's cell j sits on level-0 cell
`2^l j + (2^l - 1) / 2`.
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from genpoint3d.train.model.BasicEncoder import STRIDE, BasicEncoder


class FineFinder(nn.Module):
    """The CNN and the neighbourhood comparisons. The CNN runs once per step
    without a graph (`encode`) and again a clip at a time to carry the
    gradient back (`replay`); the comparison is checkpointed in pieces of
    `chunk` points."""

    def __init__(self, neigh: int = 32, levels: int = 4, feat_dim: int = 128, chunk: int = 256) -> None:
        super().__init__()
        self.neigh, self.levels, self.chunk = neigh, levels, chunk
        # Against the template and against the frame before, levels each.
        self.out_dim = 2 * levels * feat_dim
        self.fnet = BasicEncoder(feat_dim)
        # TAPIP3D's names and shapes (`corr_processor.*`).
        self.in_norm = nn.LayerNorm(feat_dim)
        self.posenc_mlps = nn.ModuleList(_posenc_mlp(feat_dim) for _ in range(levels))
        self.transformers = nn.ModuleList(_NeighborTransformer(feat_dim) for _ in range(levels))
        self.prev_transformers = nn.ModuleList(_NeighborTransformer(feat_dim) for _ in range(levels))

    def copy_prev(self) -> None:
        """Start the frame-before comparison as a copy of the template one."""
        self.prev_transformers.load_state_dict(self.transformers.state_dict())

    def _features(self, frames: torch.Tensor) -> torch.Tensor:
        """(T, H, W, 3) uint8 -> (T, C, H / 4, W / 4). CoTracker's input range."""
        x = frames.permute(0, 3, 1, 2).to(self.fnet.conv1.weight.dtype)
        return self.fnet(x / 127.5 - 1.0)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, T, H, W, 3) uint8 -> (B, T, C, H / 4, W / 4), with no graph."""
        return torch.stack([self._features(f) for f in frames])

    def replay(self, frames: torch.Tensor, fmap: torch.Tensor) -> None:
        """Back-propagate `fmap.grad`, left by every backward pass of the
        step, into the CNN a clip at a time."""
        if fmap.grad is None:
            return
        for f, g in zip(frames, fmap.grad):
            out = self._features(f)
            out.backward(g.to(out.dtype))

    def forward(self, fmap: torch.Tensor, fmap0: torch.Tensor, uv: torch.Tensor, uv0: torch.Tensor,
                xyz: torch.Tensor, xyz0: torch.Tensor, pos: torch.Tensor, pos0: torch.Tensor,
                query_t: torch.Tensor) -> torch.Tensor:
        """
        fmap  (B, T+1, C, h, w)  the frames being tracked, after the frame before the first
        fmap0 (B, Tq, C, h, w)   the clip's maps, the templates read from them
        uv    (B, T+1, N, 2)     the guess in each of those frames, [0, 1] across, fp32
        uv0   (B, N, 2)          each point in its query frame
        xyz   (B, T+1, 3, h, w)  each cell's 3D position; xyz0 the clip's
        pos   (B, T+1, N, 3)     the guess in 3D; pos0 (B, N, 3) the query
        query_t (B, N) long      each point's query frame in `fmap0`

        returns (B, T, N, out_dim) for the T frames after the first
        """
        out = []
        clips = max(1, self.chunk // uv.shape[2])
        for i in range(0, fmap.shape[0], clips):
            cut = slice(i, i + clips)
            part = (fmap[cut], fmap0[cut], uv[cut], uv0[cut], xyz[cut], xyz0[cut], pos[cut], pos0[cut],
                    query_t[cut])
            if torch.is_grad_enabled():
                out.append(checkpoint(self._neighbourhoods, *part, use_reentrant=False))
            else:
                out.append(self._neighbourhoods(*part))
        return torch.cat(out)

    def _neighbourhoods(self, fmap: torch.Tensor, fmap0: torch.Tensor, uv: torch.Tensor,
                        uv0: torch.Tensor, xyz: torch.Tensor, xyz0: torch.Tensor, pos: torch.Tensor,
                        pos0: torch.Tensor, query_t: torch.Tensor) -> torch.Tensor:
        """TAPIP3D's lookup (`knn_feature_4d_optimized.py`) per level: the K
        cells nearest the guess in 3D against the template, then against the
        frame before; both returned, the template's first. Every input holds
        the frame before the first tracked one first."""
        b, T, C, h, w = fmap.shape
        T -= 1
        N = pos.shape[2]
        size = uv0.new_tensor([w, h])
        q = (uv0 * size - 0.5 / STRIDE)                                    # (b, N, 2) level-0 cells
        # Each clip's point cloud brought to unit spread, as TAPIP3D does
        # (`norm_mode: isotropic`): the scale its offset encoding is built for.
        # From frame 0's alone, which every clip sees; TAPIP3D uses the whole
        # window, which a forecast would leak through.
        xyz, xyz0, pos, pos0 = xyz.float(), xyz0.float(), pos.float(), pos0.float()
        first = xyz0[:, 0]                                                  # the clip's frame 0
        mean = first.mean(dim=(2, 3))                                       # (b, 3)
        std = (first - mean[:, :, None, None]).std(dim=(1, 2, 3)).clamp_min(1e-6)
        xyz = (xyz - mean[:, None, :, None, None]) / std[:, None, None, None, None]
        xyz0 = (xyz0 - mean[:, None, :, None, None]) / std[:, None, None, None, None]
        pos = (pos - mean[:, None, None]) / std[:, None, None, None]
        pos0 = (pos0 - mean[:, None]) / std[:, None, None]
        uv_b, pos_b, uv, pos = uv[:, :-1], pos[:, :-1], uv[:, 1:], pos[:, 1:]
        out, out_prev = [], []
        for lvl in range(self.levels):
            s = 2 ** lvl
            pool = lambda t: F.avg_pool2d(t, s, stride=s) if lvl else t
            f = pool(fmap.flatten(0, 1))
            hl, wl = f.shape[-2:]
            # The cells' positions are picked, not averaged: an average of a
            # rim cell and the wall behind it is a point in mid-air.
            pick_xyz = lambda t: F.interpolate(t, size=(hl, wl), mode="nearest-exact")
            x = pick_xyz(xyz.flatten(0, 1))
            f_all = f.flatten(2).transpose(1, 2).reshape(b, T + 1, hl * wl, C)      # (b, T+1, P, C)
            x_all = x.flatten(2).transpose(1, 2).reshape(b, T + 1, hl * wl, 3)      # (b, T+1, P, 3)
            f, x = f_all[:, 1:], x_all[:, 1:]
            k = min(self.neigh, hl * wl)
            with torch.no_grad():
                near = torch.cdist(pos, x).topk(k, dim=-1, largest=False).indices      # (b, T, N, k)
            pick = lambda t, i: t.gather(-2, i.flatten(-2)[..., None].expand(*i.shape[:-2], -1, t.shape[-1])
                                         ).unflatten(-2, i.shape[-2:])
            cur_f, cur_x = pick(f, near), pick(x, near)                     # (b, T, N, k, .)
            at = (q - (s - 1) / 2) / s
            grid = 2 * at / at.new_tensor([wl - 1, hl - 1]) - 1             # (b, N, 2)

            def support(f0, x0):
                """The template in one map, (b, C, hl, wl) features and (b, P, 3)
                positions: the K cells nearest each query point, led by the
                query's own feature sampled where it is. (b, N, k + 1, C), (b, N, k, 3)."""
                with torch.no_grad():
                    near0 = torch.cdist(pos0, x0).topk(k, dim=-1, largest=False).indices
                sf, sx = pick(f0.flatten(2).transpose(1, 2), near0), pick(x0, near0)
                own = F.grid_sample(f0.float(), grid[:, :, None], align_corners=True)[..., 0].transpose(1, 2)
                return torch.cat([own[:, :, None].to(sf.dtype), sf], dim=2), sx

            # Each point's template from its own frame: the support built in
            # every query frame there is, each point keeping its own.
            sup_f = sup_x = None
            for t in query_t.unique().tolist():
                sf, sx = support(pool(fmap0[:, t]), pick_xyz(xyz0[:, t]).flatten(2).transpose(1, 2))
                m = (query_t == t)[:, :, None, None]
                sup_f = sf if sup_f is None else torch.where(m, sf, sup_f)
                sup_x = sx if sup_x is None else torch.where(m, sx, sup_x)
            # Normalised after the sampling and the pooling, as TAPIP3D has it.
            sup_f, cur_f = self.in_norm(sup_f), self.in_norm(cur_f)
            sup_off = torch.cat([torch.zeros_like(sup_x[:, :, :1]), sup_x - pos0[:, :, None]], dim=2)
            cur_off = cur_x - pos[:, :, :, None]
            enc = self.posenc_mlps[lvl]
            y = cur_f + enc(_posenc(cur_off))                                # (b, T, N, k, D)
            x = sup_f + enc(_posenc(sup_off))                                # (b, N, k+1, D)
            x = x[:, None].expand(b, T, *x.shape[1:]).flatten(0, 2)
            out.append(self.transformers[lvl](x, y.flatten(0, 2)).reshape(b, T, N, -1))
            out_prev.append(self._before(f_all[:, :-1], x_all[:, :-1], pick, uv_b, pos_b, hl, wl, s, enc,
                                         self.prev_transformers[lvl], y))
        return torch.cat(out + out_prev, dim=-1)

    def _before(self, f_b, x_b, pick, uv_b, pos_b, hl, wl, s, enc, transformer, y):
        """One level's comparison of the guess's neighbourhood `y` against the
        K cells nearest the guess in the frame before (`f_b` its cells'
        features, `x_b` positions, `uv_b` and `pos_b` the guess there), led by
        that guess's own feature. (b, T, N, D)."""
        b, T, P, C = f_b.shape
        N = pos_b.shape[2]
        k = min(self.neigh, P)
        with torch.no_grad():
            near = torch.cdist(pos_b, x_b).topk(k, dim=-1, largest=False).indices
        sup_f, sup_x = pick(f_b, near), pick(x_b, near)                       # (b, T, N, k, .)
        at = (uv_b * uv_b.new_tensor([wl * s, hl * s]) - 0.5 / STRIDE - (s - 1) / 2) / s
        grid = 2 * at / at.new_tensor([wl - 1, hl - 1]) - 1                   # (b, T, N, 2)
        maps = f_b.transpose(2, 3).reshape(b * T, C, hl, wl).float()
        own = F.grid_sample(maps, grid.flatten(0, 1)[:, :, None], align_corners=True)[..., 0]
        own = own.transpose(1, 2).reshape(b, T, N, 1, C)
        sup_f = self.in_norm(torch.cat([own.to(sup_f.dtype), sup_f], dim=3))
        sup_off = torch.cat([torch.zeros_like(sup_x[..., :1, :]), sup_x - pos_b[..., None, :]], dim=3)
        sup = sup_f + enc(_posenc(sup_off))
        return transformer(sup.flatten(0, 2), y.flatten(0, 2)).reshape(b, T, N, -1)


def _posenc(x: torch.Tensor, lo: int = -2, hi: int = 14) -> torch.Tensor:
    """x and its sin and cos at 2^lo .. 2^(hi-1) per unit: TAPIP3D's `posenc`."""
    scales = 2.0 ** torch.arange(lo, hi, device=x.device, dtype=x.dtype)
    xb = (x[..., None, :] * scales[:, None]).flatten(-2)
    return torch.cat([x, torch.sin(torch.cat([xb, xb + 0.5 * torch.pi], dim=-1))], dim=-1)


def _posenc_mlp(dim: int) -> nn.Sequential:
    """TAPIP3D's `build_mlp(99, dim, 128, 3, last_act=GELU, output_norm=LayerNorm)`,
    layer for layer, so its indices match the checkpoint's."""
    return nn.Sequential(nn.LayerNorm(99), nn.Linear(99, 128), nn.GELU(), nn.Linear(128, 128),
                         nn.GELU(), nn.Linear(128, dim), nn.GELU(), nn.LayerNorm(dim))


class _Attention(nn.Module):
    """TAPIP3D's `Attention`: q from x, k and v from the context, one SDPA."""

    def __init__(self, dim: int, heads: int = 1, head_dim: int = 128, qkv_bias: bool = False) -> None:
        super().__init__()
        self.heads, self.head_dim = heads, head_dim
        self.to_q = nn.Linear(dim, heads * head_dim, bias=qkv_bias)
        self.to_kv = nn.Linear(dim, 2 * heads * head_dim, bias=qkv_bias)
        self.to_out = nn.Linear(heads * head_dim, dim)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, N1, _ = x.shape
        split = lambda t: t.reshape(B, t.shape[1], self.heads, self.head_dim).transpose(1, 2)
        k, v = self.to_kv(context).chunk(2, dim=-1)
        out = F.scaled_dot_product_attention(split(self.to_q(x)), split(k), split(v))
        return self.to_out(out.transpose(1, 2).reshape(B, N1, -1))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.fc1, self.act, self.fc2 = nn.Linear(dim, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _CrossAttnBlock(nn.Module):
    """x reads `context`, then a feed-forward: TAPIP3D's `CrossAttnBlock`."""

    def __init__(self, dim: int, mlp_ratio: float = 1.0, heads: int = 1, head_dim: int = 128) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm_context = nn.LayerNorm(dim)
        self.cross_attn = _Attention(dim, heads, head_dim, qkv_bias=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        x = x + self.cross_attn(self.norm1(x), self.norm_context(context))
        return x + self.mlp(self.norm2(x))


class _NeighborTransformer(nn.Module):
    """One level's comparison, TAPIP3D's `NeighborTransformer` (128 wide, one
    head): the query's neighbourhood and the guess's read each other, and two
    learned output tokens sum each side into one vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.output_token_1 = nn.Parameter(torch.randn(1, dim))
        self.output_token_2 = nn.Parameter(torch.randn(1, dim))
        self.xblock1_2 = _CrossAttnBlock(dim)
        self.xblock2_1 = _CrossAttnBlock(dim)
        self.aggr1 = _Attention(dim)
        self.aggr2 = _Attention(dim)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """x (M, K+1, D) around the query, y (M, K, D) around the guess -> (M, D)."""
        new_x, new_y = self.xblock1_2(x, y), self.xblock2_1(y, x)
        t1 = self.output_token_1[None].expand(x.shape[0], -1, -1)
        t2 = self.output_token_2[None].expand(x.shape[0], -1, -1)
        return (self.aggr1(t1, new_x) + self.aggr2(t2, new_y))[:, 0]
