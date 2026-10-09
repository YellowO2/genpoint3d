"""
Correctness checks for the model wiring (steps 2 and 3).

These are properties, not performance. Each one fails loudly if a tensor is
reshaped wrongly or a mask is applied in the wrong place -- the class of bug
that otherwise shows up as "training is mysteriously bad".

  1. causality      a frame cannot see its own future
  2. mask blocks    changing a MASKED frame's features changes nothing
  3. mask passes    changing an UNMASKED frame's features does change things
  4. null is used   the null embedding receives gradient
  5. id card        the query feature reaches the output
  6. encoder shapes grid / tokens / bilinear lookup all line up
  7. locality       a patch beside the point acts, one ten patches away does not
  8. correlation    the patch that looks like the point is found, untrained
  9. window         the prior is wide on a noisy sample and narrow on a clean one
 10. match head     starts as the raw cosine, is the MLP it claims to be, trains
 11. true patch     a point placed on a patch's own surface point is in that patch
 12. top-k match    K candidates are K different places, start as the K=1 model, train

The model is built the way a run builds it (displacement target, its scale,
cost volume off) and the scene has a run's proportions: patches on a grid 0.05
apart, points 0.01-0.02 from the nearest one, features wider than the model.
Checks 7 and 8 passed for years on numbers no scene has -- a "far" patch 50
units away, gates opened by hand -- while the real model could do neither.

Run:  .venv/bin/python scripts/check_model.py
"""

import torch
import torch.nn.functional as F

from genpoint3d.data.transform import TRAJ_SCALE_DISP
from genpoint3d.geometry import batch_unproject
from genpoint3d.models.encoder import VisualEncoder, patch_centre_xyz
from genpoint3d.models.match import (
    match_accuracy_topk, match_ce, nms_peaks, raw_cosine, true_patch,
)
from genpoint3d.models.model import PointDiT

B, T, N, G, D, FEAT = 2, 8, 5, 12, 64, 48
P = G * G
SPACING = 0.05  # between neighbouring patches; Kubric at 24x24 gives 0.04-0.11
CUT = 4  # frames [0, CUT) keep their images; the rest are forecast


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {detail}")
    return ok


def model(**kw) -> PointDiT:
    """The model as scripts/train.py builds it for a run, scaled down."""
    torch.manual_seed(0)
    kw = {"costvol": False, **kw}
    return PointDiT(dim=D, depth=2, num_heads=4, cross_attn=True, feat_dim=FEAT,
                    displacement=True, traj_scale=TRAJ_SCALE_DISP, **kw).eval()


def grid_xyz() -> torch.Tensor:
    """(P, 3) patch positions, row-major, on a flat G x G grid one unit deep."""
    i = (torch.arange(G) - (G - 1) / 2) * SPACING
    ys, xs = torch.meshgrid(i, i, indexing="ij")
    return torch.stack([xs, ys, torch.ones_like(xs)], dim=-1).reshape(P, 3)


def build():
    m = model()
    x = torch.randn(B, T, N, 3)
    x[:, 0] = 0                             # frame 0 is pinned to the anchor
    k = torch.rand(B)
    grid = grid_xyz()
    # Not quite regular and not quite still, as a depth map under a moving
    # camera is not.
    pxyz = grid + torch.randn(B, T, P, 3) * 0.1 * SPACING
    anchor = grid[torch.randint(P, (B, N))] + torch.randn(B, N, 3) * 0.2 * SPACING
    ctx = torch.randn(B, T, P, FEAT)
    vm = torch.zeros(B, T, dtype=torch.bool)
    vm[:, :CUT] = True
    idc = torch.randn(B, N, FEAT)
    return m, x, k, anchor, ctx, vm, idc, pxyz


def check_causality(m, x, k, anchor, ctx, vm, idc, pxyz) -> bool:
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        x2 = x.clone()
        x2[:, CUT:] += 10.0
        pert = m(x2, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    past = (pert - base)[:, :CUT].abs().max().item()
    future = (pert - base)[:, CUT:].abs().max().item()
    return report("causality", past < 1e-6 and future > 1e-8,
                  f"past moved {past:.1e} (want 0), future moved {future:.1e} (want >0)")


def check_mask(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """The forecasting switch: masked frames must be blind to their features."""
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

        masked = ctx.clone()
        masked[:, CUT:] = torch.randn_like(masked[:, CUT:]) * 50
        a = m(x, k, anchor, context=masked, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

        seen = ctx.clone()
        seen[:, :CUT] += 1.0
        b = m(x, k, anchor, context=seen, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

    blocked = (a - base).abs().max().item()
    passed = (b - base).abs().max().item()
    return (
        report("mask blocks", blocked < 1e-6, f"masked-frame features moved output {blocked:.1e} (want 0)"),
        report("mask passes", passed > 1e-8, f"visible-frame features moved output {passed:.1e} (want >0)"),
    )


def check_patch_pos(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """Patch positions must reach the output, and be masked along with features."""
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

        seen = pxyz.clone()
        seen[:, :CUT] += 1.0
        a = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=seen)

        hidden = pxyz.clone()
        hidden[:, CUT:] = torch.randn_like(hidden[:, CUT:]) * 50
        b = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=hidden)

    used = (a - base).abs().max().item()
    blocked = (b - base).abs().max().item()
    return (
        report("patch pos used", used > 1e-8, f"visible-frame positions moved output {used:.1e} (want >0)"),
        report("patch pos masked", blocked < 1e-6, f"masked-frame positions moved output {blocked:.1e} (want 0)"),
    )


def check_locality(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """A point must be moved by the patch BESIDE it and not by one ten patches off.

    Without this the model has to discover which of P patches is its own, from
    scratch, with no supervision on attention -- and the measured result was a
    model that ignored the video entirely. Geometry answers it instead.

    Both patches are perturbed by the same amount, so any difference in effect is
    the distance prior and nothing else. Ten patch spacings is 0.5 scene units:
    the old prior, a strength of 1.3 per unit squared, gave that patch 0.72 of
    the nearby one's weight, and this check is what would have said so.
    """
    row = G // 2
    near, far = row * G + 1, row * G + 10   # 1 and 10 spacings right of patch 0
    anchor = pxyz[:, 0, row * G].unsqueeze(1).expand(B, N, 3).clone()
    x = torch.zeros_like(x)                 # every point sits on its anchor
    k = torch.ones_like(k)                  # and is clean, so the window is narrow
    bump = torch.randn(FEAT)
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        c = ctx.clone(); c[:, :, near] += bump
        a = m(x, k, anchor, context=c, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        c = ctx.clone(); c[:, :, far] += bump
        b = m(x, k, anchor, context=c, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

    d_near = (a - base).abs().max().item()
    d_far = (b - base).abs().max().item()
    return (
        report("near patch acts", d_near > 1e-8, f"patch 1 spacing away moved output {d_near:.1e} (want >0)"),
        report("far patch muted", d_far < 0.1 * d_near,
               f"patch 10 spacings away moved it {d_far:.1e}, "
               f"{d_far / max(d_near, 1e-30):.3f}x the nearby one (want <0.1)"),
    )


def check_correlation(m, x, k, anchor, ctx, vm, idc, pxyz) -> list[bool]:
    """The patch that looks like the point must be found, and found UNTRAINED.

    Each point's own feature is planted in one patch per frame, two patches
    further along the row every frame, among patches of noise. Nothing is opened
    by hand and no weight is consulted to build the matching patch: the feature
    is simply the same one. The old score put that patch at rank P/2, and its
    "best match" was the centre of the grid.

      match ranked 1st  the planted patch has the highest score of all P
      match offset      the offset handed to the model is the planted
                        displacement, to within a patch spacing
      match ignores x   and is the same whatever the noisy sample is: it is
                        evidence only while it is not the model's own guess
      match acts        of two patches equally near, the lookalike is the one read
    """
    vm = torch.ones_like(vm)
    grid = grid_xyz()
    pxyz = grid.expand(B, T, P, 3)
    rows = torch.arange(N) + 1                                  # one row per point
    home = rows * G                                             # column 0
    anchor = grid[home].expand(B, N, 3).clone()
    planted = home[None] + 2 * torch.arange(T).clamp(max=4)[:, None]   # (T, N)
    ctx = ctx.clone()
    ctx[:, torch.arange(T)[:, None], planted] = idc[:, None]
    want = (grid[planted] - grid[home]) / TRAJ_SCALE_DISP       # (T, N, 3)
    x = want.expand(B, T, N, 3) + torch.randn(B, T, N, 3) * 0.5  # a noisy guess
    x[:, 0] = 0

    seen = {}
    hooks = [
        m.cross_blocks[0].register_forward_pre_hook(
            lambda _, a, kw: seen.update(corr=kw["corr"]), with_kwargs=True),
        m.match_proj.register_forward_pre_hook(lambda _, a: seen.update(match=a[0])),
    ]
    with torch.no_grad():
        m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        found = seen["match"]
        m(torch.randn_like(x) * 3, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    for h in hooks:
        h.remove()
    moved = (seen["match"] - found).abs().max().item()

    top = seen["corr"].reshape(B, T, N, P).argmax(-1)
    hit = (top == planted).float().mean().item()
    err = (found[..., :3] - want).norm(dim=-1).max().item() * TRAJ_SCALE_DISP / SPACING
    out = [
        report("match ranked 1st", hit == 1.0, f"planted patch scored highest for {hit:.0%} of points (want 100%)"),
        report("match offset", err < 1.0, f"offset is off by at most {err:.2f} patch spacings (want <1)"),
        report("match ignores x", moved == 0.0, f"another noisy sample moved the match {moved:.1e} (want 0)"),
    ]

    # Point 0 sits on a patch; its left and right neighbours are equally near.
    # One looks exactly like it, the other exactly unlike. Every other point is
    # parked in the far corner: one close by would read both patches itself
    # and pass the change on through spatial attention.
    anchor = grid[P - 1].expand(B, N, 3).clone()
    anchor[:, 0] = grid[home[0] + 2]
    x = torch.zeros_like(x)
    like, unlike = home[0] + 1, home[0] + 3
    ctx = torch.randn_like(ctx)
    ctx[:, :, like], ctx[:, :, unlike] = idc[:, None, 0], -idc[:, None, 0]
    bump = torch.randn(FEAT) * 0.1
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        c = ctx.clone(); c[:, :, like] += bump
        a = m(x, k, anchor, context=c, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        c = ctx.clone(); c[:, :, unlike] += bump
        b = m(x, k, anchor, context=c, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    # Only point 0's own output is evidence; the others have other templates.
    d_hit = (a - base)[:, :, 0].abs().max().item()
    d_miss = (b - base)[:, :, 0].abs().max().item()
    return out + [
        report("match acts", d_hit > 1e-8, f"matching patch moved output {d_hit:.1e} (want >0)"),
        report("mismatch muted", d_miss < 0.1 * d_hit, f"non-matching patch moved it {d_miss:.1e}, "
               f"{d_miss / max(d_hit, 1e-30):.3f}x the matching one (want <0.1)"),
    ]


def check_window(x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """The scheduled prior must be `locality_wide` times wider at k = 0 than at
    k = 1, frame by frame, and at k = 1 be the fixed prior exactly.

    Each frame is given its own k, rising from 0 to 1 along the clip, so a
    width laid out against the wrong axis shows as the wrong ratio in some
    frame. Correlation is off: the bias the block receives is the prior alone.
    """
    W = 8.0
    vm = torch.ones_like(vm)
    k = torch.linspace(0, 1, T).expand(B, T)
    got = {}
    for mode in ("patch", "sched"):
        m = model(correlate=False, locality_mode=mode, locality_wide=W)
        h = m.cross_blocks[0].attn.register_forward_pre_hook(
            lambda _, a, kw, mode=mode: got.update({mode: kw["attn_mask"]}), with_kwargs=True)
        with torch.no_grad():
            m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        h.remove()
    fixed, sched = (got[n].reshape(B, T, N, P) for n in ("patch", "sched"))
    # Where neither sits on the floor or on the nearest patch's zero, the two
    # biases differ by the squared ratio of the widths and nothing else.
    use = (fixed > -90) & (fixed < -1e-3)
    ratio = torch.stack([(fixed[:, t] / sched[:, t])[use[:, t]].sqrt().median() for t in range(T)])
    want = W ** (1 - k[0])
    off = (ratio / want - 1).abs().max().item()
    same = (sched[:, -1] - fixed[:, -1]).abs().max().item()
    return (
        report("window narrows", off < 1e-3,
               f"{ratio[0]:.2f}x wider at k=0, {ratio[T // 2]:.2f}x at k={k[0, T // 2]:.2f}, "
               f"{ratio[-1]:.2f}x at k=1 (want {W:.0f}^(1-k), worst frame off by {off:.1e})"),
        report("window at k=1", same == 0.0,
               f"scheduled prior differs from the fixed one by {same:.1e} at k=1 (want 0)"),
    )


def check_costvol(x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """The support window must matter, and only from frame 0.

    Nothing else in this suite touches it: `id card` tests the single-vector
    template, and the correlation checks compare that template against patches.
    The support window is a separate path -- features gathered around the ANCHOR
    at frame 0 -- and it is the thing CoTracker has and our first attempt did not.

    Gate opened by hand; it inits at zero so an untrained model matches run6.
    Runs leave the cost volume off, so this is the one check built with it on.
    """
    m = model(costvol=True)
    m.cv_gate.data.fill_(1.0)
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        # Frame 0 is the query frame, so its patches are what the support reads.
        f0 = ctx.clone(); f0[:, 0] += 1.0
        a = m(x, k, anchor, context=f0, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        # A frame the model is not allowed to see must not reach the cost volume,
        # which reads RAW features -- before the adapter swaps in the null.
        hid = ctx.clone(); hid[:, CUT:] += 1.0
        b = m(x, k, anchor, context=hid, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    d_sup = (a - base).abs().max().item()
    d_masked = (b - base)[:, :CUT].abs().max().item()
    return (
        report("support acts", d_sup > 1e-8, f"frame-0 support moved output {d_sup:.1e} (want >0)"),
        report("cv not leaking", d_masked < 1e-8,
               f"masked-frame features moved visible output {d_masked:.1e} (want 0)"),
    )


def check_cv_optional() -> bool:
    """--costvol 0 must create no parameters, so run6/7/8 checkpoints still load."""
    extra = [n for n in model(costvol=False).state_dict() if n.startswith("cv_")]
    return report("cv optional", not extra, f"--costvol 0 adds {len(extra)} params (want 0)")


def check_bidirectional(x, k, anchor, ctx, vm, idc, pxyz) -> bool:
    """--causal 0 must actually let a frame see later frames.

    The mirror of `check_causality`. A flag that silently does nothing is worse
    than no flag, because the run it produces looks like an answer.
    """
    m = model(causal=False)
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        x2 = x.clone(); x2[:, CUT:] += 10.0
        pert = m(x2, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    past = (pert - base)[:, :CUT].abs().max().item()
    return report("bidirectional", past > 1e-8,
                  f"--causal 0: future moved the past {past:.1e} (want >0)")


def check_corr_optional() -> bool:
    """--correlate 0 must create no correlation parameters.

    Same contract as --locality 0: a checkpoint trained before this existed has
    to keep loading, which it only does if the state dict has no extra keys.
    """
    extra = [n for n in model(correlate=False).state_dict() if "corr" in n or "match" in n]
    return report("corr optional", not extra, f"--correlate 0 adds {len(extra)} params (want 0)")


def check_match_head(x, k, anchor, ctx, vm, idc, pxyz) -> list[bool]:
    """--match-learn must start as the model without it, compute the residual
    MLP it is described as (it never builds the adapted features, so the
    shortcut is checked against the obvious code), and be reachable by the
    matching loss. --match-learn 0 must add no parameters."""
    vm = torch.ones_like(vm)
    off, on = model(), model(match_learn=True)
    extra = [n for n in off.state_dict() if n.startswith("match_head")]
    same = all(torch.equal(v, on.state_dict()[n]) for n, v in off.state_dict().items())
    with torch.no_grad():
        a = off(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        b = on(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        s0 = (on.match_scores(ctx, idc, pxyz) - raw_cosine(ctx, idc)).abs().max().item()
    start = (a - b).abs().max().item()

    head = on.match_head
    torch.nn.init.normal_(head.out.weight, std=0.3)
    mlp = lambda f: F.gelu(head.inp(f)) @ head.out.weight.T
    with torch.no_grad():
        naive = torch.einsum("bnc,btpc->btnp", F.normalize(idc + mlp(idc), dim=-1),
                             F.normalize(ctx + mlp(ctx), dim=-1))
        moved = (naive - raw_cosine(ctx, idc)).abs().max().item()
        err = (on.match_scores(ctx, idc, pxyz) - naive).abs().max().item()

    target = torch.randint(P, (B, T, N))
    valid = torch.rand(B, T, N) > 0.3
    on.zero_grad()
    match_ce(on.match_scores(ctx, idc, pxyz), head.tau(), target, valid).sum().backward()
    g = {n: p.grad.abs().max().item() for n, p in head.named_parameters()}
    dead = [n for n, v in g.items() if v == 0]
    other = [n for n, p in on.named_parameters()
             if not n.startswith("match_head") and p.grad is not None and p.grad.abs().sum() > 0]
    return [
        report("head optional", not extra and same,
               f"--match-learn 0 adds {len(extra)} params (want 0); the rest initialise "
               f"{'the same' if same else 'DIFFERENTLY'} with it on"),
        report("head starts raw", start == 0.0 and s0 == 0.0,
               f"at step 0 the output differs by {start:.1e}, the scores by {s0:.1e} (want 0)"),
        report("head is the MLP", err < 1e-5 and moved > 1e-2,
               f"scores differ from normalize(f + MLP(f)) by {err:.1e} (want <1e-5),"
               f" on scores the MLP moved by {moved:.2f}"),
        report("head trains", not dead and not other,
               f"matching loss: {len(dead)} of {len(g)} head parameters without gradient {dead or ''},"
               f" {len(other)} outside the head with one (want 0, 0)"),
    ]


def check_topk(x, k, anchor, ctx, vm, idc, pxyz) -> list[bool]:
    """--match-topk K hands the trunk K distinct candidates.

      peaks distinct    on a made-up score map with two adjacent high cells
                        and a lower peak far away, the candidates are the top
                        cell and the far peak -- not the neighbour
      topk optional     K = 1 adds no parameters, and the rest initialise the
                        same with K = 4
      topk starts as 1  at step 0 the K = 4 model's output is the K = 1 one's
      rank 1 is K=1     the first five numbers at K = 4 are the K = 1 match
      topk is read      each further candidate moves the output once its
                        weights are not zero, and the loss reaches both those
                        weights and, through the candidates, the learned head
      top5 counts       the diagnostic says yes for a truth next to the 2nd
                        peak and no for one next to nothing
    """
    vm = torch.ones_like(vm)
    K = 4
    sc = torch.rand(1, 1, 2, P) * 0.1
    a, b_, far = 3 * G + 3, 3 * G + 4, 9 * G + 8
    sc[..., a], sc[..., b_], sc[..., far] = 1.0, 0.9, 0.5
    pk = nms_peaks(sc, 3)
    cheb = lambda i, j: torch.maximum((i // G - j // G).abs(), (i % G - j % G).abs())
    apart = min(cheb(pk[..., i], pk[..., j]).min().item()
                for i in range(3) for j in range(i))
    # The same answer as masking the map, the obvious way.
    ref, left = [], sc.clone()
    for _ in range(3):
        t = left.argmax(-1)
        ref.append(t)
        cells = torch.arange(P)
        left = left.masked_fill(cheb(cells, t[..., None]) <= 1, -torch.inf)
    naive = torch.equal(pk, torch.stack(ref, -1))
    right = bool((pk[..., 0] == a).all() and (pk[..., 1] == far).all())

    tgt = torch.tensor([far + 1, 6 * G + 0]).reshape(1, 1, 2)
    top5 = match_accuracy_topk(sc, tgt, torch.ones(1, 1, 2, dtype=torch.bool), 5).item()

    one, many = model(), model(match_topk=K)
    lone, lmany = model(match_learn=True), model(match_learn=True, match_topk=K)
    extra = [n for n in one.state_dict() if n.startswith("match_more")]
    same = (all(torch.equal(v, many.state_dict()[n]) for n, v in one.state_dict().items())
            and all(torch.equal(v, lmany.state_dict()[n]) for n, v in lone.state_dict().items()))
    kw = dict(context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
    seen = {}
    hooks = [one.match_proj.register_forward_pre_hook(lambda _, i: seen.update(one=i[0])),
             many.match_proj.register_forward_pre_hook(lambda _, i: seen.update(first=i[0])),
             many.match_more.register_forward_pre_hook(lambda _, i: seen.update(more=i[0]))]
    with torch.no_grad():
        o1 = one(x, k, anchor, **kw)
        oK = many(x, k, anchor, **kw)
        start = (o1 - oK).abs().max().item()
        rank1 = torch.equal(seen["one"], seen["first"])
        # Every candidate is a different place, with a lower score than the last.
        cand = torch.cat([seen["first"], seen["more"]], -1).reshape(B, T, N, K, 5)
        ordered = bool((cand[..., 1:, 3] <= cand[..., :-1, 3]).all())
        differ = (cand[..., 1:, :3] - cand[..., :-1, :3]).norm(dim=-1).min().item()
        torch.nn.init.normal_(many.match_more.weight, std=0.5)
        base = many(x, k, anchor, **kw)
        moved = []
        for j in range(1, K):
            w = many.match_more.weight.clone()
            many.match_more.weight[:, 5 * (j - 1):5 * j] = 0
            moved.append((many(x, k, anchor, **kw) - base).abs().max().item())
            many.match_more.weight.copy_(w)
    for h in hooks:
        h.remove()

    # The weights start at exactly zero and must still get a gradient there.
    lmany.zero_grad()
    h = lmany.match_more.register_forward_pre_hook(lambda _, i: seen.update(more=i[0]))
    lmany(x, k, anchor, **kw).pow(2).sum().backward(retain_graph=True)
    h.remove()
    gm = lmany.match_more.weight.grad.abs().max().item()
    # What the runners-up hand over, traced back to the learned head: only
    # them, since the head is reached through rank 1 and the attention anyway.
    gh = torch.autograd.grad(seen["more"].pow(2).sum(),
                             lmany.match_head.out.weight)[0].abs().max().item()
    return [
        report("peaks distinct", right and naive and apart >= 2 and top5 == 0.5,
               f"candidates {pk[0, 0, 0].tolist()} (want [{a}, {far}, ..], never {b_}),"
               f" closest pair {apart} cells apart (want >=2), "
               f"{'same' if naive else 'DIFFERENT'} as masking the map; "
               f"top-5 diagnostic {top5:.2f} (want 0.50)"),
        report("topk optional", not extra and same,
               f"--match-topk 1 adds {len(extra)} params (want 0); the rest initialise "
               f"{'the same' if same else 'DIFFERENTLY'} with K={K}"),
        report("topk starts as 1", start == 0.0 and rank1,
               f"at step 0 the K={K} output differs from K=1 by {start:.1e} (want 0); "
               f"rank 1 is {'the' if rank1 else 'NOT the'} K=1 match"),
        report("topk is read", min(moved) > 1e-8 and ordered and differ > 0.0,
               f"dropping candidate 2..{K} moved output {min(moved):.1e}..{max(moved):.1e}"
               f" (want >0); scores {'fall' if ordered else 'DO NOT fall'} with rank"),
        report("topk trains", gm > 0 and gh > 0,
               f"gradient on the candidates' weights {gm:.1e}, on the learned head "
               f"through candidates 2..{K} {gh:.1e} (want >0)"),
    ]


def check_true_patch() -> bool:
    """A point sitting exactly on the surface point stored for a patch must be
    assigned that patch -- on a frame that is not square, under a camera that
    has moved, at both grid sizes. And one behind the camera or outside the
    frame must be marked as having no patch."""
    H, W, Tn = 96, 160, 3
    ok, worst = True, 0
    for g in (12, 24):
        gen = torch.Generator().manual_seed(g)
        depth = 2 + torch.rand(Tn, H, W, generator=gen) * 3
        K = torch.tensor([[170., 0, W / 2], [0, 150., H / 2], [0, 0, 1]]).expand(Tn, 3, 3)
        E = torch.eye(4).repeat(Tn, 1, 1)
        for t in range(1, Tn):                      # a turn about y and a shift
            a = torch.tensor(0.1 * t)
            E[t, 0, 0], E[t, 0, 2], E[t, 2, 0], E[t, 2, 2] = a.cos(), a.sin(), -a.sin(), a.cos()
            E[t, :3, 3] = torch.tensor([0.2, -0.1, 0.3]) * t
        # Patch positions exactly as the cache builds them: frame-0 camera.
        E0 = E @ torch.linalg.inv(E[:1])
        pxyz = patch_centre_xyz(batch_unproject(depth, K, E0), g)     # (T, P, 3)
        hw = torch.tensor([[H, W]])
        got, inside = true_patch(pxyz[None], K[None], E0[None], hw, g)
        wrong = (got[0] != torch.arange(g * g)).sum().item() + (~inside).sum().item()
        # The same points seen from behind, and pushed out of the side.
        behind = pxyz[None].clone(); behind[:, 0, :, 2] *= -1
        out = pxyz[None].clone(); out[:, 0, :, 0] += 100
        leak = (true_patch(behind, K[None], E0[None], hw, g)[1][:, 0].sum().item()
                + true_patch(out, K[None], E0[None], hw, g)[1][:, 0].sum().item())
        ok &= wrong == 0 and leak == 0
        worst = max(worst, wrong + leak)
    return report("true patch", ok, f"{worst} points given the wrong patch or a patch "
                                    "they cannot have, on 12x12 and 24x24 (want 0)")


def check_grads(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool, bool]:
    """The last one is what the old matching failed: its projections sat behind
    gates initialised at zero, so at step 0 their gradient was exactly zero and
    the gates were only ever shown a random score."""
    m.zero_grad()
    idc = idc.clone().requires_grad_(True)
    m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz).sum().backward()
    null_g = m.null_ctx.grad.abs().sum().item()
    idc_g = idc.grad.abs().sum().item()
    live = {n: p.grad.abs().max().item() for n, p in m.named_parameters()
            if "match" in n or "corr" in n or "sigma" in n}
    dead = [n for n, g in live.items() if g == 0]
    return (
        report("match is live", not dead, f"{len(dead)} of {len(live)} locality and match "
               f"parameters have zero gradient at step 0 {dead or ''} (want 0)"),
        report("null is used", null_g > 0, f"null_ctx grad {null_g:.2e}"),
        report("id card", idc_g > 0, f"id_card grad {idc_g:.2e}"),
    )


def check_encoder() -> bool:
    """Shapes out of stage [1], and that patch features and patch positions
    agree on which patch is which -- they are matched by index in the model, so
    a transpose in either flatten would silently pair a feature with someone
    else's coordinates."""
    enc = VisualEncoder(image_size=64, stub=True)
    E, g = enc.dim, enc.grid
    frames = torch.randint(0, 255, (T, 128, 128, 3), dtype=torch.uint8)
    feat = enc(frames)
    tok = VisualEncoder.tokens(feat)
    uv = torch.rand(T, N, 2) * 127
    samp = enc.sample_at(feat, uv, (128, 128))
    ok = feat.shape == (T, E, g, g) and tok.shape == (T, g * g, E) and samp.shape == (T, N, E)
    ok = report("encoder shapes", ok,
                f"grid {tuple(feat.shape)}  tokens {tuple(tok.shape)}  sample {tuple(samp.shape)}")

    # Each pixel carries its own patch index as its "position", so the pooled
    # value of patch i must come back as row i.
    idx = torch.arange(g * g, dtype=torch.float32).reshape(1, 1, g, g)
    pm = F.interpolate(idx, size=(128, 128), mode="nearest").expand(T, 3, 128, 128)
    xyz = enc.patch_xyz(pm)
    want = torch.arange(g * g, dtype=torch.float32)[None, :, None].expand(T, g * g, 3)
    err = (xyz - want).abs().max().item()
    return ok & report("patch order", xyz.shape == (T, g * g, 3) and err == 0.0,
                       f"max index mismatch {err:.1f}")


def main() -> int:
    print(f"B={B} T={T} N={N} P={P} D={D} feat={FEAT}, cutoff T_C={CUT}\n")
    m, *args = build()
    ok = check_causality(m, *args)
    ok &= all(check_mask(m, *args))
    ok &= all(check_patch_pos(m, *args))
    ok &= all(check_locality(m, *args))
    ok &= all(check_correlation(m, *args))
    ok &= all(check_grads(m, *args))
    ok &= all(check_window(*args))
    ok &= check_bidirectional(*args)
    ok &= check_corr_optional()
    ok &= check_cv_optional()
    ok &= all(check_costvol(*args))
    ok &= all(check_match_head(*args))
    ok &= all(check_topk(*args))
    ok &= check_true_patch()
    ok &= check_encoder()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
