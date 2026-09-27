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

Run:  .venv/bin/python scripts/check_model.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from genpoint3d.models.encoder import VisualEncoder
from genpoint3d.models.model import PointDiT

B, T, N, P, D = 2, 8, 5, 16, 64
CUT = 4  # frames [0, CUT) keep their images; the rest are forecast


def report(name: str, ok: bool, detail: str) -> bool:
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<16} {detail}")
    return ok


def build():
    torch.manual_seed(0)
    m = PointDiT(dim=D, depth=2, num_heads=4, cond_dim=D, cross_attn=True).eval()
    x = torch.randn(B, T, N, 3)
    k = torch.rand(B)
    anchor = torch.randn(B, N, 3)
    ctx = torch.randn(B, T, P, D)
    vm = torch.zeros(B, T, dtype=torch.bool)
    vm[:, :CUT] = True
    idc = torch.randn(B, N, D)
    pxyz = torch.randn(B, T, P, 3)
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
    """A point must be moved more by patches NEAR it than by distant ones.

    Without this the model has to discover which of P patches is its own, from
    scratch, with no supervision on attention -- and the measured result was a
    model that ignored the video entirely. Geometry answers it instead.

    Both patches are perturbed by the same amount, so any difference in effect is
    the distance prior and nothing else.
    """
    P = ctx.shape[2]
    pxyz = pxyz.clone()
    pxyz[:, :, 0] = 0.0                     # patch 0 sits exactly on the points
    pxyz[:, :, 1] = 50.0                    # patch 1 is far away
    x = torch.zeros_like(x)                 # every point at the origin, beside patch 0
    with torch.no_grad():
        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        near = ctx.clone(); near[:, :, 0] += 1.0
        a = m(x, k, anchor, context=near, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        far = ctx.clone(); far[:, :, 1] += 1.0
        b = m(x, k, anchor, context=far, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

    d_near = (a - base).abs().max().item()
    d_far = (b - base).abs().max().item()
    return (
        report("near patch acts", d_near > 1e-8, f"nearby patch moved output {d_near:.1e} (want >0)"),
        report("far patch muted", d_far < d_near, f"distant patch moved it {d_far:.1e}, "
                                                  f"{d_far / max(d_near, 1e-30):.2f}x the nearby one (want <1)"),
    )


def check_correlation(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """A patch that MATCHES the query template must act more than one that does not.

    Every patch is placed at the same position here, so the distance prior is
    identical for all of them and appearance is the only thing left that can
    differentiate. Before this existed the model held the template and the patches
    but never compared them, and the ablation showed it ignored the video.

    The gates are opened by hand: they init at zero on purpose, so an untrained
    model is bit-identical to the locality-only one. This asks whether the wiring
    is there, not whether it is active at step 0.
    """
    for cb in m.cross_blocks:
        cb.corr_w.data.fill_(4.0)
    m.match_gate.data.fill_(1.0)

    pxyz = torch.ones_like(pxyz)            # every patch equidistant from every point
    with torch.no_grad():
        # The feature whose corr_k projection equals point 0's corr_q projection,
        # i.e. the patch that looks exactly like that point.
        q0 = m.corr_q(idc)[:, 0]                                    # (B, D)
        f = q0 @ torch.linalg.pinv(m.corr_k.weight).T               # (B, D_feat)
        ctx = ctx.clone()
        ctx[:, :, 0] = f[:, None]                                   # patch 0 matches
        ctx[:, :, 1] = -f[:, None]                                  # patch 1 anti-matches

        base = m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        hit = ctx.clone(); hit[:, :, 0] += 1.0
        a = m(x, k, anchor, context=hit, visual_mask=vm, id_card=idc, patch_xyz=pxyz)
        miss = ctx.clone(); miss[:, :, 1] += 1.0
        b = m(x, k, anchor, context=miss, visual_mask=vm, id_card=idc, patch_xyz=pxyz)

    # Point 0 is the one whose template was matched; only its output is evidence.
    d_hit = (a - base)[:, :, 0].abs().max().item()
    d_miss = (b - base)[:, :, 0].abs().max().item()
    return (
        report("match acts", d_hit > 1e-8, f"matching patch moved output {d_hit:.1e} (want >0)"),
        report("mismatch muted", d_miss < d_hit, f"non-matching patch moved it {d_miss:.1e}, "
               f"{d_miss / max(d_hit, 1e-30):.2f}x the matching one (want <1)"),
    )


def check_costvol(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    """The support window must matter, and only from frame 0.

    Nothing else in this suite touches it: `id card` tests the single-vector
    template, and the correlation checks compare that template against patches.
    The support window is a separate path -- features gathered around the ANCHOR
    at frame 0 -- and it is the thing CoTracker has and our first attempt did not.

    Gate opened by hand; it inits at zero so an untrained model matches run6.
    """
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
    torch.manual_seed(0)
    off = PointDiT(dim=D, depth=2, num_heads=4, cond_dim=D, cross_attn=True, costvol=False)
    extra = [n for n in off.state_dict() if n.startswith("cv_")]
    return report("cv optional", not extra, f"--costvol 0 adds {len(extra)} params (want 0)")


def check_bidirectional(x, k, anchor, ctx, vm, idc, pxyz) -> bool:
    """--causal 0 must actually let a frame see later frames.

    The mirror of `check_causality`. A flag that silently does nothing is worse
    than no flag, because the run it produces looks like an answer.
    """
    torch.manual_seed(0)
    m = PointDiT(dim=D, depth=2, num_heads=4, cond_dim=D, cross_attn=True,
                 causal=False).eval()
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
    torch.manual_seed(0)
    off = PointDiT(dim=D, depth=2, num_heads=4, cond_dim=D, cross_attn=True,
                   correlate=False)
    extra = [n for n in off.state_dict() if "corr" in n or "match" in n]
    return report("corr optional", not extra, f"--correlate 0 adds {len(extra)} params (want 0)")


def check_grads(m, x, k, anchor, ctx, vm, idc, pxyz) -> tuple[bool, bool]:
    m.zero_grad()
    idc = idc.clone().requires_grad_(True)
    m(x, k, anchor, context=ctx, visual_mask=vm, id_card=idc, patch_xyz=pxyz).sum().backward()
    null_g = m.null_ctx.grad.abs().sum().item()
    idc_g = idc.grad.abs().sum().item()
    return (
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
    print(f"B={B} T={T} N={N} P={P} D={D}, cutoff T_C={CUT}\n")
    m, *args = build()
    ok = check_causality(m, *args)
    ok &= all(check_mask(m, *args))
    ok &= all(check_patch_pos(m, *args))
    ok &= all(check_locality(m, *args))
    ok &= all(check_grads(m, *args))
    # Last: it opens the correlation gates, which mutates the model.
    ok &= check_bidirectional(*args)
    ok &= check_corr_optional()
    ok &= check_cv_optional()
    ok &= all(check_costvol(m, *args))
    ok &= all(check_correlation(m, *args))
    ok &= check_encoder()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
