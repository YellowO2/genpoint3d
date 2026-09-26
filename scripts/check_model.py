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
    ok &= check_encoder()
    print("\nALL CHECKS PASSED" if ok else "\nSOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
