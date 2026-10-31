"""Predict a batch's tracks and its training loss, without learning."""

import torch

from genpoint3d.train.score.Prediction import Prediction
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import fine_inputs, known_frame0, to_device, visual_mask
from genpoint3d.train.tools.flow import flow_matching_loss, sample
from genpoint3d.train.tools.regress import refine, regress_loss

# Scoring starts every sample from the same noise, so a checkpoint has one
# score rather than a slightly different one each time it is asked.
EVAL_SEED = 0


class Predictor:
    """Holds the model in eval mode, and the noise every sample starts from,
    until `done` hands the model back for training."""

    def __init__(self, model, judging: ScoreSettings) -> None:
        self.model, self.judging = model, judging
        model.eval()
        self.gen = torch.Generator(device=judging.device).manual_seed(EVAL_SEED)
        # The extra best-of-k samples draw from their own noise, so asking for
        # more of them never changes the first sample, or its score.
        self.more_gen = torch.Generator(device=judging.device).manual_seed(EVAL_SEED + 1)

    def __call__(self, batch: dict) -> Prediction:
        model, j = self.model, self.judging
        b = to_device(batch, j.device, j.amp)
        traj, anchor, vis, ctx, idc, pxyz = (b["traj"], b["anchor"], b["visibility"],
                                             b["context"], b["id_card"], b["patch_xyz"])
        # One cutoff for every clip, so the future scores are all the same
        # forecast; without one, every frame keeps its image.
        cutoff = None
        if j.cutoff:
            if j.cutoff >= traj.shape[1]:
                raise ValueError(f"cutoff {j.cutoff} leaves none of {traj.shape[1]} frames to forecast")
            cutoff = torch.full((traj.shape[0],), j.cutoff, device=j.device)
        vm = visual_mask(b, cutoff)
        # What the model is shown. The scores still use `vm`: hiding is a
        # change to the input, not to which frames count.
        shown = vm
        if j.hide_hidden and vm is not None:
            # A diagnostic that reads the truth, set only by `scripts/evaluate.py
            # --ablate hide-hidden`: each point's image is blanked wherever it is
            # hidden, as a forecast frame is, except frame 0, the query.
            in_view = vis.bool().clone()
            in_view[:, 0] = True
            shown = vm[..., None] & in_view

        kx0 = known_frame0(b) if j.anchor_frame0 else None
        # The same autocast as the training step: a model trained in bf16 and
        # scored in fp32 is being scored on numerics it never saw.
        cond = dict(context=ctx, visual_mask=shown, id_card=idc, patch_xyz=pxyz,
                    **fine_inputs(model, b, j.amp))
        # Regress is deterministic: k samples would be k copies of one answer.
        others = ()
        seen, logits = model.visibility, None
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=j.amp):
            if j.method == "regress":
                l, _ = regress_loss(model, traj, anchor, mask=vis, known_x0=kx0,
                                    loss_type=j.loss_type, iters=j.refine_iters,
                                    param=j.param, carry=j.carry, window=j.window,
                                    overlap=j.overlap, **cond)
                if j.support:
                    pred = _with_support(model, anchor, traj.shape[1], kx0, seen, j, cond)
                else:
                    pred = refine(model, anchor, num_frames=traj.shape[1], iters=j.refine_iters,
                                  known_x0=kx0, visibility=seen, param=j.param,
                                  carry=j.carry, window=j.window, overlap=j.overlap, reach=j.reach, **cond)
            else:
                l, _ = flow_matching_loss(model, traj, anchor, mask=vis, known_x0=kx0,
                                          loss_type=j.loss_type, **cond)
                pred = sample(model, anchor, num_frames=traj.shape[1], steps=j.steps,
                              generator=self.gen, known_x0=kx0, visibility=seen, **cond)
                others = tuple(
                    sample(model, anchor, num_frames=traj.shape[1], steps=j.steps,
                           generator=self.more_gen, known_x0=kx0, **cond)
                    for _ in range(j.samples - 1 if cutoff is not None else 0))
        if seen:
            pred, logits = pred
        return Prediction(batch=b, visual_mask=vm, tracks=pred, loss=l.item(),
                          others=others, visibility=logits)

    def done(self) -> None:
        self.model.train()


def _with_support(model, anchor, num_frames, kx0, seen, j: ScoreSettings, cond: dict):
    """Refine the queries together with a `j.support` x `j.support` grid of
    helper points over frame 0, then keep the queries' tracks only.

    TAPIP3D's test-time support grid (16 x 16, `inference_utils.py`): points
    seen almost everywhere, so a query's neighbours in the spatial attention
    always include visible points showing how its surroundings move. Each
    helper starts where frame 0's depth puts the centre of its grid cell. Only
    for a model that reads no DINOv3: a helper would need a DINOv3 id card,
    which only the cached query points have.
    """
    if cond.get("id_card") is not None or cond.get("context") is not None:
        raise ValueError("support points are built for a model without DINOv3 (--dino 0)")
    if cond.get("fine_xyz") is None:
        raise ValueError("support points start from frame 0's depth: a finder with --fine-xyz 1")
    S, B = j.support, anchor.shape[0]
    centre = (torch.arange(S, device=anchor.device, dtype=torch.float32) + 0.5) / S
    gy, gx = torch.meshgrid(centre, centre, indexing="ij")
    grid = torch.stack([gx, gy], -1).reshape(1, S * S, 1, 2).expand(B, -1, -1, -1) * 2 - 1
    helpers = torch.nn.functional.grid_sample(cond["fine_xyz"][:, 0].float(), grid,
                                              align_corners=False)[..., 0].transpose(1, 2)
    all_anchor = torch.cat([anchor, helpers.to(anchor.dtype)], dim=1)
    all_kx0 = None if kx0 is None else torch.cat([kx0, torch.zeros_like(helpers).to(kx0.dtype)], 1)
    vm = cond.get("visual_mask")
    if vm is not None and vm.dim() == 3:
        raise ValueError("support points take a per-frame visual mask, not a per-point one")
    out = refine(model, all_anchor, num_frames=num_frames, iters=j.refine_iters, known_x0=all_kx0,
                 visibility=seen, param=j.param, carry=j.carry, window=j.window,
                 overlap=j.overlap, reach=j.reach, **cond)
    N = anchor.shape[1]
    return (out[0][:, :, :N], out[1][:, :, :N]) if seen else out[:, :, :N]
