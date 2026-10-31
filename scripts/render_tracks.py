"""
Watch a checkpoint track: its predictions drawn over the RGB frames, as mp4s.

A visual sanity check beside the scores. The predictions are made exactly the
way scoring makes them -- the same `Predictor`, the same val split, the same
batch size, so the same noise -- and each clip's moving-point APD is printed,
so every video can be matched to a number.

Per point, in each frame:
    GREEN  the truth: filled where it is visible, a ring where it is hidden,
           left out behind the camera or off the image
    RED    the prediction, with a thin line to the truth, and a short trail of
           its last few positions in the point's own colour. Off the image, a
           small arrow at the border points the way it went.

Beside it, on black, every point of the clip the same way without the video
(`--panel-points`), so the whole cloud's motion shows.

With `--forecast-from C` the truth is left out: RED is the model tracking
with every frame seen, and from frame C on GREEN is the same model
forecasting with the frames from C blank, a line from each to its red twin.

Writes `<out>/<clip>.mp4` and `<out>/<clip>.png`, a contact sheet of 6 frames.

Run:  python scripts/render_tracks.py --ckpt outputs/run33/best.pt \\
          --cache ~/scratch/cache/kubric_768_3493_ctr \\
          --frames-cache ~/scratch/cache/kubric_frames384 --clips 6 --out renders
"""

import argparse
import colorsys
import dataclasses
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.Settings import Settings
from genpoint3d.train.data.ClipDataset import ClipDataset
from genpoint3d.train.data.split_clips import split_clips
from genpoint3d.train.model.build_model import build_model
from genpoint3d.train.score.Predictor import Predictor
from genpoint3d.train.score.ScoreSettings import ScoreSettings
from genpoint3d.train.tools.batch import to_metres
from genpoint3d.train.tools.metrics import _project, _to_camera_t, motion_px, tapvid3d_metrics

SIZE = 512          # drawn frame side, px
TRAIL = 8           # predicted positions kept in the trail
FPS = 6
SHEET = 6           # frames on the contact sheet
GREEN, RED, WHITE = (0, 220, 0), (0, 0, 255), (255, 255, 255)   # BGR


def pixels(pts_m: torch.Tensor, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """(T, N, 3) metres in the frame-0 camera -> (T, N, 2) pixels of the drawn
    SIZE x SIZE frame, and (T, N) depth in each frame's own camera.

    The cached frames are the native frame squashed to a square, so a point a
    fraction u of the way across natively is that fraction across here too."""
    cam = _to_camera_t(pts_m[None], b["extrinsics"])
    uv = _project(cam, b["intrinsics"])[0]
    h, w = b["hw"][0].tolist()
    uv = uv / torch.tensor([w, h], device=uv.device) * SIZE
    return uv.cpu().numpy(), cam[0, ..., 2].cpu().numpy()


def inside(p: np.ndarray, z: float) -> bool:
    return z > 0 and 0 <= p[0] < SIZE and 0 <= p[1] < SIZE


def colours(n: int) -> list[tuple[int, int, int]]:
    """n well-separated BGR colours, one per point, for following its trail."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.618034) % 1.0, 0.85, 1.0)
        out.append((int(b * 255), int(g * 255), int(r * 255)))
    return out


def border_arrow(img, p: np.ndarray, colour) -> None:
    """A small arrow at the frame's edge, pointing at an off-image `p`."""
    c = np.array([SIZE / 2, SIZE / 2])
    d = p - c
    d = d / max(np.linalg.norm(d), 1e-6)
    tip = np.clip(p, 3, SIZE - 4)
    tail = tip - 10 * d
    cv2.arrowedLine(img, tuple(int(v) for v in tail), tuple(int(v) for v in tip),
                    colour, 1, cv2.LINE_AA, tipLength=0.4)


def draw_frame(rgb: np.ndarray, t: int, gt: tuple, pr: tuple, vis: np.ndarray,
               cols: list, cutoff: int, first: int = 0, look=(GREEN, RED),
               behind: np.ndarray | None = None) -> np.ndarray:
    """One frame: rgb (R, R, 3) uint8, gt and pr each (pixels (T, N, 2),
    depth (T, N)), vis (T, N) bool. Returns SIZE x SIZE BGR.

    Drawn at twice the size and shrunk, so the small dots and thin lines come
    out smooth. `behind` (T, N) marks predictions that fall behind the visible
    surface at their pixel: drawn as a ring, not a dot, since the image there
    shows what is in front of them. Green is the truth (faded while the point is hidden), red the
    prediction, a faint line between them, and a thin trail in the point's own
    colour."""
    k = 2
    img = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), (SIZE * k, SIZE * k),
                     interpolation=cv2.INTER_CUBIC)
    (gp, gz), (pp, pz) = gt, pr
    gp, pp = gp * k, pp * k
    N = gp.shape[1]
    at = lambda xy: tuple(np.round(xy).astype(int))
    on = lambda xy, z: inside(xy / k, z)
    # `first`: the prediction is drawn from that frame on, and the trails
    # follow the first set instead.
    show = t >= first
    tp, tz = (gp, gz) if first else (pp, pz)
    g_col, p_col = look
    # The trails first, oldest faintest, so the dots sit on top of them.
    for age in range(TRAIL - 1, 0, -1):
        s = t - age
        if s < 0:
            continue
        layer = img.copy()
        for n in range(N):
            a, b = tp[s, n], tp[s + 1, n]
            if on(a, tz[s, n]) and on(b, tz[s + 1, n]):
                cv2.line(layer, at(a), at(b), cols[n], 2, cv2.LINE_AA)
        alpha = 0.7 * (1.0 - age / TRAIL)
        img = cv2.addWeighted(layer, alpha, img, 1 - alpha, 0)
    links, hidden = img.copy(), img.copy()
    for n in range(N):
        if show and on(gp[t, n], gz[t, n]) and on(pp[t, n], pz[t, n]):
            cv2.line(links, at(gp[t, n]), at(pp[t, n]), WHITE, 1, cv2.LINE_AA)
    img = cv2.addWeighted(links, 0.5, img, 0.5, 0)
    for n in range(N):
        if on(gp[t, n], gz[t, n]):
            cv2.circle(hidden if not vis[t, n] else img, at(gp[t, n]), 6, g_col, -1, cv2.LINE_AA)
    # Hidden truth at 40%: the point is still there, just not seen.
    mask = (hidden != img).any(-1)
    img[mask] = cv2.addWeighted(hidden, 0.4, img, 0.6, 0)[mask]
    for n in range(N):
        if show and on(pp[t, n], pz[t, n]):
            hollow = behind is not None and behind[t, n]
            cv2.circle(img, at(pp[t, n]), 6, p_col, 3 if hollow else -1, cv2.LINE_AA)
    img = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    for n in range(N):
        if show and not on(pp[t, n], pz[t, n]) and pz[t, n] > 0:
            border_arrow(img, pp[t, n] / k, p_col)
    label = f"{t}" + ("" if not cutoff else ("" if t < cutoff else "  forecast"))
    cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, label, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return img


def draw_points(t: int, gt: tuple, pr: tuple, vis: np.ndarray, first: int = 0,
                look=(GREEN, RED)) -> np.ndarray:
    """Every point at frame t on black, SIZE x SIZE BGR: the first set
    (faded while hidden), the second from frame `first` on, a faint line
    between them. By default the truth green and the prediction red."""
    img = np.zeros((SIZE, SIZE, 3), np.uint8)
    show, (g_col, p_col) = t >= first, look
    faded = tuple(c * 2 // 5 for c in g_col)
    (gp, gz), (pp, pz) = gt, pr
    at = lambda xy: tuple(np.round(xy).astype(int))
    for n in range(gp.shape[1]):
        if show and inside(gp[t, n], gz[t, n]) and inside(pp[t, n], pz[t, n]):
            cv2.line(img, at(gp[t, n]), at(pp[t, n]), (90, 90, 90), 1, cv2.LINE_AA)
    for n in range(gp.shape[1]):
        if inside(gp[t, n], gz[t, n]):
            cv2.circle(img, at(gp[t, n]), 3, g_col if vis[t, n] else faded, -1, cv2.LINE_AA)
    for n in range(gp.shape[1]):
        if show and inside(pp[t, n], pz[t, n]):
            cv2.circle(img, at(pp[t, n]), 3, p_col, -1, cv2.LINE_AA)
    cv2.putText(img, f"all {gp.shape[1]} points", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                WHITE, 1, cv2.LINE_AA)
    return img


def write_video(frames: list[np.ndarray], path: Path) -> str:
    """H.264 if this OpenCV build has it, else MPEG-4 part 2. Returns which."""
    height, width = frames[0].shape[:2]
    for code in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*code), FPS, (width, height))
        if w.isOpened():
            for f in frames:
                w.write(f)
            w.release()
            return code
    raise RuntimeError(f"OpenCV could not open an mp4 writer for {path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cache", required=True, help="feature cache directory")
    p.add_argument("--frames-cache", required=True,
                   help="RGB frames of --cache's clips, from scripts/cache_frames.py")
    p.add_argument("--clips", type=int, default=6, help="the first K val clips")
    p.add_argument("--clip", nargs="*", default=None,
                   help="these clips by name instead of the val split")
    p.add_argument("--points", type=int, default=32,
                   help="query points drawn, the most moving first")
    p.add_argument("--panel-points", type=int, default=0,
                   help="also draw this many points on black beside the video; 0 none")
    p.add_argument("--forecast-from", type=int, default=0, metavar="C",
                   help="draw tracking (red) against forecasting from frame C (green), no truth")
    p.add_argument("--eval-cutoff", type=int, default=0, metavar="C",
                   help="forecast: frames C on get no image, as in evaluate.py")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--batch", type=int, default=None,
                   help="clips per batch; the run's own by default. 1 for clips with different numbers of points")
    p.add_argument("--out", required=True, help="directory for the mp4s and PNGs")
    p.add_argument("--draw-truth-as-pred", action="store_true",
                   help="debug: draw the truth as the prediction; red must sit on green")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    targs = ckpt["args"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and torch.cuda.is_bf16_supported()

    # The trainer's own split, so "the first K val clips" are the clips its
    # val APD was measured on.
    split = split_clips(args.cache, targs.get("val_frac", 0.1), targs.get("seed", 0))
    if args.clip:
        clips = [Path(args.cache) / f"{c}.pt" for c in args.clip]
    else:
        clips = split.val[: args.clips]
    method = targs.get("method", "flow")
    fine, fine_xyz = targs.get("fine", 0), bool(targs.get("fine_xyz", 0))
    ds = ClipDataset(clips, targs["points"], resample=False,
                     norm_mode=targs.get("norm_mode", "median"),
                     target=targs.get("target", "absolute"),
                     fix_idcard=bool(targs.get("fix_idcard", 0)),
                     frames=(ClipDataset.frames_dir(args.frames_cache, clips,
                                                    targs.get("fine_res", 384), depth=fine_xyz)
                             if fine else None),
                     features=bool(targs.get("dino", 1)), depth=fine_xyz)
    # The trainer's val batch size: flow sampling draws its noise per batch,
    # so this is what makes these the very samples the val APD scored.
    loader = DataLoader(ds, batch_size=args.batch or targs.get("batch", 1), shuffle=False, num_workers=2)

    model = build_model(Settings(**targs), split.feat_dim, split.has_feats).to(device)
    model.load_state_dict(ckpt["model"])
    judging = ScoreSettings(device=device, steps=args.sample_steps, amp=amp,
                            anchor_frame0=bool(targs.get("anchor_frame0", 0)),
                            loss_type=targs.get("loss_type", "l21"), method=method,
                            refine_iters=targs.get("refine_iters", 4),
                            param=targs.get("param", "position"),
                            carry=bool(targs.get("carry", 0)), window=targs.get("window", 0),
                            overlap=targs.get("overlap", 0), cutoff=args.eval_cutoff)
    predictor = Predictor(model, judging)
    # The same model, the frames from C on blank: what it forecasts there.
    forecaster = (Predictor(model, dataclasses.replace(judging, cutoff=args.forecast_from))
                  if args.forecast_from else None)
    print(f"step {ckpt['step']} of {args.ckpt} | {device} | {method} | {len(clips)} clips"
          f" | drawing {args.points} of {targs['points']} points", flush=True)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    k = 0
    for batch in loader:
        # The scorer is wrapped in no_grad (score/main.py); calling the
        # Predictor directly is not, and would keep every activation.
        with torch.no_grad():
            pred = predictor(batch)
            ahead = forecaster(batch) if forecaster else None
        b = pred.batch
        gt_m = to_metres(b["traj"], b)
        pr_m = gt_m if args.draw_truth_as_pred else to_metres(pred.tracks.float(), b)
        vis = b["visibility"].bool()
        # The scorer's moving-point APD: points that move more than 4 px, no
        # rescale (`track_scores`).
        moving_px = motion_px(gt_m, vis, b["intrinsics_256"], b["extrinsics"])
        mv = vis & (moving_px > 4)[:, None]
        apd_mv = tapvid3d_metrics(pr_m, gt_m, mv, b["intrinsics_256"], b["extrinsics"],
                                  scaling="none", per_clip=True)["average_pts_within_thresh"]
        apd = tapvid3d_metrics(pr_m, gt_m, vis, b["intrinsics_256"], b["extrinsics"],
                               per_clip=True)["average_pts_within_thresh"]
        for i in range(gt_m.shape[0]):
            name = clips[k].stem
            k += 1
            one = {key: b[key][i:i + 1] for key in ("intrinsics", "extrinsics", "hw")}
            pick = moving_px[i].argsort(descending=True)[: args.points]
            gt = pixels(gt_m[i][:, pick], one)
            pr = pixels(pr_m[i][:, pick], one)
            # Predictions farther than the surface the frame shows at their
            # pixel (its depth at the CNN cell there), by more than 3%.
            behind = None
            if b.get("depth") is not None:
                d = b["depth"][i].float().cpu().numpy()                     # (T, h, w)
                h, w = d.shape[1:]
                cu = np.clip(np.round(pr[0][..., 0] / SIZE * w - 0.125), 0, w - 1).astype(int)
                cv = np.clip(np.round(pr[0][..., 1] / SIZE * h - 0.125), 0, h - 1).astype(int)
                surface = d[np.arange(d.shape[0])[:, None], cv, cu]
                behind = pr[1] > surface * 1.03
            v = vis[i][:, pick].cpu().numpy()
            rgb = np.load(Path(args.frames_cache) / f"{name}.npy")
            cols = colours(len(pick))
            # Tracking against forecasting: the first set is the tracking, red,
            # the second the forecast, green from frame C on.
            how = dict(first=args.forecast_from, look=(RED, GREEN)) if ahead else {}
            fc_m = to_metres(ahead.tracks.float(), b) if ahead else None
            if ahead:
                gt, pr = pr, pixels(fc_m[i][:, pick], one)
            frames = [draw_frame(rgb[t], t, gt, pr, v, cols, args.eval_cutoff or args.forecast_from,
                                 behind=None if ahead else behind, **how)
                      for t in range(rgb.shape[0])]
            if args.panel_points:
                every = moving_px[i].argsort(descending=True)[: args.panel_points]
                first_m, second_m = (pr_m, fc_m) if ahead else (gt_m, pr_m)
                gt_all, pr_all = pixels(first_m[i][:, every], one), pixels(second_m[i][:, every], one)
                v_all = vis[i][:, every].cpu().numpy()
                frames = [np.hstack([f, draw_points(t, gt_all, pr_all, v_all, **how)])
                          for t, f in enumerate(frames)]
            codec = write_video(frames, out / f"{name}.mp4")
            at = np.linspace(0, len(frames) - 1, SHEET).round().astype(int)
            cv2.imwrite(str(out / f"{name}.png"), np.hstack([frames[t] for t in at]))
            n_mv = int((moving_px[i] > 4).sum())
            print(f"  {name}  APD moving {apd_mv[i].item():.3f} ({n_mv} moving points)"
                  f"  APD all {apd[i].item():.3f}  -> {out / name}.mp4 [{codec}]", flush=True)
        if k >= len(clips):
            break
    predictor.done()
    if forecaster:
        forecaster.done()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
