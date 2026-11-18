"""The points the adding rule gives a clip, to turn and look at beside the clip's
video, at a few gaps and flat-patch distances: can the scene be told from them?
A point is drawn as what it stores, with no joins guessed: a disc at its place,
in its pixel's colour, facing the way its surface faces (read off the depth in
its query frame and turned with its body), as wide as its nearest points that
face alike are far. Picked as training picks them (`ClipDataset._spawned`),
with no model.

  python scripts/points_view.py --cache CACHE --frames FRAMES --clips 0 40 --gaps 0.025 --flats 0 0.01 --out points_view.html
"""

import argparse
import base64
import io
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.ClipDataset import DEPTH, _near, _query_frames, _spawned
from genpoint3d.train.tools.farthest_first import UNIT
from genpoint3d.train.tools.kabsch import kabsch

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--frames", required=True)
p.add_argument("--clips", type=int, nargs="+", default=[0])
p.add_argument("--gaps", type=float, nargs="+", default=[0.025])
p.add_argument("--flats", type=float, nargs="+", default=[0.0, 0.01], help="metres a body's points explain surface within; 0 for the spacing alone")
p.add_argument("--limit", type=int, default=4000, help="the most points a clip holds")
p.add_argument("--asked", type=int, default=256)
p.add_argument("--out", default="points_view.html")
args = p.parse_args()

ALIKE = 0.7  # two points face alike if their facings are within 45 degrees
NEAR = 3     # a disc reaches its third nearest point that faces alike


def _facing(clip: CachedClip, depth: torch.Tensor) -> torch.Tensor:
    """(N, 3) the way each point's surface faces in its query frame, toward the camera, read off the depth there."""
    (T, h, w), (H, W), q = depth.shape, clip.hw, _query_frames(clip)
    v, u = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    cell = (clip.query_uv / torch.tensor([W, H]) * torch.tensor([w, h])).long()
    out = torch.zeros(len(q), 3)
    for t in q.unique().tolist():
        K, z = clip.intrinsics[t].float(), depth[t]
        xyz = torch.stack([((u + 0.5) / w * W - K[0, 2]) / K[0, 0] * z, ((v + 0.5) / h * H - K[1, 2]) / K[1, 1] * z, z])   # (3, h, w) the camera's
        pad = torch.nn.functional.pad(xyz[None], (1, 1, 1, 1), mode="replicate")[0]
        n = torch.linalg.cross(pad[:, 1:-1, 2:] - pad[:, 1:-1, :-2], pad[:, 2:, 1:-1] - pad[:, :-2, 1:-1], dim=0)
        n = torch.nn.functional.normalize(n, dim=0)
        n = torch.where((n * xyz).sum(0, keepdim=True) > 0, -n, n)
        i = q == t
        out[i] = n[:, cell[i, 1].clamp(0, h - 1), cell[i, 0].clamp(0, w - 1)].T @ clip.extrinsics[t].float()[:3, :3]
    return out


def _turns(clip: CachedClip) -> dict:
    """Each body's turn (T, 3, 3) from the first frame, fitted on its points."""
    return {k: kabsch(torch.ones(len(clip.traj), int((clip.segment == k).sum())), clip.traj[:1, clip.segment == k].float(),
                      clip.traj[:, clip.segment == k].float())[0] for k in clip.segment.unique().tolist()}


def _discs(got: CachedClip, facing: torch.Tensor, turns: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """(T, N, 3) each point's facing in every frame, turned with its body, and (T, N) its disc's reach in metres."""
    T, N = got.visibility.shape
    q = _query_frames(got)
    turn = torch.stack([turns[k] for k in got.segment.tolist()], 1)                       # (T, N, 3, 3)
    way = (turn @ turn[q, torch.arange(N)].transpose(1, 2) @ facing[:, :, None])[..., 0]
    reach = torch.full((T, N), UNIT / 2)
    for t in range(T):
        on = q <= t
        apart = torch.cdist(got.traj[t].float(), got.traj[t].float())
        apart[~((got.segment[:, None] == got.segment[None]) & (way[t] @ way[t].T > ALIKE) & on[None])] = float("inf")
        near = apart.sort(1).values[:, 1:NEAR + 1]                                          # without itself
        far = torch.where(near.isfinite(), near, near.new_zeros(())).max(1).values
        reach[t] = torch.where(far > 0, far, reach[t])
    return way, reach


def _jpeg(frame: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(frame).save(out, "JPEG", quality=80)
    return base64.b64encode(out.getvalue()).decode()


files = sorted(Path(args.cache).expanduser().glob("*.pt"))
views = []
for i in args.clips:
    whole = CachedClip.from_dict(torch.load(files[i], weights_only=False))
    video = np.load(Path(args.frames).expanduser() / f"{whole.seq_id}.npy")                # (T, r, r, 3)
    depth = torch.from_numpy(np.load(Path(args.frames).expanduser() / (whole.seq_id + DEPTH))).float()   # (T, r / 4, r / 4)
    r = video.shape[1]
    for gap, flat in [(g, f) for g in args.gaps for f in args.flats]:
        torch.manual_seed(0)
        clip = _near(whole)
        T, N = clip.visibility.shape
        got, _ = _spawned(clip, args.limit, args.asked, gap, torch.ones(T, dtype=torch.bool), torch.randperm(N), flat)
        q, n = _query_frames(got), torch.arange(got.traj.shape[1])
        at = (got.query_uv / torch.tensor([got.hw[1], got.hw[0]]) * r).long().clamp(0, r - 1)
        way, reach = _discs(got, _facing(got, depth), _turns(clip))
        views.append({
            "name": f"clip {whole.seq_id}, gap {gap:g}, flat {flat * 100:g} cm: {len(q)} points, {int((q == 0).sum())} in the first frame",
            "way": (way * 100).round().int().tolist(), "reach": (reach * 100).round().int().tolist(),   # (T, N, 3), (T, N) cm
            "video": [_jpeg(f) for f in video],
            "xyz": (got.traj.float() * 100).round().int().tolist(),                        # (T, N, 3) cm, the first frame's camera
            "seen": got.visibility.int().tolist(), "born": q.tolist(),
            "rgb": video[q.numpy(), at[:, 1].numpy(), at[:, 0].numpy()].tolist(),
            "object": got.segment.int().tolist(),
        })
        print(views[-1]["name"], flush=True)

PAGE = """<!doctype html><meta charset=utf-8><title>Points view</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;overflow:hidden}#bar{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;height:34px}
canvas{display:block}#video{position:fixed;right:8px;top:58px;width:320px;border:1px solid #555}</style>
<div id=bar><select id=which></select><label>frame <input id=t type=range min=0 value=0> <span id=tn></span></label>
<label><input id=colour type=checkbox> colour by object</label><label><input id=discs type=checkbox checked> discs</label>
<label>point size <input id=size type=range min=1 max=10 value=3></label>
<span>drag to turn, right-drag to move, wheel to zoom; dim: not seen in this frame</span></div>
<img id=video title="the clip's video at this frame">
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.178.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.178.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const V = __DATA__, $ = i => document.getElementById(i), BAR = 50;
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000), controls = new OrbitControls(camera, renderer.domElement);
scene.background = new THREE.Color(0x111111);
const points = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({vertexColors: true, size: 0.03}));
const disc = new THREE.CircleGeometry(1, 20), skin = new THREE.MeshBasicMaterial({side: THREE.DoubleSide}), AHEAD = new THREE.Vector3(0, 0, 1);
let discs = null;
scene.add(points);
V.forEach((x, i) => $('which').add(new Option(x.name, i)));
let v;
// The clip's points are in the first frame's camera, in cm, x right, y down, z ahead; three.js looks down -z with y up.
const place = p => [p[0] / 100, -p[1] / 100, -p[2] / 100];
const paint = (i, dim) => {
  const c = $('colour').checked ? new THREE.Color().setHSL(v.object[i] * 67 % 360 / 360, v.object[i] ? 0.8 : 0, 0.6)
                                : new THREE.Color().setRGB(v.rgb[i][0] / 255, v.rgb[i][1] / 255, v.rgb[i][2] / 255, THREE.SRGBColorSpace);
  return c.multiplyScalar(dim).toArray();
};
function show() {
  const t = +$('t').value, here = [];
  $('tn').textContent = t;
  $('video').src = 'data:image/jpeg;base64,' + v.video[t];
  v.xyz[t].forEach((p, i) => { if (v.born[i] <= t) here.push(i); });
  points.geometry.dispose();
  points.geometry = new THREE.BufferGeometry();
  points.geometry.setAttribute('position', new THREE.Float32BufferAttribute(here.flatMap(i => place(v.xyz[t][i])), 3));
  points.geometry.setAttribute('color', new THREE.Float32BufferAttribute(here.flatMap(i => paint(i, v.seen[t][i] ? 1 : 0.3)), 3));
  points.material.size = +$('size').value / 100;
  if (discs) { scene.remove(discs); discs.dispose(); }
  discs = new THREE.InstancedMesh(disc, skin, here.length);
  const m = new THREE.Matrix4(), turn = new THREE.Quaternion(), c = new THREE.Color();
  here.forEach((i, j) => {
    turn.setFromUnitVectors(AHEAD, new THREE.Vector3(...place(v.way[t][i])).normalize());
    const r = v.reach[t][i] / 100;
    discs.setMatrixAt(j, m.compose(new THREE.Vector3(...place(v.xyz[t][i])), turn, new THREE.Vector3(r, r, r)));
    discs.setColorAt(j, c.fromArray(paint(i, 0.8)));
  });
  discs.visible = $('discs').checked;
  scene.add(discs);
}
function pick() {
  v = V[$('which').value];
  $('t').max = v.xyz.length - 1;
  const first = v.xyz[0].filter((p, i) => v.born[i] == 0).map(place), mid = [0, 1, 2].map(k => first.reduce((s, p) => s + p[k], 0) / Math.max(first.length, 1));
  camera.position.set(0, 0, 0);
  controls.target.set(...mid);
  controls.update();
  show();
}
function size() { const w = innerWidth, h = innerHeight - BAR; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
$('which').onchange = pick;
['t', 'size', 'colour', 'discs'].forEach(i => $(i).oninput = show);
addEventListener('resize', size);
size(); pick();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>"""
Path(args.out).expanduser().write_text(PAGE.replace("__DATA__", json.dumps(views, separators=(",", ":"))))
print(f"written {args.out}")
