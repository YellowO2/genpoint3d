"""The points the adding rule gives a clip, to turn and look at beside the clip's
video, at a few gaps and flat-patch distances: can the scene be told from them?
A point stores its place, its pixel's colour and its bars: when it is added,
one to each of its nearest points of its body that the body's surface runs
straight to (read off the clip's own tracks here; the model's to say). Three
points barred to each other are a wall. Picked as training picks them
(`ClipDataset._spawned`), with no model.

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
from genpoint3d.train.data.ClipDataset import _near, _query_frames, _spawned

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

NEAR = 8     # a point added is barred among this many of its nearest points
SAG = 0.02   # metres a bar may run off its body's surface, beyond how far apart the surface's tracks are there
ALONG = torch.tensor([0.25, 0.5, 0.75])


def _bars(got: CachedClip, clip: CachedClip) -> list:
    """The bars (B, 2): each point, when added, to those of its NEAR nearest points of its body there by then
    that the body's surface runs straight to, the surface being the clip's own tracks."""
    q, bars = _query_frames(got), set()
    for f in q.unique().tolist():
        new, pos, surface = torch.nonzero(q == f)[:, 0], got.traj[f].float(), clip.traj[f].float()
        spaced = torch.cat([torch.cdist(surface[i:i + 500], surface).masked_fill(clip.segment[i:i + 500, None] != clip.segment[None], float("inf"))
                            .topk(2, largest=False).values[:, 1] for i in range(0, len(surface), 500)])     # (D,) each track to its nearest
        apart = torch.cdist(pos[new], pos)
        apart[~((got.segment[new, None] == got.segment[None]) & (q <= f)[None])] = float("inf")
        apart[torch.arange(len(new)), new] = float("inf")
        near, who = apart.topk(min(NEAR, len(pos)), largest=False)
        a, b = new[:, None].expand_as(who)[near.isfinite()], who[near.isfinite()]
        on = torch.ones(len(a), dtype=torch.bool)
        for i in range(0, len(a), 500):                                                     # a few at a time, for memory
            j = slice(i, i + 500)
            mid = (pos[a[j], None] + ALONG[:, None] * (pos[b[j]] - pos[a[j]])[:, None]).flatten(0, 1)
            off, at = torch.cdist(mid, surface).masked_fill(got.segment[a[j]].repeat_interleave(len(ALONG))[:, None] != clip.segment[None], float("inf")).min(1)
            on[j] = (off <= SAG + spaced[at]).view(-1, len(ALONG)).all(1)
        bars |= {tuple(sorted(p)) for p in torch.stack([a[on], b[on]], 1).tolist()}
    return sorted(bars)


def _walls(bars: list) -> list:
    """The walls (F, 3): three points barred to each other."""
    joined = {}
    for a, b in bars:
        joined.setdefault(a, set()).add(b)
    return [[a, b, c] for a, b in bars for c in sorted(joined.get(a, set()) & joined.get(b, set()))]


def _jpeg(frame: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(frame).save(out, "JPEG", quality=80)
    return base64.b64encode(out.getvalue()).decode()


files = sorted(Path(args.cache).expanduser().glob("*.pt"))
views = []
for i in args.clips:
    whole = CachedClip.from_dict(torch.load(files[i], weights_only=False))
    video = np.load(Path(args.frames).expanduser() / f"{whole.seq_id}.npy")                # (T, r, r, 3)
    r = video.shape[1]
    for gap, flat in [(g, f) for g in args.gaps for f in args.flats]:
        torch.manual_seed(0)
        clip = _near(whole)
        T, N = clip.visibility.shape
        got, _ = _spawned(clip, args.limit, args.asked, gap, torch.ones(T, dtype=torch.bool), torch.randperm(N), flat)
        q, n = _query_frames(got), torch.arange(got.traj.shape[1])
        at = (got.query_uv / torch.tensor([got.hw[1], got.hw[0]]) * r).long().clamp(0, r - 1)
        bars = _bars(got, clip)
        views.append({
            "name": f"clip {whole.seq_id}, gap {gap:g}, flat {flat * 100:g} cm: {len(q)} points, {len(bars)} bars",
            "bars": bars, "walls": _walls(bars),
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
<label><input id=colour type=checkbox> colour by object</label><label><input id=faces type=checkbox checked> walls</label><label><input id=bars type=checkbox checked> bars</label>
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
const mesh = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({vertexColors: true, side: THREE.DoubleSide}));
const bars = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color: 0xffffff, transparent: true, opacity: 0.35}));
scene.add(mesh, bars, points);
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
  mesh.geometry.dispose();
  mesh.geometry = new THREE.BufferGeometry();
  mesh.geometry.setAttribute('position', new THREE.Float32BufferAttribute(v.xyz[t].flatMap(place), 3));
  mesh.geometry.setAttribute('color', new THREE.Float32BufferAttribute(v.xyz[t].flatMap((p, i) => paint(i, 0.8)), 3));
  mesh.geometry.setIndex(v.walls.filter(f => f.every(i => v.born[i] <= t)).flat());
  mesh.visible = $('faces').checked;
  bars.geometry.dispose();
  bars.geometry = new THREE.BufferGeometry();
  bars.geometry.setAttribute('position', new THREE.Float32BufferAttribute(v.xyz[t].flatMap(place), 3));
  bars.geometry.setIndex(v.bars.filter(f => f.every(i => v.born[i] <= t)).flat());
  bars.visible = $('bars').checked;
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
['t', 'size', 'colour', 'faces', 'bars'].forEach(i => $(i).oninput = show);
addEventListener('resize', size);
size(); pick();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>"""
Path(args.out).expanduser().write_text(PAGE.replace("__DATA__", json.dumps(views, separators=(",", ":"))))
print(f"written {args.out}")
