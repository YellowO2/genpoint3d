"""The world a clip was made from, to turn and look at beside the clip's video:
what the stored world should come to be. Each object is its own low-poly shape,
all sides, the one Kubric's physics engine used (a Google Scanned Object's
collision mesh): its corners the points, its edges the bars, its faces the
walls, placed as Kubric had it in every frame. No model.

  python scripts/world_view.py --objects OBJECTS --frames FRAMES --assets DIR --clips 000040 --out world_view.html

OBJECTS is scripts/movi_objects.py's; a model missing from DIR is fetched from Kubric's public store.
"""

import argparse
import base64
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

STORE = "https://storage.googleapis.com/kubric-public/assets/GSO/{}.tar.gz"

p = argparse.ArgumentParser()
p.add_argument("--objects", required=True)
p.add_argument("--frames", required=True)
p.add_argument("--assets", required=True)
p.add_argument("--clips", nargs="+", required=True)
p.add_argument("--out", default="world_view.html")
args = p.parse_args()
assets = Path(args.assets).expanduser()


def model(name: str) -> tuple[list, list, list]:
    """A model's corners (V, 3) in metres, its walls (F, 3) and its colour, the mean of its texture."""
    home = assets / name
    if not home.exists():
        home.mkdir(parents=True)
        with urllib.request.urlopen(STORE.format(name)) as r, tarfile.open(fileobj=io.BytesIO(r.read())) as tar:
            for m in tar.getmembers():
                if m.isfile() and Path(m.name).name in ("collision_geometry.obj", "texture.png"):
                    (home / Path(m.name).name).write_bytes(tar.extractfile(m).read())
    corners, walls = [], []
    for line in (home / "collision_geometry.obj").read_text().splitlines():
        part = line.split()
        if part[:1] == ["v"]:
            corners.append([float(x) for x in part[1:4]])
        elif part[:1] == ["f"]:
            i = [int(x.split("/")[0]) - 1 for x in part[1:]]
            walls += [[i[0], i[k], i[k + 1]] for k in range(1, len(i) - 1)]
    colour = np.asarray(Image.open(home / "texture.png").convert("RGB").resize((16, 16))).reshape(-1, 3).mean(0)
    return corners, walls, colour.round().astype(int).tolist()


def jpeg(frame: np.ndarray) -> str:
    out = io.BytesIO()
    Image.fromarray(frame).save(out, "JPEG", quality=80)
    return base64.b64encode(out.getvalue()).decode()


views = []
for clip in args.clips:
    d = np.load(Path(args.objects).expanduser() / f"{clip}.npz")
    bodies = []
    for k, name in enumerate(d["asset"].tolist()):
        corners, walls, colour = model(name)
        bodies.append({"name": name, "corners": (np.array(corners) * float(d["scale"][k])).round(4).tolist(), "walls": walls, "colour": colour,
                       "place": d["positions"][k].round(4).tolist(), "turn": d["quaternions"][k].round(5).tolist()})   # (T, 3), (T, 4) w first
    points = sum(len(b["corners"]) for b in bodies)
    views.append({"name": f"clip {clip}: {len(bodies)} objects, {points} points", "bodies": bodies,
                  "camera": d["camera_positions"].round(4).tolist(), "camera_turn": d["camera_quaternions"].round(5).tolist(),
                  "fov": float(np.degrees(2 * np.arctan(float(d["sensor_width"]) / 2 / float(d["focal_length"])))),
                  "video": [jpeg(f) for f in np.load(Path(args.frames).expanduser() / f"{clip}.npy")]})
    print(views[-1]["name"], "; points an object:", sorted(len(b["corners"]) for b in bodies), flush=True)

PAGE = """<!doctype html><meta charset=utf-8><title>World view</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif;overflow:hidden}#bar{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap;height:34px}
canvas{display:block}#video{position:fixed;right:8px;top:58px;width:320px;border:1px solid #555}</style>
<div id=bar><select id=which></select><label>frame <input id=t type=range min=0 value=0> <span id=tn></span></label>
<label><input id=walls type=checkbox checked> walls</label><label><input id=bars type=checkbox checked> bars</label>
<button id=cam>the video's camera</button><span>drag to turn, right-drag to move, wheel to zoom</span></div>
<img id=video title="the clip's video at this frame">
<script type=importmap>{"imports":{"three":"https://unpkg.com/three@0.178.0/build/three.module.js","three/addons/":"https://unpkg.com/three@0.178.0/examples/jsm/"}}</script>
<script type=module>
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
const V = __DATA__, $ = i => document.getElementById(i), BAR = 50;
const renderer = new THREE.WebGLRenderer({antialias: true});
renderer.setPixelRatio(Math.min(devicePixelRatio, 1.5));
document.body.appendChild(renderer.domElement);
const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.05, 500), controls = new OrbitControls(camera, renderer.domElement);
camera.up.set(0, 0, 1);                                                    // Kubric's world: z up, the floor at z = 0
scene.background = new THREE.Color(0x111111);
scene.add(new THREE.HemisphereLight(0xffffff, 0x444444, 1.6));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(3, -4, 8);
scene.add(sun);
const floor = new THREE.Mesh(new THREE.CircleGeometry(12, 48), new THREE.MeshLambertMaterial({color: 0x555555, side: THREE.DoubleSide}));
const lines = new THREE.GridHelper(24, 24, 0x777777, 0x666666);
lines.rotation.x = Math.PI / 2;
lines.position.z = 0.002;
scene.add(floor, lines);
V.forEach((x, i) => $('which').add(new Option(x.name, i)));
let v, made = [];
function pick() {
  made.forEach(m => { scene.remove(m.walls, m.bars); m.walls.geometry.dispose(); m.bars.geometry.dispose(); });
  v = V[$('which').value];
  $('t').max = v.video.length - 1;
  made = v.bodies.map(b => {
    const shape = new THREE.BufferGeometry();
    shape.setAttribute('position', new THREE.Float32BufferAttribute(b.corners.flat(), 3));
    shape.setIndex(b.walls.flat());
    shape.computeVertexNormals();
    const walls = new THREE.Mesh(shape, new THREE.MeshLambertMaterial({color: new THREE.Color().setRGB(b.colour[0] / 255, b.colour[1] / 255, b.colour[2] / 255, THREE.SRGBColorSpace), flatShading: true, side: THREE.DoubleSide}));
    const bars = new THREE.LineSegments(new THREE.WireframeGeometry(shape), new THREE.LineBasicMaterial({color: 0x000000, transparent: true, opacity: 0.45}));
    scene.add(walls, bars);
    return {walls, bars, b};
  });
  look();
  show();
}
function show() {
  const t = +$('t').value;
  $('tn').textContent = t;
  $('video').src = 'data:image/jpeg;base64,' + v.video[t];
  for (const m of made) for (const o of [m.walls, m.bars]) {
    o.position.set(...m.b.place[t]);
    o.quaternion.set(m.b.turn[t][1], m.b.turn[t][2], m.b.turn[t][3], m.b.turn[t][0]);
    o.visible = $(o === m.walls ? 'walls' : 'bars').checked;
  }
}
function look() {                                                          // from where the video's camera is, toward what it looks at
  const t = +$('t').value, q = v.camera_turn[t], at = new THREE.Vector3(...v.camera[t]);
  const ahead = new THREE.Vector3(0, 0, -1).applyQuaternion(new THREE.Quaternion(q[1], q[2], q[3], q[0]));
  camera.fov = v.fov;
  camera.position.copy(at);
  controls.target.copy(at).addScaledVector(ahead, at.length());
  size();
  controls.update();
}
function size() { const w = innerWidth, h = innerHeight - BAR; renderer.setSize(w, h); camera.aspect = w / h; camera.updateProjectionMatrix(); }
$('which').onchange = pick;
$('cam').onclick = look;
['t', 'walls', 'bars'].forEach(i => $(i).oninput = show);
addEventListener('resize', size);
size(); pick();
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
</script>"""
Path(args.out).expanduser().write_text(PAGE.replace("__DATA__", json.dumps(views, separators=(",", ":"))))
print(f"written {args.out}")
