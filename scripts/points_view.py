"""The points the adding rule gives a clip, as a point cloud to turn and look at
(no video, each point in its pixel's colour), at a few gaps: is the density right?
Picked as training picks them (`ClipDataset._spawned`), with no model.

  python scripts/points_view.py --cache CACHE --frames FRAMES --clips 0 40 --gaps 0.025 0.035 0.05 --out points_view.html
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from genpoint3d.CachedClip import CachedClip
from genpoint3d.train.data.ClipDataset import _near, _query_frames, _spawned

p = argparse.ArgumentParser()
p.add_argument("--cache", required=True)
p.add_argument("--frames", required=True)
p.add_argument("--clips", type=int, nargs="+", default=[0])
p.add_argument("--gaps", type=float, nargs="+", default=[0.025, 0.035, 0.05])
p.add_argument("--limit", type=int, default=4000, help="the most points a clip holds")
p.add_argument("--asked", type=int, default=256)
p.add_argument("--out", default="points_view.html")
args = p.parse_args()

files = sorted(Path(args.cache).expanduser().glob("*.pt"))
views = []
for i in args.clips:
    whole = CachedClip.from_dict(torch.load(files[i], weights_only=False))
    video = np.load(Path(args.frames).expanduser() / f"{whole.seq_id}.npy")                # (T, r, r, 3)
    r = video.shape[1]
    for gap in args.gaps:
        torch.manual_seed(0)
        clip = _near(whole)
        T, N = clip.visibility.shape
        got, _ = _spawned(clip, args.limit, args.asked, gap, torch.ones(T, dtype=torch.bool), torch.randperm(N))
        q, n = _query_frames(got), torch.arange(got.traj.shape[1])
        at = (got.query_uv / torch.tensor([got.hw[1], got.hw[0]]) * r).long().clamp(0, r - 1)
        views.append({
            "name": f"clip {whole.seq_id}, gap {gap:g}: {len(q)} points, {int((q == 0).sum())} in the first frame",
            "xyz": (got.traj.float() * 100).round().int().tolist(),                        # (T, N, 3) cm, the first frame's camera
            "seen": got.visibility.int().tolist(), "born": q.tolist(),
            "rgb": video[q.numpy(), at[:, 1].numpy(), at[:, 0].numpy()].tolist(),
            "object": (got.segment if got.segment is not None else torch.zeros(len(q))).int().tolist(),
        })
        print(views[-1]["name"], flush=True)

PAGE = """<!doctype html><meta charset=utf-8><title>Points view</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px sans-serif}#bar{padding:8px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
canvas{display:block;width:100vw;height:calc(100vh - 50px);cursor:grab}</style>
<div id=bar><select id=which></select><label>frame <input id=t type=range min=0 value=0> <span id=tn></span></label>
<label><input id=colour type=checkbox> colour by object</label><label>point size <input id=size type=range min=1 max=8 value=3></label>
<span>drag to turn, wheel to zoom; faded: not seen in this frame</span></div><canvas id=c></canvas>
<script>const V=__DATA__;const $=i=>document.getElementById(i),c=$('c'),g=c.getContext('2d');let yaw=0,pitch=0,zoom=1,v;
V.forEach((x,i)=>$('which').add(new Option(x.name,i)));
function pick(){v=V[$('which').value];$('t').max=v.xyz.length-1;let s=[0,0,0],k=0;v.xyz[0].forEach((p,i)=>{if(v.born[i]==0){s[0]+=p[0];s[1]+=p[1];s[2]+=p[2];k++}});v.mid=s.map(a=>a/Math.max(k,1));draw()}
function hue(k){return k?`hsl(${k*67%360},80%,60%)`:'#888'}
function draw(){c.width=c.clientWidth;c.height=c.clientHeight;g.fillStyle='#111';g.fillRect(0,0,c.width,c.height);const t=+$('t').value;$('tn').textContent=t;
const cy=Math.cos(yaw),sy=Math.sin(yaw),cp=Math.cos(pitch),sp=Math.sin(pitch),f=c.height*1.1*zoom,D=v.mid[2],sz=+$('size').value,out=[];
v.xyz[t].forEach((p,i)=>{if(v.born[i]>t)return;let x=p[0]-v.mid[0],y=p[1]-v.mid[1],z=p[2]-v.mid[2];
let x1=cy*x+sy*z,z1=-sy*x+cy*z,y1=cp*y-sp*z1,z2=sp*y+cp*z1+D;if(z2<10)return;out.push([z2,c.width/2+f*x1/z2,c.height/2+f*y1/z2,i])});
out.sort((a,b)=>b[0]-a[0]);for(const [z,x,y,i] of out){g.globalAlpha=v.seen[t][i]?1:0.25;
g.fillStyle=$('colour').checked?hue(v.object[i]):`rgb(${v.rgb[i]})`;const s=Math.max(sz*D/z,1);g.fillRect(x-s/2,y-s/2,s,s)}g.globalAlpha=1}
let down=null;c.onmousedown=e=>down=[e.clientX,e.clientY];onmouseup=()=>down=null;
onmousemove=e=>{if(!down)return;yaw+=(e.clientX-down[0])/200;pitch+=(e.clientY-down[1])/200;down=[e.clientX,e.clientY];draw()};
c.onwheel=e=>{e.preventDefault();zoom*=e.deltaY<0?1.1:0.9;draw()};
['which'].forEach(i=>$(i).onchange=pick);['t','size','colour'].forEach(i=>$(i).oninput=draw);onresize=draw;pick();</script>"""
Path(args.out).expanduser().write_text(PAGE.replace("__DATA__", json.dumps(views, separators=(",", ":"))))
print(f"written {args.out}")
