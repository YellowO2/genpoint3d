"""Render one simple Kubric clip with 3D point tracks, in the layout our reader takes.

Runs inside the Kubric container (it needs bpy and kubric), one scene per call:

    apptainer exec --nv kubruntu.sif python3 scripts/kubric_simple_worker.py \
        --job-dir ~/scratch/kubric_simple/000000 --seed 0 --resolution 256

The scene is simpler and longer than MOVi: 1-3 objects (scanned GSO objects,
sometimes a plain sphere, cube or cylinder) on a floor under an HDRI dome, a
fixed camera, 48-96 frames at 12 fps. Every object is fully in view in the
first frame. Some fall from a height at the start; the others rest on the
floor and are tossed up at a later frame, so there is something happening
through the whole clip. Tosses and drops get sideways velocity and spin, and
some are aimed at another object so they collide.

Output, the zbww/tapip3d-kubric layout read by
genpoint3d/prepare/KubricSequenceDataset.py:

    <job-dir>/<name>.npy               coords, occluded, traj_3d, intrinsics,
                                       matrix_world, depth_range
    <job-dir>/frames/{t:03d}.png       RGB frame t
    <job-dir>/frames/{t:03d}_depth.png distance from the camera, 16 bit over depth_range
    <job-dir>/metadata.json            flags, seed, assets and the launch events

The point tracks are a numpy port of track_points() in Kubric's
challenges/point_tracking/dataset.py (the TAP-Vid Kubric code): query points on
a strided grid over all frames, objects followed through their object
coordinates and 3D boxes, background points unprojected from depth, and a point
counted occluded when the depth or segment at its projection disagrees, it
faces away from the camera, or it leaves the image. Unlike that code it keeps
the 3D world positions, which is what traj_3d is.
"""

import itertools
import json
import logging
import os

import bpy
import kubric as kb
import numpy as np
from kubric.renderer import Blender
from kubric.simulator import PyBullet

FPS = 12
PRIMITIVES = ["sphere", "cube", "cylinder"]

parser = kb.ArgumentParser()
parser.add_argument("--min_frames", type=int, default=48)
parser.add_argument("--max_frames", type=int, default=96)
parser.add_argument("--min_objects", type=int, default=1)
parser.add_argument("--max_objects", type=int, default=3)
parser.add_argument("--primitive_prob", type=float, default=0.2,
                    help="chance an object is a plain sphere/cube/cylinder rather than GSO")
parser.add_argument("--tracks", type=int, default=2048, help="point tracks per clip")
parser.add_argument("--samples_per_pixel", type=int, default=64)
parser.add_argument("--use_gpu", action="store_true", help="render with Cycles on CUDA")
parser.add_argument("--split", choices=["train", "test"], default="train",
                    help="which 90/10 split of GSO objects and HDRIs to draw from")
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.set_defaults(frame_rate=FPS, step_rate=240, resolution="256")
FLAGS = parser.parse_args()

# The clip length is drawn from the seed before kb.setup, which builds the scene
# from FLAGS.frame_end.
if FLAGS.seed is None:
  FLAGS.seed = int(np.random.randint(0, 2**31 - 1))
FLAGS.frame_start = 1
FLAGS.frame_end = int(np.random.RandomState(FLAGS.seed).randint(FLAGS.min_frames,
                                                                FLAGS.max_frames + 1))

scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)
T = scene.frame_end - scene.frame_start + 1
H, W = scene.resolution[1], scene.resolution[0]

simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True,
                   samples_per_pixel=FLAGS.samples_per_pixel)
if FLAGS.use_gpu:
  prefs = bpy.context.preferences.addons["cycles"].preferences
  prefs.compute_device_type = "CUDA"
  prefs.get_devices()
  for d in prefs.devices:
    d.use = d.type == "CUDA"
  bpy.context.scene.cycles.device = "GPU"
  logging.info("Cycles devices: %s", [(d.name, d.type, d.use) for d in prefs.devices])

kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Background: an HDRI lights the scene and is projected onto a dome whose
# flat bottom is the floor, as in MOVi.
train_hdri, test_hdri = hdri_source.get_test_split(fraction=0.1)
hdri_id = rng.choice(train_hdri if FLAGS.split == "train" else test_hdri)
background_hdri = hdri_source.create(asset_id=hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)

floor_friction = rng.uniform(0.3, 0.6)
floor_restitution = rng.uniform(0.5, 0.8)
dome = kubasic.create(asset_id="dome", name="dome", friction=floor_friction,
                      restitution=floor_restitution, static=True, background=True)
scene += dome
texture_node = dome.linked_objects[renderer].data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)


# --- Objects
train_gso, test_gso = gso.get_test_split(fraction=0.1)
active_gso = train_gso if FLAGS.split == "train" else test_gso


def new_object():
  """A GSO object or, with primitive_prob, a coloured KuBasic primitive, about 1-2 units across."""
  size = rng.uniform(1.0, 2.0)
  if rng.uniform() < FLAGS.primitive_prob:
    shape = rng.choice(PRIMITIVES)
    obj = kubasic.create(asset_id=shape, scale=size / 2)
    color = kb.random_hue_color(rng=rng)
    if rng.uniform() < 0.5:
      obj.material = kb.PrincipledBSDFMaterial(color=color, metallic=1.0, roughness=0.2, ior=2.5)
    else:
      obj.material = kb.PrincipledBSDFMaterial(color=color, metallic=0., ior=1.25, roughness=0.7,
                                               specular=0.33)
    obj.metadata["asset_id"] = shape
  else:
    obj = gso.create(asset_id=rng.choice(active_gso))
    obj.scale = size / np.max(obj.bounds[1] - obj.bounds[0])
    obj.metadata["asset_id"] = obj.asset_id
  obj.metadata["size"] = size
  obj.friction = rng.uniform(0.3, 0.7)
  obj.restitution = rng.uniform(0.4, 0.8)
  return obj


def random_spin():
  """Angular velocity in rad/s: none, or up to ~1.5 turns a second about a random axis."""
  if rng.uniform() < 0.3:
    return (0., 0., 0.)
  axis = rng.normal(size=3)
  return tuple(axis / np.linalg.norm(axis) * rng.uniform(2., 9.))


def sideways(obj, others, speed):
  """A horizontal velocity: toward another object half the time (to make a
  collision likely), else in a random direction."""
  if others and rng.uniform() < 0.5:
    target = np.array(others[rng.randint(len(others))].position)
    d = (target - np.array(obj.position))[:2]
  else:
    d = rng.normal(size=2)
  d = d / (np.linalg.norm(d) + 1e-8) * speed
  return np.array([d[0], d[1], 0.])


num_objects = rng.randint(FLAGS.min_objects, FLAGS.max_objects + 1)
# The first object always falls at the start; each other one falls at the start
# too, or rests on the floor and is tossed up later.
falls = [True] + [bool(rng.uniform() < 0.4) for _ in range(num_objects - 1)]

resting = []
for is_falling in falls:
  if not is_falling:
    obj = new_object()
    scene += obj
    kb.move_until_no_overlap(obj, simulator, spawn_region=[(-2.5, -2.5, 0.2), (2.5, 2.5, 1.5)],
                             rng=rng)
    resting.append(obj)
if resting:
  simulator.run(frame_start=-60, frame_end=0)  # let them settle on the floor
  for obj in resting:
    obj.velocity = (0., 0., 0.)
    obj.angular_velocity = (0., 0., 0.)

falling = []
for is_falling in falls:
  if is_falling:
    obj = new_object()
    scene += obj
    kb.move_until_no_overlap(obj, simulator, spawn_region=[(-2., -2., 1.5), (2., 2., 3.5)],
                             rng=rng)
    falling.append(obj)
objects = falling + resting

events = []
for obj in falling:
  others = [o for o in objects if o is not obj]
  obj.velocity = tuple(sideways(obj, others, rng.uniform(0., 2.)))
  obj.angular_velocity = random_spin()
  events.append({"object": obj.metadata["asset_id"], "frame": 0, "kind": "drop",
                 "velocity": list(obj.velocity), "spin": list(obj.angular_velocity)})

# Tosses for the resting objects, spread over the clip, plus sometimes a second
# toss of any object late in the clip.
toss_frames = sorted(int(f) for f in rng.randint(int(0.15 * T), int(0.7 * T) + 1, size=len(resting)))
tosses = [(f, obj) for f, obj in zip(toss_frames, resting)]
if rng.uniform() < 0.5 and T >= 40:
  tosses.append((int(rng.randint(int(0.6 * T), T - 12)), objects[rng.randint(len(objects))]))
tosses.sort(key=lambda x: x[0])


# --- Camera: fixed, looking at the middle of the floor from 6-9 units away,
# resampled until every object is fully in view in the first frame.
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)


def all_in_view(margin=0.05):
  for obj in objects:
    for corner in obj.bbox_3d:
      u, v, front = scene.camera.project_point(corner)
      if front <= 0 or not (margin < u < 1 - margin and margin < v < 1 - margin):
        return False
  return True


for attempt in range(200):
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=6., outer_radius=9., offset=1.0)
  scene.camera.look_at((rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(0.3, 1.2)))
  if all_in_view():
    break
else:
  raise RuntimeError("no camera position sees every object in the first frame")
logging.info("Camera at %s after %d tries", scene.camera.position, attempt + 1)


# --- Simulate in pieces, one per toss. After each run Kubric resets PyBullet to
# the state at its last frame, so the toss is applied there and the next run
# starts from that same frame.
logging.info("Simulating %d frames with %d objects, tosses at %s", T, len(objects),
             [f for f, _ in tosses])
start = 0
collisions = []
for frame, obj in tosses:
  if frame > start:
    _, c = simulator.run(frame_start=start, frame_end=frame)
    collisions += c
    start = frame
  others = [o for o in objects if o is not obj]
  obj.velocity = tuple(sideways(obj, others, rng.uniform(0.5, 2.5)) +
                       np.array([0., 0., rng.uniform(3., 6.)]))
  obj.angular_velocity = random_spin()
  events.append({"object": obj.metadata["asset_id"], "frame": frame, "kind": "toss",
                 "velocity": list(obj.velocity), "spin": list(obj.angular_velocity)})
_, c = simulator.run(frame_start=start, frame_end=scene.frame_end)
collisions += c


# --- Render
logging.info("Rendering %d frames at %dx%d ...", T, W, H)
data = renderer.render()

kb.compute_visibility(data["segmentation"], scene.assets)
visible = [a for a in scene.foreground_assets if np.max(a.metadata["visibility"]) > 0]
data["segmentation"] = kb.adjust_segmentation_idxs(data["segmentation"], scene.assets, visible)


# --- Point tracks (numpy port of track_points in challenges/point_tracking)
frames = range(scene.frame_start, scene.frame_end + 1)
intrinsics = np.asarray(scene.camera.intrinsics, np.float32)  # normalised, Kubric signs
cams = []
for f in frames:
  with scene.camera.at_frame(f):
    cams.append(np.asarray(scene.camera.matrix_world, np.float32))
matrix_world = np.stack(cams)  # (T, 4, 4) camera-to-world, Blender axes
cam_pos = matrix_world[:, :3, 3]

depth = data["depth"][..., 0].astype(np.float32)  # (T, H, W) distance from the camera
seg = data["segmentation"][..., 0].astype(np.int32)
obj_coords = data["object_coordinates"].astype(np.float32) / 65535. - 0.5  # in each object's box
normals = data["normal"].astype(np.float32) / 65535. * 2. - 1.

boxes = []
quats = []
for a in visible:
  per_frame = []
  for f in frames:
    with a.at_frame(f):
      per_frame.append(a.bbox_3d)
  boxes.append(np.stack(per_frame))
  quats.append(a.get_values_over_time("quaternion"))


def quat_to_rot(q):
  """(T, 4) wxyz quaternions -> (T, 3, 3) rotation matrices."""
  w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
  return np.stack([
      np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
      np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
      np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
  ], -2)


def num_to_sample(counts, total, max_frac=0.1):
  """Points per segment: an even share of what is still needed, capped at
  max_frac of the segment's grid points (get_num_to_sample)."""
  order = np.argsort(counts)
  out = np.zeros(len(counts), np.int64)
  remaining = total
  for k, s in enumerate(order):
    want = int(round(remaining / (len(order) - k)))
    out[s] = min(want, int(round(counts[s] * max_frac)))
    remaining -= out[s]
  return out


def project(world):
  """(N, T, 3) world points -> (N, T, 2) pixel coords, (N, T) in-front flag, (N, T) distance."""
  homo = np.concatenate([world, np.ones_like(world[..., :1])], -1)
  cam = np.einsum("tij,ntj->nti", np.linalg.inv(matrix_world), homo)[..., :3]
  proj = np.einsum("ij,ntj->nti", intrinsics, cam)
  uv = proj[..., :2] / proj[..., 2:3] * np.array([W, H], np.float32)
  dist = np.linalg.norm(world - cam_pos[None], axis=-1)
  return uv, proj[..., 2] > 0, dist


def occluded_by_depth_and_segment(uv, dist, seg_id):
  """A projected point is occluded if all 4 neighbouring pixels are nearer
  than 0.99 x its own distance, or if none of the 4 has its segment."""
  x = uv[..., 0] - 0.5
  y = uv[..., 1] - 0.5
  x0 = np.clip(np.floor(x).astype(np.int64), 0, W - 1)
  y0 = np.clip(np.floor(y).astype(np.int64), 0, H - 1)
  x1 = np.clip(x0 + 1, 0, W - 1)
  y1 = np.clip(y0 + 1, 0, H - 1)
  t = np.arange(T)[None, :]
  d = np.maximum.reduce([depth[t, y0, x0], depth[t, y1, x0], depth[t, y0, x1], depth[t, y1, x1]])
  other_seg = np.logical_and.reduce([seg[t, yy, xx] != seg_id
                                     for yy, xx in ((y0, x0), (y1, x0), (y0, x1), (y1, x1))])
  return (d < dist * 0.99) | other_seg


def trusted_normals():
  """(T, H, W) True where the normal map is smooth. A rough map usually comes
  from a texture's normal map, not the mesh, and the facing test fails there."""
  pad = np.pad(normals, ((0, 0), (1, 1), (1, 1), (0, 0)))
  rough = 0
  for i in (0, 2):
    for j in (0, 2):
      diff = np.sum((pad[:, i:i + H, j:j + W] - normals) ** 2, -1)
      rough = rough + (diff > 0.05 ** 2)
  return rough <= 2


stride = 4
offset = rng.randint(0, stride, size=3)
tt, yy, xx = np.meshgrid(np.arange(offset[0], T, stride), np.arange(offset[1], H, stride),
                         np.arange(offset[2], W, stride), indexing="ij")
grid = np.stack([tt, yy, xx], -1).reshape(-1, 3)  # (t, y, x) query candidates
grid_seg = seg[grid[:, 0], grid[:, 1], grid[:, 2]]
counts = np.bincount(grid_seg, minlength=len(visible) + 1)
per_seg = num_to_sample(counts, FLAGS.tracks)
trust = trusted_normals()

all_uv, all_occ, all_xyz, all_query = [], [], [], []
for i in range(len(visible) + 1):
  candidates = grid[grid_seg == i]
  if per_seg[i] == 0 or len(candidates) == 0:
    continue
  q = candidates[rng.choice(len(candidates), size=min(per_seg[i], len(candidates)), replace=False)]
  qt, qy, qx = q[:, 0], q[:, 1], q[:, 2]
  if i == 0:
    # Background does not move: unproject from the query frame's depth and
    # keep that position for the whole clip.
    raster = np.stack([(qx + 0.5) / W, (qy + 0.5) / H, np.ones(len(q))], -1)
    rays = raster @ np.linalg.inv(intrinsics).T
    rays = rays / np.linalg.norm(rays, axis=-1, keepdims=True) * depth[qt, qy, qx][:, None]
    world = np.einsum("nij,nj->ni", matrix_world[qt],
                      np.concatenate([rays, np.ones((len(q), 1))], -1))[:, :3]
    world = np.repeat(world[:, None], T, axis=1)
    faces_away = np.zeros((len(q), T), bool)
  else:
    # Object point: its coordinate inside the object's box, carried through
    # the box's pose in every frame.
    corners = np.array(list(itertools.product([-.5, .5], repeat=3)), np.float32)
    corners = np.concatenate([corners, np.ones((8, 1), np.float32)], -1)
    box = boxes[i - 1]
    local_to_world = np.stack([np.linalg.lstsq(corners, np.concatenate([b, np.ones((8, 1))], -1),
                                               rcond=None)[0] for b in box])  # (T, 4, 4)
    local = np.concatenate([obj_coords[qt, qy, qx], np.ones((len(q), 1))], -1)
    world = np.einsum("nj,tjk->ntk", local, local_to_world)
    world = world[..., :3] / world[..., 3:]
    # Facing test: carry the query frame's normal with the object's rotation.
    rot = quat_to_rot(quats[i - 1].astype(np.float64))
    n_obj = np.einsum("nji,nj->ni", rot[qt], normals[qt, qy, qx])  # R^T n
    n_world = np.einsum("tij,nj->nti", rot, n_obj)
    away = np.sum(n_world * (world - cam_pos[None]), -1) > 0
    away &= ~away[np.arange(len(q)), qt][:, None]  # the query faces away: a mesh fault, ignore
    faces_away = away & trust[qt, qy, qx][:, None]
  uv, in_front, dist = project(world)
  occ = ~in_front | occluded_by_depth_and_segment(uv, dist, i) | faces_away
  occ |= (uv[..., 0] < 0) | (uv[..., 0] > W) | (uv[..., 1] < 0) | (uv[..., 1] > H)
  all_uv.append(uv)
  all_occ.append(occ)
  all_xyz.append(world)
  all_query.append(q)

coords = np.concatenate(all_uv).astype(np.float32)
occluded = np.concatenate(all_occ)
traj_3d = np.concatenate(all_xyz).astype(np.float32)
queries = np.concatenate(all_query).astype(np.int32)

# Sanity: each point projects back onto its own query pixel at its query frame.
n = np.arange(len(queries))
err = np.linalg.norm(coords[n, queries[:, 0]] - (queries[:, [2, 1]] + 0.5), axis=-1)
logging.info("Tracks: %d points, query reprojection error median %.3f px, max %.3f px; "
             "%d visible in frame 0", len(queries), np.median(err), np.max(err),
             int(np.sum(~occluded[:, 0])))


# --- Write
name = os.path.basename(os.path.normpath(str(output_dir)))
frames_dir = output_dir / "frames"
frames_dir.mkdir(parents=True, exist_ok=True)
depth_range = np.array([depth.min(), depth.max()], np.float32)
depth_u16 = np.round((depth - depth_range[0]) / (depth_range[1] - depth_range[0]) * 65535)
depth_u16 = depth_u16.astype(np.uint16)
for t in range(T):
  kb.write_png(data["rgba"][t, ..., :3], frames_dir / f"{t:03d}.png")
  kb.write_png(depth_u16[t, ..., None], frames_dir / f"{t:03d}_depth.png")

np.save(output_dir / f"{name}.npy", {
    "coords": coords,                 # (N, T, 2) pixels
    "occluded": occluded,             # (N, T)
    "traj_3d": traj_3d,               # (N, T, 3) world
    "intrinsics": np.repeat(intrinsics[None], T, 0),  # (T, 3, 3) normalised
    "matrix_world": matrix_world,     # (T, 4, 4) camera-to-world
    "depth_range": depth_range,
    "queries": queries,               # (N, 3) t, y, x of each point's query
})

with open(output_dir / "metadata.json", "w") as fp:
  json.dump({
      "seed": FLAGS.seed, "frames": T, "fps": FPS, "resolution": [H, W],
      "background": hdri_id, "floor": {"friction": floor_friction, "restitution": floor_restitution},
      "objects": [{"asset_id": o.metadata["asset_id"], "size": o.metadata["size"],
                   "friction": o.friction, "restitution": o.restitution,
                   "visible": o in visible} for o in objects],
      "events": events,
      "collisions": len(collisions),
  }, fp, indent=1, default=float)

kb.done()
