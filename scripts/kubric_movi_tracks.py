# Kubric's challenges/movi/movi_def_worker.py (Apache-2.0, The Kubric Authors),
# unchanged up to its end, with additions marked "[genpoint3d]": a spin for
# each thrown object, 0-2 rolling balls or cylinders, their flags and --tracks,
# and before kb.done() the 3D point tracks written in the layout our
# reader takes (genpoint3d/prepare/KubricSequenceDataset.py). The scene --
# objects, throws, materials, camera, physics -- is MOVi's own; clip length,
# object counts and camera mode are its existing flags.
#
#   apptainer exec --nv kubruntu.sif python3 scripts/kubric_movi_tracks.py \
#       --job-dir OUT/000000 --seed 0 --frame_end 72 --camera linear_movement
#
# Out, besides MOVi's own files: OUT/<name>.npy (coords (N, T, 2) px, occluded,
# traj_3d (N, T, 3) world, intrinsics, matrix_world, depth_range, queries) and
# OUT/frames/{t:03d}.png, {t:03d}_depth.png. The tracks are a numpy port of
# track_points() in challenges/point_tracking/dataset.py that keeps the 3D
# positions, which that code drops.

# Copyright 2026 The Kubric Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""

"""

import itertools
import logging
import os

import bpy
import kubric as kb
from kubric.simulator import PyBullet
from kubric.renderer import Blender
import numpy as np


# --- Some configuration values
# the region in which to place objects [(min), (max)]
STATIC_SPAWN_REGION = [(-7, -7, 0), (7, 7, 10)]
DYNAMIC_SPAWN_REGION = [(-5, -5, 1), (5, 5, 5)]
VELOCITY_RANGE = [(-4., -4., 0.), (4., 4., 0.)]

# --- CLI arguments
parser = kb.ArgumentParser()
parser.add_argument("--objects_split", choices=["train", "test"],
                    default="train")
# Configuration for the objects of the scene
parser.add_argument("--min_num_static_objects", type=int, default=10,
                    help="minimum number of static (distractor) objects")
parser.add_argument("--max_num_static_objects", type=int, default=20,
                    help="maximum number of static (distractor) objects")
parser.add_argument("--min_num_dynamic_objects", type=int, default=1,
                    help="minimum number of dynamic (tossed) objects")
parser.add_argument("--max_num_dynamic_objects", type=int, default=3,
                    help="maximum number of dynamic (tossed) objects")
# Configuration for the floor and background
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)
parser.add_argument("--backgrounds_split", choices=["train", "test"],
                    default="train")

parser.add_argument("--camera", choices=["fixed_random", "linear_movement", "linear_movement_linear_lookat"],
                    default="fixed_random")
parser.add_argument("--max_camera_movement", type=float, default=4.0)
parser.add_argument("--max_motion_blur", type=float, default=0.0)


# Configuration for the source of the assets
parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")
parser.add_argument("--save_state", dest="save_state", action="store_true")
parser.set_defaults(save_state=False, frame_end=24, frame_rate=12,
                    resolution=256)
# [genpoint3d]
parser.add_argument("--tracks", type=int, default=2048, help="point tracks to sample")
parser.add_argument("--max_spin", type=float, default=10.0, help="largest spin at a throw, rad/s")
parser.add_argument("--min_num_rolling_objects", type=int, default=0)
parser.add_argument("--max_num_rolling_objects", type=int, default=2)
FLAGS = parser.parse_args()

# --- Common setups & resources
scene, rng, output_dir, scratch_dir = kb.setup(FLAGS)

motion_blur = rng.uniform(0, FLAGS.max_motion_blur)
if motion_blur > 0.0:
  logging.info(f"Using motion blur strength {motion_blur}")

simulator = PyBullet(scene, scratch_dir)
renderer = Blender(scene, scratch_dir, use_denoising=True, samples_per_pixel=64,
                   motion_blur=motion_blur)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Populate the scene
# background HDRI
train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
if FLAGS.backgrounds_split == "train":
  logging.info("Choosing one of the %d training backgrounds...", len(train_backgrounds))
  hdri_id = rng.choice(train_backgrounds)
else:
  logging.info("Choosing one of the %d held-out backgrounds...", len(test_backgrounds))
  hdri_id = rng.choice(test_backgrounds)
background_hdri = hdri_source.create(asset_id=hdri_id)
#assert isinstance(background_hdri, kb.Texture)
logging.info("Using background %s", hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)

# Dome
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=1.0,
                      restitution=0.0,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)



def get_linear_camera_motion_start_end(
    movement_speed: float,
    inner_radius: float = 8.,
    outer_radius: float = 12.,
    z_offset: float = 0.1,
):
  """Sample a linear path which starts and ends within a half-sphere shell."""
  while True:
    camera_start = np.array(kb.sample_point_in_half_sphere_shell(inner_radius,
                                                                 outer_radius,
                                                                 z_offset))
    direction = rng.rand(3) - 0.5
    movement = direction / np.linalg.norm(direction) * movement_speed
    camera_end = camera_start + movement
    if (inner_radius <= np.linalg.norm(camera_end) <= outer_radius and
        camera_end[2] > z_offset):
      return camera_start, camera_end

def get_linear_lookat_motion_start_end(
    inner_radius: float = 1.0,
    outer_radius: float = 4.0,
):
  """Sample a linear path which goes through the workspace center."""
  while True:
    # Sample a point near the workspace center that the path travels through
    camera_through = np.array(
        kb.sample_point_in_half_sphere_shell(0.0, inner_radius, 0.0)
    )
    while True:
      # Sample one endpoint of the trajectory
      camera_start = np.array(
          kb.sample_point_in_half_sphere_shell(0.0, outer_radius, 0.0)
      )
      if camera_start[-1] < inner_radius:
        break

    # Continue the trajectory beyond the point in the workspace center, so the
    # final path passes through that point.
    continuation = rng.rand(1) * 0.5
    camera_end = camera_through + continuation * (camera_through - camera_start)

    # Second point will probably be closer to the workspace center than the
    # first point.  Get extra augmentation by randomly swapping first and last.
    if rng.rand(1)[0] < 0.5:
      tmp = camera_start
      camera_start = camera_end
      camera_end = tmp
    return camera_start, camera_end


# Camera
logging.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)
if FLAGS.camera == "fixed_random":
  scene.camera.position = kb.sample_point_in_half_sphere_shell(
      inner_radius=7., outer_radius=9., offset=0.1)
  scene.camera.look_at((0, 0, 0))
elif (
    FLAGS.camera == "linear_movement"
    or FLAGS.camera == "linear_movement_linear_lookat"
):

  is_panning = FLAGS.camera == "linear_movement_linear_lookat"
  camera_inner_radius = 6.0 if is_panning else 8.0
  camera_start, camera_end = get_linear_camera_motion_start_end(
      movement_speed=rng.uniform(low=0., high=FLAGS.max_camera_movement)
  )
  if is_panning:
    lookat_start, lookat_end = get_linear_lookat_motion_start_end()

  # linearly interpolate the camera position between these two points
  # while keeping it focused on the center of the scene
  # we start one frame early and end one frame late to ensure that
  # forward and backward flow are still consistent for the last and first frames
  for frame in range(FLAGS.frame_start - 1, FLAGS.frame_end + 2):
    interp = ((frame - FLAGS.frame_start + 1) /
              (FLAGS.frame_end - FLAGS.frame_start + 3))
    scene.camera.position = (interp * np.array(camera_start) +
                             (1 - interp) * np.array(camera_end))
    if is_panning:
      scene.camera.look_at(
          interp * np.array(lookat_start)
          + (1 - interp) * np.array(lookat_end)
      )
    else:
      scene.camera.look_at((0, 0, 0))
    scene.camera.keyframe_insert("position", frame)
    scene.camera.keyframe_insert("quaternion", frame)


# ---- Object placement ----
train_split, test_split = gso.get_test_split(fraction=0.1)
if FLAGS.objects_split == "train":
  logging.info("Choosing one of the %d training objects...", len(train_split))
  active_split = train_split
else:
  logging.info("Choosing one of the %d held-out objects...", len(test_split))
  active_split = test_split



# add STATIC objects
num_static_objects = rng.randint(FLAGS.min_num_static_objects,
                                 FLAGS.max_num_static_objects+1)
logging.info("Randomly placing %d static objects:", num_static_objects)
for i in range(num_static_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=STATIC_SPAWN_REGION,
                           rng=rng)
  obj.friction = 1.0
  obj.restitution = 0.0
  obj.metadata["is_dynamic"] = False
  logging.info("    Added %s at %s", obj.asset_id, obj.position)


logging.info("Running 100 frames of simulation to let static objects settle ...")
_, _ = simulator.run(frame_start=-100, frame_end=0)


# stop any objects that are still moving and reset friction / restitution
for obj in scene.foreground_assets:
  if hasattr(obj, "velocity"):
    obj.velocity = (0., 0., 0.)
    obj.friction = 0.5
    obj.restitution = 0.5


dome.friction = FLAGS.floor_friction
dome.restitution = FLAGS.floor_restitution



# Add DYNAMIC objects
num_dynamic_objects = rng.randint(FLAGS.min_num_dynamic_objects,
                                  FLAGS.max_num_dynamic_objects+1)
logging.info("Randomly placing %d dynamic objects:", num_dynamic_objects)
for i in range(num_dynamic_objects):
  obj = gso.create(asset_id=rng.choice(active_split))
  assert isinstance(obj, kb.FileBasedObject)
  scale = rng.uniform(0.75, 3.0)
  obj.scale = scale / np.max(obj.bounds[1] - obj.bounds[0])
  obj.metadata["scale"] = scale
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=DYNAMIC_SPAWN_REGION,
                           rng=rng)
  obj.velocity = (rng.uniform(*VELOCITY_RANGE) -
                  [obj.position[0], obj.position[1], 0])
  obj.metadata["is_dynamic"] = True
  logging.info("    Added %s at %s", obj.asset_id, obj.position)
  # [genpoint3d] a spin at the throw, about a random axis: thrown things turn,
  # and turning surfaces are what MOVi's spinless throws rarely show.
  axis = rng.normal(size=3)
  obj.angular_velocity = tuple(axis / np.linalg.norm(axis) * rng.uniform(0., FLAGS.max_spin))


# [genpoint3d] Rolling objects: a ball or a cylinder on its side, set on the
# floor and pushed, spinning as it rolls so it does not skid at the start.
num_rolling = rng.randint(FLAGS.min_num_rolling_objects, FLAGS.max_num_rolling_objects + 1)
for i in range(num_rolling):
  shape = rng.choice(["sphere", "cylinder"])
  size = rng.uniform(0.6, 1.4)
  obj = kubasic.create(asset_id=shape, scale=size / 2)
  obj.material = kb.PrincipledBSDFMaterial(color=kb.random_hue_color(rng=rng), roughness=0.6)
  if shape == "cylinder":
    obj.quaternion = kb.Quaternion(axis=[1, 0, 0], degrees=90)      # on its side
  obj.metadata["is_dynamic"] = True
  scene += obj
  kb.move_until_no_overlap(obj, simulator, spawn_region=[(-4, -4, size / 2), (4, 4, size / 2 + 0.05)],
                           rng=rng)
  speed = rng.uniform(1., 4.)
  heading = -np.array(obj.position[:2]) + rng.uniform(-2., 2., size=2)
  heading = heading / (np.linalg.norm(heading) + 1e-8)
  if shape == "cylinder":
    # Laid on its side its axis is along world y, so it can only roll along x.
    heading = np.array([np.sign(rng.uniform(-1., 1.)), 0.])
  obj.velocity = (*(heading * speed), 0.)
  # Rolling without slipping: spin (v / r) about up x heading.
  obj.angular_velocity = (-heading[1] * speed / (size / 2), heading[0] * speed / (size / 2), 0.)
  logging.info("    Rolling %s at %s", shape, obj.position)


if FLAGS.save_state:
  logging.info("Saving the simulator state to '%s' prior to the simulation.",
               output_dir / "scene.bullet")
  simulator.save_state(output_dir / "scene.bullet")

# Run dynamic objects simulation
logging.info("Running the simulation ...")
animation, collisions = simulator.run(frame_start=0,
                                      frame_end=scene.frame_end+1)

# --- Rendering
if FLAGS.save_state:
  logging.info("Saving the renderer state to '%s' ",
               output_dir / "scene.blend")
  renderer.save_state(output_dir / "scene.blend")


logging.info("Rendering the scene ...")
data_stack = renderer.render()

# --- Postprocessing
kb.compute_visibility(data_stack["segmentation"], scene.assets)
visible_foreground_assets = [asset for asset in scene.foreground_assets
                             if np.max(asset.metadata["visibility"]) > 0]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True)

data_stack["segmentation"] = kb.adjust_segmentation_idxs(
    data_stack["segmentation"],
    scene.assets,
    visible_foreground_assets)
scene.metadata["num_instances"] = len(visible_foreground_assets)

# Save to image files
kb.write_image_dict(data_stack, output_dir)
kb.post_processing.compute_bboxes(data_stack["segmentation"],
                                  visible_foreground_assets)

# --- Metadata
logging.info("Collecting and storing metadata for each object.")
kb.write_json(filename=output_dir / "metadata.json", data={
    "flags": vars(FLAGS),
    "metadata": kb.get_scene_metadata(scene),
    "camera": kb.get_camera_info(scene.camera),
    "instances": kb.get_instance_info(scene, visible_foreground_assets),
})
kb.write_json(filename=output_dir / "events.json", data={
    "collisions":  kb.process_collisions(
        collisions, scene, assets_subset=visible_foreground_assets),
})

# [genpoint3d]
# --- Point tracks (numpy port of track_points in challenges/point_tracking)
T = scene.frame_end - scene.frame_start + 1
W, H = scene.resolution
frames = range(scene.frame_start, scene.frame_end + 1)
intrinsics = np.asarray(scene.camera.intrinsics, np.float32)  # normalised, Kubric signs
cams = []
for f in frames:
  with scene.camera.at_frame(f):
    cams.append(np.asarray(scene.camera.matrix_world, np.float32))
matrix_world = np.stack(cams)  # (T, 4, 4) camera-to-world, Blender axes
cam_pos = matrix_world[:, :3, 3]

depth = data_stack["depth"][..., 0].astype(np.float32)  # (T, H, W) distance from the camera
seg = data_stack["segmentation"][..., 0].astype(np.int32)
obj_coords = data_stack["object_coordinates"].astype(np.float32) / 65535. - 0.5  # in each object's box
normals = data_stack["normal"].astype(np.float32) / 65535. * 2. - 1.

boxes = []
quats = []
for a in visible_foreground_assets:
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
counts = np.bincount(grid_seg, minlength=len(visible_foreground_assets) + 1)
per_seg = num_to_sample(counts, FLAGS.tracks)
trust = trusted_normals()

all_uv, all_occ, all_xyz, all_query = [], [], [], []
for i in range(len(visible_foreground_assets) + 1):
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
  kb.write_png(data_stack["rgba"][t, ..., :3], frames_dir / f"{t:03d}.png")
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

kb.done()
