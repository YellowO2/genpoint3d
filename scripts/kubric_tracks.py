"""3D point tracks from a rendered Kubric scene: a numpy port of track_points()
in Kubric's challenges/point_tracking/dataset.py that keeps the 3D positions,
which that code drops, and each point's object. Shared by our renderer
(scripts/kubric_movi_tracks.py, inside Kubric's container) and the MOVi-F
converter (scripts/movi_to_raw.py, from Google's released shards): numpy only.

Inputs, T frames of H x W: depth (T, H, W) distance from the camera; seg
(T, H, W) 0 the background, k the k-th visible object; obj_coords (T, H, W,
3) each pixel's place in its object's box, in [-0.5, 0.5]; normals (T, H, W,
3) in [-1, 1]; boxes (K, T, 8, 3) each visible object's box corners in the
world; quats (K, T, 4) its rotation, wxyz; intrinsics (3, 3) normalised, with
Kubric's signs; matrix_world (T, 4, 4) camera to world, Blender's axes;
optionally centres (K, T, 3) each visible object's centre of mass in the
world (Kubric's object position: the physics body's).
"""

import itertools

import numpy as np


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


def point_tracks(depth, seg, obj_coords, normals, boxes, quats, intrinsics, matrix_world, n_tracks, rng,
                 centre=0.5, centres=None):
  """dict of coords (N, T, 2) pixels, occluded (N, T), traj_3d (N, T, 3)
  world, queries (N, 3) t, y, x, segment (N,) each point's object, and with
  `centres`, centre_3d (N, T, 3) its object's centre of mass in the world
  (NaN on the background). `centre` is where in its pixel a pixel's values
  were rendered: 0.5 for a render at its own size, 0.25 for MOVi-F's 256 px,
  every other pixel of its 512 px."""
  T, H, W = depth.shape
  cam_pos = matrix_world[:, :3, 3]

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
  counts = np.bincount(grid_seg, minlength=len(boxes) + 1)
  # The objects that move (a box corner more than 1 cm from where it started)
  # get half the points, at up to all their candidates, so the still
  # background does not crowd out the moving surfaces; the rest share the remainder.
  moves = np.array([False] + [bool(np.abs(b - b[:1]).max() > 0.01) for b in boxes])
  per_seg = np.zeros(len(counts), np.int64)
  if moves.any():
    per_seg[moves] = num_to_sample(counts[moves], n_tracks // 2, max_frac=1.0)
  per_seg[~moves] = num_to_sample(counts[~moves], n_tracks - per_seg.sum())
  trust = trusted_normals()

  all_uv, all_occ, all_xyz, all_query, all_seg, all_centre = [], [], [], [], [], []
  for i in range(len(boxes) + 1):
    candidates = grid[grid_seg == i]
    if per_seg[i] == 0 or len(candidates) == 0:
      continue
    q = candidates[rng.choice(len(candidates), size=min(per_seg[i], len(candidates)), replace=False)]
    qt, qy, qx = q[:, 0], q[:, 1], q[:, 2]
    if i == 0:
      # Background does not move: unproject from the query frame's depth and
      # keep that position for the whole clip.
      raster = np.stack([(qx + centre) / W, (qy + centre) / H, np.ones(len(q))], -1)
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
      rot = quat_to_rot(np.asarray(quats[i - 1], np.float64))
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
    all_seg.append(np.full(len(q), i, np.int32))
    if centres is not None:
      at = np.full((T, 3), np.nan) if i == 0 else np.asarray(centres[i - 1], np.float64)
      all_centre.append(np.repeat(at[None], len(q), 0))

  out = {"coords": np.concatenate(all_uv).astype(np.float32), "occluded": np.concatenate(all_occ),
         "traj_3d": np.concatenate(all_xyz).astype(np.float32),
         "queries": np.concatenate(all_query).astype(np.int32), "segment": np.concatenate(all_seg)}
  if centres is not None:
    out["centre_3d"] = np.concatenate(all_centre).astype(np.float32)
  return out
