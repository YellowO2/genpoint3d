"""
Masking helper for the tracking <-> forecasting unification.

Core idea (from Gen-points): pick a random cutoff frame T_C per training
sample. Frames before T_C get real image conditioning ("tracking" regime).
Frames from T_C onward get a null/blank conditioning signal ("forecasting"
regime). Re-randomized every time you draw a sample, not precomputed.
"""

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class MaskedSample:
    seq_id: str
    T_C: int                     # cutoff frame index; frames >= T_C are "unobserved"
    frames: np.ndarray           # (T, H, W, 3) original, unmodified
    conditioning_frames: np.ndarray  # (T, H, W, 3) with frames >= T_C zeroed out
    visual_mask: np.ndarray      # (T,) bool, True = has real visual conditioning
    depths: np.ndarray
    traj_3d: np.ndarray
    coords_2d: np.ndarray
    visibility: np.ndarray
    intrinsics: np.ndarray
    extrinsics: np.ndarray


def sample_cutoff(T: int, min_context: int = 1, max_context: int | None = None) -> int:
    """
    Pick T_C uniformly at random.
    T_C = T      -> pure tracking (all frames observed)
    T_C = min_context -> mostly forecasting (only the first few frames observed)
    """
    if max_context is None:
        max_context = T
    return random.randint(min_context, max_context)


def apply_conditioning_mask(sample, T_C: int) -> MaskedSample:
    """
    sample: a KubricSample from dataset.py
    Zeroes out (rather than deletes) frames >= T_C, and returns a boolean
    mask so the model/feature-extractor knows which frames are "real" vs
    "blank placeholder" -- the actual null-embedding substitution happens
    inside the model's conditioning path, this just prepares the inputs.
    """
    T = sample.frames.shape[0]
    conditioning_frames = sample.frames.copy()
    conditioning_frames[T_C:] = 0

    visual_mask = np.zeros(T, dtype=bool)
    visual_mask[:T_C] = True

    return MaskedSample(
        seq_id=sample.seq_id,
        T_C=T_C,
        frames=sample.frames,
        conditioning_frames=conditioning_frames,
        visual_mask=visual_mask,
        depths=sample.depths,
        traj_3d=sample.traj_3d,
        coords_2d=sample.coords_2d,
        visibility=sample.visibility,
        intrinsics=sample.intrinsics,
        extrinsics=sample.extrinsics,
    )


def make_training_sample(sample, min_context: int = 1) -> MaskedSample:
    """Convenience wrapper: sample a fresh T_C and apply it, in one call."""
    T = sample.frames.shape[0]
    T_C = sample_cutoff(T, min_context=min_context)
    return apply_conditioning_mask(sample, T_C)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import KubricSequenceDataset

    root = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/scratch/data/kubric_test")
    ds = KubricSequenceDataset(root, num_query_points=256)
    sample = ds[0]

    for _ in range(5):
        masked = make_training_sample(sample)
        print(f"seq={masked.seq_id}  T_C={masked.T_C}  "
              f"observed_frames={masked.visual_mask.sum()}/{len(masked.visual_mask)}")
