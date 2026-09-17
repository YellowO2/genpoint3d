"""
Builds a Rerun recording comparing:
    /full   -- the complete sequence: RGB, unprojected point cloud, and the
               full 3D trajectory, every frame.
    /masked -- the same sequence, but frames >= T_C have no visual
               conditioning (no image/point cloud), matching what the model
               actually sees during training under the tracking<->forecasting
               masking scheme (masking.py). Points are dimmed for those frames.

This is a data/pipeline demo, not a model output -- no predictions, just
showing what the model will see.

Usage:
    python visualize.py <root_dir> [seq_id] [out.rrd]

Then, on your local machine (not the remote GPU box): download the .rrd and
either drag it into https://rerun.io/viewer, or `pip install rerun-sdk` and
run `rerun out.rrd`.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from genpoint3d.data.kubric import KubricSequenceDataset
from genpoint3d.data.masking import make_training_sample
from genpoint3d.viz import rerun_log as rl

import rerun
import rerun.blueprint as rrb


def build_comparison_recording(root_dir: str, seq_id: str, out_path: str, num_query_points: int = 256):
    ds = KubricSequenceDataset(root_dir, seq_ids=[seq_id], num_query_points=num_query_points)
    sample = ds[0]
    masked = make_training_sample(sample, min_context=max(1, sample.frames.shape[0] // 3))

    rl.setup_visualizer(serve=False)

    rl.log_video("full", sample.frames, sample.intrinsics, sample.extrinsics, sample.depths)
    rl.log_trajectory("full", "gt", sample.intrinsics, sample.extrinsics, sample.traj_3d, sample.visibility)

    rl.log_video("masked", masked.conditioning_frames, masked.intrinsics, masked.extrinsics, masked.depths, active_mask=masked.visual_mask)
    effective_visibs = masked.visibility & masked.visual_mask[:, None]
    rl.log_trajectory("masked", "gt", masked.intrinsics, masked.extrinsics, masked.traj_3d, effective_visibs)

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/full", name=f"Full (T_C={masked.T_C})"),
            rrb.Spatial3DView(origin="/masked", name="Masked (forecast region dimmed)"),
        ),
        collapse_panels=True,
    )
    rerun.send_blueprint(blueprint)

    rl.save_recording(Path(out_path))
    print(f"Wrote {out_path}  (T_C={masked.T_C}, {sample.frames.shape[0]} frames)")


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "local/data/kubric_test"
    seq_id = sys.argv[2] if len(sys.argv) > 2 else "000000"
    out_path = sys.argv[3] if len(sys.argv) > 3 else "local/outputs/comparison.rrd"
    build_comparison_recording(root, seq_id, out_path)
