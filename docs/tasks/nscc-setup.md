# NSCC runbook (ASPIRE 2A, PBS Pro)

Your half of the parallel split: get data and weights onto NSCC while the
encoder is being built. Everything here runs on the **login node** except the
final job.

---

## 0. Before anything — unlock DINOv3 (2 min, do this first)

DINOv3 is gated. Without this, step 3 cannot run.

1. Open https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m
2. Log in, accept the terms ("agree to share your contact information")
3. Make a token: https://huggingface.co/settings/tokens (role: `read`)

Then on NSCC:

```bash
cd ~/scratch/genpoint3d
module load python/3.11.7-gcc11
source ~/scratch/venvs/fyp/bin/activate
pip install -U huggingface_hub
hf auth login          # paste the token
```

Verify it worked:

```bash
hf download facebook/dinov3-vits16-pretrain-lvd1689m --quiet && echo "DINOv3 OK"
```

If that 403s, the terms were not accepted on the right account.

---

## 1. Clone the repo

```bash
cd ~/scratch
git clone https://github.com/YellowO2/genpoint3d.git
cd genpoint3d
```

## 2. Environment

Login nodes have no GPU, so `nvidia-smi` is missing there — that is normal and
not a problem. Install the CUDA build anyway; it is the compute nodes that
matter.

```bash
module load python/3.11.7-gcc11
module load cuda/12.2.2
source ~/scratch/venvs/fyp/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy opencv-python-headless huggingface_hub
```

## 3. Download Kubric

Check the size before committing to it:

```bash
hf download zbww/tapip3d-kubric --repo-type dataset --dry-run 2>/dev/null | tail -3
```

Then fetch ~500 sequences to **scratch**, never `$HOME`:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
seqs = [f"{i:06d}" for i in range(500)]
snapshot_download(
    repo_id="zbww/tapip3d-kubric", repo_type="dataset",
    allow_patterns=[f"{s}/*" for s in seqs],
    local_dir="/home/users/ntu/yhuang01/scratch/kubric",
    max_workers=8,
)
PY
```

Interrupted downloads resume — just rerun it.

## 4. Cache the transform and the DINOv3 features

Decoding 48 PNGs per clip costs ~1.3 s and re-encoding DINOv3 every epoch
would dominate training, so both are done once up front.

This one wants a GPU (it runs DINOv3 over every frame), so submit it rather
than running it on the login node:

```bash
python scripts/preprocess.py \
  --root ~/scratch/kubric \
  --out  ~/scratch/cache/kubric500.pt \
  --features --feat-dim 256 --image-size 384 --points 256
```

Expect ~7 MB/clip, so ~3.6 GB for 500 clips. On CPU it is ~3 s/clip (25 min);
on a GPU, a few minutes.

Sanity check before trusting it:

```bash
python -c "
import torch; b=torch.load('$HOME/scratch/cache/kubric500.pt', weights_only=False)
c=b['clips'][0]
print(len(b['clips']),'clips | feat_dim',b['feat_dim'])
print({k:(tuple(v.shape) if hasattr(v,'shape') else v) for k,v in c.items() if k!='seq_id'})
"
```

---

## 5. Submit a job (PBS, not SLURM)

Find your project code and the GPU queue first — these are account-specific:

```bash
qstat -Q                    # queue names
myprojects                  # project codes and SU balance
```

Then `job.pbs` (fill in `-P` and `-q`):

```bash
#!/bin/bash
#PBS -N genpoint3d
#PBS -l select=1:ncpus=16:ngpus=1:mem=64gb
#PBS -l walltime=04:00:00
#PBS -q normal
#PBS -P <YOUR_PROJECT_CODE>
#PBS -j oe
#PBS -o /home/users/ntu/yhuang01/scratch/genpoint3d/outputs/

cd ~/scratch/genpoint3d
module load python/3.11.7-gcc11
module load cuda/12.2.2
source ~/scratch/venvs/fyp/bin/activate

nvidia-smi                  # now this works -- we are on a GPU node

python scripts/train.py \
  --cache ~/scratch/cache/kubric500.pt \
  --steps 20000 --batch 16 --points 128 \
  --dim 256 --depth 6 --heads 4 \
  --out outputs/run500
```

Watch the `VAL ... APD` line. That is the whole experiment: the TAP-Vid-3D
`average_pts_within_thresh`, scored in metres, which is what other papers
report. `baseline` beside it is the same measure applied to the mean
trajectory -- if APD is not clearly above it, nothing generalised.

```bash
qsub -q normal -P personal-yhuang01 scripts/train.pbs
qstat -u $USER              # watch it
tail -f ~/scratch/train.live.log
python scripts/plot_log.py outputs/run3493/log.json
```

---

## Notes

- `$SCRATCH` is unset on this system; use `~/scratch` explicitly.
- `/scratch` is usually purged periodically -- keep checkpoints you care about
  somewhere durable.
- Section 5 trains pure **tracking**: every frame keeps its image, so the
  tracking/forecasting mask is off. That is deliberate -- forecasting is much
  harder, and training both at once splits the signal. Turn masking on only
  once tracking is learning.
- DINOv3 is cached, not run during training, so the GPU only ever sees the
  17.9M-parameter model. The 21M encoder is frozen and already spent.
