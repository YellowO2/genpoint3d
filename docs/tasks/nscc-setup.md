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

## 4. Cache the transform

Decoding 48 PNGs per clip costs ~1.3 s; caching makes training ~30x cheaper.
One-off, CPU only, safe on the login node for 500 clips (~12 min).

```bash
python scripts/preprocess.py \
  --root ~/scratch/kubric \
  --out  ~/scratch/cache/kubric500.pt
```

Expect roughly 0.1 MB/clip, so ~50 MB total.

---

## 5. Submit a job (PBS, not SLURM)

Find your project code and the GPU queue first — these are account-specific:

```bash
qstat -Q                    # queue names
glsproject 2>/dev/null || echo "check your NSCC welcome email for the -P code"
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

```bash
qsub job.pbs
qstat -u $USER              # watch it
```

---

## Notes

- `$SCRATCH` is unset on this system; use `~/scratch` explicitly.
- `/scratch` is usually purged periodically — keep checkpoints you care about
  somewhere durable.
- The command in section 5 trains **without images**, which is a weak
  experiment (see `docs/map.html`). It is here so the pipeline is proven
  end-to-end on the cluster. The real run comes once step 3 lands.
