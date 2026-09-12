# Gotchas

Things that cost real time. Written down so they only cost it once.

---

## Downloading the Kubric dataset — huggingface_hub HANGS on NSCC

**Symptom:** `hf download` or `snapshot_download` runs with no error, no
progress bar, and no files on disk. Forever. The process stays alive.

**Cause:** HuggingFace serves file data through **Xet** now. The redirect goes
to `us.aws.cdn.hf.co/xet-bridge-us/...`, and the `hf_xet` client that
`huggingface_hub` uses does not get through on NSCC.

**How to confirm in 5 seconds:**

```bash
curl -sIL -m 30 "https://huggingface.co/datasets/zbww/tapip3d-kubric/resolve/main/000000/000000.npy" | grep -E "^HTTP|^location"
```

If `location:` contains `xet-bridge` while the final line says `HTTP/2 200`,
plain HTTPS works and only the Xet client is broken.

**Fix:** use `scripts/download_kubric.py` — plain `urllib` + a thread pool, no
huggingface_hub at all. Skips files already on disk, so it resumes.

```bash
python scripts/download_kubric.py --out ~/scratch/kubric --clips 500
```

`HF_HUB_DISABLE_XET=1` may also work but has not been verified here.

**Note:** small gated *model* weights (DINOv3, 82 MB) downloaded fine with
`hf download` on NSCC. The problem is specific to the dataset's Xet-backed LFS
files, so "DINOv3 downloaded OK" does not mean the dataset will.

---

## The dataset has 11,000 clips, not 1,000

The HF API caps a directory listing at 1000 entries, so `zbww/tapip3d-kubric`
looks like it has 1000 sequences. It goes up to `010999`. Probe directly
rather than trusting the listing.

---

## Never pass hundreds of `allow_patterns`

`huggingface_hub` matches every file against every pattern. 500 patterns x
~25,000 files = 12.5M Python comparisons, which looks exactly like a hang.

Use one pattern with a character class instead:

```
allow_patterns=["000[0-4]*/*"]     # sequences 000000-000499
```

---

## NSCC: how to submit a job

**Always submit like this:**

```bash
qsub -q normal -P personal-yhuang01 job.pbs
```

Learned the hard way: `-q ai` looks right for ML work but rejects personal
projects, and omitting `-P` is rejected outright.

A project code IS required, and the account only has
`personal-yhuang01` (100,000 SU; GPU billed at 64 SU/hour, so ~1,560
GPU-hours). Personal projects are **rejected by the `ai` queue**, so
everything goes through `-q normal`:

```bash
qsub -q normal -P personal-yhuang01 job.pbs
```

Routing inside `normal` is automatic from the resources requested:

| resources | internal queue | walltime allowed |
| --- | --- | --- |
| ngpus=1, long | g1 | 2h - 24h |
| 1 - 4 gpus, short | gdev | 1s - 2h |
| ncpus=1 only | q1 | 2h - 24h |
| 1 - 128 cpus, short | qdev | 1s - 2h |

Walltime is a **cap, not a reservation** — the job ends when the script ends.
But exceeding it kills the job, so leave headroom.

---

## `hf` on the PATH does not mean `huggingface_hub` is in the venv

The `hf` CLI can come from `~/.local/bin` (user install) while the active venv
has no `huggingface_hub` at all. `hf auth whoami` works; `import
huggingface_hub` raises ModuleNotFoundError.

Check the venv, not the PATH:

```bash
python -c "import huggingface_hub; print(huggingface_hub.__version__)"
```

---

## Backgrounding a heredoc does not work

```bash
nohup python - > log 2>&1 <<'PY' &     # BROKEN -- exits immediately
```

`&` redirects stdin from /dev/null, so `python -` gets nothing. Write the
script to a file first, then `nohup python file.py &`.

---

## Never feed all-zero tokens to the model

`AdaRMSNorm` conditioning is purely **multiplicative** — it predicts a scale
for the norm. `rms_norm(0) = 0`, so a zero token stream stays zero through the
entire network and **no gradient flows at all**. The loss sits frozen at a
constant and nothing errors.

Cost us 400 wasted training steps in `overfit.py --regression`, which fed
`torch.zeros_like(x1)` as tokens. The control now feeds the anchor repeated
across time.

---

## The denoising target must have ~unit variance

Flow matching mixes the target with `x0 ~ N(0, I)`. A target at std 0.025 is
40x quieter than the noise, so the model learns to output `-x0` — and the loss
still falls, so it looks like it is working.

This is why `TRAJ_SCALE` exists in `data/transform.py`, and why
`scripts/check_transform.py` asserts `0.3 < target_std < 3.0`.

---

## Measured timings — fill these in as we learn them

So we can size walltime and plan runs instead of guessing.

| what | hardware | measured |
| --- | --- | --- |
| download 500 clips (24.5k files) | NSCC login | ~45 min, throttled to ~10 files/s |
| preprocess (PNG decode + DINOv3) | A100 shared (`gdev`) | **4.0 s/clip** -> 463 clips = ~31 min |
| preprocess | Mac CPU (MPS) | 4.7 s/clip |
| cache size | - | **7.3 MB/clip** -> 463 clips = 3.4 GB |
| raw Kubric on disk | - | 13 MB/clip -> 500 clips = 6.5 GB |
| train step, 17.9M, batch 16, N=128 | A100 **shared** | **0.8 s/it** -> 20k steps = 4.4 h |
| validation (50 sampling steps, 56 clips) | A100 | **35 s** per VAL |

First real result, 2026-09-12: 17.9M model, 322 train / 56 val clips, 500
steps -> **val ratio 0.309** (rmse 0.267 vs 0.864 baseline). Generalises to
unseen clips, tracking mode.

Useful conversions:
- 460 clips at batch 16 = ~29 steps per epoch, so 3000 steps = ~104 epochs.
- 500 clips x 24 frames at 24 fps = **8 minutes of video total**. It is a small
  dataset; do not over-read a weak result.
