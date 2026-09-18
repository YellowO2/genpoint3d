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

Listing `zbww/tapip3d-kubric` via the HF tree API returns exactly 1000 folders
ending at `000999`, which looks like the whole dataset. It is not -- sequences
exist up to `010999`. Verified by requesting `001000`, `004999`, `009999`
directly and binary-searching the upper bound.

Whatever the cause (pagination, most likely), do not trust a listing that
comes back at a suspiciously round number. Probe past it.

---

## Never pass hundreds of `allow_patterns`

`huggingface_hub` matches every file against every pattern. 500 patterns x
~25,000 files = 12.5M Python comparisons, which looks exactly like a hang.

Use one pattern with a character class instead:

```
allow_patterns=["000[0-4]*/*"]     # sequences 000000-000499
```

---

## NSCC: everything computational goes through the scheduler

From the login MOTD:

> All computational jobs must run via scheduler, including pre- and
> post-processing jobs.

That covers more than training. Cache patching, preprocessing, format
conversion -- anything that runs for minutes and uses real CPU belongs in a
job, not on the login node. Login nodes are shared, and a long multi-threaded
run there gets killed.

The exceptions are genuinely interactive: `git pull`, a `--limit 10` smoke
test, `ls`, editing files. Downloads are the awkward case -- compute nodes may
have no internet, so `download_kubric.py` runs on the login node out of
necessity.

A CPU-only job (no `ngpus=`) routes to the CPU queue and does not wait behind
GPU work.

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

## Checking SU usage: `myprojects`, not `glsproject`

`myprojects` is the command that reports project codes and Service Unit usage.

```bash
myprojects                                   # balance
myprojects -p personal-yhuang01 -l           # per-user breakdown
myprojects -p personal-yhuang01 -l -s 2026-09-01 -e 2026-09-30
```

Budget: 100,000 SU on `personal-yhuang01`, GPU billed at **64 SU/hour**, so
~1,560 GPU-hours total. A 4-hour preprocessing job is ~256 SU -- about 0.25%.
Cost is not the constraint; walltime and queue waits are.

---

## Disk and inode quota on NSCC — not a constraint, stop checking

`df -h ~/scratch` shows the **shared** filesystem (9.5 PB), which says nothing
about your own limit. Lustre quotas are per-user:

```bash
lfs quota -h -u $USER /scratch
```

Measured 2026-09-16: **100 TB block quota, 200M inode quota** (default
settings, not something we requested). Usage at the time was 19 GB / 61k files.

At 13 MB and ~49 files per raw clip, the entire 11,000-clip dataset would be
~143 GB and ~540k files — **0.14% of the block quota and 0.27% of the inodes**.

So: disk is never the reason to download fewer clips. Preprocessing time and
training time are.

---

## Preprocessing was 90% idle: 48 serial file reads per clip

**Symptom:** preprocessing ran at **46 s/clip** on NSCC, against 4.0 s/clip
measured earlier on the same hardware with the same code.

**Diagnosis** -- `qstat -f <jobid> | grep resources_used` settles it in one line:

```
cput       = 00:27:44     <- actual CPU work
walltime   = 04:49:21     <- time elapsed
cpupercent = 112          <- ~1.1 cores busy of the 16 requested
```

28 minutes of work in 4h49m. The job was **waiting**, not computing.

**Cause:** `/scratch` is Lustre, a network filesystem. Each clip needs 48 small
files and `_load_frames_and_depths` read them in a plain `for` loop, so 48
round trips happened one after another with the GPU idle throughout.

The earlier 4.0 s/clip was not a fair baseline -- those clips had just been
downloaded, so they were still in page cache.

**Fix:** a `ThreadPoolExecutor` over the frame loop (`data/kubric.py`).
Threads, not processes: the time is spent waiting on I/O, and both the file
read and `cv2.imdecode` release the GIL. Tune with `KUBRIC_READ_WORKERS`.

Verified byte-identical output; 4x faster even on a local SSD, where there is
no network latency to hide.

**General lesson:** when a job is far slower than its own past self, check
`cput` against `walltime` BEFORE optimising anything. Idle time and slow
compute need opposite fixes.

---

## Measured timings — fill these in as we learn them

So we can size walltime and plan runs instead of guessing.

| what | hardware | measured |
| --- | --- | --- |
| download 500 clips (24.5k files) | NSCC login | ~45 min, throttled to ~10 files/s |
| download 3500 clips (171k files) | NSCC login | **8 h** at 6 files/s (2026-09-16 overnight) |
| preprocess, serial reads | A100 shared | 46 s/clip (job 90% idle -- see above) |
| preprocess, 16 threaded reads | A100 shared (`g1`) | **3.7 s/clip** -> 2712 clips = ~2.7 h |
| preprocess | Mac CPU (MPS) | 4.7 s/clip |
| cache size | - | **7.3 MB/clip** -> 463 clips = 3.4 GB |
| raw Kubric on disk | - | 13 MB/clip -> 500 clips = 6.5 GB |
| train step, 17.9M, batch 16, N=128 | A100 **shared** | **0.8 s/it** -> 20k steps = 4.4 h |
| validation (50 sampling steps, 56 clips) | A100 | **35 s** per VAL |

First real result, 2026-09-12: 17.9M model, 322 train / 56 val clips, 500
steps -> val ratio 0.309 (the homemade metric, since removed). Generalises to
unseen clips, tracking mode.

Useful conversions:
- 460 clips at batch 16 = ~29 steps per epoch, so 3000 steps = ~104 epochs.
- 500 clips x 24 frames at 24 fps = **8 minutes of video total**. It is a small
  dataset; do not over-read a weak result.


Kubrics Data is from tappid