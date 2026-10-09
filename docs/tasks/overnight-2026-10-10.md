# Overnight plan, 2026-10-10

Written before the user went to sleep, so the plan survives a context compact.
The user asked for work to continue unattended. Report in the morning, short.

## State when written
- Running on NSCC (g1): `run16_48` -- 48x48 grid (cache `kubric_768_3493`),
  cosine correlation + whole-frame match, `--locality 0`, 12k steps. Val moving
  0.396 at step 4000 (24x24 was 0.343 at the same step, 0.371 at the end).
- Finished: `run15_sched`, `run15_read`, `run15_noloc` (24x24). All three the
  same: val moving ~0.37, val overall ~0.59, train overall ~0.70. The window
  variants make no difference yet.
- Per-tolerance, moving points, val, 24x24: <1px 0.15, <4px 0.31, <16px 0.73.
  "Nothing moves": 0.15 / 0.22 / 0.40. TAPIP3D: 0.78 / 0.97 / 1.00.
  So it finds the object most of the time and has no precision.
- A background agent is building `--method regress` (no noise; start at
  "nothing moves", look, correct, 4 iterations) next to `--method flow`.

## Time limit
The cluster connection is good for about 4 hours from 03:13 local
time on 2026-10-10, then it drops. Anything that must start has to be
submitted before then; chain later jobs with `-W depend=` so they start
without us. Jobs already queued or running are not affected by the drop.

No training on the laptop beyond about 5 steps: it is a weak MacBook Air.

## To do, in order, without asking
1. When the agent reports and its tests pass: `git pull` on the cluster, a
   short smoke test, then a regress run otherwise identical to `run15_noloc`
   (24x24, same flags, add `--method regress`). Ask for only as much walltime
   as the job needs.
2. When `run16_48` ends: score it like for like (`fit_check.pbs` with
   `N=100,POINTS=256,SCALING=none,MOTION=4`) and read the per-tolerance numbers.
3. If regress beats flow at equal compute: run regress on the 48x48 cache.
4. If there is still a free lane: B6 from `docs/issues.md` (patch positions
   that are not an average over the patch).

## Rules that still apply
- One ssh attempt at a time, `-o BatchMode=yes`. If it fails, stop and say so
  in the morning; no retry loops, never `ssh -O exit`.
- Never `git clean`, `git checkout -- .` or `reset --hard` on either machine.
  Only `qdel` our own jobs, by id. Sync code to the cluster only with git.
- A queued job runs whatever code is on the cluster when it STARTS, so pass
  every flag that matters explicitly.
- Commits as the user, no co-author or session lines.

## Order of attack (user, 03:30)

Work down the tolerances, on train first: rough trajectory (<16px, train 0.86
against TAPIP3D 1.00), then <8, <4, <2. Keep testing fixes after the
regress-against-flow test ends; do not stop at one result.

## Results so far (06:35)

Moving points only, no rescale. "Nothing moves" scores about 0.25. Val / train.

| Run | Grid | What changed | Steps | Val | Train |
|---|---|---|---|---|---|
| run15_noloc | 24 | diffusion, baseline | 10,000 | 0.370 | 0.430 |
| run16_48 | 48 | finer grid | 12,000 | 0.444 | 0.494 |
| run18_ctr_flow | 24 | sharper patch positions (B6) | 10,000 of 12,000 | 0.393 | 0.443 |
| run19_c24 | 24 | B6, short schedule (control) | 5,000 | 0.372 | 0.392 |
| run19_m24 | 24 | B6 + trained matcher (B11) | 5,000 | 0.405 | 0.422 |
| run19_mk24 | 24 | B6 + trained matcher + top-4 candidates (B12) | 5,000 | 0.404 | 0.421 |
| run17_rg_sched | 24 | no-noise method, shrinking window | 2,500 | 0.384 | 0.420 |
| run17_rg_noloc | 24 | no-noise method, whole frame only | 2,500 | 0.381 | 0.413 |
| run21_rg_m24 | 24 | no-noise + B6 + trained matcher | 2,000 of 4,000 | 0.441 | 0.455 |
| run20_m48 | 48 | diffusion + B6 + trained matcher | 4,000 of 12,000 | 0.456 | 0.461 |

Per tolerance, val, moving points (<1, <2, <4, <8, <16 px):

| Run | <1 | <2 | <4 | <8 | <16 |
|---|---|---|---|---|---|
| nothing moves | 0.15 | 0.17 | 0.22 | 0.32 | 0.40 |
| run15_sched (24, 12k) | 0.15 | 0.19 | 0.31 | 0.53 | 0.73 |
| run19_m24 (5k) | 0.15 | 0.20 | 0.34 | 0.57 | 0.77 |
| run17_rg_sched (2.5k) | 0.15 | 0.19 | 0.31 | 0.53 | 0.74 |
| run21_rg_m24 (2k) | 0.16 | 0.22 | 0.39 | 0.62 | 0.81 |
| run20_m48 (4k) | 0.16 | 0.25 | 0.42 | 0.64 | 0.81 |
| run16_48 (12k) | 0.17 | 0.26 | 0.44 | 0.63 | 0.80 |
| TAPIP3D | 0.78 | 0.91 | 0.97 | 0.99 | 1.00 |

What this says:

- The whole-frame match is the limit on the rough trajectory. Untrained, for
  moving points, the best match is within one patch of the truth 69% of the
  time (24 grid) and 58% (48 grid); the truth is among the best five 89% / 81%.
- Training the match (B11) takes 69% to 78% within 2,500 steps and then stops
  rising. It is worth +0.03 on the moving score.
- Showing the trunk four candidates instead of one (B12) adds nothing.
- Sharper patch positions (B6) are worth about +0.02.
- The no-noise method reaches in 2,500 steps what diffusion needs 12,000 for,
  and its validation is five times faster. The window makes little difference.
- Nothing has moved the <1px number off the "nothing moves" value on the 24
  grid. Only the finer grid moves <1 and <2.

Queued when the connection dropped (all read their flags from the qsub line):

| Run | Lane | What |
|---|---|---|
| run19_md24 | gdev | trained matcher with a wider head (`--match-dim 256`) |
| run22_rg_m48_short | gdev | no-noise + B6 + matcher on 48 grid, 1,400 steps |
| run24_rg_mc24 | gdev | run21 plus the local cost volume (`--costvol 1`), for precision |
| run24_rg_mc48_short | gdev | the same on the 48 grid, 1,400 steps |
| run23_rg_m48 | g1, after run20_m48 | no-noise + B6 + matcher on 48 grid, 6,000 steps |

Updated 06:50. The wider matcher head (run19_md24) is at 0.390 val at step 2,500 against 0.381 for the narrow one, with the match within one patch 79% against 78%.
