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
