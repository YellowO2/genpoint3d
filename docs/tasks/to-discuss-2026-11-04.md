# To discuss before the next runs (4 Nov)

## Check what was just built (`f35cca2`, `c0d0077`)

1. Soft visibility: the image kept, the head's last call fed back; the
   before/at-query-frame input; the gap defaults (half the clips, 2-12 frames).
2. The frame-before comparison: off unless both frames have their image.
3. Group vectors: a point new to a window starts from zeros.
4. Scoring: references at the query frame; 26 scores kept.

## Understanding the loop

5. How the sweeps (fbfb) run over the windows; what a backward sweep does
   differently; why a window's first frame does not move forward.
6. A point first seen inside a window steps out both ways from its query
   frame, but the window is told one direction: is the before/at input enough?
7. What motion cannot be continued from 4 frames of past.

## Design ideas to settle

8. Past positions as input: how many frames, how small the overlap can be.
9. The template: the latest clear look, or a memory the model updates.
10. The spawn rule: new points where new surface appears.
11. Whole-frame comparison: removed in the cleanup; needed eventually?
12. `--refine-iters` unused under a schedule; steps measured in window
    lengths, so a clip shorter than a window changes their units.

## Before any run

13. The run plan: three runs (A, A + rope-current, A + past positions) or two.
14. Speed: test runs should be faster (3 rounds), fbfb is 4.
15. Data: the mixed loader (our clips and the old Kubric), PointOdyssey,
    busier Kubric scenes.
16. Tests after training: fall test, Kubric drop test, ball view angle
    (spin_up_10), standard scoring.

## Housekeeping

17. `genpoint3d/train/data/check_target_scale.py` is no longer called: keep or delete.
