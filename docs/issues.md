# Issues: one list

Everything found on 2026-10-09, in plain words. One place, so nothing has to be
remembered. Update the status column as things change.

Reference points, moving points only (how many land within tolerance):
"nothing moves" 0.25, our model about 0.28, TAPIP3D 0.93 (on clips it trained on).

## A. Fixed

| # | What was wrong | Commit |
|---|---|---|
| A1 | The model looked for image patches near the wrong location. | 03e6f58 |
| A2 | Scoring used tolerances half the size the benchmark uses. | acd5173 |
| A3 | Frame numbers were fed in a form the time code does not expect, so neighbouring frames looked unrelated. | 2867786 |
| A4 | The point's "what do I look like" feature was read from slightly the wrong spot. | 2867786 |
| A5 | Positions were rounded too coarsely in training, and validation used different precision from training. | 2867786 |
| A6 | The loss weighting had a per-clip scale error. | 2867786 |
| A7 | Our score was computed under different rules from the reference tracker's (clip weighting, empty clips, number of points). | f3995c0 |

## B. Not fixed, most important first

| # | What is wrong | Kind | Size of job | Status |
|---|---|---|---|---|
| B1 | **The model cannot read the patch under its own point.** "Look near yourself" is about 100x too weak, and training cannot strengthen it. | design | small | done 3f4ced0: the prior is one patch spacing wide, measured on each frame's grid, and its width trains multiplicatively. Not yet in a run. Still centred on the noisy guess, so early in denoising it points at the wrong patches (B5). Then `--locality-mode sched` (default for new runs): the window starts `--locality-wide` spacings wide at pure noise and narrows to one as the sample gets clean; both ends train. It takes almost nothing off the true patch at any noise level (the fixed window takes more than 5 logits off it for 75% of moving points at k = 0), but spreads the prior thin: less of it lands near the truth, not more. In a toy with lookalike patches it beat the fixed window on 4 seeds of 4 by about 5% in loss, and 4 did better than 8. Where the guess is already right it is worse than the fixed window. Not yet in a run. |
| B2 | **"Find what looks like me" starts out random**, so the "best match position" is just the middle of the scene and the model ignores it. | design | small | done 3f4ced0: matching is plain cosine similarity, on from step 0, and the best match is handed over as a displacement the model can copy. Then: the match is searched over the whole frame by appearance and refined in the winner's 3x3, so it no longer depends on the noisy guess. Not yet in a run. |
| B3 | Our self-checks pass for the wrong reasons (made-up numbers unlike real scenes). | test | small | done 61a27fe: check_model.py uses a run's settings and real spacing and fails on the old model; check_learning.py trains two small tasks to prove the image gets read. |
| B4 | The loss type pushes the model toward "nothing moves" when it is unsure. The paper uses the plain squared loss. | design | one flag | run14 queued |
| B5 | The model outputs "which way to push the guess" instead of the clean answer, and does not re-look at the image around its improving guess. Both reference trackers do. | design | medium | built, not yet in a run: `--method regress` trains the same network with no noise, the way TAPIP3D and CoTracker work. The guess starts at "nothing moves" and is corrected `--refine-iters` times (default 4), each correction scored against the truth, later ones weighted more (0.8 per step back, as TAPIP3D). The noise-level input becomes the iteration clock, so with `--locality-mode sched` the first look is wide and the last is one patch wide, around the improving guess. Run it as the flow run with `--method regress` added and nothing else changed. A step costs 4 forward and backward passes instead of 1, in the memory of 1. It answers one question: is the noise the bottleneck, or the features? Checked on a laptop for correctness only (a perfect-answer test, gradients, `--method flow` unchanged bit for bit); whether it LEARNS better is not tested. |
| B6 | Each patch's stored 3D position is an average, so it is blurry at object edges (typically 21 cm off). | design | small | code done, re-cache pending: each patch now stores the pointmap value at its centre pixel, one real surface point, never a blend. New caches get this from `preprocess.py`; the two existing caches still hold the averages until `scripts/recache_xyz.pbs` is run on each (CPU only, features untouched, writes a new directory). Not yet in a run. |
| B7 | Points that are hidden in some frames get no training signal there. | design | small | open |
| B8 | Coarse image grid and last-layer features only. The paper uses a finer grid and several layers. | design | medium | open |
| B9 | Over half the points we train on never move. | design | small | open |
| B10 | We train far less than the references (12-30k steps against 200k). | resources | - | open |
| B11 | **The search for "the patch that looks like me" picks the right patch only about half the time** (47% on the 24x24 grid, 60% on 48x48), on frozen features, and nothing trains it. | design | small | built, not yet in a run: `--match-learn 1` adds a small trainable correction to the features the search compares (it starts as today's search exactly) and a loss that names the right patch for every visible point in every frame after the first. The right patch is where the true position lands in that frame's image, from the camera. Works with `--method flow` and `--method regress`; the loss is computed once per step in both. Every validation now also logs how often the search is right (`match_acc`), and right to within one patch (`match_acc_near`), for all visible points and for moving ones, on val and train clips, with or without the flag: on the VAL line as `match exact/near mv exact/near`. Checked on a laptop for correctness only (flag off is unchanged bit for bit, the right patch is right on a made-up scene, gradients arrive); whether the search gets BETTER is not tested. |

## C. Checked and correct (stop worrying about these)

Data loading, colours, depth, camera maths, 3D tracks matching the 2D ones,
the target encode/decode round trip, the flow-matching loss and sampler (exact
on a perfect-answer test), the training loop, learning rate, EMA, gradient
accumulation, the scorer (matches TAPIP3D's own to 7 decimals, with camera
motion), the inputs given to TAPIP3D, and the network's wiring (masks, shapes,
attention, gradients).

## Runs in flight

| Run | What it tests |
|---|---|
| run13_fixed | A1-A6 together, same settings as the control |
| run14_l2 | run13 plus B4 (squared loss) |
