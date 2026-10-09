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
| B1 | **The model cannot read the patch under its own point.** "Look near yourself" is about 100x too weak, and training cannot strengthen it. | design | small | next |
| B2 | **"Find what looks like me" starts out random**, so the "best match position" is just the middle of the scene and the model ignores it. | design | small | next |
| B3 | Our self-checks pass for the wrong reasons (made-up numbers unlike real scenes). | test | small | with B1, B2 |
| B4 | The loss type pushes the model toward "nothing moves" when it is unsure. The paper uses the plain squared loss. | design | one flag | run14 queued |
| B5 | The model outputs "which way to push the guess" instead of the clean answer, and does not re-look at the image around its improving guess. Both reference trackers do. | design | medium | after B1, B2 |
| B6 | Each patch's stored 3D position is an average, so it is blurry at object edges (typically 21 cm off). | design | small | open |
| B7 | Points that are hidden in some frames get no training signal there. | design | small | open |
| B8 | Coarse image grid and last-layer features only. The paper uses a finer grid and several layers. | design | medium | open |
| B9 | Over half the points we train on never move. | design | small | open |
| B10 | We train far less than the references (12-30k steps against 200k). | resources | - | open |

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
