# Runs

One `log.json` per run.

APD is `average_pts_within_thresh` from TAP-Vid-3D, scored in metres. Higher is
better.

| # | run | commit | steps | best val APD | test APD | what changed |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `run3493` | `cabc639` | 30000 | 0.0516 | 0.0105† | first full-dataset run, 3144 train clips |
| 2 | `run3493_apdloss` | `0796ab2` | 30000 | 0.2100 | — | depth weighted loss + frame 0 starting pinned + l21 instead of l2 |
| 3 | `run3493_disp` | `ef50476` | 30000 | 0.2487 | — | per-point displacement target instead of absolute |
| 4 | `run4_adapter` | `c9d05ec` | 28000 | 0.2740 | 0.2859 | visual head moved into the model, so it trains (was frozen at random init upstream of the cache) |
| 5 | `run5_locality` | `27c16b9` | 12000 | 0.2940 | 0.2774 | cross-attention biased towards patches near each point's current estimate |
| 6 | `run6_corr` | `c93f990` | 12000 | **0.3540** | **0.3618** | query template compared against every patch: correlation steers the lookup, and its soft-argmax feeds in where the template matches |
| 7 | `run7_30k` | `a13f19b` | 30000 | **0.3880** | — | same as run 6, 30k steps instead of 12k |
| 8 | `run8_bidir` | `a13f19b` | 30000 | 0.3750 | — | `--causal 0`: a frame may attend to later frames |

`compare.png` overlays every run: `python runs/plot.py runs/*.json`
Legend labels are positional -- run 1 is the first file on the command line.

† Scored before `81c5900`, which taught `evaluate.py` to read `target`,
`norm_mode`, `anchor_frame0` and `loss_type` from the checkpoint. Before that it
used its own defaults, so a displacement-trained model was scored as absolute.
Re-score before quoting this number.

The static baseline (query point assumed never to move) is **0.2520** on the test
cache and 0.241 on val. That is the bar, not zero.

## The ablation, and what settled the diagnosis

`evaluate.py --ablate` destroys one visual input at a time. A score that does not
move names a pathway the model is not using. Run on the test cache:

| condition | run4 | run5 | run6 |
| --- | --- | --- | --- |
| intact | 0.2859 | 0.2774 | **0.3618** |
| swap-features (another clip's video) | 0.88x | 0.91x | **0.57x** |
| noise-features | 0.91x | 0.90x | 0.66x |
| shuffle-pos | 0.95x | 0.99x | 0.74x |
| swap-id (another clip's templates) | 0.90x | 0.95x | 0.73x |
| blind | 0.05x | 0.05x | 0.07x |

**Runs 4 and 5 were not reading the video.** Given another clip's footage they
scored 0.2510 and 0.2522 against a static baseline of 0.2520 -- the wrong video
cost them nothing, because they had learned a generic motion prior instead.

A point had to discover which of 576 patches was its own, unsupervised, while
"predict barely any motion" cut the loss immediately. Gradient descent took the
easy win. CoTracker3 and TAPIP3D both refuse that search: they sample features at
the point's current estimate and compare against the query template, iteratively.

Run 5 narrowed the search by distance -- but distance is computed from the path
and the patch positions, so **no pixel enters it** and it carries no information
about the image. The ablation measured exactly that: 0.99x on `shuffle-pos`.

Run 6 added the comparison. Every pathway went from ignored to used, and
`swap-features` at 0.2056 now falls **below** the static baseline: a wrong image
actively misleads the model, which only happens if it is relying on the image.

`blind` is a bad control -- it deletes the cross-attention layers entirely, an
architecture never seen in training, so its collapse says nothing about whether
the features are useful. The honest ablations keep the shape and corrupt the
content.

## Causal vs bidirectional, and a loss/metric disagreement

| | best val APD | val_loss at end |
| --- | --- | --- |
| run7, causal | **0.3880** | 0.2775 |
| run8, bidirectional | 0.3750 | **0.2122** |

Bidirectional **generalises better and overfits far less** -- its val_loss barely
moved from 16k to 30k (0.2068 -> 0.2122) where run7's climbed 0.2473 -> 0.2775 --
and still scores WORSE on APD.

**Our loss and our metric disagree.** The loss is l21 on 3D displacement; APD
counts points inside depth-scaled thresholds. Cutting mean error while landing
fewer points inside the tight bands is exactly what smoothing toward the mean
does, and bidirectional attention smooths.

Note this contradicts the closest reference: genpt sets
`causal_attn_masking: False`. It costs us 0.013 APD. The likely difference is the
target: **we regress raw metric 3D and every reference reparameterises** -- DELTA
predicts `log(d_t / d_1)` and ablated it against inverse and Euclidean depth,
genpt splits coordinate, visibility and confidence losses. Keep `--causal 1`
until the loss is in the geometry the metric scores.

## Known open items

- Runs 5 and 6 peaked on their last step, so 30k was tried: run 7 reached 0.388
  and val_loss turned up after ~16k. **Not schedule-limited any more; now
  data-limited.** Downloading 3,144 -> ~5,600 clips (DELTA trains on 5,632).
- Image augmentation is unavailable to us: we cache DINOv3 features, not pixels,
  so flips and colour jitter would need the backbone re-run. Geometric jitter on
  `patch_xyz` (DELTA's depth noise) is the version that works on a feature cache.
- Per-frame APD: frame 0 scores 1.0000 and decays to 0.3251 by frame 23. Error
  accumulates across the rollout. Rollout training and diffusion forcing address
  this; `flow.py` already accepts `per_frame_k`.
- The encoder takes `last_hidden_state` at one scale. The paper's spec is
  multiple layers concatenated and upsampled 2x. At 384px with patch 16 the grid
  is 24x24, so one patch covers 16 pixels -- coarser than a tracker needs.
