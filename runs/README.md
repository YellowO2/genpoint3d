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

## Known open items

- Best APD lands on the LAST step in runs 5 and 6, with the cosine schedule
  already at zero. Run 4 peaked at 10k and 18k further steps changed nothing, so
  12k was chosen -- but these two may be schedule-limited rather than converged.
- Per-frame APD: frame 0 scores 1.0000 and decays to 0.3251 by frame 23. Error
  accumulates across the rollout. Rollout training and diffusion forcing address
  this; `flow.py` already accepts `per_frame_k`.
- The encoder takes `last_hidden_state` at one scale. The paper's spec is
  multiple layers concatenated and upsampled 2x. At 384px with patch 16 the grid
  is 24x24, so one patch covers 16 pixels -- coarser than a tracker needs.
