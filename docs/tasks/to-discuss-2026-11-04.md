# To discuss before the next runs (4 Nov)

## With the user, before the runs

1. Points 2, 3 and 4 of the fixes (`f35cca2`, `c0d0077`), so they are known
   to be right: the frame-before comparison off unless both frames have
   their image; group vectors from zeros for new points; scoring from the
   query frame, 26 scores.
2. What the runs train on: our 240 clips alone (fzL lost Kubric skill that
   way) or mixed with the old Kubric.
3. The run plan: three runs (A, A + rope-current, A + past positions) or two.

## With the user, later

- The template: the latest clear look, or a memory the model updates.
- The spawn rule: new points where new surface appears.
- Whole-frame comparison, removed in the cleanup: needed eventually?

## Decided without discussion

- Gaps: half the clips, 2-12 frames.
- Past positions: offsets from the guess at the window's edge, the side the
  sweep came from, "don't know" before frame 0 and before a point's query
  frame until a backward sweep fills it.
- `--refine-iters` refused beside a schedule rather than silently unused.
- Test runs kept fast; after training, the fall test, the Kubric drop test,
  the ball view angle (spin_up_10) and standard scoring.
- A point first seen inside a window: the before/at-query-frame input tells
  it which frames are its past; raised again only if late points do badly.
