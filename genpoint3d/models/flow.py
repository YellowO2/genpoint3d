"""
YOUR TASK. Write pseudocode for the three functions below, then we check it.

(The previous version is in git at commit 42fe257 if you ever want to peek --
but try first.)

================================================================================
WHAT THIS FILE IS FOR
================================================================================

`model.py` gives us a network. Here is its exact contract:

    velocity = model(x, k, anchor)

    x        (B, T, N, 3)   some 3D trajectory
    k        (B,)           a number in [0, 1] -- "how noisy is x"
    anchor   (B, N, 3)      where each point started, at frame 0
    velocity (B, T, N, 3)   a 3-vector per point per frame

That network is just a function. It does not know how to train itself, and it
does not know how to produce a trajectory from nothing. This file supplies
both: the training objective, and the sampler.

================================================================================
THE ONE IDEA YOU CANNOT DERIVE -- read this carefully
================================================================================

We want: start from random noise, end up with a real trajectory.

Flow matching says: draw noise `x0` and take a real trajectory `x1`, then
imagine walking in a straight line from one to the other.

    k = 0.0    ────────────────────────────────►  k = 1.0
    x0                                            x1
    pure noise                                    real trajectory

    any point on that line:   x_k = (1 - k) * x0 + k * x1

Concretely, if x0 = 10 and x1 = 2:
    k = 0.00  ->  x_k = 10       (all noise)
    k = 0.25  ->  x_k = 8
    k = 0.50  ->  x_k = 6        (halfway)
    k = 1.00  ->  x_k = 2        (the real thing)

Now: if you walk that line at constant speed, what is your velocity?
It is `x1 - x0` -- the same at every point on the line. In the example, -8.

**That is what we train the network to predict.** Show it `x_k` and tell it
`k`; it must output `x1 - x0`.

Why this is useful: at sample time we have no `x1`. But if the network can
look at any point and say "the real data is *that way*", we can start at pure
noise and keep stepping in the direction it points until we arrive.

================================================================================
YOUR TASK
================================================================================

Write pseudocode (or plain English, numbered steps) for each function. Do not
worry about torch syntax or edge cases -- I want the logic. Think about shapes
where it matters.

--------------------------------------------------------------------------------
1.  sample_k(batch_size) -> (B,) numbers in [0, 1]

    During training we pick a random `k` for each sample. Uniform would be the
    obvious choice, and it is WRONG -- the paper reports this as critical.

    Think: at k = 0.02 the input is basically pure noise. At k = 0.98 it is
    basically the answer already. Which values of k is the network actually
    learning something hard at? Should we spend our training steps evenly?

    (You do not need to know the formula. Say what property you want, and
    which end of the range you want more of.)

--------------------------------------------------------------------------------
2.  flow_matching_loss(model, x1, anchor, mask) -> a single number

    x1     (B, T, N, 3)  the real trajectory (ground truth)
    anchor (B, N, 3)     conditioning, passed straight to the model
    mask   (B, T, N)     True where the point is visible

    One training step. Everything you need is in the section above.

    Two things to be careful about:
      - `k` is (B,) but `x1` is (B, T, N, 3). How do you multiply them?
      - what is `mask` for, and where in the calculation does it apply?

--------------------------------------------------------------------------------
3.  sample(model, anchor, num_frames, steps) -> (B, T, N, 3)

    No ground truth here. Produce a trajectory out of nothing.

    Where do you start? What does the model tell you at each stage? How far do
    you move each time, and how do you know when to stop?

    (Hint: `steps` is how many times you ask the model. If you take `steps`
    equal hops from k=0 to k=1, how big is one hop?)
================================================================================
"""
