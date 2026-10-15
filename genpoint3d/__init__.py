"""A 3D point tracker, in two machines that meet at the cache.

    prepare/   raw Kubric clips -> cached clips   python -m genpoint3d.prepare.main
    train/     cached clips -> checkpoints        python -m genpoint3d.train.main

CachedClip (with ScaleStats and NormStats) is the cache's format, the one
thing both sides share. scripts/ holds the tools around them: evaluation,
visualisation, benchmarks, checks and the cluster's job files.
"""
