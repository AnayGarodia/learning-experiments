"""Sweep extension per verifier Flag 1: r1/r2, both archs, LRs 6e-3 and
1e-2, 2 seeds. Same protocol as train.py (unclipped — clipped variants
only if a divergence makes the edge condition undecidable)."""
from train import run_one

JOBS = [dict(arch=a, rung=r, lr=lr, seed=s)
        for r in ("r1", "r2") for a in ("transformer", "lstm")
        for lr in (6e-3, 1e-2) for s in (0, 1)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", required=True)
    i, w = map(int, ap.parse_args().shard.split("/"))
    for j, kw in enumerate(JOBS):
        if j % w == i:
            r = run_one(**kw)
            print(j, kw, r["curve"][-1], flush=True)
