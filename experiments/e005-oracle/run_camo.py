import argparse, itertools
from train import run_arm

JOBS = list(itertools.product(("uniform", "drop_noise"), (0.5, 0.75), range(3)))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); i, w = map(int, a.shard.split("/"))
    for j, (arm, np_, s) in enumerate(JOBS):
        if j % w == i:
            r = run_arm(arm, seed=s, noise_p=np_, camouflage=True)
            print(j, arm, np_, s, r["curve"][-1], flush=True)
