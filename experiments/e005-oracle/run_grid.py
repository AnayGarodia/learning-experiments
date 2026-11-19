import argparse, itertools
from train import run_arm, ARMS

def jobs():
    for arm, np_, s in itertools.product(ARMS, (0.25, 0.5, 0.75), range(3)):
        yield dict(arm=arm, noise_p=np_, seed=s)
    for flip, s in itertools.product((0.1, 0.25, 0.5), range(3)):
        yield dict(arm="drop_noise", noise_p=0.5, oracle_flip=flip, seed=s)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", default="0/1")
    a = ap.parse_args(); i, w = map(int, a.shard.split("/"))
    for j, kw in enumerate(jobs()):
        if j % w == i:
            r = run_arm(device="cpu", total_tokens=6_000_000, **kw)
            print(j, r["arm"], kw.get("noise_p"), kw.get("oracle_flip", 0),
                  r["curve"][-1], flush=True)
