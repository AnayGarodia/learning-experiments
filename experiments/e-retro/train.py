"""e-retro runner. One run = (arch, rung, lr, seed). Verdict logic is in
DESIGN.md; this only records curves."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load, batches, val_windows
from models import RUNGS, ByteTransformer, ByteLSTM

HERE = Path(__file__).parent
RUNS = HERE / "runs"
BATCH, SEQ = 32, 256
BUDGET = {"r1": 10_000_000, "r2": 20_000_000, "r3": 30_000_000}  # bytes
LRS = (3e-4, 1e-3, 3e-3)


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def evaluate(model, vw, device):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(vw), 64):
            b = torch.from_numpy(vw[i : i + 64]).to(device)
            loss = F.cross_entropy(model(b[:, :-1]).transpose(1, 2), b[:, 1:],
                                   reduction="sum")
            tot += loss.item(); n += b[:, 1:].numel()
    model.train()
    return tot / n  # nats/byte


def run_one(arch, rung, lr, seed, device="cpu", budget=None):
    train_arr, val_arr = load()
    vw = val_windows(val_arr)
    name, (d, L), h = next(r for r in RUNGS if r[0] == rung)
    torch.manual_seed(seed)
    model = (ByteTransformer(d, L) if arch == "transformer" else ByteLSTM(h)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    budget = budget or BUDGET[rung]
    steps = budget // (BATCH * SEQ)
    gen = batches(train_arr, BATCH, SEQ, seed=seed)
    curve, t0 = [], time.time()
    for step in range(steps):
        b = torch.from_numpy(next(gen)).to(device)
        loss = F.cross_entropy(model(b[:, :-1]).transpose(1, 2), b[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            vl = evaluate(model, vw, device)
            curve.append({"tokens": (step + 1) * BATCH * SEQ,
                          "val_nats": round(vl, 4),
                          "val_bpb": round(vl / np.log(2), 4)})
    rec = {"arch": arch, "rung": rung, "lr": lr, "seed": seed,
           "nonemb_params": model.nonemb_params(), "budget": budget,
           "git_sha": git_sha(), "wall_s": round(time.time() - t0, 1),
           "curve": curve}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{arch}_{rung}_lr{lr:.0e}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    return rec


JOBS = [dict(arch=a, rung=r[0], lr=lr, seed=s)
        for a in ("transformer", "lstm") for r in RUNGS
        for lr in LRS for s in (0, 1)]

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        for arch in ("transformer", "lstm"):
            r = run_one(arch, "r1", 1e-3, 0, budget=400_000)
            print("smoke", arch, r["nonemb_params"], r["curve"][-1], f"{r['wall_s']}s",
                  flush=True)
    else:
        i, w = map(int, a.shard.split("/"))
        for j, kw in enumerate(JOBS):
            if j % w == i:
                r = run_one(**kw)
                print(j, kw, r["curve"][-1], flush=True)
