"""Pre-registered fairness re-run: LSTM WITH hidden-state carry across
256-byte windows (truncated BPTT over contiguous lanes), train AND eval.
Records use arch='lstm_carry'."""
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data import load
from models import RUNGS, ByteLSTM
from train import BATCH, SEQ, BUDGET, git_sha

RUNS = Path(__file__).parent / "runs"


def lanes(arr, n_lanes, seq):
    lane_len = len(arr) // n_lanes
    starts = np.arange(n_lanes) * lane_len
    pos = 0
    while True:
        reset = pos + seq + 1 > lane_len
        if reset:
            pos = 0
        x = np.stack([arr[s + pos: s + pos + seq + 1] for s in starts]).astype(np.int64)
        yield x, reset or pos == 0
        pos += seq


def eval_carry(model, val_arr, device, n_lanes=64):
    model.eval()
    lane_len = len(val_arr) // n_lanes
    n_win = (lane_len - 1) // SEQ
    tot, n = 0.0, 0
    with torch.no_grad():
        h = c = None
        gen = lanes(val_arr, n_lanes, SEQ)
        for _ in range(n_win):
            x, reset = next(gen)
            if reset:
                h = c = None
            b = torch.from_numpy(x).to(device)
            e = model.emb(b[:, :-1])
            out, (h, c) = model.lstm(e) if h is None else model.lstm(e, (h, c))
            loss = F.cross_entropy(model.head(model.ln(out)).transpose(1, 2),
                                   b[:, 1:], reduction="sum")
            tot += loss.item(); n += b[:, 1:].numel()
    model.train()
    return tot / n


def run_carry(rung, lr, seed, device="cpu"):
    train_arr, val_arr = load()
    _, _, hid = next(r for r in RUNGS if r[0] == rung)
    torch.manual_seed(seed)
    model = ByteLSTM(hid).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    steps = BUDGET[rung] // (BATCH * SEQ)
    gen = lanes(train_arr, BATCH, SEQ)
    h = c = None
    curve, t0 = [], time.time()
    for step in range(steps):
        x, reset = next(gen)
        if reset:
            h = c = None
        b = torch.from_numpy(x).to(device)
        e = model.emb(b[:, :-1])
        out, (h, c) = model.lstm(e) if h is None else model.lstm(e, (h, c))
        h, c = h.detach(), c.detach()
        loss = F.cross_entropy(model.head(model.ln(out)).transpose(1, 2), b[:, 1:])
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % max(1, steps // 10) == 0 or step == steps - 1:
            vl = eval_carry(model, val_arr, device)
            curve.append({"tokens": (step + 1) * BATCH * SEQ,
                          "val_nats": round(vl, 4),
                          "val_bpb": round(vl / np.log(2), 4)})
    rec = {"arch": "lstm_carry", "rung": rung, "lr": lr, "seed": seed,
           "nonemb_params": model.nonemb_params(), "budget": BUDGET[rung],
           "git_sha": git_sha(), "wall_s": round(time.time() - t0, 1),
           "curve": curve}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"lstm_carry_{rung}_lr{lr:.0e}_s{seed}_carry.json").write_text(
        json.dumps(rec, indent=1))
    return rec


JOBS = ([dict(rung="r1", lr=lr, seed=s) for lr in (3e-3, 6e-3) for s in (0, 1)]
        + [dict(rung="r2", lr=lr, seed=s) for lr in (3e-3, 6e-3) for s in (0, 1)]
        + [dict(rung="r3", lr=lr, seed=s) for lr in (1e-3, 3e-3) for s in (0, 1)])

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--shard", required=True)
    i, w = map(int, ap.parse_args().shard.split("/"))
    for j, kw in enumerate(JOBS):
        if j % w == i:
            r = run_carry(**kw)
            print(j, kw, r["curve"][-1], flush=True)
