"""e011 runner. One arm = one set of prediction horizons on the same
trunk. Aux heads train the trunk; ONLY the main next-token head is
ever scored. Aux-head FLOPs are charged (each head is a vocab-sized
projection, comparable to the main head, so records carry both a
token count and a FLOP-equivalent count)."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from corpus import CorpusGen, VOCAB
from model import TinyGPT

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SEQ, BATCH = 256, 64

ARMS = {                       # aux horizons beyond the main +1 head
    "next_token": [],
    "multi_horizon": [4, 16, 64],
    "annealed": [4, 16, 64],   # aux weight decays linearly to 0
    "mtp4": [2, 3, 4],
}


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def evaluate(model, device, long_density, n_batches=8):
    """Per-scale consequence accuracy (main head, position before the
    consequence) + structure loss, on a held-out stream."""
    g = CorpusGen(seed=9999, long_density=long_density)
    model.eval()
    s_loss, s_n = 0.0, 0
    hit = {3: 0, 4: 0, 5: 0}
    tot = {3: 0, 4: 0, 5: 0}
    with torch.no_grad():
        for _ in range(n_batches):
            x, lab = g.batch(BATCH, SEQ)
            xt = torch.from_numpy(x).to(device)
            logits = model(xt)[:, :-1]
            tgt = xt[:, 1:]
            tlab = torch.from_numpy(lab).to(device)[:, 1:]
            ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="none").view(tgt.shape)
            m = tlab == 1
            s_loss += ls[m].sum().item(); s_n += m.sum().item()
            pred = logits.argmax(-1)
            for c in (3, 4, 5):
                mc = tlab == c
                hit[c] += (pred[mc] == tgt[mc]).sum().item()
                tot[c] += mc.sum().item()
    model.train()
    return (s_loss / max(s_n, 1),
            {c: round(hit[c] / tot[c], 4) if tot[c] else None for c in (3, 4, 5)})


def run_arm(arm, seed, long_density, device="cpu", total_tokens=6_000_000,
            eval_every=40):
    horizons = ARMS[arm]
    torch.manual_seed(seed)
    gen = CorpusGen(seed=1000 + seed, long_density=long_density)
    model = TinyGPT(max_len=SEQ).to(device)
    aux_heads = nn.ModuleList(nn.Linear(128, VOCAB) for _ in horizons).to(device)
    params = list(model.parameters()) + list(aux_heads.parameters())
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.1)
    steps = total_tokens // (BATCH * SEQ)
    # FLOP-equivalents: charge each aux head as one extra head-projection
    # per token relative to the control's (trunk + 1 head) = 1.0 unit
    head_frac = 0.30  # measured-ish share of a forward that is the head at d=128/V=5000; disclosed constant
    fe_per_step = 1.0 + head_frac * len(horizons)
    curve = []
    t0 = time.time()
    for step in range(steps):
        x, _ = gen.batch(BATCH, SEQ)
        xt = torch.from_numpy(x).to(device)
        feats = model.features(xt)
        logits = model.head(feats)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                               xt[:, 1:].reshape(-1))
        aux_w = 1.0 if arm != "annealed" else max(0.0, 1.0 - step / steps)
        if horizons and aux_w > 0:
            for h, head in zip(horizons, aux_heads):
                al = F.cross_entropy(head(feats[:, :-h]).reshape(-1, VOCAB),
                                     xt[:, h:].reshape(-1))
                loss = loss + aux_w * al
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            sl, acc = evaluate(model, device, long_density)
            curve.append({"tokens": (step + 1) * BATCH * SEQ,
                          "flop_eq_tokens": int((step + 1) * BATCH * SEQ * fe_per_step),
                          "struct_loss": round(sl, 4),
                          "acc_short": acc[3], "acc_med": acc[4], "acc_long": acc[5]})
    rec = {"arm": arm, "seed": seed, "long_density": long_density,
           "horizons": horizons, "head_frac": head_frac,
           "total_tokens": total_tokens, "git_sha": git_sha(),
           "wall_s": round(time.time() - t0, 1), "curve": curve}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{arm}_ld{long_density}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import argparse, itertools
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        r = run_arm("multi_horizon", 0, 0.25, total_tokens=800_000)
        print("smoke:", r["curve"][-1], f"wall={r['wall_s']}s")
    else:
        jobs = list(itertools.product(ARMS, (0.25, 1.0), range(3)))
        i, w = map(int, a.shard.split("/"))
        for j, (arm, ld, s) in enumerate(jobs):
            if j % w == i:
                r = run_arm(arm, s, ld)
                print(j, arm, ld, s, r["curve"][-1], flush=True)
