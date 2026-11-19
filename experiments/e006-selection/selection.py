"""e006: data-selection policies at honestly-charged compute.
Cost unit = forward-equivalents (train step = 3: fwd + ~2x bwd;
scoring a candidate batch = 1). All curves plotted vs total cost."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from corpus import CorpusGen
from model import TinyGPT

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SEQ, BATCH, CANDS = 256, 64, 4

POLICIES = ("random", "self_uncertainty", "oracle", "anti")


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def evaluate(model, device, n_batches=8):
    g = CorpusGen(seed=9999)
    model.eval()
    s_loss, s_n, c_hit, c_n = 0.0, 0, 0, 0
    with torch.no_grad():
        for _ in range(n_batches):
            x, lab = g.batch(BATCH, SEQ)
            xt = torch.from_numpy(x).to(device)
            logits = model(xt)[:, :-1]
            tgt, tlab = xt[:, 1:], torch.from_numpy(lab).to(device)[:, 1:]
            ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 tgt.reshape(-1), reduction="none").view(tgt.shape)
            m = tlab == 1
            s_loss += ls[m].sum().item(); s_n += m.sum().item()
            mc = tlab == 3
            c_hit += (logits.argmax(-1)[mc] == tgt[mc]).sum().item()
            c_n += mc.sum().item()
    model.train()
    return s_loss / s_n, c_hit / c_n


def score_pool(model, x, lab, policy, device):
    """Per-SEQUENCE scores over a pool (v2: example-level selection).
    Returns one score per row."""
    with torch.no_grad():
        xt = torch.from_numpy(x).to(device)
        logits = model(xt)[:, :-1]
        tlab = torch.from_numpy(lab).to(device)[:, 1:]
        ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             xt[:, 1:].reshape(-1), reduction="none").view(tlab.shape)
        if policy == "self_uncertainty":
            m = tlab >= 0
        else:  # oracle / anti: loss on learnable classes only
            m = (tlab == 1) | (tlab == 3)
        s = (ls * m).sum(1) / m.sum(1).clamp(min=1)
    return s.cpu().numpy()


def run_policy(policy, seed, dup_p, device="cpu", budget_fe=1500):
    torch.manual_seed(seed)
    gen = CorpusGen(seed=1000 + seed, dup_p=dup_p)
    model = TinyGPT(max_len=SEQ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    spent, curve = 0, []
    t0 = time.time()
    next_eval = 30
    while spent < budget_fe:
        if policy == "random":
            x, lab = gen.batch(BATCH, SEQ)
            spent += 3
        else:
            # v2: pool of CANDS*BATCH sequences, keep the BATCH best per policy
            xp, labp = gen.batch(CANDS * BATCH, SEQ)
            scores = score_pool(model, xp, labp, policy, device)
            order = np.argsort(scores)
            idx = order[-BATCH:] if policy != "anti" else order[:BATCH]
            x, lab = xp[idx], labp[idx]
            spent += 3 + CANDS
        xt = torch.from_numpy(x).to(device)
        logits = model(xt)[:, :-1]
        tlab = torch.from_numpy(lab).to(device)[:, 1:]
        ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             xt[:, 1:].reshape(-1), reduction="none").view(tlab.shape)
        loss = ls[tlab >= 0].mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if spent >= next_eval:
            sl, ca = evaluate(model, device)
            curve.append({"cost_fe": spent, "struct_loss": round(sl, 4),
                          "cons_acc": round(ca, 4)})
            next_eval += 30
    rec = {"policy": policy, "seed": seed, "dup_p": dup_p, "budget_fe": budget_fe,
           "git_sha": git_sha(), "wall_s": round(time.time() - t0, 1), "curve": curve}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{policy}_dup{dup_p}_s{seed}.json").write_text(json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import argparse, itertools
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        r = run_policy("oracle", 0, 0.5, budget_fe=300)
        print(r["policy"], r["curve"][-1], f"wall={r['wall_s']}s")
    else:
        jobs = list(itertools.product(POLICIES, (0.0, 0.5, 0.9), range(3)))
        i, w = map(int, a.shard.split("/"))
        for j, (pol, dp, s) in enumerate(jobs):
            if j % w == i:
                r = run_policy(pol, s, dp)
                print(j, pol, dp, s, r["curve"][-1], flush=True)
