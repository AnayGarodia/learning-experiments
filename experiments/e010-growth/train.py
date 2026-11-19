"""e010 runner. FLOP accounting: 6 * nonemb_params * tokens (standard
approximation); grown runs are charged the donor's FLOPs (Du's honest
convention)."""
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from corpus import CorpusGen
from model import GPT, grow_depth_x2

HERE = Path(__file__).parent
RUNS = HERE / "runs"
CKPTS = HERE / "ckpts"
SEQ, BATCH = 256, 64


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def evaluate(model, device):
    g = CorpusGen(seed=9999)
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for _ in range(6):
            x, lab = g.batch(BATCH, SEQ)
            xt = torch.from_numpy(x).to(device)
            tlab = torch.from_numpy(lab).to(device)[:, 1:]
            logits = model(xt)[:, :-1]
            ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                 xt[:, 1:].reshape(-1), reduction="none").view(tlab.shape)
            m = tlab >= 0
            tot += ls[m].sum().item(); n += m.sum().item()
    model.train()
    return tot / n


def train_tokens(model, tokens, seed, device, flop_base=0, eval_every=30):
    gen = CorpusGen(seed=1000 + seed)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    steps = tokens // (BATCH * SEQ)
    P = model.nonemb_params()
    curve = []
    for step in range(steps):
        x, _ = gen.batch(BATCH, SEQ)
        xt = torch.from_numpy(x).to(device)
        loss = F.cross_entropy(model(xt)[:, :-1].reshape(-1, model.head.out_features),
                               xt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            fl = flop_base + 6 * P * (step + 1) * BATCH * SEQ
            curve.append({"flops": fl, "tokens": (step + 1) * BATCH * SEQ,
                          "loss": round(evaluate(model, device), 4)})
    return curve


def run(kind, seed, layers=4, donor_tokens=0, growth=None, cont_tokens=6_000_000,
        device="cpu"):
    """kind: scratch | grown. Grown: train donor at `layers`, grow x2,
    continue. Scratch: train at `layers` directly."""
    torch.manual_seed(seed)
    t0 = time.time()
    rec = {"kind": kind, "seed": seed, "layers": layers, "growth": growth,
           "donor_tokens": donor_tokens, "cont_tokens": cont_tokens,
           "git_sha": git_sha()}
    if kind == "scratch":
        m = GPT(n_layers=layers)
        rec["nonemb_params"] = m.nonemb_params()
        rec["curve"] = train_tokens(m, cont_tokens, seed, device)
        name = f"scratch_L{layers}_s{seed}.json"
    else:
        CKPTS.mkdir(exist_ok=True)
        ck = CKPTS / f"donor_L{layers}_t{donor_tokens}_s{seed}.pt"
        donor = GPT(n_layers=layers)
        if ck.exists():
            donor.load_state_dict(torch.load(ck, weights_only=True))
            donor_curve = []
        else:
            donor_curve = train_tokens(donor, donor_tokens, seed, device)
            torch.save(donor.state_dict(), ck)
        flop_base = 6 * donor.nonemb_params() * donor_tokens
        g = grow_depth_x2(donor, shrink_perturb=(growth == "shrink_perturb"),
                          seed=seed).to(device)
        rec["nonemb_params"] = g.nonemb_params()
        rec["donor_curve"] = donor_curve
        rec["curve"] = train_tokens(g, cont_tokens, seed, device, flop_base=flop_base)
        name = f"grown_{growth}_L{layers}x2_dt{donor_tokens}_s{seed}.json"
    rec["wall_s"] = round(time.time() - t0, 1)
    RUNS.mkdir(exist_ok=True)
    (RUNS / name).write_text(json.dumps(rec, indent=1))
    return rec


JOBS = []
for s in range(3):
    JOBS.append(dict(kind="scratch", seed=s, layers=4))
    JOBS.append(dict(kind="scratch", seed=s, layers=8))
    for growth in ("copy", "shrink_perturb"):
        for dt in (1_000_000, 3_000_000, 9_000_000):
            JOBS.append(dict(kind="grown", seed=s, layers=4, growth=growth,
                             donor_tokens=dt))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        r = run("grown", 0, growth="copy", donor_tokens=400_000, cont_tokens=400_000)
        print("smoke:", r["nonemb_params"], "params,", r["curve"][-1], f"{r['wall_s']}s")
    else:
        i, w = map(int, a.shard.split("/"))
        for j, kw in enumerate(JOBS):
            if j % w == i:
                r = run(device="cpu", **kw)
                print(j, r.get("growth"), kw, r["curve"][-1], flush=True)
