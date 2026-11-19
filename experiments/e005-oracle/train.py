"""One arm = one class-weight vector over oracle labels.
Next-token framing: the loss/label at position t refers to predicting
token t+1, so weights use the LABEL OF THE TARGET token."""
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from corpus import CorpusGen, CONS0, K
from model import TinyGPT

HERE = Path(__file__).parent
RUNS = HERE / "runs"
SEQ = 256
BATCH = 64

ARMS = {                      # weights per class (fork, structure, noise, consequence)
    "uniform":      (1.0, 1.0, 1.0, 1.0),
    "drop_noise":   (1.0, 1.0, 0.0, 1.0),
    "struct_only":  (0.0, 1.0, 0.0, 1.0),
    "drop_forks":   (0.0, 1.0, 1.0, 1.0),
}


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def evaluate(model, gen_seed, device, n_batches=8, camouflage=False):
    """Structure loss + consequence accuracy on held-out stream."""
    g = CorpusGen(seed=gen_seed, camouflage=camouflage)
    model.eval()
    s_loss, s_n, c_hit, c_n = 0.0, 0, 0, 0
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
            mc = tlab == 3
            c_hit += (logits.argmax(-1)[mc] == tgt[mc]).sum().item()
            c_n += mc.sum().item()
    return s_loss / s_n, c_hit / c_n


def run_arm(arm, seed, device="cpu", total_tokens=6_000_000, oracle_flip=0.0,
            noise_p=0.5, eval_every=5, camouflage=False):
    w = torch.tensor(ARMS[arm])
    torch.manual_seed(seed)
    gen = CorpusGen(seed=1000 + seed, noise_p=noise_p, camouflage=camouflage)
    model = TinyGPT(max_len=SEQ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    steps = total_tokens // (BATCH * SEQ)
    curve = []
    t0 = time.time()
    rng = np.random.default_rng(seed)
    for step in range(steps):
        x, lab = gen.batch(BATCH, SEQ)
        if oracle_flip > 0:  # titration: corrupt oracle labels
            flip = rng.random(lab.shape) < oracle_flip
            lab = np.where(flip, rng.integers(0, 4, lab.shape), lab)
        xt = torch.from_numpy(x).to(device)
        tlab = torch.from_numpy(lab).to(device)[:, 1:]
        logits = model(xt)[:, :-1]
        ls = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                             xt[:, 1:].reshape(-1), reduction="none").view(tlab.shape)
        wt = w.to(device)[tlab.clamp(min=0)] * (tlab >= 0)
        loss = (ls * wt).sum() / wt.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
        model.train()
        if (step + 1) % eval_every == 0 or step == steps - 1:
            sl, ca = evaluate(model, gen_seed=9999, device=device, camouflage=camouflage)
            curve.append({"tokens": (step + 1) * BATCH * SEQ,
                          "struct_loss": round(sl, 4), "cons_acc": round(ca, 4)})
            model.train()
    rec = {"arm": arm, "seed": seed, "oracle_flip": oracle_flip, "camouflage": camouflage,
           "noise_p": noise_p, "total_tokens": total_tokens,
           "git_sha": git_sha(), "wall_s": round(time.time() - t0, 1),
           "curve": curve}
    RUNS.mkdir(exist_ok=True)
    (RUNS / f"{arm}_f{oracle_flip}_np{noise_p}_c{int(camouflage)}_s{seed}.json").write_text(
        json.dumps(rec, indent=1))
    return rec


if __name__ == "__main__":
    import sys
    r = run_arm(sys.argv[1] if len(sys.argv) > 1 else "uniform", seed=0,
                total_tokens=800_000)
    print(r["arm"], r["curve"][-1], f"wall={r['wall_s']}s")
