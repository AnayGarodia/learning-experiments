"""Train the 3-size ladder, saving 5 checkpoints across training
(dossier constraint b: oracle fraction as a function of training tokens)."""
import json
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from corpus import CorpusGen
from model import SIZES, SwiGPT

HERE = Path(__file__).parent
CKPTS = HERE / "ckpts"
SEQ, BATCH = 256, 64
TOTAL_TOKENS = 6_000_000


def git_sha():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                              capture_output=True, text=True).stdout.strip()
    except OSError:
        return "unknown"


def train(size, seed=0, device="cpu", total_tokens=TOTAL_TOKENS):
    torch.manual_seed(seed)
    gen = CorpusGen(seed=1000 + seed)
    m = SwiGPT(**SIZES[size]).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1)
    steps = total_tokens // (BATCH * SEQ)
    # 5 log-spaced checkpoints (fractions of training)
    marks = sorted({max(1, int(steps * f)) for f in (0.04, 0.1, 0.25, 0.5, 1.0)})
    CKPTS.mkdir(exist_ok=True)
    t0 = time.time()
    for step in range(1, steps + 1):
        x, _ = gen.batch(BATCH, SEQ)
        xt = torch.from_numpy(x).to(device)
        loss = F.cross_entropy(m(xt)[:, :-1].reshape(-1, m.head.out_features),
                               xt[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step in marks:
            tok = step * BATCH * SEQ
            torch.save({"state": m.state_dict(), "tokens": tok, "size": size,
                        "seed": seed, "git_sha": git_sha()},
                       CKPTS / f"{size}_s{seed}_t{tok}.pt")
            print(f"{size} ckpt @ {tok} tokens, loss {loss.item():.3f}, "
                  f"{time.time()-t0:.0f}s", flush=True)
    return marks


if __name__ == "__main__":
    import sys
    size = sys.argv[1] if len(sys.argv) > 1 else "s"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tt = int(sys.argv[3]) if len(sys.argv) > 3 else TOTAL_TOKENS
    train(size, seed=seed, total_tokens=tt)
