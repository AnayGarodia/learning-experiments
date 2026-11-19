"""E1: product-key sparse memory (Lample et al. 2019) vs dense MLP, forgetting vs capacity.
Reuses harness.py streams/metrics. Fine-tune only: no replay, no regularizer."""
import argparse, json, math, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from harness import DEV, make_stream, batches, accuracy, run, FineTune


class PKM(nn.Module):
    """Product-key memory: N = n*n slots addressed by two half-queries; top-k over k*k candidates."""
    def __init__(self, d, N, k=32, heads=1):
        super().__init__()
        self.n = int(math.isqrt(N)); assert self.n * self.n == N
        self.k, self.d = k, d; self.kh = min(k, self.n)
        self.q = nn.Sequential(nn.Linear(d, d), nn.BatchNorm1d(d))  # BN on query as in Lample et al. 2019; without it the value table never trains
        self.keys = nn.Parameter(torch.randn(2, self.n, d // 2) / math.sqrt(d // 2))
        self.values = nn.EmbeddingBag(N, d, mode="sum", sparse=True)
        nn.init.normal_(self.values.weight, std=d ** -0.5)

    def forward(self, x):
        q = self.q(x); q1, q2 = q.chunk(2, -1)
        s1, i1 = (q1 @ self.keys[0].T).topk(self.kh, -1)  # B,kh
        s2, i2 = (q2 @ self.keys[1].T).topk(self.kh, -1)
        s = (s1[:, :, None] + s2[:, None, :]).flatten(1)  # B,k*k
        idx = (i1[:, :, None] * self.n + i2[:, None, :]).flatten(1)
        s, j = s.topk(min(self.k, self.kh * self.kh), -1); idx = idx.gather(1, j)
        w = F.softmax(s, -1)
        return self.values(idx, per_sample_weights=w)


class Net(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(784, 128), nn.ReLU()); self.block = block; self.head = nn.Linear(128, 10)
    def forward(self, x): return self.head(self.block(self.trunk(x)))


class RandomHashPKM(nn.Module):
    """E1c control: addressing is a FIXED random projection of the raw input; only the value table trains."""
    def __init__(self, N, k=32):
        super().__init__(); self.pkm = PKM(128, N, k); self.proj = nn.Linear(784, 128, bias=False)
        self.pkm.q = nn.Sequential(nn.Linear(128, 128), nn.LayerNorm(128))
        for p in list(self.proj.parameters()) + list(self.pkm.q.parameters()) + [self.pkm.keys]: p.requires_grad_(False)
        with torch.no_grad(): self.pkm.keys.normal_(std=1.0)
    def forward(self, x): return self.pkm(self.proj(x))


class RHNet(nn.Module):
    def __init__(self, N, k):
        super().__init__(); self.block = RandomHashPKM(N, k); self.head = nn.Linear(128, 10)
    def forward(self, x): return self.head(self.block(x))


def make_model(kind, N, k):
    if kind == "rhash": return RHNet(N, k).to(DEV)
    if kind == "pkm": block = PKM(128, N, k)
    else: block = nn.Sequential(nn.Linear(128, N), nn.ReLU(), nn.Linear(N, 128))  # dense, width N
    return Net(block).to(DEV)


class Method(FineTune):
    def __init__(self, kind="pkm", N=1024, k=32, lr=0.1, epochs=1, bs=64, seed=0, freeze="none", **kw):
        self.freeze = freeze; torch.manual_seed(seed)
        self.model = make_model(kind, N, k)
        if freeze == "head":
            for p in self.model.head.parameters(): p.requires_grad_(False)
        vals = [p for n, p in self.model.named_parameters() if n.endswith("values.weight")]
        rest = [p for n, p in self.model.named_parameters() if not n.endswith("values.weight") and p.requires_grad]
        self.opts = ([torch.optim.Adam(rest, lr=lr)] if rest else []) + ([torch.optim.SparseAdam(vals, lr=lr)] if vals else [])
        self.epochs, self.bs = epochs, bs; self.g = torch.Generator().manual_seed(seed)

    def step(self, x, y):
        loss = F.cross_entropy(self.model(x), y)
        for o in self.opts: o.zero_grad()
        loss.backward()
        for o in self.opts: o.step()

    def train_task(self, x, y, t):
        super().train_task(x, y, t)
        if t == 0 and self.freeze in ("addr", "addr+head"):  # freeze addressing path (and optionally head) after the first task
            frozen = ["trunk", "block.q", "block.keys"] + (["head"] if self.freeze == "addr+head" else [])
            for n, p in self.model.named_parameters():
                if any(n.startswith(f) for f in frozen): p.requires_grad_(False)
            for mod in self.model.modules():
                if isinstance(mod, nn.BatchNorm1d): mod.eval(); mod.momentum = 0.0; mod.keep_eval = True
            rest = [p for n, p in self.model.named_parameters() if p.requires_grad and not n.endswith("values.weight")]
            self.opts = ([torch.optim.Adam(rest, lr=self.opts[0].param_groups[0]["lr"])] if rest else []) + self.opts[1:]

    def predict(self, x): return self.model(x)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stream", default="split"); p.add_argument("--tasks", type=int, default=5)
    p.add_argument("--kind", default="pkm"); p.add_argument("--Ns", default="256,1024,4096,16384,65536")
    p.add_argument("--k", type=int, default=32); p.add_argument("--seeds", type=int, default=3); p.add_argument("--lr", type=float, default=3e-3); p.add_argument("--freeze", default="none")
    a = p.parse_args()
    out = []
    for N in map(int, a.Ns.split(",")):
        rows = []
        for s in range(a.seeds):
            t0 = time.time(); stream = make_stream(a.stream, a.tasks, seed=s)
            r = run(Method, stream, kind=a.kind, N=N, k=a.k, lr=a.lr, seed=s, freeze=a.freeze); r.update(seed=s, N=N, kind=a.kind, secs=round(time.time() - t0, 1)); rows.append(r)
        acc = np.array([r["acc"] for r in rows]); bwt = np.array([r["bwt"] for r in rows])
        print(f"{a.stream:9s} {a.kind:5s} {a.freeze:9s} N={N:6d} ACC {acc.mean():.3f}±{acc.std():.3f}  BWT {bwt.mean():+.3f}±{bwt.std():.3f}  ({rows[0]['secs']}s/seed)", flush=True)
        out += rows
    json.dump(out, open(f"results/results_e1_{a.stream}_{a.kind}_{a.freeze}.json", "w"))
