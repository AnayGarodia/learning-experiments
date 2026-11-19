"""E3: the allocation law on FIXED pretrained features. Split-CIFAR-100, 10 tasks x 10 classes, class-incremental.

Features: torchvision ResNet18 (ImageNet weights), 512-d, precomputed once (frozen).
Methods (all with a FROZEN random readout unless noted, so only the value table trains):
  rhash      : fixed random-projection product-key addressing, top-k of N slots, values train.   (E1d condition)
  learned    : product-key addressing trained jointly, values train. Reports slot utilization.  (collapse condition)
  ncm        : nearest-class-mean on the frozen features. Zero training, zero forgetting. The honest bar.
  headft     : a trainable linear head on frozen features, fine-tuned task by task. Standard lower bound.
Metrics: ACC, BWT, utilization (slots touched by all tasks / N), all per Lopez-Paz.
"""
import argparse, json, math, os, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms, models

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
FEAT = "/tmp/cifar100_resnet18_feats.pt"


def features():
    if os.path.exists(FEAT): return torch.load(FEAT, weights_only=True)
    tf = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); net.fc = nn.Identity(); net = net.to(DEV).eval()
    out = {}
    for split in ["train", "test"]:
        ds = datasets.CIFAR100("/tmp/cifar100", train=split == "train", download=True, transform=tf)
        dl = torch.utils.data.DataLoader(ds, batch_size=256, num_workers=0)
        xs, ys = [], []
        with torch.no_grad():
            for x, y in dl: xs.append(net(x.to(DEV)).cpu()); ys.append(y)
        out[split] = (torch.cat(xs), torch.cat(ys)); print(split, out[split][0].shape, flush=True)
    torch.save(out, FEAT); return out


def stream(feats, seed):
    g = torch.Generator().manual_seed(seed); order = torch.randperm(100, generator=g)
    (xtr, ytr), (xte, yte) = feats["train"], feats["test"]
    tasks = []
    for t in range(10):
        cls = order[10 * t:10 * t + 10]
        mtr = torch.isin(ytr, cls); mte = torch.isin(yte, cls)
        tasks.append(tuple(z.to(DEV) for z in (xtr[mtr], ytr[mtr], xte[mte], yte[mte])))
    return tasks


class PKM(nn.Module):
    def __init__(self, d_in, N, k, d_out, learned):
        super().__init__(); self.n = int(math.isqrt(N)); self.k = k; self.kh = min(k, self.n)
        self.q = nn.Sequential(nn.Linear(d_in, 128), nn.LayerNorm(128)); self.keys = nn.Parameter(torch.randn(2, self.n, 64))
        self.values = nn.EmbeddingBag(N, d_out, mode="sum", sparse=True); nn.init.zeros_(self.values.weight)
        if not learned:
            for p in list(self.q.parameters()) + [self.keys]: p.requires_grad_(False)
    def route(self, x):
        q1, q2 = self.q(x).chunk(2, -1)
        s1, i1 = (q1 @ self.keys[0].T).topk(self.kh, -1); s2, i2 = (q2 @ self.keys[1].T).topk(self.kh, -1)
        s = (s1[:, :, None] + s2[:, None, :]).flatten(1); idx = (i1[:, :, None] * self.n + i2[:, None, :]).flatten(1)
        s, j = s.topk(min(self.k, self.kh * self.kh), -1); return idx.gather(1, j), F.softmax(s, -1)
    def forward(self, x):
        idx, w = self.route(x); return self.values(idx, per_sample_weights=w)


class MemNet(nn.Module):
    """values -> frozen random readout -> 100 logits."""
    def __init__(self, N, k, learned, d_mem=128):
        super().__init__(); self.mem = PKM(512, N, k, d_mem, learned); self.head = nn.Linear(d_mem, 100)
        for p in self.head.parameters(): p.requires_grad_(False)
    def forward(self, x): return self.head(self.mem(x))


@torch.no_grad()
def acc(model, x, y):
    model.eval(); a = (model(x).argmax(1) == y).float().mean().item(); model.train(); return a


def train_epochs(model, opts, x, y, epochs, bs, g):
    for _ in range(epochs):
        idx = torch.randperm(len(x), generator=g).to(DEV)
        for i in range(0, len(x), bs):
            j = idx[i:i + bs]; loss = F.cross_entropy(model(x[j]), y[j])
            for o in opts: o.zero_grad()
            loss.backward()
            for o in opts: o.step()


def run_mem(tasks, N, k, learned, lr, epochs, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = MemNet(N, k, learned).to(DEV)
    vals = [m.mem.values.weight]; rest = [p for n, p in m.named_parameters() if p.requires_grad and not n.endswith("values.weight")]
    opts = [torch.optim.SparseAdam(vals, lr=lr)] + ([torch.optim.Adam(rest, lr=lr)] if rest else [])
    T = 10; R = np.zeros((T + 1, T)); touched = set()
    for i, (xtr, ytr, xte, yte) in enumerate(tasks):
        train_epochs(m, opts, xtr, ytr, epochs, 64, g)
        for j in range(T): R[i + 1, j] = acc(m, tasks[j][2], tasks[j][3])
        with torch.no_grad(): touched |= set(m.mem.route(xte)[0].flatten().tolist())
    return R, len(touched) / N


def run_ncm(tasks):
    T = 10; R = np.zeros((T + 1, T)); means = {}
    for i, (xtr, ytr, xte, yte) in enumerate(tasks):
        for c in ytr.unique().tolist(): means[c] = xtr[ytr == c].mean(0)
        C = torch.stack([means[c] for c in sorted(means)]); labels = torch.tensor(sorted(means), device=DEV)
        for j in range(T):
            x, y = tasks[j][2], tasks[j][3]; pred = labels[torch.cdist(x, C).argmin(1)]; R[i + 1, j] = (pred == y).float().mean().item()
    return R, float("nan")


def run_headft(tasks, lr, epochs, seed):
    torch.manual_seed(seed); g = torch.Generator().manual_seed(seed); m = nn.Linear(512, 100).to(DEV); opt = torch.optim.Adam(m.parameters(), lr=lr)
    T = 10; R = np.zeros((T + 1, T))
    for i, (xtr, ytr, xte, yte) in enumerate(tasks):
        train_epochs(m, [opt], xtr, ytr, epochs, 64, g)
        for j in range(T): R[i + 1, j] = acc(m, tasks[j][2], tasks[j][3])
    return R, float("nan")


def metrics(R):
    T = R.shape[1]; return dict(acc=R[T].mean(), bwt=np.mean([R[T, j] - R[j + 1, j] for j in range(T - 1)]), diag=np.mean([R[j + 1, j] for j in range(T)]))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="rhash"); p.add_argument("--Ns", default="4096,16384,65536,262144,1048576")
    p.add_argument("--k", type=int, default=32); p.add_argument("--seeds", type=int, default=3); p.add_argument("--epochs", type=int, default=3); p.add_argument("--lr", type=float, default=3e-3)
    a = p.parse_args(); feats = features(); out = []
    Ns = [int(n) for n in a.Ns.split(",")] if a.method in ("rhash", "learned") else [0]
    for N in Ns:
        rows = []
        for s in range(a.seeds):
            t0 = time.time(); tasks = stream(feats, s)
            if a.method in ("rhash", "learned"): R, util = run_mem(tasks, N, a.k, a.method == "learned", a.lr, a.epochs, s)
            elif a.method == "ncm": R, util = run_ncm(tasks)
            else: R, util = run_headft(tasks, a.lr, a.epochs, s)
            r = metrics(R); r.update(N=N, seed=s, util=util, method=a.method, R=R.round(4).tolist(), secs=round(time.time() - t0, 1)); rows.append(r)
        A = np.array([r["acc"] for r in rows]); B = np.array([r["bwt"] for r in rows]); U = np.nanmean([r["util"] for r in rows]); D = np.mean([r["diag"] for r in rows])
        print(f"cifar100 {a.method:8s} N={N:8d} ACC {A.mean():.3f}±{A.std():.3f} BWT {B.mean():+.3f}±{B.std():.3f} diag {D:.3f} util {U:.3f} ({rows[0]['secs']}s/seed)", flush=True)
        out += rows
    json.dump(out, open(f"results_e3_{a.method}.json", "w"))
