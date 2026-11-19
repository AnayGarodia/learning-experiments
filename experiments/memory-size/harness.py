"""Tiny continual-learning harness. One file, no framework.

Streams: permuted-MNIST (domain-incremental), split-MNIST (class-incremental, single head).
Metrics: ACC, BWT (forgetting), FWT, per Lopez-Paz & Ranzato 2017.
Methods: any object with .train_task(loader, task_id) and .predict(x) -> logits.
"""
import argparse, json, time, sys
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision import datasets, transforms

DEV = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
ROOT = "/tmp/mnist"


def load_mnist():
    tf = transforms.Compose([transforms.ToTensor()])
    tr = datasets.MNIST(ROOT, train=True, download=True, transform=tf)
    te = datasets.MNIST(ROOT, train=False, download=True, transform=tf)
    xtr = tr.data.float().div(255).view(-1, 784); ytr = tr.targets
    xte = te.data.float().div(255).view(-1, 784); yte = te.targets
    return xtr, ytr, xte, yte


def make_stream(kind, n_tasks, seed=0):
    """Returns list of (train_x, train_y, test_x, test_y) tensors on DEV."""
    g = torch.Generator().manual_seed(seed)
    xtr, ytr, xte, yte = load_mnist()
    tasks = []
    if kind == "permuted":
        for t in range(n_tasks):
            perm = torch.arange(784) if t == 0 else torch.randperm(784, generator=g)
            tasks.append((xtr[:, perm], ytr, xte[:, perm], yte))
    elif kind == "split":
        assert n_tasks == 5
        for t in range(5):
            cls = [2 * t, 2 * t + 1]
            mtr = (ytr == cls[0]) | (ytr == cls[1]); mte = (yte == cls[0]) | (yte == cls[1])
            tasks.append((xtr[mtr], ytr[mtr], xte[mte], yte[mte]))
    else:
        raise ValueError(kind)
    return [tuple(z.to(DEV) for z in task) for task in tasks]


def mlp(width=256, depth=2, n_out=10):
    layers, d = [], 784
    for _ in range(depth):
        layers += [nn.Linear(d, width), nn.ReLU()]; d = width
    layers += [nn.Linear(d, n_out)]
    return nn.Sequential(*layers).to(DEV)


def batches(x, y, bs, g):
    idx = torch.randperm(len(x), generator=g, device="cpu").to(x.device)
    for i in range(0, len(x), bs):
        j = idx[i:i + bs]
        yield x[j], y[j]


@torch.no_grad()
def accuracy(model, x, y, bs=2048):
    model.eval(); correct = 0
    for i in range(0, len(x), bs):
        correct += (model(x[i:i + bs]).argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    for mod in model.modules():  # modules flagged frozen stay in eval (BatchNorm running stats)
        if getattr(mod, 'keep_eval', False): mod.eval()
    return correct / len(x)


class FineTune:
    """Plain SGD on each task in sequence. Lower bound."""
    def __init__(self, lr=0.01, epochs=1, bs=64, seed=0, **kw):
        self.model = mlp(); self.opt = torch.optim.SGD(self.model.parameters(), lr=lr)
        self.epochs, self.bs = epochs, bs; self.g = torch.Generator().manual_seed(seed)

    def step(self, x, y):
        loss = F.cross_entropy(self.model(x), y)
        self.opt.zero_grad(); loss.backward(); self.opt.step()

    def train_task(self, x, y, t):
        for _ in range(self.epochs):
            for xb, yb in batches(x, y, self.bs, self.g):
                self.step(xb, yb)

    def predict(self, x): return self.model(x)


class Replay(FineTune):
    """Reservoir buffer of past examples, mixed 1:1 into every batch."""
    def __init__(self, buffer=500, **kw):
        super().__init__(**kw); self.cap = buffer; self.bx, self.by, self.seen = [], [], 0

    def step(self, x, y):
        if self.bx:
            bx = torch.stack(self.bx); by = torch.stack(self.by)
            j = torch.randint(len(bx), (len(x),), generator=self.g).to(x.device)
            x = torch.cat([x, bx[j]]); y = torch.cat([y, by[j]])
        super().step(x, y)

    def train_task(self, x, y, t):
        super().train_task(x, y, t)
        for xi, yi in zip(x, y):  # reservoir sampling
            self.seen += 1
            if len(self.bx) < self.cap: self.bx.append(xi); self.by.append(yi)
            else:
                r = int(torch.randint(self.seen, (1,), generator=self.g))
                if r < self.cap: self.bx[r] = xi; self.by[r] = yi


class Joint(FineTune):
    """Retrain on the union of all tasks so far. Upper bound (not continual)."""
    def __init__(self, **kw):
        super().__init__(**kw); self.xs, self.ys = [], []

    def train_task(self, x, y, t):
        self.xs.append(x); self.ys.append(y)
        self.model = mlp(); self.opt = torch.optim.SGD(self.model.parameters(), lr=self.opt.param_groups[0]["lr"])
        super().train_task(torch.cat(self.xs), torch.cat(self.ys), t)


METHODS = {"finetune": FineTune, "replay": Replay, "joint": Joint}


def run(method, stream, **kw):
    T = len(stream); R = np.zeros((T + 1, T))  # R[i, j] = acc on task j after training i tasks (row 0 = untrained)
    m = method(**kw)
    for j in range(T): R[0, j] = accuracy(m.model, stream[j][2], stream[j][3])
    for i, (xtr, ytr, xte, yte) in enumerate(stream):
        m.train_task(xtr, ytr, i)
        for j in range(T): R[i + 1, j] = accuracy(m.model, stream[j][2], stream[j][3])
    acc = R[T, :].mean()
    bwt = np.mean([R[T, j] - R[j + 1, j] for j in range(T - 1)])
    fwt = np.mean([R[j, j] - R[0, j] for j in range(1, T)])
    return dict(acc=acc, bwt=bwt, fwt=fwt, R=R.round(4).tolist())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stream", default="permuted"); p.add_argument("--tasks", type=int, default=10)
    p.add_argument("--methods", default="finetune,replay,joint"); p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--epochs", type=int, default=1); p.add_argument("--lr", type=float, default=0.1)
    a = p.parse_args()
    for name in a.methods.split(","):
        rows = []
        for s in range(a.seeds):
            t0 = time.time(); stream = make_stream(a.stream, a.tasks, seed=s)
            r = run(METHODS[name], stream, seed=s, epochs=a.epochs, lr=a.lr); r["seed"] = s; r["secs"] = round(time.time() - t0, 1)
            rows.append(r)
        accs = np.array([r["acc"] for r in rows]); bwts = np.array([r["bwt"] for r in rows])
        print(f"{a.stream:9s} {name:9s} ACC {accs.mean():.3f}±{accs.std():.3f}  BWT {bwts.mean():+.3f}  FWT {np.mean([r['fwt'] for r in rows]):+.3f}  ({rows[0]['secs']}s/seed)", flush=True)
        with open(f"results/results_{a.stream}_{name}.json", "w") as f: json.dump(rows, f)
