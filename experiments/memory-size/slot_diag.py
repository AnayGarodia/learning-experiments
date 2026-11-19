"""Slot utilization and overlap per task for the frozen-addressing PKM (E1b). Retrains 1 seed, records touched slots."""
import sys, numpy as np, torch
from harness import make_stream
import memory
N = int(sys.argv[1]); freeze = sys.argv[2] if len(sys.argv) > 2 else "addr+head"
stream = make_stream("permuted", 10, seed=0); m = memory.Method(kind="pkm", N=N, k=32, lr=3e-3, seed=0, freeze=freeze)
def touched(x):
    b = m.model.block; m.model.eval(); s = set()
    with torch.no_grad():
        for i in range(0, 5000, 500):
            h = m.model.trunk(x[i:i+500]); q = b.q(h); q1, q2 = q.chunk(2, -1)
            i1 = (q1 @ b.keys[0].T).topk(b.kh, -1)[1]; i2 = (q2 @ b.keys[1].T).topk(b.kh, -1)[1]
            sc = ((q1 @ b.keys[0].T).topk(b.kh, -1)[0][:, :, None] + (q2 @ b.keys[1].T).topk(b.kh, -1)[0][:, None, :]).flatten(1)
            idx = (i1[:, :, None] * b.n + i2[:, None, :]).flatten(1); j = sc.topk(min(b.k, b.kh * b.kh), -1)[1]
            s |= set(idx.gather(1, j).flatten().tolist())
    m.model.train(); return s
sets = []
for t, (xtr, ytr, xte, yte) in enumerate(stream):
    m.train_task(xtr, ytr, t); sets.append(touched(xte))
    print(f"N={N} task{t} slots touched by 5000 test ex: {len(sets[-1])} ({100*len(sets[-1])/N:.1f}%)", flush=True)
J = np.zeros((10, 10))
for i in range(10):
    for j in range(10): J[i, j] = len(sets[i] & sets[j]) / len(sets[i] | sets[j])
print("mean Jaccard overlap between different tasks: %.3f" % J[~np.eye(10, dtype=bool)].mean())
print("union of all tasks: %d of %d" % (len(set.union(*sets)), N))
