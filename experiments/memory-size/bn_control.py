"""Control for E1's BatchNorm confound: train PKM on the stream, then re-estimate BN running stats on a mix of
all tasks (no weight update) and re-evaluate. If BWT is unchanged, BN stats are not the source of forgetting."""
import sys, numpy as np, torch
from harness import make_stream, accuracy
import memory
stream_kind, N = sys.argv[1], int(sys.argv[2]); T = 10 if stream_kind == "permuted" else 5
for seed in range(2):
    stream = make_stream(stream_kind, T, seed=seed); m = memory.Method(kind="pkm", N=N, k=32, lr=3e-3, seed=seed)
    R = np.zeros((T + 1, T))
    for i, (xtr, ytr, xte, yte) in enumerate(stream):
        m.train_task(xtr, ytr, i)
        for j in range(T): R[i + 1, j] = accuracy(m.model, stream[j][2], stream[j][3])
    bwt_before = np.mean([R[T, j] - R[j + 1, j] for j in range(T - 1)]); acc_before = R[T].mean()
    # re-estimate BN stats on all tasks, weights frozen
    for mod in m.model.modules():
        if isinstance(mod, torch.nn.BatchNorm1d): mod.reset_running_stats(); mod.momentum = None
    m.model.train()
    with torch.no_grad():
        for (xtr, ytr, _, _) in stream:
            for i in range(0, 5000, 500): m.model(xtr[i:i + 500])
    m.model.eval()
    accs = [accuracy(m.model, stream[j][2], stream[j][3]) for j in range(T)]
    bwt_after = np.mean([accs[j] - R[j + 1, j] for j in range(T - 1)])
    print(f"{stream_kind} N={N} seed={seed} ACC {acc_before:.3f} -> {np.mean(accs):.3f}   BWT {bwt_before:+.3f} -> {bwt_after:+.3f}", flush=True)
