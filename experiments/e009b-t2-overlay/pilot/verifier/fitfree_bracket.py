"""Fit-free bracket on M. No interpolation at all: F_frontier(C_o) must lie
between the last frontier rung BELOW C_o and the first frontier rung AT OR
ABOVE C_o. Any fit is a guess inside that bracket, so if the bracket sits
wholly on one side of a threshold the fit cannot change the verdict."""
import sys
from pathlib import Path
import numpy as np

E = Path("./experiments/e009b-t2-overlay")
sys.path.insert(0, str(E))
import analyze as A

recs = A.load()
by_level = {}
for r in recs:
    by_level.setdefault(r["tokens"], []).append(r)
for v in by_level.values():
    v.sort(key=lambda r: A.LADDER.index(r["size"]))


def bracket(front, c, fo):
    fs = np.array([f for f, _ in front]); cs = np.array([x for _, x in front])
    if c > cs[-1]:
        return fs[-1] / fo, np.inf, "above-support"
    if c < cs[0]:
        return 0.0, fs[0] / fo, "below-support"
    j = int(np.searchsorted(cs, c))          # first frontier rung with cap >= c
    lo = fs[j - 1] if j > 0 else 0.0
    return lo / fo, fs[j] / fo, "ok"


for t in sorted(by_level):
    rl = by_level[t]
    i = np.arange(rl[0]["eval_seqs"])
    print(f"\n===== {t:,} tokens =====")
    for axis in ("acc", "cons"):
        dense = [(A.eff_flops(r, "full", i), A.stats(r, "full", axis, i)) for r in rl]
        front = A.frontier(dense)
        print(f" [{axis}] frontier: " +
              " ".join(f"{f/1e6:.3f}M@{c:.4f}" for f, c in front))
        for r in rl:
            for o in ("depth_arg", "joint_arg"):
                fo = A.eff_flops(r, o, i); co = A.stats(r, o, axis, i)
                lo, hi, flag = bracket(front, co, fo)
                pw = A.frontier_flops(front, co, "pw")[0] / fo
                lg = A.frontier_flops(front, co, "logit")[0] / fo
                ll = A.frontier_flops(front, co, "loglin")[0] / fo
                warn = ""
                if flag == "ok" and not (lo <= ll <= hi):
                    warn = "  <-- LOGLIN OUTSIDE BRACKET (misspecified)"
                print(f"   {r['size']:4} {o:10} M_bracket [{lo:6.2f},{hi:6.2f}] "
                      f"pw {pw:5.2f} logit {lg:5.2f} loglin {ll:5.2f} {flag}{warn}")
