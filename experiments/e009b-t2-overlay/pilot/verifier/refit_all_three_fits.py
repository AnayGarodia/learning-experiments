"""Verifier re-fit: headline multiplier + CI under EACH of the three fits,
independently of analyze.py's verdict path (which only checks pw vs logit)."""
import sys, glob, json
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

for t in sorted(by_level):
    rl = by_level[t]
    n = rl[0]["eval_seqs"]
    full = np.arange(n)
    print(f"\n===== {t:,} tokens =====")
    for axis in ("acc", "cons"):
        for fit in ("pw", "logit", "loglin"):
            m, front = A.level_multipliers(rl, axis, full, fit)
            ci, _ = A.bootstrap(rl, axis, fit)
            cand = [(m[(r["size"], "joint_arg")], r["size"]) for r in rl]
            inside = [c for c in cand if c[0][1] == "ok"]
            if not inside:
                print(f"  [{axis}][{fit:6}] no in-support point")
                continue
            (M, flag, fo, co), rung = max(inside, key=lambda x: x[0][0])
            lo, hi = ci[(rung, "joint_arg")]
            v = A.verdict(lo, hi)
            # also: headline over ALL points incl. out-of-support (pre-amendment rule)
            (M2, f2, _, _), rung2 = max(cand, key=lambda x: x[0][0])
            lo2, hi2 = ci[(rung2, "joint_arg")]
            print(f"  [{axis}][{fit:6}] in-support headline rung={rung:4} M={M:6.2f} "
                  f"CI[{lo:6.2f},{hi:6.2f}] -> {v}"
                  f"   || all-points headline {rung2:4} M={M2:6.2f} "
                  f"CI[{lo2:6.2f},{hi2:6.2f}] flag={f2} -> {A.verdict(lo2,hi2)}")
