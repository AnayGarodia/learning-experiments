"""e-retro analysis per DESIGN.md verdict rule. Partial-data tolerant.
Carry-arm runs use arch name 'lstm_carry' and are shown alongside."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RUNS = Path(__file__).parent / "runs"
RUNGS = ("r1", "r2", "r3")


def main():
    cells = defaultdict(list)
    for f in sorted(RUNS.glob("*.json")):
        r = json.loads(f.read_text())
        cells[(r["arch"], r["rung"], r["lr"])].append(r["curve"][-1]["val_nats"])

    best = {}
    for (arch, rung, lr), vals in sorted(cells.items()):
        m = float(np.mean(vals))
        cur = best.get((arch, rung))
        if cur is None or m < cur[1]:
            best[(arch, rung)] = (lr, m, float(np.ptp(vals)), len(vals))

    archs = sorted({a for (a, _, _) in cells})
    print(f"{'rung':5} {'arch':14} {'best_lr':9} {'val_nats':9} {'bpb':8} {'seed_ptp':9} n")
    for rung in RUNGS:
        for arch in archs:
            b = best.get((arch, rung))
            if b:
                lr, m, sp, n = b
                print(f"{rung:5} {arch:14} {lr:<9.0e} {m:<9.4f} {m/np.log(2):<8.4f} {sp:<9.4f} {n}")

    print("\nall sweep points:")
    for (arch, rung, lr), vals in sorted(cells.items(), key=lambda kv: (kv[0][1], kv[0][0], kv[0][2])):
        print(f"  {rung} {arch:14} lr={lr:.0e}: {np.mean(vals):.4f} (n={len(vals)}, ptp={np.ptp(vals):.4f})")

    print("\ngap (lstm - transformer, nats; positive = transformer wins):")
    gaps = {}
    for rung in RUNGS:
        t, l = best.get(("transformer", rung)), best.get(("lstm", rung))
        if t and l:
            gaps[rung] = l[1] - t[1]
            print(f"  {rung}: {gaps[rung]:+.4f} (spread t={t[2]:.4f} l={l[2]:.4f})")
        c = best.get(("lstm_carry", rung))
        if c and t:
            print(f"  {rung} carry: {c[1] - t[1]:+.4f} (lstm_carry {c[1]:.4f} @ {c[0]:.0e}, spread {c[2]:.4f})")

    if len(gaps) == 3:
        wins = all(g > 0 for g in gaps.values())
        spreads = [max(best[("transformer", r)][2], best[("lstm", r)][2]) for r in RUNGS]
        tol = max(spreads)
        trend_ok = gaps["r2"] >= gaps["r1"] - tol and gaps["r3"] >= gaps["r2"] - tol
        v = "PASS" if (wins and trend_ok) else ("PARTIAL" if wins else "FAIL")
        print(f"\nVERDICT (literal pre-registered rule): {v}")


if __name__ == "__main__":
    main()
