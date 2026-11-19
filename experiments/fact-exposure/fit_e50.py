"""E50 = exposures/pass at which recall crosses 0.5, per run and probe kind, by log-linear interpolation.
Then a power-law fit E50 ~ N^-a across model sizes. Usage: python fit_e50.py results/results_*.json"""
import json, sys, math
import numpy as np

rows = []
for f in sys.argv[1:]:
    r = json.load(open(f)); by = [x for x in r["log"] if "by_exposure" in x]
    if not by: continue
    by = by[-1]["by_exposure"]; N = r["nparams"]; tag = r["args"]["tag"]
    kinds = sorted({k.split(":")[1] for k in by})
    for kind in kinds:
        pts = sorted((int(k.split(":")[0]), v) for k, v in by.items() if k.endswith(":" + kind))
        e50 = None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if y0 < 0.5 <= y1: e50 = math.exp(math.log(x0) + (0.5 - y0) / (y1 - y0) * (math.log(x1) - math.log(x0))); break
        if e50 is None and pts and pts[-1][1] >= 0.5: e50 = pts[0][0]
        rows.append((tag, N, kind, e50, pts))
        print(f"{tag:28s} N={N/1e6:6.1f}M {kind:15s} E50={'>'+str(pts[-1][0]) if e50 is None else f'{e50:6.1f}'}  " + " ".join(f"{x}:{y:.2f}" for x, y in pts))

# power-law fit per kind over runs that share an arm (tag suffix after the size token)
import collections
grp = collections.defaultdict(list)
for tag, N, kind, e50, _ in rows:
    if e50: grp[(tag.split("_")[-1], kind)].append((N, e50))
for (arm, kind), pts in sorted(grp.items()):
    if len({p[0] for p in pts}) < 2: continue
    x = np.log([p[0] for p in pts]); y = np.log([p[1] for p in pts]); a, b = np.polyfit(x, y, 1)
    print(f"fit {arm:10s} {kind:12s}: E50 = {math.exp(b):.3g} * N^{a:+.2f}   ({len(pts)} sizes)")

if __name__ == "__main__" and len(sys.argv) == 1:
    # self-check on a synthetic curve
    pts = [(1, .05), (3, .06), (10, .2), (30, .45), (100, .8)]
    x0, y0, x1, y1 = 30, .45, 100, .8; e = math.exp(math.log(x0) + (0.5 - y0) / (y1 - y0) * (math.log(x1) - math.log(x0)))
    assert 30 < e < 100 and abs(e - 35.7) < 1, e; print("selfcheck ok", round(e, 1))
