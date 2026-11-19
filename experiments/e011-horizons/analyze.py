"""e011: tokens-to-criterion per arm, both accountings, + artifact pack.

Two accountings are reported for every cell, because the auxiliary heads
are not free: RAW tokens (matched data) and FLOP-EQUIVALENT tokens
(matched compute, the record's flop_eq_tokens). Multi-horizon arms pay
their head cost only in the second.
"""
import json
import glob
import collections
import statistics
from pathlib import Path

HERE = Path(__file__).parent
CRIT = ("acc_long", 0.90)   # pre-registered long-range criterion
ARMS = ("next_token", "mtp4", "multi_horizon", "annealed")


def crossing(curve, key, thresh, cost="tokens"):
    """First cost at which the metric reaches the threshold; None if never."""
    for pt in curve:
        if pt.get(key) is not None and pt[key] >= thresh:
            return pt[cost]
    return None


def main():
    recs = [json.loads(open(f).read()) for f in glob.glob(str(HERE / "runs/*.json"))]
    cells = collections.defaultdict(list)
    for r in recs:
        cells[(r["long_density"], r["arm"])].append(r)

    key, thresh = CRIT
    rows = []
    for ld in sorted({r["long_density"] for r in recs}):
        base = None
        for arm in ARMS:
            rs = sorted(cells.get((ld, arm), []), key=lambda r: r["seed"])
            if not rs:
                continue
            raw = [crossing(r["curve"], key, thresh, "tokens") for r in rs]
            fe = [crossing(r["curve"], key, thresh, "flop_eq_tokens") for r in rs]
            n_reached = sum(x is not None for x in raw)
            med_raw = statistics.median([x for x in raw if x]) if n_reached else None
            med_fe = statistics.median([x for x in fe if x]) if n_reached else None
            if arm == "next_token":
                base = (med_raw, med_fe)
            rows.append(dict(ld=ld, arm=arm, n=len(rs), reached=n_reached,
                             raw=med_raw, fe=med_fe,
                             mult_raw=(med_raw / base[0]) if base and base[0] and med_raw else None,
                             mult_fe=(med_fe / base[1]) if base and base[1] and med_fe else None,
                             spread=(max(x for x in raw if x) / min(x for x in raw if x))
                             if n_reached == len(rs) and min(x for x in raw if x) else None))

    hdr = f"{'ld':>5} {'arm':<14} {'n':>2} {'hit':>3} {'tokens':>9} {'flop-eq':>9} {'x_raw':>6} {'x_fe':>6} {'spread':>7}"
    print(f"criterion: {key} >= {thresh}   (x = ratio vs next_token; >1 means SLOWER)")
    print(hdr)
    for r in rows:
        f = lambda v, w=9, p=0: (f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}")
        print(f"{r['ld']:>5} {r['arm']:<14} {r['n']:>2} {r['reached']:>3} "
              f"{f(r['raw'])} {f(r['fe'])} {f(r['mult_raw'],6,2)} {f(r['mult_fe'],6,2)} "
              f"{f(r['spread'],7,2)}")

    # grid-resolution guard (the artifact class that voided e005/e006 run 1)
    evals = sorted({pt["tokens"] for r in recs for pt in r["curve"]})
    step = evals[1] - evals[0] if len(evals) > 1 else 0
    print(f"\neval grid: {len(evals)} points, spacing {step:,} tokens "
          f"({step / evals[-1]:.0%} of the run) — any two arms within one "
          f"spacing of each other are NOT separated by this instrument.")
    for ld in sorted({r["ld"] for r in rows}):
        vals = [r["raw"] for r in rows if r["ld"] == ld and r["raw"]]
        if vals and max(vals) - min(vals) < step:
            print(f"  ld={ld}: ALL arms within one grid step -> UNDER-RESOLVED, no ordering claim")

    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    with open(art / "results.csv", "w") as fh:
        fh.write("long_density,arm,seeds,seeds_reaching_criterion,"
                 "median_tokens_to_criterion,median_flopeq_tokens_to_criterion,"
                 "multiplier_vs_next_token_raw,multiplier_vs_next_token_flopeq,seed_spread\n")
        for r in rows:
            g = lambda v: ("" if v is None else f"{v:.4f}")
            fh.write(f"{r['ld']},{r['arm']},{r['n']},{r['reached']},"
                     f"{g(r['raw'])},{g(r['fe'])},{g(r['mult_raw'])},{g(r['mult_fe'])},{g(r['spread'])}\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        lds = sorted({r["ld"] for r in rows})
        fig, axes = plt.subplots(1, len(lds), figsize=(6 * len(lds), 4.5), squeeze=False)
        for ax, ld in zip(axes[0], lds):
            for arm in ARMS:
                rs = sorted(cells.get((ld, arm), []), key=lambda r: r["seed"])
                if not rs:
                    continue
                xs = [pt["tokens"] for pt in rs[0]["curve"]]
                ys = [statistics.median([r["curve"][i][key] for r in rs])
                      for i in range(len(rs[0]["curve"]))]
                ax.plot(xs, ys, marker="o", label=arm)
            ax.axhline(thresh, ls="--", c="k", lw=1)
            ax.set_title(f"long_density = {ld}")
            ax.set_xlabel("training tokens")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel(key)
        axes[0][0].legend()
        fig.suptitle("e011: long-range accuracy vs training (median of 3 seeds)")
        fig.tight_layout()
        fig.savefig(art / "results.png", dpi=150)
        print("wrote artifacts/results.csv artifacts/results.png")
    except ImportError:
        print("wrote artifacts/results.csv (no matplotlib)")


if __name__ == "__main__":
    main()
