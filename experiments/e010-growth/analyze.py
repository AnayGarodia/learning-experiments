"""e010: savings-decay curve, exponent-vs-prefactor, donor-convergence sweep.

Three pre-registered objects (dossier 010):
  (a) savings(horizon) = FLOPs_scratch(loss)/FLOPs_grown(loss) - 1 at
      several loss horizons — the DECAY CURVE, the thing nobody has
      measured with error bars;
  (b) exponent vs prefactor from L(C) = A * C^(-alpha) fits per family
      (G1/G2 discriminator: does growth change the slope or only the
      offset?);
  (c) savings vs donor_tokens (G3's regime-artifact discriminator).

Grown runs are charged the donor's FLOPs (set in DESIGN before data).
"""
import json
import glob
import math
import random
import collections
import statistics
from pathlib import Path

HERE = Path(__file__).parent
SEED = 0  # bootstrap determinism (no wall-clock randomness)


def flops_at_loss(curve, target):
    """FLOPs at which the curve first reaches `target` loss, log-linearly
    interpolated between the bracketing points. None if never reached."""
    prev = None
    for pt in curve:
        if pt["loss"] <= target:
            if prev is None:
                return pt["flops"]
            # interpolate in (log C, loss)
            l0, l1 = prev["loss"], pt["loss"]
            c0, c1 = math.log(prev["flops"]), math.log(pt["flops"])
            if l0 == l1:
                return pt["flops"]
            frac = (l0 - target) / (l0 - l1)
            return math.exp(c0 + frac * (c1 - c0))
        prev = pt
    return None


def fit_power_law(points):
    """Least squares on log L = log A - alpha * log C. Returns (alpha, logA)."""
    pts = [(math.log(c), math.log(l)) for c, l in points if c > 0 and l > 0]
    n = len(pts)
    if n < 3:
        return None, None
    mx = sum(p[0] for p in pts) / n
    my = sum(p[1] for p in pts) / n
    den = sum((p[0] - mx) ** 2 for p in pts)
    if den == 0:
        return None, None
    slope = sum((p[0] - mx) * (p[1] - my) for p in pts) / den
    return -slope, my - slope * mx


def bootstrap_ci(values, n=2000, lo=2.5, hi=97.5):
    if len(values) < 2:
        return (None, None)
    rng = random.Random(SEED)
    means = []
    for _ in range(n):
        s = [values[rng.randrange(len(values))] for _ in values]
        means.append(sum(s) / len(s))
    means.sort()
    return (means[int(lo / 100 * n)], means[int(hi / 100 * n)])


def label(r):
    if r["kind"] == "scratch":
        return f"scratch_L{r['layers']}"
    return f"grown_{r['growth']}_dt{r['donor_tokens']}"


def main():
    recs = [json.loads(open(f).read()) for f in glob.glob(str(HERE / "runs/*.json"))]
    if not recs:
        print("no records")
        return
    by_arm = collections.defaultdict(list)
    for r in recs:
        by_arm[label(r)].append(r)

    print(f"{len(recs)} records, {len(by_arm)} arms\n")

    # The comparison baseline is scratch at the GROWN size (L=8): growth is
    # only a win if it beats training the final architecture from scratch.
    base = by_arm.get("scratch_L8", [])
    if not base:
        print("WARNING: scratch_L8 missing — savings are not computable yet.")
    base_curves = [r["curve"] for r in base]

    # Loss horizons must lie in the band EVERY arm actually traverses:
    #   lower edge = the highest "best loss" any arm reaches (below this,
    #                some arm never gets there, so savings are undefined);
    #   upper edge = the lowest starting loss any arm has (above this, some
    #                arm was already past it at its first eval, so its
    #                crossing is unresolved rather than early).
    # Grown runs inherit a trained donor and therefore START low, which
    # makes this band narrow by construction. Its width is an instrument
    # limit and is reported, not hidden.
    band_lo = max(r["curve"][-1]["loss"] for r in recs)
    band_hi = min(r["curve"][0]["loss"] for r in base) if base else \
        min(r["curve"][0]["loss"] for r in recs)
    print(f"comparable loss band: [{band_lo:.4f}, {band_hi:.4f}] "
          f"(width {band_hi - band_lo:.4f})")
    # A grown arm may already be BELOW a horizon at its first eval (it
    # inherits a trained donor). Its crossing is then unresolved-early, so
    # its savings is a LOWER BOUND, flagged with '>' rather than dropped.
    early = {arm: sum(1 for r in rs if r["curve"][0]["loss"] <= band_lo)
             for arm, rs in by_arm.items()}
    if any(early.values()):
        names = [a for a, n in early.items() if n]
        print(f"  NOTE: {len(names)} arm(s) start at or below the lower edge "
              f"({', '.join(sorted(names))}). Their crossings happened before "
              f"the first eval, so their savings are LOWER BOUNDS, marked '>'. "
              f"Resolving them needs evals during the first {band_lo:.3f}-nat "
              f"descent, which this grid does not have.")
    if band_hi <= band_lo:
        print("  BAND EMPTY even against the scratch_L8 baseline: every arm, "
              "including the baseline, starts below the hardest commonly "
              "reached loss. Savings are not computable at any horizon.")
        horizons = []
    else:
        horizons = [round(band_lo + i * (band_hi - band_lo) / 4, 4)
                    for i in (3, 2, 1)]  # early (high loss) -> late (low loss)
        if band_hi - band_lo < 0.05:
            print("  WARNING: band narrower than 0.05 nats. The three horizons "
                  "are nearly the same point, so this yields ONE savings "
                  "number, not a decay CURVE. Do not read a decay shape off "
                  "it. Fix by extending the scratch baselines or evaluating "
                  "the grown arms from an earlier checkpoint.")
        print(f"loss horizons (early->late): {horizons}")
    print()

    # Exponent fits are restricted to a COMMON COMPUTE BAND. Fitting each
    # family over its own full curve compares slopes measured at different
    # points of the same decelerating descent: a grown model starts from a
    # trained donor, so it begins mid-curve where the slope is already flat,
    # and would show a "smaller exponent" purely from where it sits. That is
    # a confound, not a scaling result.
    # Comparison set = the arms G2 actually contrasts: the grown family and
    # the SAME-SIZE scratch baseline. scratch_L4 is the donor-scale
    # reference and ends at far lower total compute, so including it would
    # empty the band for a reason unrelated to growth.
    cmp_set = [r for r in recs if r["kind"] == "grown" or r["layers"] == 8]
    fit_lo = max(r["curve"][0]["flops"] for r in cmp_set) if cmp_set else 0
    fit_hi = min(r["curve"][-1]["flops"] for r in cmp_set) if cmp_set else 0
    print(f"common compute band for exponent fits: "
          f"[{fit_lo:.3g}, {fit_hi:.3g}] FLOPs")
    if fit_hi <= fit_lo:
        print("  BAND EMPTY: exponents are NOT comparable across families "
              "(no shared compute range). Reporting per-family fits only, "
              "flagged as non-comparable.\n")
    else:
        span = fit_hi / fit_lo
        print(f"  ({span:.1f}x of compute range shared)")
        if span < 3.0:
            print("  EXPONENTS NOT ANSWERABLE: a power-law slope fitted over "
                  f"{span:.1f}x of x-range is dominated by curvature and noise, "
                  "not by the exponent. The alphas below are reported for the "
                  "record and MUST NOT be compared across families. G1-vs-G2 "
                  "(exponent change vs prefactor change) cannot be decided by "
                  "this grid; deciding it needs the grown and scratch families "
                  "to share at least ~10x of compute, i.e. longer scratch runs "
                  "or an earlier growth point.")
        print()

    def in_band(pt):
        return fit_hi > fit_lo and fit_lo <= pt["flops"] <= fit_hi

    hdr = f"{'arm':<28} {'n':>2} {'alpha':>7} {'logA':>7} " + \
          " ".join(f"{'sav@' + str(h):>12}" for h in horizons)
    print(hdr)
    rows = []
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        banded = [(pt["flops"], pt["loss"]) for r in rs for pt in r["curve"]
                  if in_band(pt)]
        pts = banded if len(banded) >= 3 else \
            [(pt["flops"], pt["loss"]) for r in rs for pt in r["curve"]]
        alpha, logA = fit_power_law(pts)
        savings = []
        for h in horizons:
            if not base_curves:
                savings.append(None)
                continue
            b = [x for x in (flops_at_loss(c, h) for c in base_curves) if x]
            g = [x for x in (flops_at_loss(r["curve"], h) for r in rs) if x]
            if not b or not g:
                savings.append(None)
                continue
            bound = any(r["curve"][0]["loss"] <= h for r in rs)
            savings.append((statistics.median(b) / statistics.median(g) - 1.0, bound))
        f = lambda v, w, p: (f"{v:>{w}.{p}f}" if v is not None else f"{'--':>{w}}")

        def fs(s):
            if s is None:
                return f"{'--':>12}"
            val, bound = s
            return f"{('>' if bound else '') + format(val, '.3f'):>12}"
        print(f"{arm:<28} {len(rs):>2} {f(alpha,7,4)} {f(logA,7,3)} " +
              " ".join(fs(s) for s in savings))
        rows.append(dict(arm=arm, n=len(rs), alpha=alpha, logA=logA,
                         savings=[None if s is None else s[0] for s in savings],
                         savings_bound=[None if s is None else s[1] for s in savings],
                         donor=rs[0]["donor_tokens"],
                         growth=rs[0]["growth"], kind=rs[0]["kind"]))

    print("\n(savings > 0 means growth reached that loss with FEWER total "
          "FLOPs than scratch_L8, donor cost included)")

    # Censoring census. A '>' cell means the arm was already past that loss at
    # its first eval, so its crossing is unmeasured and the number is only a
    # lower bound. Two arms with the same donor budget then report IDENTICAL
    # censored values (both fall back to the same first-eval FLOPs) — this is
    # the censoring, NOT the e006-style "arms are secretly identical" bug. The
    # discriminator: uncensored cells differ (see dt=1M at the late horizon).
    cells = [(r["arm"], i, v, b) for r in rows
             for i, (v, b) in enumerate(zip(r["savings"], r["savings_bound"]))
             if v is not None]
    grown_cells = [c for c in cells if c[0].startswith("grown")]
    censored = [c for c in grown_cells if c[3]]
    if grown_cells:
        pct = 100 * len(censored) / len(grown_cells)
        print(f"\nCENSORING CENSUS: {len(censored)}/{len(grown_cells)} grown "
              f"savings cells ({pct:.0f}%) are lower bounds, not measurements.")
        if pct > 50:
            print("  The savings measurement is therefore MOSTLY UNRESOLVED. "
                  "Grown arms inherit a trained donor and are already below "
                  "most horizons at their first eval, so this grid cannot see "
                  "where they crossed. This is a DESIGN fix, not a "
                  "more-seeds fix: evaluate grown arms during their first "
                  "descent (dense early evals right after the growth step), "
                  "or compare at matched compute rather than matched loss. "
                  "Do not report a savings-decay curve from this table.")

    # MATCHED-COMPUTE COMPARISON (censoring-free).
    # Matched-LOSS savings need a crossing point, and grown arms cross before
    # their first eval, so 89% of those cells are censored. Matched COMPUTE
    # needs no crossing: every curve has a loss at every compute level it
    # covers. Inside the shared band we can therefore ask the same question
    # without censoring — at equal total FLOPs (donor charged), who is ahead?
    if fit_hi > fit_lo and base_curves:
        def loss_at_flops(curve, c):
            prev = None
            for pt in curve:
                if pt["flops"] >= c:
                    if prev is None:
                        return pt["loss"] if pt["flops"] == c else None
                    f = (math.log(c) - math.log(prev["flops"])) / \
                        (math.log(pt["flops"]) - math.log(prev["flops"]))
                    return prev["loss"] + f * (pt["loss"] - prev["loss"])
                prev = pt
            return None

        probes = [fit_lo * (fit_hi / fit_lo) ** (i / 4) for i in range(5)]
        print("\nMATCHED-COMPUTE comparison (censoring-free, inside the shared "
              "band). Negative delta = grown arm has LOWER loss than "
              "scratch_L8 at the same total FLOPs:")
        print(f"  {'arm':<30} " + " ".join(f"{c:>10.3g}" for c in probes))
        base_at = [statistics.median([x for x in
                                      (loss_at_flops(c, p) for c in base_curves)
                                      if x is not None] or [float("nan")])
                   for p in probes]
        print(f"  {'scratch_L8 (baseline loss)':<30} " +
              " ".join(f"{b:>10.4f}" for b in base_at))
        for arm in sorted(by_arm):
            if not arm.startswith("grown"):
                continue
            rs = by_arm[arm]
            cells = []
            for p, b in zip(probes, base_at):
                vals = [x for x in (loss_at_flops(r["curve"], p) for r in rs)
                        if x is not None]
                if not vals:
                    cells.append(None)
                    continue
                sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
                cells.append((statistics.median(vals) - b, sd))
            print(f"  {arm:<30} " +
                  " ".join(f"{v[0]:>+7.4f}+-{v[1]:.4f}" if v is not None
                           else f"{'--':>15}" for v in cells))
        print("  (this is the comparison the matched-loss table cannot make; "
              "it is narrow — only 1.2x of compute — but it is uncensored)")

    # (b) exponent vs prefactor: G1/G2 discriminator
    sc = next((r for r in rows if r["arm"] == "scratch_L8"), None)
    if sc and sc["alpha"]:
        print(f"\nexponent check (G2 predicts UNCHANGED exponent, G1 a change):")
        print(f"  scratch_L8 alpha = {sc['alpha']:.4f}")
        for r in rows:
            if r["kind"] == "grown" and r["alpha"]:
                d = r["alpha"] - sc["alpha"]
                print(f"  {r['arm']:<28} alpha = {r['alpha']:.4f}  (delta {d:+.4f})")
        print("  NOTE: per-arm alphas are pooled-point fits, not per-seed; treat"
              " deltas as descriptive until the per-seed CI below separates them.")

    # per-seed alpha CIs where seeds allow
    print("\nper-seed exponent CIs (bootstrap over seeds):")
    for arm in sorted(by_arm):
        rs = by_arm[arm]
        alphas = []
        for r in rs:
            banded = [(pt["flops"], pt["loss"]) for pt in r["curve"] if in_band(pt)]
            a, _ = fit_power_law(banded if len(banded) >= 3 else
                                 [(pt["flops"], pt["loss"]) for pt in r["curve"]])
            if a:
                alphas.append(a)
        if len(alphas) >= 2:
            lo, hi = bootstrap_ci(alphas)
            print(f"  {arm:<28} alpha {statistics.mean(alphas):.4f} "
                  f"[{lo:.4f}, {hi:.4f}]  (n={len(alphas)} seeds)")

    # (c) G3: does the donor-convergence sweep move or flip savings?
    print("\ndonor-convergence sweep (G3: flat => growth mechanics; "
          "sign flip => regime artifact):")
    for growth in ("copy", "shrink_perturb"):
        line = [r for r in rows if r["growth"] == growth]
        line.sort(key=lambda r: r["donor"])
        if line:
            cells = "  ".join(
                f"dt={r['donor']//1000}k:" +
                (f"{r['savings'][-1]:+.3f}" if r["savings"] and r["savings"][-1] is not None else "--")
                for r in line)
            print(f"  {growth:<16} (late horizon) {cells}")

    art = HERE / "artifacts"
    art.mkdir(exist_ok=True)
    with open(art / "results.csv", "w") as fh:
        fh.write("arm,kind,growth,donor_tokens,seeds,alpha,logA," +
                 ",".join(f"savings_at_loss_{h}" for h in horizons) + "\n")
        for r in rows:
            g = lambda v: ("" if v is None else f"{v:.6f}")
            fh.write(f"{r['arm']},{r['kind']},{r['growth']},{r['donor']},{r['n']},"
                     f"{g(r['alpha'])},{g(r['logA'])}," +
                     ",".join(g(s) for s in r["savings"]) + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for arm in sorted(by_arm):
            rs = by_arm[arm]
            xs = [pt["flops"] for pt in rs[0]["curve"]]
            ys = [statistics.median([r["curve"][i]["loss"] for r in rs])
                  for i in range(len(rs[0]["curve"]))]
            style = "--" if rs[0]["kind"] == "scratch" else "-"
            axes[0].plot(xs, ys, style, marker=".", label=arm)
        axes[0].set_xscale("log"); axes[0].set_yscale("log")
        axes[0].set_xlabel("total FLOPs (donor charged)")
        axes[0].set_ylabel("loss")
        axes[0].set_title("loss vs total compute")
        axes[0].legend(fontsize=7); axes[0].grid(alpha=0.3)

        for growth, mark in (("copy", "o"), ("shrink_perturb", "s")):
            line = sorted([r for r in rows if r["growth"] == growth],
                          key=lambda r: r["donor"])
            for i, h in enumerate(horizons):
                xs = [r["donor"] for r in line if r["savings"][i] is not None]
                ys = [r["savings"][i] for r in line if r["savings"][i] is not None]
                if xs:
                    axes[1].plot(xs, ys, marker=mark, alpha=0.5 + 0.25 * i,
                                 label=f"{growth} @loss {h}")
        axes[1].axhline(0, c="k", lw=1)
        axes[1].set_xscale("log")
        axes[1].set_xlabel("donor tokens")
        axes[1].set_ylabel("savings vs scratch_L8")
        axes[1].set_title("savings-decay + donor-convergence sweep")
        axes[1].legend(fontsize=7); axes[1].grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(art / "results.png", dpi=150)
        print("\nwrote artifacts/results.csv artifacts/results.png")
    except ImportError:
        print("\nwrote artifacts/results.csv (no matplotlib)")


if __name__ == "__main__":
    main()
