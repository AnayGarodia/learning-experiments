"""The overlay: oracle-skipped compute against the dense inference
frontier, and the horizontal gap at matched capability (the multiplier).

Verdict rule is pre-registered in DESIGN.md: M_lower >= 2.0 kills T2;
M_upper <= 1.6 means T2 survives and adaptive compute dies as a lever;
anything else is INCONCLUSIVE, as is disagreement between the two
primary capability axes.

The rule is applied to the FIT-FREE BRACKET on M (verifier ruling,
DESIGN.md): F_frontier(C_oracle) is bounded by the last measured
frontier rung under C_oracle and the first at or over it, and every
monotone fit lies inside. Thresholds, statistic and the max-over-rungs
headline are unchanged from pre-registration; only the freedom to
choose a fit is removed, which can make a verdict harder to call but
never easier. Three point fits are still reported, the originally
pre-registered global log-linear among them, flagged wherever it falls
outside the bracket (i.e. wherever it contradicts a measured rung).

Capability axes: next-token accuracy and consequence accuracy, the two
metrics the argmax criterion preserves by construction. LOSS IS NOT
PRESERVED by that criterion and is plotted only as a diagnostic of what
argmax-matching discards; no verdict may be read off it.

Bootstrap uses common random sequences: one resample of eval-sequence
indices is applied to every point at a training level at once, so the
frontier and the oracle points move together and the CI contains only
the noise the comparison actually carries.

Usage: python analyze.py
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
LADDER = ["xxs", "xs0", "xs1", "xs", "s", "m", "l", "xl"]
AXES = {"acc": "next-token accuracy", "cons": "consequence accuracy"}
ORACLES = ["depth_arg", "joint_arg", "depth_kl", "joint_kl"]
KILL_T2, SURVIVE_T2 = 2.0, 1.6
NBOOT = 1000



def cell(rec):
    """The analysis unit is (training level, SEED), never the level alone.

    Two seeds at one level are two independent draws of the training recipe.
    Pooling them into one Pareto envelope lets a lucky seed at a small rung
    undercut the frontier and call it a result — DESIGN's uninterpretable
    condition 4 ("the Pareto envelope is measuring recipe noise"), and
    decision 0012's corollary that per-seed values are reported, never means.
    So every frontier, every multiplier and every gate is computed WITHIN a
    seed, and the verdict combines seeds conservatively.
    """
    return (rec["tokens"], rec["seed"])


def clabel(c):
    return f"{c[0]:,} tok seed{c[1]}"


def load():
    recs = [json.loads(Path(f).read_text()) for f in glob.glob(str(HERE / "runs/*.json"))]
    if not recs:
        raise SystemExit("no runs/*.json — run overlay.py first")
    # convert the per-sequence lists once; the bootstrap indexes them ~10^6
    # times and np.asarray on a Python list dominates the runtime otherwise
    for r in recs:
        r["cons_n_per_seq"] = np.asarray(r["cons_n_per_seq"])
        for p in r["points"].values():
            for k in ("acc_hit", "cons_hit", "loss_sum", "agree_hit",
                      "skip_frac_per_seq"):
                p[k] = np.asarray(p[k])
    return sorted(recs, key=lambda r: (r["tokens"], LADDER.index(r["size"])))


def stats(rec, point, axis, idx):
    """Capability of one point on `axis`, over resampled sequences `idx`."""
    p = rec["points"][point]
    if axis == "acc":
        hit = p["acc_hit"][idx]
        n = np.full(len(idx), rec["n_pred_per_seq"])
    elif axis == "cons":
        hit = p["cons_hit"][idx]
        n = rec["cons_n_per_seq"][idx]
    else:  # loss: diagnostic only, lower is better
        return p["loss_sum"][idx].sum() / (len(idx) * rec["n_pred_per_seq"])
    return hit.sum() / max(n.sum(), 1)


def eff_flops(rec, point, idx):
    sk = rec["points"][point]["skip_frac_per_seq"][idx].mean()
    return rec["flops_head"] + (1.0 - sk) * rec["flops_block"]


def eff_flops_kv(rec, point, idx):
    """Same, but charging the KV FLOOR a skipped (token, block) still owes.

    A skipped pass is NOT free: later positions attend to that position at
    that block, so its keys and values must exist. CALM's state copy pays
    the kv projection off the passthrough state; temporal reuse pays it off
    the reused state (DESIGN already flags temporal reuse as 'FLOPs-free but
    KV-full'). Cost is 2*d*(2d) = 4 d^2 per skipped (token, block).

    d is recovered from the head term (head = 2 d V, V = 5000) so this needs
    no torch import. Reported as a DIAGNOSTIC column, never as the verdict
    statistic — the verdict statistic is the pre-registered zero-charge one.
    It moves M DOWN, i.e. away from a kill, so it cannot rescue a verdict.
    """
    sk = rec["points"][point]["skip_frac_per_seq"][idx].mean()
    d = rec["flops_head"] / (2 * 5000)
    return (rec["flops_head"] + (1.0 - sk) * rec["flops_block"]
            + sk * 4 * d * d * rec["L"])


def frontier(points):
    """Pareto envelope: (flops, capability) sorted by flops, keeping only
    rungs no cheaper rung already matches."""
    pts = sorted(points, key=lambda t: t[0])
    out = []
    best = -np.inf
    for f, c in pts:
        if c > best:
            out.append((f, c))
            best = c
    return out


def frontier_flops(front, c, fit="pw"):
    """Dense FLOPs/token needed for capability c. Returns (value, flag)."""
    fs = np.array([f for f, _ in front], dtype=float)
    cs = np.array([c_ for _, c_ in front], dtype=float)
    if len(front) == 1:
        return fs[0], "single-point-frontier"
    if c > cs[-1]:
        return fs[-1], "above-support"   # lower bound on F_frontier
    if c < cs[0]:
        return fs[0], "below-support"    # upper bound on F_frontier
    if fit == "pw":  # piecewise-linear in (log10 F, capability)
        return 10 ** np.interp(c, cs, np.log10(fs)), "ok"
    if fit == "logit":  # piecewise-linear in (log10 F, logit capability)
        def lg(v):
            v = np.clip(v, 1e-4, 1 - 1e-4)
            return np.log(v / (1 - v))
        return 10 ** np.interp(lg(c), lg(cs), np.log10(fs)), "ok"
    b, a = np.polyfit(np.log10(fs), cs, 1)  # cap = a + b*log10(F); diagnostic
    if abs(b) < 1e-12:
        return fs[0], "flat-fit"
    return 10 ** ((c - a) / b), "ok"


def frontier_bracket(front, c):
    """FIT-FREE bounds on F_frontier(c): [last frontier rung below c, first
    frontier rung at or above c]. Every monotone fit lies inside this
    interval by construction, so a verdict whose bracket sits wholly on one
    side of a threshold cannot be changed by the choice of fit.

    Out of support falls out for free and correctly: above support gives
    (F_max, inf) — a valid lower bound, so KILLED stays callable and
    SURVIVES does not; below support gives (0, F_min) — the mirror image.
    """
    fs = np.array([f for f, _ in front], dtype=float)
    cs = np.array([c_ for _, c_ in front], dtype=float)
    if c > cs[-1]:
        return fs[-1], np.inf
    if c < cs[0]:
        return 0.0, fs[0]
    j = int(np.searchsorted(cs, c))
    return (fs[j - 1] if j > 0 else 0.0), fs[j]


def level_multipliers(recs_at_level, axis, idx, fit="pw"):
    """M = F_frontier(C_oracle) / F_oracle, per rung per oracle config."""
    dense = [(eff_flops(r, "full", idx), stats(r, "full", axis, idx))
             for r in recs_at_level]
    front = frontier(dense)
    out = {}
    for r in recs_at_level:
        for o in ORACLES:
            fo = eff_flops(r, o, idx)
            co = stats(r, o, axis, idx)
            ff, flag = frontier_flops(front, co, fit)
            out[(r["size"], o)] = (ff / fo, flag, fo, co)
    return out, front


def level_brackets(recs_at_level, axis, idx):
    """Fit-free [M_lo, M_hi] per (rung, oracle) at one training level."""
    dense = [(eff_flops(r, "full", idx), stats(r, "full", axis, idx))
             for r in recs_at_level]
    front = frontier(dense)
    out = {}
    for r in recs_at_level:
        for o in ORACLES:
            fo = eff_flops(r, o, idx)
            lo, hi = frontier_bracket(front, stats(r, o, axis, idx))
            out[(r["size"], o)] = (lo / fo, hi / fo)
    return out


_BOOT_CACHE = {}
_BRACKET_CACHE = {}


def bootstrap_bracket(recs_at_level, axis):
    """CI on the fit-free bracket: 2.5th pct of M_lo, 97.5th pct of M_hi."""
    key = (cell(recs_at_level[0]), axis)
    if key in _BRACKET_CACHE:
        return _BRACKET_CACHE[key]
    n = recs_at_level[0]["eval_seqs"]
    rng = np.random.default_rng(0)  # same common random sequences as bootstrap()
    keys = [(r["size"], o) for r in recs_at_level for o in ORACLES]
    los = {k: [] for k in keys}
    his = {k: [] for k in keys}
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)
        b = level_brackets(recs_at_level, axis, idx)
        for k in keys:
            los[k].append(b[k][0])
            his[k].append(b[k][1])
    out = {k: (float(np.percentile(los[k], 2.5)),
               float(np.percentile(np.clip(his[k], 0, 1e12), 97.5))
               if np.isfinite(his[k]).all() else np.inf) for k in keys}
    _BRACKET_CACHE[key] = out
    return out


def bootstrap(recs_at_level, axis, fit="pw"):
    key = (cell(recs_at_level[0]), axis, fit)
    if key in _BOOT_CACHE:
        return _BOOT_CACHE[key]
    n = recs_at_level[0]["eval_seqs"]
    rng = np.random.default_rng(0)  # same resamples for every axis and fit
    keys = [(r["size"], o) for r in recs_at_level for o in ORACLES]
    draws = {k: [] for k in keys}
    caps = {r["size"]: [] for r in recs_at_level}
    for _ in range(NBOOT):
        idx = rng.integers(0, n, n)  # common random sequences
        m, _ = level_multipliers(recs_at_level, axis, idx, fit)
        for k in keys:
            draws[k].append(m[k][0])
        for r in recs_at_level:
            caps[r["size"]].append(stats(r, "full", axis, idx))
    ci = {k: tuple(np.percentile(v, [2.5, 97.5])) for k, v in draws.items()}
    cap_ci = {s: (np.percentile(v, 2.5), np.percentile(v, 97.5))
              for s, v in caps.items()}
    _BOOT_CACHE[key] = (ci, cap_ci)
    return ci, cap_ci


def gate(recs_at_level, axis, cap_ci):
    """Dynamic-range gate: some adjacent-rung capability gap must exceed
    5x the bootstrap CI half-width, else the frontier is degenerate."""
    full = np.arange(recs_at_level[0]["eval_seqs"])
    caps = [stats(r, "full", axis, full) for r in recs_at_level]
    half = max((hi - lo) / 2 for lo, hi in cap_ci.values())
    gaps = [abs(caps[i + 1] - caps[i]) for i in range(len(caps) - 1)]
    biggest = max(gaps) if gaps else 0.0
    return biggest >= 5 * half, biggest, half


def verdict(m_lo, m_hi):
    if m_lo >= KILL_T2:
        return "T2 KILLED (adaptivity premium real)"
    if m_hi <= SURVIVE_T2:
        return "T2 SURVIVES (adaptive compute dies as a lever)"
    return "INCONCLUSIVE"


def main():
    recs = load()
    by_level = {}
    for r in recs:
        by_level.setdefault(cell(r), []).append(r)
    for v in by_level.values():
        v.sort(key=lambda r: LADDER.index(r["size"]))
    levels = sorted(by_level)          # (tokens, seed) cells
    tok_levels = sorted({t for t, _ in levels})
    full_idx = {t: np.arange(by_level[t][0]["eval_seqs"]) for t in levels}

    # ---- instrument checks first (they can invalidate everything below)
    print("=== instrument: e009 cross-check on shared checkpoints ===")
    e009_runs = HERE.parent / "e009-redundancy" / "runs"
    for r in recs:
        f = e009_runs / f"{r['size']}_s{r['seed']}_t{r['tokens']}.json"
        if f.exists():
            o = json.loads(f.read_text())
            c = r["e009_cmp"]
            print(f"  {r['size']:4} t={r['tokens']:>8} depth_arg {c['depth_frac_arg']:.4f} "
                  f"vs e009 {o['depth_frac_arg']:.4f} | joint {c['joint_block_arg']:.4f} "
                  f"vs {o['joint_block_arg']:.4f} | temporal {c['temporal_arg']:.4f} "
                  f"vs {o['temporal_arg']:.4f}")

    # Controls: the joint arm executes BETTER than depth-only despite
    # skipping more. That is only credible if matched-density controls
    # score BELOW depth-only. Violation means the masking is suspect.
    print("\n=== instrument: control ordering (must be random < wrong-neighbour"
          " < depth-only) ===")
    for r in recs:
        p = r["points"]
        if "ctl_random_mask" not in p:
            continue
        n = r["eval_seqs"] * r["eval_seq_len"]
        a = {k: p[k]["agree_hit"].sum() / n
             for k in ("ctl_random_mask", "ctl_wrong_neighbour", "depth_arg",
                       "joint_arg")}
        ok = a["ctl_random_mask"] < a["ctl_wrong_neighbour"] < a["depth_arg"]
        print(f"  {r['size']:4} t={r['tokens']:>8} random {a['ctl_random_mask']:.4f} "
              f"< wrong-neighbour {a['ctl_wrong_neighbour']:.4f} "
              f"< depth {a['depth_arg']:.4f} | joint {a['joint_arg']:.4f}  "
              f"{'OK' if ok else '*** ORDERING VIOLATED — masking suspect ***'}")

    print("\n=== dense ladder: capability and FLOPs per training level ===")
    gates = {}
    for t in levels:
        rl = by_level[t]
        print(f"\n-- {clabel(t)}")
        print(f"  {'rung':4} {'FLOPs/tok':>11} {'acc':>7} {'cons':>7} {'loss':>7} "
              f"{'joint skip':>10} {'eff FLOPs':>11} {'agree':>7}")
        for r in rl:
            i = full_idx[t]
            jf = eff_flops(r, "joint_arg", i)
            ag = r["points"]["joint_arg"]["agree_hit"].sum() / (
                r["eval_seqs"] * r["eval_seq_len"])
            print(f"  {r['size']:4} {r['flops_full']:>11,} "
                  f"{stats(r,'full','acc',i):>7.4f} {stats(r,'full','cons',i):>7.4f} "
                  f"{stats(r,'full','loss',i):>7.4f} "
                  f"{r['points']['joint_arg']['skip_frac']:>10.4f} {jf:>11,.0f} "
                  f"{ag:>7.4f}")
        for axis in AXES:
            ok, biggest, half = gate(rl, axis, bootstrap(rl, axis)[1])
            gates[(t, axis)] = (ok, biggest, half)
            print(f"  dynamic range [{axis}]: max adjacent gap {biggest:.4f} vs "
                  f"CI half-width {half:.4f} -> gap/CI {biggest/max(half,1e-9):.1f}x, "
                  f"gate {'PASS' if ok else 'FAIL (degenerate frontier)'}")

    # ---- the overlay and the multipliers
    print("\n=== overlay: multiplier M = F_frontier(C_oracle) / F_oracle ===")
    results = {}
    for t in levels:
        rl = by_level[t]
        for axis in AXES:
            for fit in ("pw", "logit", "loglin"):
                m_pw, front = level_multipliers(rl, axis, full_idx[t], fit)
                ci, _ = bootstrap(rl, axis, fit)
                for k, (M, flag, fo, co) in m_pw.items():
                    results[(t, axis, fit, k)] = (M, ci[k], flag, fo, co)
        print(f"\n-- {clabel(t)}   (frontier rungs: "
              f"{[f'{f/1e6:.2f}M' for f, _ in frontier([(eff_flops(r,'full',full_idx[t]), stats(r,'full','acc',full_idx[t])) for r in rl])]})")
        bkf = {a: level_brackets(rl, a, full_idx[t]) for a in AXES}
        print(f"  {'rung':4} {'oracle':10} {'axis':5} {'eff FLOPs':>11} {'cap':>7} "
              f"{'fit-free M':>14} {'M(pw)':>7} {'M(logit)':>8} {'M(loglin)':>9} "
              f"{'kvM':>6} {'flag':>16}")
        for r in rl:
            for o in ORACLES:
                for axis in AXES:
                    M, ci, flag, fo, co = results[(t, axis, "pw", (r["size"], o))]
                    M2 = results[(t, axis, "logit", (r["size"], o))][0]
                    M3 = results[(t, axis, "loglin", (r["size"], o))][0]
                    blo, bhi = bkf[axis][(r["size"], o)]
                    # KV-floor diagnostic: same M, honest FLOPs denominator
                    kvM = M * fo / eff_flops_kv(r, o, full_idx[t])
                    bad = "" if blo <= M3 <= bhi else " LOGLIN-OUT-OF-BRACKET"
                    print(f"  {r['size']:4} {o:10} {axis:5} {fo:>11,.0f} {co:>7.4f} "
                          f"[{blo:>5.2f},{bhi:>6.2f}] {M:>7.2f} {M2:>8.2f} "
                          f"{M3:>9.2f} {kvM:>6.2f} {flag:>16}{bad}")

    # ---- primary level and verdict, per the pre-registered rule
    print("\n=== verdict (pre-registered rule, DESIGN.md) ===")
    verdicts = {}
    for axis in AXES:
        # a token-level qualifies only if the gate passes on EVERY seed
        qualifying = [tk for tk in tok_levels
                      if all(gates[(c, axis)][0] for c in levels if c[0] == tk)]
        if not qualifying:
            print(f"  [{axis}] no training level passes the dynamic-range gate "
                  f"on every seed -> UNINTERPRETABLE on this corpus")
            verdicts[axis] = ("UNINTERPRETABLE", None, None, None)
            continue
        tok = max(qualifying)
        # The verdict runs on the FIT-FREE bracket (verifier ruling,
        # 2026-08-16). M is bounded below by the last dense frontier rung
        # under C_oracle and above by the first rung at or over it; every
        # monotone fit lies inside. Thresholds, statistic and "max over
        # rungs" headline are unchanged from pre-registration — only the
        # fit-choice freedom is removed, and removing it can only make a
        # verdict HARDER to call, never easier. Point fits are reported.
        #
        # Seeds combine CONSERVATIVELY and per-seed, never by averaging:
        # a kill must hold on the WORST seed, a survive must hold on the
        # worst seed too. Both directions therefore need seed agreement.
        per_seed = {}
        for c in [c for c in levels if c[0] == tok]:
            bk = bootstrap_bracket(by_level[c], axis)
            cand = [(bk[(r["size"], "joint_arg")], r["size"]) for r in by_level[c]]
            (klo, _), krung = max(cand, key=lambda x: x[0][0])  # best case for KILL
            (_, shi), srung = max(cand, key=lambda x: x[0][1])  # worst for SURVIVE
            rs = next(r for r in by_level[c] if r["size"] == srung)
            kv = shi * (eff_flops(rs, "joint_arg", full_idx[c])
                        / eff_flops_kv(rs, "joint_arg", full_idx[c]))
            per_seed[c] = (klo, krung, shi, srung, kv)
            print(f"  [{axis}] {clabel(c)}: KILL-side max lower bound "
                  f"{klo:.2f} ({krung}), SURVIVE-side max upper bound "
                  f"{shi:.2f} ({srung}), KV-floor SURVIVE-side {kv:.2f} | "
                  f"point fits at {srung}: pw "
                  f"{results[(c, axis, 'pw', (srung, 'joint_arg'))][0]:.2f} "
                  f"logit {results[(c, axis, 'logit', (srung, 'joint_arg'))][0]:.2f} "
                  f"loglin {results[(c, axis, 'loglin', (srung, 'joint_arg'))][0]:.2f}")
        klo = min(p[0] for p in per_seed.values())   # kill must hold on worst seed
        shi = max(p[2] for p in per_seed.values())   # survive must hold on worst
        kv = max(p[4] for p in per_seed.values())
        if klo >= KILL_T2:
            v = "T2 KILLED (adaptivity premium real)"
        elif shi <= SURVIVE_T2 and kv <= SURVIVE_T2:
            v = "T2 SURVIVES (adaptive compute dies as a lever)"
        elif shi <= SURVIVE_T2 or kv <= SURVIVE_T2:
            v = (f"INCONCLUSIVE (the two FLOPs accountings disagree: "
                 f"zero-charge {shi:.2f}, KV-floor {kv:.2f}, against "
                 f"{SURVIVE_T2})")
        else:
            v = (f"INCONCLUSIVE (fit-free bracket straddles: worst-seed "
                 f"KILL lower bound {klo:.2f} < {KILL_T2}, worst-seed "
                 f"SURVIVE upper bound {shi:.2f} > {SURVIVE_T2})")
        c0 = min(per_seed)
        srung0 = per_seed[c0][3]
        print(f"  [{axis}] PRIMARY LEVEL {tok:,} tokens, both seeds combined "
              f"conservatively -> KILL {klo:.2f} / SURVIVE {shi:.2f} "
              f"(KV-floor {kv:.2f}) -> {v}")
        verdicts[axis] = (v, c0, srung0,
                          results[(c0, axis, "pw", (srung0, "joint_arg"))][:2])
    vs = {v[0] for v in verdicts.values()}
    combined = vs.pop() if len(vs) == 1 else "INCONCLUSIVE (primary axes disagree)"
    print(f"  COMBINED: {combined}")

    write_artifacts(recs, by_level, levels, full_idx, results, gates, verdicts, combined)


def write_artifacts(recs, by_level, levels, full_idx, results, gates, verdicts, combined):
    ART.mkdir(exist_ok=True)
    cols = ["size", "seed", "tokens", "nonemb", "flops_full", "flops_head",
            "config", "skip_frac", "flops_eff", "flops_eff_kvfloor",
            "M_acc_bracket_lo", "M_acc_bracket_hi",
            "M_cons_bracket_lo", "M_cons_bracket_hi",
            "acc", "cons", "loss",
            "argmax_agreement", "M_acc_pw", "M_acc_lo", "M_acc_hi",
            "M_cons_pw", "M_cons_lo", "M_cons_hi", "M_acc_logit",
            "M_cons_logit", "M_acc_loglin_diag", "flag"]
    brack = {(t, a): level_brackets(by_level[t], a, full_idx[t])
             for t in levels for a in AXES}
    with open(ART / "results.csv", "w") as fh:
        fh.write(",".join(cols) + "\n")
        for r in recs:
            cl = cell(r)
            i = full_idx[cl]
            for cfg in ["full"] + ORACLES:
                p = r["points"][cfg]
                ag = p["agree_hit"].sum() / (r["eval_seqs"] * r["eval_seq_len"])
                key = (cl, "acc", "pw", (r["size"], cfg))
                if key in results:
                    M, ci, flag, _, _ = results[key]
                    Mc, cic, _, _, _ = results[(cl, "cons", "pw", (r["size"], cfg))]
                    M2 = results[(cl, "acc", "logit", (r["size"], cfg))][0]
                    M2c = results[(cl, "cons", "logit", (r["size"], cfg))][0]
                    M3 = results[(cl, "acc", "loglin", (r["size"], cfg))][0]
                    extra = [M, ci[0], ci[1], Mc, cic[0], cic[1], M2, M2c, M3, flag]
                else:
                    extra = [""] * 9 + ["dense-reference"]
                bk = [round(v, 4) for a in AXES
                      for v in brack[(cl, a)].get((r["size"], cfg), ("", ""))
                      ] if cfg != "full" else ["", "", "", ""]
                fh.write(",".join(str(v) for v in [
                    r["size"], r["seed"], r["tokens"], r["nonemb"], r["flops_full"],
                    r["flops_head"], cfg, p["skip_frac"], round(eff_flops(r, cfg, i), 1),
                    round(eff_flops_kv(r, cfg, i), 1), *bk,
                    round(stats(r, "full" if cfg == "full" else cfg, "acc", i), 5),
                    round(stats(r, "full" if cfg == "full" else cfg, "cons", i), 5),
                    round(stats(r, "full" if cfg == "full" else cfg, "loss", i), 5),
                    round(ag, 5)] + [round(v, 4) if isinstance(v, float) else v
                                     for v in extra]) + "\n")
    print(f"\nwrote {ART/'results.csv'}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("no matplotlib — skipped results.png")
    else:
        prim = verdicts["cons"][1] if verdicts["cons"][1] else levels[0]
        other = levels[-1] if levels[-1] != prim else levels[0]
        show = [prim] if other == prim else [prim, other]
        fig, axes = plt.subplots(2, 2, figsize=(13, 9))
        for ax, t in zip(axes[0], show):
            rl = by_level[t]
            i = full_idx[t]
            dense = [(eff_flops(r, "full", i), stats(r, "full", "cons", i)) for r in rl]
            fr = frontier(dense)
            ax.plot([f for f, _ in fr], [c for _, c in fr], "k-o", lw=2,
                    label="dense frontier", zorder=3)
            for f, c in dense:
                ax.plot(f, c, "ko", mfc="none", ms=9)
            for r in rl:
                for o, mk, col in (("depth_arg", "s", "tab:blue"),
                                   ("joint_arg", "^", "tab:red")):
                    ax.plot(eff_flops(r, o, i), stats(r, o, "cons", i), mk, color=col,
                            ms=8, label=o if r is rl[0] else None)
                    ax.annotate("", xy=(eff_flops(r, o, i), stats(r, o, "cons", i)),
                                xytext=(eff_flops(r, "full", i), stats(r, "full", "cons", i)),
                                arrowprops=dict(arrowstyle="->", color=col, alpha=0.35))
            ok = gates[(t, "cons")][0]
            ax.set_xscale("log")
            ax.set_title(f"{clabel(t)}"
                         + (" [PRIMARY]" if t == prim else "")
                         + ("" if ok else "  [DEGENERATE: gate FAIL]"))
            ax.set_xlabel("inference FLOPs / token")
            ax.set_ylabel("consequence accuracy")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        ax = axes[1][0]
        for o, col in (("depth_arg", "tab:blue"), ("joint_arg", "tab:red"),
                       ("joint_kl", "tab:green")):
            for size in LADDER:
                pts = [c for c in levels if (c, "cons", "pw", (size, o)) in results
                       and c[1] == levels[0][1]]
                if not pts:
                    continue
                ax.plot([c[0] for c in pts],
                        [results[(c, "cons", "pw", (size, o))][0] for c in pts],
                        "-o", color=col, alpha=0.6, ms=4,
                        label=o if size == LADDER[0] else None)
        ax.axhline(2.0, color="k", ls="--", label="2x kill line")
        ax.axhline(1.0, color="gray", ls=":")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("training tokens"); ax.set_ylabel("multiplier M (consequence acc)")
        ax.set_title("horizontal gap vs the dense frontier"); ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        ax = axes[1][1]
        t = levels[-1]
        i = full_idx[t]
        for r in by_level[t]:
            ax.plot(eff_flops(r, "full", i), stats(r, "full", "loss", i), "ko")
            ax.plot(eff_flops(r, "joint_arg", i), stats(r, "joint_arg", "loss", i), "r^")
        ax.set_xscale("log")
        ax.set_xlabel("inference FLOPs / token"); ax.set_ylabel("cross-entropy loss")
        ax.set_title("LOSS — NOT PRESERVED BY THE ORACLE CRITERION\n"
                     "(diagnostic only; no verdict may be read here)", fontsize=9)
        ax.grid(alpha=0.3)
        fig.suptitle(f"e009b T2 overlay — oracle ceiling vs dense inference frontier "
                     f"— {combined}")
        fig.tight_layout()
        fig.savefig(ART / "results.png", dpi=150)
        print(f"wrote {ART/'results.png'}")

    caveats(recs, by_level, levels, full_idx, gates, verdicts, combined)


def caveats(recs, by_level, levels, full_idx, gates, verdicts, combined):
    t_last = levels[-1]
    i = full_idx[t_last]
    drift = min(r["points"]["joint_arg"]["agree_hit"].sum() /
                (r["eval_seqs"] * r["eval_seq_len"]) for r in by_level[t_last])
    dd = min(r["points"]["depth_arg"]["agree_hit"].sum() /
             (r["eval_seqs"] * r["eval_seq_len"]) for r in by_level[t_last])
    heads = ", ".join(f"{r['size']} {100*r['flops_head']/r['flops_full']:.1f}%"
                      for r in by_level[t_last])
    lines = [
        "# e009b — caveats (auto-generated by analyze.py)",
        "",
        f"Verdict as run: **{combined}**",
        "",
        "## The oracle is post-hoc and non-realizable — every number is a CEILING",
        "",
        "The skip decisions require the full model's own final logits (the",
        "depth criterion is 'this intermediate readout already argmax-matches",
        "the final answer'). No router, gate, or predictor can reproduce them.",
        "The correct reading of a large multiplier is 'an upper bound worth",
        "chasing'; the correct reading of a small one is 'no mechanism at this",
        "granularity can help, because the thing that sees the answer cannot'.",
        "",
        "## What the argmax criterion does and does not preserve",
        "",
        "Accuracy-type metrics are preserved by construction wherever the",
        "criterion holds; loss is not, and the loss panel exists only to show",
        "what argmax-matching discards. A T2 verdict read off loss is invalid.",
        "",
        "## Measured drift: the union is not automatically jointly valid",
        "",
        f"Executing all skips at once retains {drift:.4f} argmax agreement with",
        f"the full model for the joint arm and {dd:.4f} for depth-only, at the",
        "final checkpoint (worst rung). e009 reported the union of independent",
        "counterfactuals, which is an upper bound; capability here is measured",
        "on the executed configuration, so any interaction cost is visible as",
        "lost capability rather than hidden.",
        "",
        "## Temporal reuse is charged zero FLOPs but keeps its KV",
        "",
        "The joint arm skips a block at a position while later positions still",
        "attend to that position's true keys and values there. A real",
        "implementation would have to produce them. Depth-skip has no such",
        "problem (CALM-style state copy). The joint arm is therefore doubly a",
        "ceiling; the depth-only arm is the implementation-faithful one.",
        "",
        "Note the sting: the faithful arm is also the one that loses the most",
        "capability when executed. Depth-only skipping at these fractions is",
        "not capability-preserving in practice, because frozen early-exit",
        "states feed later blocks' attention for OTHER positions. Controls",
        "confirming this is drift and not a masking bug are in",
        "pilot/controls.txt and are re-run per cell by overlay.py.",
        "",
        "## The joint arm REPAIRS the depth arm — a post-hoc-only effect",
        "",
        "Joint executes better than depth-only despite skipping strictly more.",
        "The verifier localised it (VERIFIER RULING in DESIGN.md): the gain is",
        "entirely on mid-exit positions (agreement 0.34 -> 0.81 at s, 0.34 ->",
        "0.97 at l), it comes from reuse applied to the position's OWN state",
        "and not from context, and it is not a denoising effect — the copied",
        "value is MORE drifted in L2 than the value it replaces, and injecting",
        "noise, zeros, or the passthrough state at the SAME certified slots is",
        "worse than doing nothing. Only the genuine t-1 value works (t-2 works",
        "partly). So the temporal certificate is a valid-substitute certificate",
        "and applying it overwrites the depth arm's execution drift with a",
        "value the clean model verified. That repair needs the clean model, so",
        "no realizable mechanism has it: the joint arm's agreement is inflated",
        "relative to any implementable scheme. Direction: generous to the",
        "oracle, hence conservative for a T2-survives reading. The joint arm's",
        "high agreement is NOT evidence that the union of skips is jointly",
        "valid.",
        "",
        "## A skipped block is not free: the KV floor",
        "",
        "Later positions attend to a skipped position at that block, so its",
        "keys and values must still be produced (2*d*2d = 4 d^2 per skipped",
        "(token, block)). The pre-registered accounting charges skips zero and",
        "is retained as the verdict statistic; `flops_eff_kvfloor` in",
        "results.csv reports the honest figure alongside. The floor moves M",
        "DOWN (away from a kill) and so cannot rescue a verdict.",
        "",
        "## The unembedding is a floor no oracle can skip",
        "",
        f"Head share of inference FLOPs: {heads}. Block-level skip fractions",
        "overstate end-to-end savings at small rungs, which is exactly where",
        "the dense frontier is cheapest — this is a real structural limit on",
        "adaptive compute, not an artifact of the accounting.",
        "",
        "## Corpus: the capability axis saturates",
        "",
        "The e005 synthetic corpus is mostly irreducible entropy by",
        "construction (fork tokens, uniform noise spans), so next-token",
        "accuracy has a hard ceiling that every rung reaches by end of",
        "training. Levels failing the dynamic-range gate below are reporting a",
        "fact about the corpus, not about adaptivity:",
    ]
    for t in levels:
        for axis in ("acc", "cons"):
            ok, biggest, half = gates[(t, axis)]
            lines.append(f"- {clabel(t)} [{axis}]: max adjacent gap {biggest:.4f}, "
                         f"CI half-width {half:.4f}, gate "
                         f"{'PASS' if ok else 'FAIL (degenerate)'}")
    lines += [
        "",
        "## Frontier resolution — and why the verdict is fit-free",
        "",
        "Adjacent rungs are 2.19-2.95x apart in inference FLOPs, the same",
        "order as the 2x verdict line, so interpolation between rungs would be",
        "load-bearing. It is not allowed to be: the verdict runs on the",
        "FIT-FREE BRACKET (the last frontier rung below C_oracle and the first",
        "at or above it), which every monotone fit lies inside. Three point",
        "fits are still reported (piecewise-linear in capability, piecewise-",
        "linear in logit capability, and the originally pre-registered global",
        "least-squares log-linear). The global log-linear fit falls OUTSIDE",
        "the bracket in most in-support cases — it does not pass through the",
        "measured rungs and cannot represent a saturating frontier — and is",
        "retained only as `M_acc_loglin_diag` with a printed violation flag.",
        "",
        "## Iso-token, not iso-training-FLOPs",
        "",
        "All rungs see 6M tokens, so larger rungs got more training compute",
        "than a compute-optimal allocation would give them. This raises the",
        "high-FLOPs end of the frontier and therefore inflates the multiplier:",
        "generous to the oracle, conservative for a T2-survives conclusion.",
        "",
        "## Scope",
        "",
        "One synthetic corpus, 6M tokens, 0.049M-24.8M non-embedding params,",
        "CPU, block-level granularity. The transferable quantities are the",
        "direction of the multiplier and its dependence on training level, not",
        "its value.",
    ]
    (ART / "CAVEATS.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {ART/'CAVEATS.md'}")


if __name__ == "__main__":
    main()
