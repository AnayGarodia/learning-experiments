"""e004g analysis: retention-vs-k curves, the K1/K2/K3 discriminators,
artifacts/results.csv, artifacts/results.png, artifacts/CAVEATS.md.

No binning, no smoothing, no interpolation: every plotted point is a
measured point (grid-snapping guard 4 in DESIGN.md). Retention is
plotted against BOTH k and cumulative optimizer steps, because
"retention decays with skill count" and "retention decays with total
gradient steps" are different findings and are otherwise confounded
(guard 3).
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs"
ART = HERE / "artifacts"
MDE_1SEED, MDE_3SEED = 0.060, 0.035
# skills 0-3 are type A (constant answer) and are FIXED-OFFSET COPYABLE at
# ~0.95 in the gate. They ride along as the dossier's fact-like arm; they must
# never carry a conclusion about skills, so every headline read excludes them.
COPYABLE = {0, 1, 2, 3}


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def rows(runs):
    """One row per (arm, seed, k, measured skill). `just_after` is that
    skill's unprompted accuracy at the step it was consolidated, which is
    the denominator retention is expressed against."""
    out = []
    for r in runs:
        just = {}
        for st in r["steps"]:
            just.setdefault(str(st["skill"]), st["unprompted"][str(st["skill"])])
            for sk, acc in st["unprompted"].items():
                # Skills not yet written are EMITTED (retention None): leakage
                # into an unwritten skill is as informative as loss of a
                # written one, and dropping them hid it from results.csv.
                out.append({
                    "arm": r["arm"], "seed": r["seed"], "k": st["k"],
                    "cum_steps": st["cum_steps"], "consolidated_skill": st["skill"],
                    "measured_skill": int(sk), "age": st["k"] - (
                        r["order"].index(int(sk)) + 1),
                    "written": sk in just, "copyable": int(sk) in COPYABLE,
                    "unprompted": acc, "just_after": just.get(sk),
                    "retention": (round(acc / just[sk], 4)
                                  if just.get(sk) else None),
                    "prompted_on_top": st["prompted_on_top"][sk],
                    "reprompt_gain": round(st["prompted_on_top"][sk] - acc, 4),
                    "generality_nats": st["generality_nats"],
                    "unknown_marker_agree": st["unknown_marker_agree"],
                    "unknown_marker_applies_rule": st.get(
                        "unknown_marker_applies_rule", {}).get(sk),
                    # newest rule bleeding into THIS already-trained marker
                    "cross_marker_leak": st.get(
                        "cross_marker_leak", {}).get(sk),
                    "write_tripwire_ok": st["write_tripwire_ok"],
                    # off-length probe, final k only (None elsewhere)
                    "hard_unprompted": st.get("hard_unprompted", {}).get(sk),
                    "hard_reprompt_gain": (
                        round(st["hard_prompted_on_top"][sk]
                              - st["hard_unprompted"][sk], 4)
                        if sk in st.get("hard_unprompted", {}) else None),
                })
    return out


CLEAN_AGE = 2   # VERIFIER: rows younger than this are conditionality-confounded
# A never-written skill reading above this means some OTHER skill's answer
# satisfies its marker, i.e. the pool has a coinciding pair and every number
# here is inflated by an unknown amount. This was 1.0 in the published output
# and was reported as a field nobody read. It is now BLOCKING.
UNPROMPTED_BLOCK = 0.20
EXCLUDE = set()   # `--exclude 7,13`: skills dropped from every read, recorded
HEADROOM = 1.0 - MDE_1SEED   # a skill already above this cannot show a +MDE gain


def _mean(xs):
    return round(float(np.mean(xs)), 4) if len(xs) else None


def three_way(kl, ce, x, tol):
    """Which factor moved a metric: the WRITE objective, the REPLAY objective,
    or both. PRE-REGISTERED 2026-08-17, before `ce_write_kl_replay` ran:
    "sits with arm Y" means within one tolerance of Y AND more than one
    tolerance from the other endpoint. If the endpoints are less than TWO
    tolerances apart, nothing is decomposable -- there is then no gap position
    that is far from both -- and the read is UNREADABLE rather than a verdict.

    The 2*tol guard was added because the identifiability dry-run FAILED
    without it (2026-08-17): with the grid's measured endpoints and the
    single-seed MDE, a genuinely half-way arm was mislabelled
    REPLAY_OBJECTIVE. The tolerance for this read is therefore MDE_3SEED,
    which is the resolution of the 3-seed arm means actually being compared
    (the same choice `snell_edge_resolvable` already makes)."""
    out = {"kl_write_kl_replay": kl, "ce_write_ce_replay": ce,
           "ce_write_kl_replay": x, "tol": round(tol, 4)}
    if None in (kl, ce, x):
        out["call"] = "ARM_MISSING"
        return out
    if abs(ce - kl) <= 2 * tol:
        out["call"] = "UNREADABLE_ENDPOINTS_TOO_CLOSE"
        out["gap"] = round(ce - kl, 4)
        return out
    near_ce, near_kl = abs(x - ce) <= tol, abs(x - kl) <= tol
    out["call"] = ("UNRESOLVED_ARMS_TOO_CLOSE" if near_ce and near_kl else
                   "WRITE_OBJECTIVE" if near_ce else
                   "REPLAY_OBJECTIVE" if near_kl else "MIXED_DECOMPOSITION")
    out["fraction_of_gap_from_kl"] = round((x - kl) / (ce - kl), 3)
    return out


def verdict(rs):
    """K1 vs K2 vs K3, read only on the persistent-adapter arms."""
    v = {}
    pers = [r for r in rs if r["arm"] in ("consolidate", "no_replay")]
    if not pers:
        return {"note": "no persistent-adapter runs present"}
    for arm in sorted({r["arm"] for r in rs}):
        # headline reads exclude the copyable type-A skills entirely
        a = [r for r in rs if r["arm"] == arm and r["written"]
             and not r["copyable"]]
        a_copy = [r for r in rs if r["arm"] == arm and r["written"]
                  and r["copyable"] and r["age"] > 0]
        unwritten = [r for r in rs if r["arm"] == arm and not r["written"]]
        prior = [r for r in a if r["age"] > 0]          # skills already written
        if not a:
            continue
        v[arm] = {
            "n_rows": len(a),
            "mean_just_after": round(float(np.mean(
                [r["just_after"] for r in a])), 4),
            "min_retention_prior": round(min((r["retention"] for r in prior),
                                             default=float("nan")), 4),
            "mean_retention_prior": round(float(np.mean(
                [r["retention"] for r in prior])), 4) if prior else None,
            "retention_at_max_k": round(float(np.mean(
                [r["retention"] for r in prior
                 if r["k"] == max(x["k"] for x in a)])), 4) if prior else None,
            "mean_reprompt_gain": round(float(np.mean(
                [r["reprompt_gain"] for r in a])), 4),
            "generality_drift_nats": round(
                max(r["generality_nats"] for r in a)
                - min(r["generality_nats"] for r in a), 4),
            # per-seed drift, because the three-way write-objective read below
            # compares arms on this axis and a mean could hide a split
            "generality_drift_nats_per_seed": {
                str(sd): round(
                    max(r["generality_nats"] for r in a if r["seed"] == sd)
                    - min(r["generality_nats"] for r in a if r["seed"] == sd), 4)
                for sd in sorted({r["seed"] for r in a})},
            # `generality_drift_nats` above pools all seeds into one max-minus-
            # min, so it also absorbs BETWEEN-seed level differences. On the
            # published grid that inflated direct_ft to 0.3183 while its
            # per-seed drifts were 0.048 / 0.278 / 0.056 -- one seed carrying
            # the whole figure. The seed-mean is the drift actually per run,
            # and the SD is what says whether an arm difference is resolvable.
            "generality_drift_nats_seedmean": None,      # filled just below
            "generality_drift_nats_seedsd": None,
            "min_unknown_marker_agree": round(
                min(r["unknown_marker_agree"] for r in a), 3),
            "all_writes_landed": all(r["write_tripwire_ok"] for r in a),
            # if this is high the write attached the rule to the FORMAT, not
            # to the marker, and no interference reading is safe
            "max_unknown_marker_applies_rule": (
                max([r["unknown_marker_applies_rule"] for r in a
                     if r["unknown_marker_applies_rule"] is not None],
                    default=None)),
            # VERIFIER RULING (K0): the pre-registered probe read PER AGE
            # instead of collapsed to a max. A rule is format-attached when
            # freshly written and becomes marker-attached once a LATER write
            # takes the format default, so `max` over all rows is pinned at
            # ~1.0 by the newest write at every k and can never clear 0.20.
            # Age is the axis the probe is informative on.
            "umar_by_age": {
                str(g): _mean([r["unknown_marker_applies_rule"] for r in a
                               if r["age"] == g
                               and r["unknown_marker_applies_rule"] is not None])
                for g in sorted({r["age"] for r in a})},
            "max_umar_age_ge_clean": max(
                [r["unknown_marker_applies_rule"] for r in a
                 if r["age"] >= CLEAN_AGE
                 and r["unknown_marker_applies_rule"] is not None],
                default=None),
            # per-seed retention at that seed's max k. Decision 0012
            # corollary: print per-seed values, never only the mean.
            "retention_at_max_k_per_seed": {
                str(sd): _mean([r["retention"] for r in prior
                                if r["seed"] == sd and r["k"] == max(
                                    x["k"] for x in a if x["seed"] == sd)])
                for sd in sorted({r["seed"] for r in prior})},
            "retention_clean_per_seed": {
                str(sd): _mean([r["retention"] for r in prior
                                if r["seed"] == sd and r["age"] >= CLEAN_AGE])
                for sd in sorted({r["seed"] for r in prior})},
            # CEILING CENSORING on the K1/K2 discriminator: a skill already at
            # unprompted >= HEADROOM cannot show a +MDE re-prompting gain, so
            # its row carries no information about K1 vs K2.
            "reprompt_skills_total": len({r["measured_skill"] for r in a}),
            "reprompt_skills_with_headroom": len(
                {r["measured_skill"] for r in a if r["unprompted"] <= HEADROOM}),
            "mean_reprompt_gain_headroom": _mean(
                [r["reprompt_gain"] for r in a if r["unprompted"] <= HEADROOM]),
            # four of the six non-degenerate skills are ONE mechanism ("copy
            # the input and decorate it"). A mean over them is a mean over a
            # narrow family, so the per-skill breakdown is mandatory reading.
            "retention_prior_by_skill": {
                str(sk): _mean([r["retention"] for r in prior
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in prior})},
            # the fact-like arm, reported beside the skills but never mixed in
            "retention_prior_typeA_copyable": _mean(
                [r["retention"] for r in a_copy if r["retention"] is not None]),
            "n_rows_typeA_copyable": len(a_copy),
            # off-length diagnostic (see HARD_WORDS in consolidate.py). NOT a
            # K1/K2 verdict: the write never saw a word this long.
            "offlength_unprompted": _mean(
                [r["hard_unprompted"] for r in a
                 if r["hard_unprompted"] is not None]),
            "offlength_reprompt_gain": _mean(
                [r["hard_reprompt_gain"] for r in a
                 if r["hard_reprompt_gain"] is not None]),
            # K0 proper: the newest rule bleeding into markers the run has
            # ALREADY trained. Reported per k, because k=1 is structurally
            # different -- with one rule written there is nothing to
            # condition on and leak is uninformative.
            "cross_marker_leak_by_k": {
                str(k): round(float(np.mean(
                    [r["cross_marker_leak"] for r in a
                     if r["k"] == k and r["cross_marker_leak"] is not None])), 4)
                for k in sorted({r["k"] for r in a})
                if any(r["k"] == k and r["cross_marker_leak"] is not None
                       for r in a)},
            # leakage the other way: a skill scoring above ~0 before it was
            # ever written means the format, not the marker, carries the rule
            "max_unprompted_before_write": (
                max([r["unprompted"] for r in unwritten], default=None)),
        }
    for arm, a in v.items():
        d = list((a.get("generality_drift_nats_per_seed") or {}).values())
        if d:
            a["generality_drift_nats_seedmean"] = round(float(np.mean(d)), 4)
            a["generality_drift_nats_seedsd"] = round(float(np.std(d)), 4)

    c = v.get("consolidate", {})
    ceil = v.get("separate_adapters", {})
    ft = v.get("direct_ft", {})
    norep = v.get("no_replay", {})
    xw = v.get("ce_write_kl_replay", {})

    # nats tolerance: the largest within-arm across-seed spread of the drift,
    # floored at 0.02. Formula pre-registered; the value is measured from the
    # same runs because no prior seed-noise estimate for nats drift exists.
    tol_nats = max([0.02] + [a["generality_drift_nats_seedsd"]
                             for a in v.values()
                             if a.get("generality_drift_nats_seedsd")])

    # ---- POOL GATE, checked before anything else ---------------------------
    mub = [x for x in (v[arm].get("max_unprompted_before_write") for arm in v)
           if x is not None]
    pool_blocks = bool(mub) and max(mub) > UNPROMPTED_BLOCK

    # ---- K0, on the PRE-REGISTERED probe read per age (verifier ruling) ----
    umar_clean = c.get("max_umar_age_ge_clean")
    k0_blocks = (umar_clean is not None and umar_clean > 0.20) or pool_blocks

    # ---- retention: COMPOUNDS / WALL / INCONCLUSIVE ------------------------
    # Read on the per-seed mean, not on a raw min over every row: at grid
    # scale the persistent arm has ~45 retention rows per arm and binomial
    # noise alone puts one of them under 0.90 whenever true retention is
    # below ~0.95. The min is kept as a diagnostic, not as the verdict.
    per_seed = list((c.get("retention_clean_per_seed")
                     or c.get("retention_at_max_k_per_seed") or {}).values())
    per_seed = [x for x in per_seed if x is not None]
    if not per_seed:
        ret_call = "INCONCLUSIVE_NO_ROWS"
    elif min(per_seed) > 0.90:
        ret_call = "COMPOUNDS"
    elif max(per_seed) <= 0.90:
        ret_call = "WALL"
    else:
        ret_call = "INCONCLUSIVE_SEEDS_SPLIT"

    # ---- re-prompting gain: censoring-aware --------------------------------
    n_head = c.get("reprompt_skills_with_headroom", 0)
    g = c.get("mean_reprompt_gain_headroom")
    if n_head < 2 or g is None:
        rep_call = "INCONCLUSIVE_CEILING_CENSORED"
    elif g > MDE_1SEED:
        rep_call = "FREES_CONTEXT"
    elif abs(g) <= MDE_1SEED:
        rep_call = "CACHE_ONLY"
    else:
        rep_call = "NEGATIVE_GAIN"

    v["_reading"] = {
        # BLOCKING: pool coincidence. Every K flag below is False while set.
        "POOL_max_unprompted_before_write": max(mub) if mub else None,
        "POOL_BLOCKS": pool_blocks,
        "POOL_excluded_skills": sorted(EXCLUDE),
        # CHECKED FIRST: an unconditional write confounds every rival below.
        # VERIFIER RULING 2026-08-16: the pre-registered probe is kept and
        # `cross_marker_leak` is REJECTED as the reading rule (it is near
        # arithmetically implied by retention, so it cannot independently
        # confirm conditionality). The probe is read at age >= CLEAN_AGE,
        # where the pre-registered threshold has discriminating power.
        "K0_preregistered_value_max_all_rows": c.get(
            "max_unknown_marker_applies_rule"),
        "K0_umar_by_age": c.get("umar_by_age"),
        "K0_max_umar_age_ge_clean": umar_clean,
        "K0_clean_age": CLEAN_AGE,
        "K_readings_blocked": k0_blocks,
        "cross_marker_leak_by_k": c.get("cross_marker_leak_by_k") or {},
        "cross_marker_leak_note": (
            "diagnostic only; not the K0 reading rule (verifier ruling)"),
        # the two questions are read SEPARATELY: retention (K1 vs K3) is not
        # censored, the re-prompting gain (K1 vs K2) usually is.
        "retention_call": ret_call,
        "retention_per_seed": c.get("retention_clean_per_seed"),
        "retention_clean_rows": sum(
            1 for r in rs if r["arm"] == "consolidate" and r["written"]
            and r["age"] >= CLEAN_AGE),
        "reprompt_call": rep_call,
        "reprompt_headroom_skills": f"{n_head}/{c.get('reprompt_skills_total')}",
        "reprompt_gain_headroom_only": g,
        "K1_compounding": bool(not k0_blocks and ret_call == "COMPOUNDS"
                               and rep_call == "FREES_CONTEXT"),
        "K2_cache_not_organ": bool(not k0_blocks and ret_call == "COMPOUNDS"
                                   and rep_call == "CACHE_ONLY"),
        "K3_interference_wall": bool(not k0_blocks and ret_call == "WALL"),
        # The swappable-adapter ceiling has retention 1.0 BY CONSTRUCTION
        # (each skill owns its weights), so the informative comparison is
        # what each skill reaches when written with vs without interference.
        "ceiling_note": "separate_adapters retention == 1.0 by construction",
        "write_quality_vs_ceiling": (
            round(c["mean_just_after"] - ceil["mean_just_after"], 4)
            if c and ceil else None),
        # dossier 004: Snell's ~9-point edge must replicate or the premise
        # falls. NOT computed before the verifier pass; direct_ft unrun.
        "snell_edge_consolidate_minus_direct_ft": (
            round(c["mean_just_after"] - ft["mean_just_after"], 4)
            if c and ft else None),
        "snell_edge_resolvable": (
            None if not (c and ft) else
            abs(c["mean_just_after"] - ft["mean_just_after"]) > MDE_3SEED),
        # ---- TASK A: the three-way write-objective decomposition -----------
        # KL/KL (consolidate) vs CE/KL (ce_write_kl_replay) vs CE/CE
        # (direct_ft). The grid's finding 3 could not separate the write
        # objective from the replay objective; these two reads do.
        "write_objective_retention": three_way(
            c.get("mean_retention_prior"), ft.get("mean_retention_prior"),
            xw.get("mean_retention_prior"), MDE_3SEED),
        # read on the SEED-MEAN drift with a tolerance equal to the largest
        # across-seed SD, because the pooled max-minus-min absorbs between-seed
        # level differences (see generality_drift_nats_seedmean).
        "write_objective_nats": three_way(
            c.get("generality_drift_nats_seedmean"),
            ft.get("generality_drift_nats_seedmean"),
            xw.get("generality_drift_nats_seedmean"), tol_nats),
        "write_objective_nats_pooled_diagnostic": {
            arm: v[arm].get("generality_drift_nats") for arm in
            ("consolidate", "ce_write_kl_replay", "direct_ft") if arm in v},
        "write_objective_per_seed": {
            arm: {"retention_clean": v[arm].get("retention_clean_per_seed"),
                  "retention_prior_mean": v[arm].get("mean_retention_prior"),
                  "nats_drift": v[arm].get("generality_drift_nats_per_seed")}
            for arm in ("consolidate", "ce_write_kl_replay", "direct_ft")
            if arm in v},
        "write_objective_note": (
            "the write objective is context distillation (Snell, Klein & "
            "Zhong, arXiv 2209.15189); CE-vs-distillation forgetting is "
            "published (SDFT arXiv 2601.19897, DiSC arXiv 2602.16093). Any "
            "reading here is a reproduction, and SDFT reports the OPPOSITE "
            "sign at 3B, below which this 0.5B run sits."),
        "replay_effect_consolidate_minus_no_replay": (
            round(c["retention_at_max_k"] - norep["retention_at_max_k"], 4)
            if c.get("retention_at_max_k") is not None
            and norep.get("retention_at_max_k") is not None else None),
        "offlength_unprompted": c.get("offlength_unprompted"),
        "offlength_reprompt_gain": c.get("offlength_reprompt_gain"),
        "offlength_note": ("diagnostic: the write is distilled on 5-letter "
                           "words only, so a prompted-beats-baked result here "
                           "is partly off-distribution, not a K1 win"),
        "max_k_reached": max((r["k"] for r in rs), default=None),
        "MDE_single_seed": MDE_1SEED,
        "caution": ("a 'no cliff' reading bounds nothing beyond the k "
                    "actually run; the sequential-editing collapse this "
                    "design is contrasted against appears at k~32"),
    }
    return v


def figure(rs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({r["arm"] for r in rs})
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    rs = [r for r in rs if r["written"]]
    for arm in arms:
        a = [r for r in rs if r["arm"] == arm and r["age"] > 0]
        if a:
            ks = sorted({r["k"] for r in a})
            ax[0].plot(ks, [np.mean([r["retention"] for r in a if r["k"] == k])
                            for k in ks], "o-", label=arm)
            cs = sorted({r["cum_steps"] for r in a})
            ax[1].plot(cs, [np.mean([r["retention"] for r in a
                                     if r["cum_steps"] == c]) for c in cs],
                       "s--", label=arm)
        b = [r for r in rs if r["arm"] == arm]
        ks = sorted({r["k"] for r in b})
        ax[2].plot(ks, [np.mean([r["reprompt_gain"] for r in b if r["k"] == k])
                        for k in ks], "^-", label=arm)
    # K0 panel: the PRE-REGISTERED probe against AGE since the skill's own
    # write. This is the axis it discriminates on -- against k it is pinned
    # near 1.0 by whichever skill was written last (verifier ruling).
    for arm in arms:
        b = [r for r in rs if r["arm"] == arm
             and r["unknown_marker_applies_rule"] is not None]
        if b:
            gs = sorted({r["age"] for r in b if r["age"] >= 0})
            ax[3].plot(gs, [np.mean([r["unknown_marker_applies_rule"]
                                     for r in b if r["age"] == g])
                            for g in gs], "d-", label=arm)
    ax[3].axhline(0.20, color="k", ls=":", lw=1)
    ax[3].axvline(CLEAN_AGE - 0.5, color="r", ls="--", lw=1)
    ax[3].set(xlabel="age (writes since this skill's own write)",
              ylim=(-0.05, 1.05),
              ylabel="own rule under an UNCLAIMED marker",
              title="K0: conditionality is acquired (dotted=threshold,\n"
                    "red=rows cleared for K1/K2/K3)")
    ax[0].axhline(0.90, color="k", ls=":", lw=1)
    # FIXED y-limits: autoscale turned a 0.0015 spread into a visual trend,
    # which is the presentation form of the grid-snapping artifact.
    ax[0].set(xlabel="k (skills consolidated)", ylim=(0, 1.05),
              ylabel="retention of PRIOR skills", title="retention vs k (K1/K3)")
    ax[1].set(xlabel="cumulative optimizer steps", ylabel="retention",
              ylim=(0, 1.05), title="retention vs compute (confound check)")
    ax[2].axhline(MDE_1SEED, color="k", ls=":", lw=1)
    ax[2].axhline(-MDE_1SEED, color="k", ls=":", lw=1)
    ax[2].set(xlabel="k", ylabel="prompted_on_top - unprompted",
              title="re-prompting gain (K1/K2), dotted = MDE")
    for x in ax:
        x.legend(fontsize=8); x.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(path, dpi=130)


# ---- decision 0012 clause 1: identifiability dry-run -----------------------
# "Simulate the predicted result and show the verdict rule fires." Two grids
# passed their MDE and still could not call a verdict; this is the check that
# would have caught both. Runs in milliseconds, no model, no training.

# just-consolidated accuracy per skill, taken from the gate's PROMPTED column
# for the six non-degenerate skills (upper, double, title, first, plural,
# triple). The spread matters: four of six saturate, which is what censors
# the re-prompting discriminator.
# the grid's actual skill ids: 6 non-degenerate (upper, double, title, first,
# plural, triple) plus the two type-A copyables that extend k to the
# pre-registered 8 and are excluded from every headline read.
SYNTH_SKILLS = [5, 7, 8, 9, 11, 13, 0, 1]
SYNTH_JUST = {5: 0.94, 7: 1.00, 8: 1.00, 9: 0.985, 11: 0.795, 13: 1.00,
              0: 1.00, 1: 1.00}


def synth(mech, skills=SYNTH_SKILLS, seeds=(0, 1, 2), noise=0.02, rng_seed=0,
          pen=None, nats=None, fade=None):
    """Synthetic run records for one rival mechanism, in runs/*.json shape.

    `pen` / `nats` override the per-arm retention penalty and generality drift;
    they exist for the TASK A write-objective dry-run and default to the
    original behaviour."""
    pen = pen or {"direct_ft": 0.09}
    nats, fade = nats or {}, fade or {}
    rng = np.random.default_rng(rng_seed)
    runs = []
    for arm in ("consolidate", "separate_adapters", "direct_ft", "no_replay",
                "ce_write_kl_replay"):
        for sd in seeds:
            order = [int(skills[i]) for i in
                     np.random.default_rng(100 + sd).permutation(len(skills))]
            steps = []
            for i, s in enumerate(order):
                seen = order if arm != "separate_adapters" else [s]
                unp, top, umar = {}, {}, {}
                for j in seen:
                    pos = order.index(j)
                    if pos > i:                      # not written yet
                        unp[str(j)], top[str(j)] = 0.0, SYNTH_JUST[j]
                        umar[str(j)] = 0.0
                        continue
                    age = i - pos
                    base = SYNTH_JUST[j] - pen.get(arm, 0.0)
                    if mech == "K3" and age:         # editing-style decay
                        base *= max(0.0, 1 - (0.15 if arm == "no_replay"
                                              else 0.09) * age)
                    if age and fade.get(arm):    # arm-specific forgetting rate
                        base *= max(0.0, 1 - fade[arm] * age)
                    gain = 0.10 if (mech == "K1" and arm != "separate_adapters") else 0.0
                    e = float(rng.normal(0, noise))
                    unp[str(j)] = round(min(1.0, max(0.0, base + e)), 3)
                    top[str(j)] = round(min(1.0, unp[str(j)] + gain), 3)
                    # K0 pattern measured in the pilot: the newest write holds
                    # the format default, older rules are marker-conditional
                    umar[str(j)] = 1.0 if age == 0 else (0.5 if age == 1 else 0.0)
                steps.append({
                    "k": i + 1, "skill": s, "cum_steps": 60 * (i + 1),
                    "unprompted": unp, "prompted_on_top": top,
                    "generality_nats": round(2.84 + nats.get(arm, 0.0)
                                             * i / max(1, len(order) - 1), 4),
                    "unknown_marker_agree": 0.0,
                    "unknown_marker_applies_rule": umar,
                    "cross_marker_leak": {}, "write_tripwire_ok": True})
            runs.append({"arm": arm, "seed": sd, "order": order, "steps": steps})
    return runs


def dryrun():
    """Each rival's PREDICTED outcome must produce its own verdict category.
    Any INCONCLUSIVE here means the design is broken (decision 0012 c.1)."""
    want = {"K1": ("COMPOUNDS", "FREES_CONTEXT"),
            "K2": ("COMPOUNDS", "CACHE_ONLY"),
            "K3": ("WALL", None)}
    ok = True
    for mech, (ret, rep) in want.items():
        rd = verdict(rows(synth(mech)))["_reading"]
        got_r, got_p = rd["retention_call"], rd["reprompt_call"]
        fired = [k for k in ("K1_compounding", "K2_cache_not_organ",
                             "K3_interference_wall") if rd[k]]
        good = (got_r == ret and (rep is None or got_p == rep)
                and fired == [{"K1": "K1_compounding",
                               "K2": "K2_cache_not_organ",
                               "K3": "K3_interference_wall"}[mech]]
                and not rd["K_readings_blocked"])
        ok &= good
        print(f"  {mech}: retention={got_r} reprompt={got_p} "
              f"headroom={rd['reprompt_headroom_skills']} "
              f"snell={rd['snell_edge_consolidate_minus_direct_ft']} "
              f"replay={rd['replay_effect_consolidate_minus_no_replay']} "
              f"fired={fired} -> {'OK' if good else 'BROKEN'}")
    print("identifiability dry-run:", "PASS" if ok else "FAIL")
    assert ok, "a predicted outcome does not produce its own verdict"
    dryrun_write()


# ---- TASK A identifiability dry-run (decision 0012 clause 1) ---------------
# Three worlds, one per interpretation rule pre-committed in the brief. The
# fade rates are set so the KL and CE endpoints reproduce the grid's measured
# clean-pool retention gap (0.92 vs 0.83) and nats gap (0.10 vs 0.32).
KL_FADE, CE_FADE = 0.025, 0.055     # reproduce the grid's 0.92 / 0.83
WRITE_WORLDS = {
    # world -> (fade rate for ce_write_kl_replay, nats drift for it, expected)
    "write_objective": (CE_FADE, 0.32, "WRITE_OBJECTIVE"),
    "replay_objective": (KL_FADE, 0.10, "REPLAY_OBJECTIVE"),
    "in_between": ((KL_FADE + CE_FADE) / 2, 0.21, "MIXED_DECOMPOSITION"),
}


def dryrun_write():
    ok = True
    for world, (f, n, want) in WRITE_WORLDS.items():
        rd = verdict(rows(synth(
            "K2",
            fade={"consolidate": KL_FADE, "no_replay": KL_FADE,
                  "direct_ft": CE_FADE, "ce_write_kl_replay": f},
            nats={"consolidate": 0.10, "no_replay": 0.08,
                  "separate_adapters": 0.05, "direct_ft": 0.32,
                  "ce_write_kl_replay": n})))["_reading"]
        r, g = rd["write_objective_retention"], rd["write_objective_nats"]
        good = r["call"] == want and g["call"] == want
        ok &= good
        print(f"  {world}: retention {r['kl_write_kl_replay']}/"
              f"{r['ce_write_kl_replay']}/{r['ce_write_ce_replay']} -> "
              f"{r['call']} | nats {g['kl_write_kl_replay']}/"
              f"{g['ce_write_kl_replay']}/{g['ce_write_ce_replay']} -> "
              f"{g['call']} (tol {g['tol']}) -> {'OK' if good else 'BROKEN'}")
    print("write-objective dry-run:", "PASS" if ok else "FAIL")
    assert ok, "a predicted write-objective world does not produce its verdict"


def main():
    ART.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print("no runs in", RUNS); return
    rs = [r for r in rows(runs) if r["measured_skill"] not in EXCLUDE]
    with (ART / "results.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader(); w.writerows(rs)
    v = verdict(rs)
    (ART / "verdict.json").write_text(json.dumps(v, indent=1))
    figure(rs, ART / "results.png")
    print(json.dumps(v, indent=1))
    print(f"\n{len(rs)} rows -> artifacts/results.csv, results.png, verdict.json")
    # spread check (decision 0010 item 4)
    for arm in sorted({r["arm"] for r in rs}):
        u = [r["unprompted"] for r in rs if r["arm"] == arm]
        print(f"spread[{arm}] unprompted sd={np.std(u):.4f} "
              f"min={min(u):.3f} max={max(u):.3f}")
    if v.get("_reading", {}).get("POOL_BLOCKS"):
        raise SystemExit(
            "\n*** POOL GATE BLOCKS THIS READ ***\n"
            f"a never-written skill reads "
            f"{v['_reading']['POOL_max_unprompted_before_write']} "
            f"(> {UNPROMPTED_BLOCK}): some other skill's answer satisfies its "
            "marker, so retention is inflated by an unknown amount.\n"
            "Artifacts were still written, for diagnosis only. Re-run with "
            "--exclude <the coinciding skills> to get a readable number.")


if __name__ == "__main__":
    import sys
    if "--exclude" in sys.argv:
        EXCLUDE = {int(x) for x in
                   sys.argv[sys.argv.index("--exclude") + 1].split(",")}
    if "--dryrun" in sys.argv:
        dryrun()
    else:
        main()
