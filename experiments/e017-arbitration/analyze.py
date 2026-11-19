"""e017 analysis: BOTH quantities, always, and a 2-D verdict.

The reading order is enforced in code and is not negotiable:

  0. writes must land, or an arm is not a retention measurement at all;
  1. the WRITE-TAX check (decision 0012 clause 7a): retention is a RATIO to
     each skill's own just-after accuracy, so an intervention that weakens the
     write inflates retention mechanically. Absolute prior accuracy is emitted
     beside every ratio and an arm whose writes are taxed is flagged;
  2. DYNAMIC RANGE: the no-replay baseline must visibly degrade in the k range
     run, or nothing has been stressed and no arm may be read;
  3. KNOWN-POSITIVE: replay25 must reproduce e004g/e015 retention inside a
     band fixed before the pilot was analysed. Outside it, the harness has
     drifted and nothing is readable;
  4. only then the 2-D read: (is the newest write CONDITIONAL) x (is retention
     RESTORED), per arm. A1 needs both. A3 -- the clean deflation -- is
     arbitration restored WITHOUT retention, and it is only visible because
     both quantities are reported at every k.

No binning, no smoothing. Per-skill, per-seed and per-k values are printed for
every headline number (decision 0012 clause 5).
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs"
ART = HERE / "artifacts"

MDE_1SEED, MDE_3SEED = 0.060, 0.035
EVAL_RESOLUTION = 1 / 200
CLEAN_AGE = 2            # e004g verifier: younger rows are conditionality-confounded
UMAR_BLOCK = 0.20        # e004g's pre-registered conditionality threshold
# VERIFIER EDIT 2026-08-17, made WITH PILOT DATA VISIBLE (which weakens it; see
# DESIGN.md VERIFIER RULING item 3). The relative criterion
# `umar_age0 <= replay25 + MDE` is vacuous if the reference arm's own pooled
# umar_age0 lands high -- an arm that is plainly unconditional would pass, and
# RETENTION_WITHOUT_ARBITRATION could never fire. The relative test now carries
# an ABSOLUTE FLOOR: whatever replay25 does, an arm claiming restored
# arbitration must fire under an unowned cue less than half the time.
UMAR_REL_CAP = 0.50
DYNRANGE = 0.85          # no_replay must fall to <= this
HALF_GAP = 0.50          # dossier 017 A1 kill line: beat the baseline by more
                         # than half the replay gap, or A1 is killed
# KNOWN-POSITIVE BAND, fixed here BEFORE any e017 number was read. Anchored to
# e015's pilot at the same base, suite, rank, write budget and replay schedule:
# 0.6787 on its N=12 suite, 0.6915 on the prefix-free subset this experiment
# runs. +-20% of 0.69. Outside this band the harness has drifted.
KNOWN_POSITIVE_BAND = (0.55, 0.83)
BASELINE, REFERENCE = "no_replay", "replay25"
ARB_ARMS = ("negcue", "cueanchor", "reserve", "selfdistil")
# NOT a candidate fix and deliberately excluded from the verdict: LwF's own
# authors report the method failing when the proxy inputs do not exercise the
# old behaviour. `selfdistil_offcue` is that configuration, so it is a
# PREDICTED NEGATIVE -- if it matches `selfdistil`, our retention is not coming
# from the mechanism claimed and the implementation is wrong.
LWF_CONTROL = "selfdistil_offcue"
COPYABLE = {0, 1, 2, 3}


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def rows(runs):
    """One row per (arm, seed, k, measured skill)."""
    out = []
    for r in runs:
        just = {}
        for st in r["steps"]:
            just.setdefault(str(st["skill"]), st["unprompted"][str(st["skill"])])
            for sk, acc in st["unprompted"].items():
                age = st["k"] - (r["order"].index(int(sk)) + 1)
                out.append({
                    "arm": r["arm"], "seed": r["seed"], "base": r.get("base"),
                    "n_params": r.get("n_params"), "rank": r.get("rank"),
                    "n_skills": r.get("n_skills"), "tag": r.get("tag", ""),
                    "k": st["k"], "cum_steps": st["cum_steps"],
                    "consolidated_skill": st["skill"], "measured_skill": int(sk),
                    "age": age, "written": sk in just,
                    "copyable": int(sk) in COPYABLE,
                    "unprompted": acc, "just_after": just.get(sk),
                    "freegen": st.get("freegen", {}).get(sk),   # e021
                    "retention": (round(acc / just[sk], 4)
                                  if just.get(sk) else None),
                    "unknown_marker_applies_rule":
                        st["unknown_marker_applies_rule"].get(sk),
                    "diag_rank": st.get("diag_rank", {}).get(sk),
                    "newest_rule_under_j":
                        st.get("newest_rule_under_j", {}).get(sk),
                    "generality_nats": st["generality_nats"],
                    "write_tripwire_ok": st["write_tripwire_ok"],
                    "n_aux_examples": st.get("n_aux_examples"),
                    "target_fidelity":
                        (st.get("selfdistil_target_fidelity") or {}).get(sk),
                    "cueanchor_weight_sd":
                        (st.get("cueanchor_weight_stats") or {}).get("sd"),
                })
    return out


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def per_arm(rs):
    """Everything each arm is, before any comparison is made."""
    v = {}
    for arm in sorted({r["arm"] for r in rs}):
        a = [r for r in rs if r["arm"] == arm and r["written"]
             and not r["copyable"]]
        if not a:
            continue
        prior = [r for r in a if r["age"] > 0]
        clean = [r for r in prior if r["age"] >= CLEAN_AGE]
        # BOUNDED mean, per the e015 verifier's pre-stated procedure: drop
        # conditionality-blocking rows rather than suppressing the whole arm.
        cleanb = [r for r in clean
                  if (r["unknown_marker_applies_rule"] or 0) <= UMAR_BLOCK]
        seeds = sorted({r["seed"] for r in a})
        maxk = max(r["k"] for r in a)
        half = (maxk + 1) // 2
        age0 = [r for r in a if r["age"] == 0
                and r["unknown_marker_applies_rule"] is not None]
        d = {
            "bases": sorted({r["base"] for r in a}),
            "n_params": max(r["n_params"] or 0 for r in a),
            "rank": max(r["rank"] or 0 for r in a),
            "n_skills": max(r["n_skills"] or 0 for r in a),
            "max_k": maxk, "n_rows_clean": len(clean),
            "n_rows_clean_bounded": len(cleanb),
            "seeds": seeds,
            # ---- write quality. An arm whose writes do not land is not a
            # high-retention arm, it is a failed arm.
            "all_writes_landed": all(r["write_tripwire_ok"] for r in a),
            "failed_write_skills": sorted({
                r["measured_skill"] for r in a if not r["write_tripwire_ok"]}),
            "mean_just_after": _mean([r["just_after"] for r in a]),
            "just_after_by_skill": {
                str(sk): _mean([r["just_after"] for r in a
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in a})},
            # ---- retention (RATIO), per seed / skill / age / k, never only
            # the mean.
            "retention_clean_per_seed": {
                str(sd): _mean([r["retention"] for r in cleanb
                                if r["seed"] == sd]) for sd in seeds},
            "retention_clean_unbounded_per_seed": {
                str(sd): _mean([r["retention"] for r in clean
                                if r["seed"] == sd]) for sd in seeds},
            "retention_clean_by_skill": {
                str(sk): _mean([r["retention"] for r in cleanb
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in cleanb})},
            "retention_by_age": {
                str(g): _mean([r["retention"] for r in prior if r["age"] == g])
                for g in sorted({r["age"] for r in prior})},
            "retention_clean_by_k": {
                str(kk): _mean([r["retention"] for r in cleanb if r["k"] == kk])
                for kk in sorted({r["k"] for r in cleanb})},
            "retention_early_k": _mean([r["retention"] for r in cleanb
                                        if r["k"] <= half]),
            "retention_late_k": _mean([r["retention"] for r in cleanb
                                       if r["k"] > half]),
            "min_retention_row": round(min(
                (r["retention"] for r in prior), default=float("nan")), 4),
            # ---- retention (ABSOLUTE). The ratio's denominator is the arm's
            # own write, so a taxed write inflates it; this cannot.
            "prior_abs_clean": _mean([r["unprompted"] for r in cleanb]),
            "prior_abs_clean_by_skill": {
                str(sk): _mean([r["unprompted"] for r in cleanb
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in cleanb})},
            # ---- CONDITIONALITY OF THE NEWEST WRITE. The other half of the
            # mandate: never report retention without this.
            "umar_age0": _mean([r["unknown_marker_applies_rule"] for r in age0]),
            "umar_age0_per_seed": {
                str(sd): _mean([r["unknown_marker_applies_rule"] for r in age0
                                if r["seed"] == sd]) for sd in seeds},
            "umar_age0_by_k": {
                str(kk): _mean([r["unknown_marker_applies_rule"] for r in age0
                                if r["k"] == kk])
                for kk in sorted({r["k"] for r in age0})},
            "umar_age0_by_skill": {
                str(sk): _mean([r["unknown_marker_applies_rule"] for r in age0
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in age0})},
            "umar_by_age": {
                str(g): _mean([r["unknown_marker_applies_rule"] for r in a
                               if r["age"] == g])
                for g in sorted({r["age"] for r in a})},
            "max_umar_age_ge_clean": max(
                [r["unknown_marker_applies_rule"] for r in clean
                 if r["unknown_marker_applies_rule"] is not None],
                default=None),
            "n_rows_blocking": len(clean) - len(cleanb),
            # ---- masking vs erasure (e015 amendment 2), on at every k here
            "diag_rank_prior_by_k": {
                str(kk): _mean([r["diag_rank"] for r in prior if r["k"] == kk])
                for kk in sorted({r["k"] for r in prior})
                if any(r["diag_rank"] is not None
                       for r in prior if r["k"] == kk)},
            "newest_rule_under_prior_cue": _mean(
                [r["newest_rule_under_j"] for r in prior]),
            # ---- arm-specific mechanism certification
            "selfdistil_target_fidelity_by_k": {
                str(kk): _mean([r["target_fidelity"] for r in prior
                                if r["k"] == kk])
                for kk in sorted({r["k"] for r in prior})
                if any(r["target_fidelity"] is not None
                       for r in prior if r["k"] == kk)},
            "cueanchor_weight_sd": _mean([r["cueanchor_weight_sd"] for r in a]),
            "generality_drift_nats": round(
                max(r["generality_nats"] for r in a)
                - min(r["generality_nats"] for r in a), 4),
            "max_unprompted_before_write": max(
                [r["unprompted"] for r in rs if r["arm"] == arm
                 and not r["written"]], default=None),
            "aux_examples_at_max_k": max(
                [r["n_aux_examples"] or 0 for r in a], default=0),
        }
        # skill-balanced mean (e015 verifier's residual: row means are
        # order-weighted, since clean rows per skill run 1..k by write position)
        bs = [x for x in d["retention_clean_by_skill"].values() if x is not None]
        d["retention_clean_skill_balanced"] = (
            round(float(np.mean(bs)), 4) if bs else None)
        d["retention_clean_mean"] = _mean(
            list(d["retention_clean_per_seed"].values()))
        # BLOCKING (2026-08-17): see e004g DESIGN 'POOL GATE'. A never-written
        # skill above the 0.20 control level means another skill's answer
        # satisfies its marker; the arm is unreadable, not merely annotated.
        d["pool_blocked"] = bool(d["max_unprompted_before_write"] is not None
                                 and d["max_unprompted_before_write"]
                                 > UMAR_BLOCK)
        d["k0_blocked"] = bool(d["pool_blocked"]
                               or (d["max_umar_age_ge_clean"] is not None
                                   and d["max_umar_age_ge_clean"] > UMAR_BLOCK))
        v[arm] = d
    return v


def reading(v):
    """The pre-registered verdict rules, in the order DESIGN.md fixes them."""
    out = {"MDE_single_seed": MDE_1SEED, "MDE_3seed": MDE_3SEED,
           "eval_resolution": EVAL_RESOLUTION, "clean_age": CLEAN_AGE,
           "dynamic_range_threshold": DYNRANGE, "half_gap_line": HALF_GAP,
           "known_positive_band": list(KNOWN_POSITIVE_BAND)}

    def ret(arm):
        return v.get(arm, {}).get("retention_clean_mean")

    def seedvals(arm):
        return [x for x in v.get(arm, {}).get(
            "retention_clean_per_seed", {}).values() if x is not None]

    # ---- BOTH QUANTITIES, side by side, for every arm, always ---------------
    out["both_quantities_by_arm"] = {
        a: {"umar_age0": v[a]["umar_age0"],
            "retention_clean_mean": v[a]["retention_clean_mean"],
            "retention_skill_balanced": v[a]["retention_clean_skill_balanced"],
            "prior_abs_clean": v[a]["prior_abs_clean"],
            "mean_just_after": v[a]["mean_just_after"],
            "writes_landed": v[a]["all_writes_landed"]}
        for a in sorted(v)}
    out["arms_with_failed_writes"] = [a for a in v if not v[a]["all_writes_landed"]]
    out["arms_k0_blocked"] = [a for a in v if v[a]["k0_blocked"]]
    out["blocking_rows_by_arm"] = {a: v[a]["n_rows_blocking"] for a in sorted(v)}

    # ---- 1. WRITE TAX (decision 0012 clause 7a) ----------------------------
    # Retention is acc_now / acc_just_after for the SAME skill on the SAME
    # model, so base competence normalises out -- but an aux loss that taxes
    # the WRITE lowers the denominator and inflates the ratio. That confound is
    # specific to this experiment and is checked, not assumed.
    ja_base = v.get(BASELINE, {}).get("mean_just_after")
    out["mean_just_after_by_arm"] = {a: v[a]["mean_just_after"] for a in sorted(v)}
    out["write_taxed_arms"] = sorted(
        a for a in v if ja_base is not None
        and (v[a]["mean_just_after"] or 0) < ja_base - MDE_3SEED)
    out["write_tax_note"] = (
        "an arm in `write_taxed_arms` writes measurably weaker than the "
        "baseline, so its RATIO retention is inflated by construction. Read "
        "`prior_abs_clean` for that arm, not `retention_clean_mean`.")

    # ---- 2. dynamic range ---------------------------------------------------
    base = seedvals(BASELINE)
    if BASELINE not in v:
        out["dynamic_range"] = "NOT_RUN"
    elif not base or v[BASELINE]["n_rows_clean_bounded"] == 0:
        out["dynamic_range"] = "NO_CLEAN_ROWS"
    elif not v[BASELINE]["all_writes_landed"]:
        out["dynamic_range"] = "BASELINE_WRITES_FAILED"
    elif max(base) <= DYNRANGE:
        out["dynamic_range"] = "OK"
    else:
        out["dynamic_range"] = "NO_DYNAMIC_RANGE"
    out["baseline_retention_per_seed"] = v.get(BASELINE, {}).get(
        "retention_clean_per_seed")
    out["baseline_umar_age0"] = v.get(BASELINE, {}).get("umar_age0")

    # ---- 3. known-positive --------------------------------------------------
    r_ref = ret(REFERENCE)
    out["known_positive_replay25"] = r_ref
    if r_ref is None:
        out["known_positive"] = "NOT_RUN"
    elif KNOWN_POSITIVE_BAND[0] <= r_ref <= KNOWN_POSITIVE_BAND[1]:
        out["known_positive"] = "OK"
    else:
        out["known_positive"] = "OUT_OF_BAND"

    # ---- 4. the 2-D read ----------------------------------------------------
    out["verdict"] = None
    if out["dynamic_range"] != "OK":
        out["verdict"] = "SUPPRESSED_" + out["dynamic_range"]
        out["verdict_note"] = {
            "NO_DYNAMIC_RANGE":
                "the no-replay baseline did NOT visibly degrade in the k range "
                "run, so every arm looking fine is consistent with nothing "
                "having been stressed. This is the e004g k=3 ambiguity and it "
                "is a verdict, not a result.",
            "BASELINE_WRITES_FAILED":
                "the baseline's writes did not land, so its retention rows are "
                "not a measurement of anything.",
            "NOT_RUN": "the no-replay baseline was not run.",
            "NO_CLEAN_ROWS": "no conditionality-clean baseline rows exist.",
        }[out["dynamic_range"]]
        return out
    if out["known_positive"] != "OK":
        out["verdict"] = "SUPPRESSED_KNOWN_POSITIVE_" + out["known_positive"]
        out["verdict_note"] = (
            f"replay25 retention {r_ref} is outside the pre-registered band "
            f"{KNOWN_POSITIVE_BAND} anchored on e015/e004g. The harness has "
            "drifted; no arm is readable.")
        return out

    b, r_umar = ret(BASELINE), v[REFERENCE]["umar_age0"]
    g_replay = round(r_ref - b, 4)
    out["retention_baseline_mean"] = b
    out["gap_replay25_minus_noreplay"] = g_replay
    arb_thr = min(round((r_umar or 0) + MDE_3SEED, 4), UMAR_REL_CAP)
    out["arb_restored_threshold"] = arb_thr
    out["arb_restored_threshold_capped"] = bool(
        round((r_umar or 0) + MDE_3SEED, 4) > UMAR_REL_CAP)
    arms = {}
    for a in ARB_ARMS:
        if a not in v or not v[a]["all_writes_landed"]:
            continue
        gap = round(ret(a) - b, 4)
        signs = {s: (v[a]["retention_clean_per_seed"][s]
                     > v[BASELINE]["retention_clean_per_seed"][s])
                 for s in v[a]["retention_clean_per_seed"]
                 if s in v[BASELINE]["retention_clean_per_seed"]}
        gap_e = round((v[a]["retention_early_k"] or 0)
                      - (v[BASELINE]["retention_early_k"] or 0), 4)
        gap_l = round((v[a]["retention_late_k"] or 0)
                      - (v[BASELINE]["retention_late_k"] or 0), 4)
        g_e = round((v[REFERENCE]["retention_early_k"] or 0)
                    - (v[BASELINE]["retention_early_k"] or 0), 4)
        g_l = round((v[REFERENCE]["retention_late_k"] or 0)
                    - (v[BASELINE]["retention_late_k"] or 0), 4)
        arms[a] = {
            "umar_age0": v[a]["umar_age0"],
            "arb_restored_vs_replay":
                bool(v[a]["umar_age0"] is not None
                     and v[a]["umar_age0"] <= arb_thr),
            "arb_restored_full": bool(v[a]["umar_age0"] is not None
                                      and v[a]["umar_age0"] <= UMAR_BLOCK),
            "retention_gap": gap,
            "fraction_of_replay_gap": (round(gap / g_replay, 3)
                                       if g_replay > MDE_3SEED else None),
            "retention_restored": bool(g_replay > MDE_3SEED
                                       and gap >= HALF_GAP * g_replay
                                       and len(set(signs.values())) <= 1),
            "beats_baseline_per_seed": signs,
            "gap_early_k": gap_e, "gap_late_k": gap_l,
            "replay_gap_early_k": g_e, "replay_gap_late_k": g_l,
            "decays_with_k": bool(g_e > MDE_3SEED and g_l > MDE_3SEED
                                  and gap_e >= HALF_GAP * g_e
                                  and gap_l < HALF_GAP * g_l),
            "write_taxed": a in out["write_taxed_arms"],
            "prior_abs_clean": v[a]["prior_abs_clean"],
        }
    out["arms"] = arms
    # LwF's predicted negative. Reported before the verdict, never inside it.
    if LWF_CONTROL in v and "selfdistil" in v:
        d = round(ret("selfdistil") - ret(LWF_CONTROL), 4)
        out["lwf_negative_control"] = {
            "selfdistil": ret("selfdistil"), LWF_CONTROL: ret(LWF_CONTROL),
            "difference": d,
            "control_behaved": bool(d > MDE_3SEED),
            "note": ("the off-cue proxy must retain measurably LESS than the "
                     "on-cue one. If it does not, `selfdistil`'s retention is "
                     "not coming from preserving the old cue's function and "
                     "the arm may not be read as LwF at all.")}
    if g_replay <= MDE_3SEED:
        out["verdict"] = "SUPPRESSED_NO_REPLAY_GAP"
        out["verdict_note"] = (
            "rehearsal bought nothing resolvable at this k, so there is no gap "
            "for an arbitration intervention to close a half of. The dossier's "
            "kill line is defined against that gap and cannot be applied.")
        return out
    if not arms:
        out["verdict"] = "INCONCLUSIVE_NO_ARBITRATION_ARM"
        return out
    restored = [a for a in arms if arms[a]["retention_restored"]]
    decaying = [a for a in arms if arms[a]["decays_with_k"]]
    arb_ok = [a for a in arms if arms[a]["arb_restored_vs_replay"]]
    best = max(arms, key=lambda a: arms[a]["retention_gap"])
    out["best_arm_by_retention"] = best
    out["arms_retention_restored"] = sorted(restored)
    out["arms_arbitration_restored"] = sorted(arb_ok)
    if restored and set(restored) & set(arb_ok):
        out["verdict"] = "A1_ARBITRATION_FIXES_RETENTION"
        out["verdict_note"] = (
            "at least one intervention that stores no old data restores BOTH "
            "the conditionality of the newest write and more than half the "
            "retention rehearsal buys.")
    elif decaying:
        # DEFECT FOUND BY THE DRY-RUN: this test must run BEFORE the pooled
        # retention read, not after it. Clean rows are heavily weighted toward
        # large k (30 of 36 at k > 5 when K = 10), so an arm that holds
        # retention early and loses it late posts a POOLED gap below the
        # half-line and was being reported as A3 -- exactly the wrong verdict,
        # since the arbitration fix demonstrably worked at small k. A2 is a
        # statement about the SLOPE and is now read from the slope.
        out["verdict"] = "A2_ARBITRATION_INSUFFICIENT_AT_LARGE_K"
        out["arms_decaying_with_k"] = sorted(decaying)
        out["verdict_note"] = (
            "an intervention restores retention at small k and falls below the "
            "half-gap line by the largest k run. Arbitration is necessary and "
            "not sufficient in the k range measured.")
    elif restored:
        out["verdict"] = "RETENTION_WITHOUT_ARBITRATION"
        out["verdict_note"] = (
            "retention is restored by an arm whose newest write is still "
            "unconditional. The mechanism is then NOT the one this experiment "
            "pre-registered and the arm must be re-read before it is believed.")
    elif arb_ok:
        out["verdict"] = "A3_ARBITRATION_WITHOUT_RETENTION"
        out["verdict_note"] = (
            "conditionality of the newest write is restored to the rehearsal "
            "level WITHOUT restoring retention. The conditionality asymmetry "
            "is then a symptom and not the mechanism -- the clean deflation, "
            "and it is only visible because both quantities are reported.")
    else:
        out["verdict"] = "A0_NO_INTERVENTION_WORKED"
        out["verdict_note"] = (
            "no arbitration intervention beat the no-replay baseline by more "
            "than half the replay gap and none restored conditionality. A1 is "
            "killed by the dossier's own written kill condition.")
    out["caution"] = (
        "a 'restored' reading bounds nothing beyond the k actually run; "
        "sequential editing collapses at k~32. Retention is a RATIO to each "
        "skill's own just-after value -- read `mean_just_after` and "
        "`prior_abs_clean` beside every retention number.")
    return out


def read_all(rs):
    v = per_arm(rs)
    return v, reading(v)


def figure(rs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({r["arm"] for r in rs})
    w = [r for r in rs if r["written"] and not r["copyable"]]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for arm in arms:
        a = [r for r in w if r["arm"] == arm and r["age"] >= CLEAN_AGE]
        if a:
            ks = sorted({r["k"] for r in a})
            ax[0].plot(ks, [np.mean([r["retention"] for r in a if r["k"] == k])
                            for k in ks], "o-", label=arm)
            ax[1].plot(ks, [np.mean([r["unprompted"] for r in a if r["k"] == k])
                            for k in ks], "o-", label=arm)
        z = [r for r in w if r["arm"] == arm and r["age"] == 0
             and r["unknown_marker_applies_rule"] is not None]
        if z:
            ks = sorted({r["k"] for r in z})
            ax[2].plot(ks, [np.mean([r["unknown_marker_applies_rule"]
                                     for r in z if r["k"] == k]) for k in ks],
                       "d-", label=arm)
        g = [r for r in w if r["arm"] == arm and r["diag_rank"] is not None
             and r["age"] > 0]
        if g:
            ks = sorted({r["k"] for r in g})
            ax[3].semilogy(ks, [np.mean([r["diag_rank"] for r in g
                                         if r["k"] == k]) for k in ks],
                           "^-", label=arm)
    ax[0].axhline(DYNRANGE, color="r", ls=":", lw=1)
    ax[0].set(xlabel="k", ylabel="retention of prior skills (ratio)",
              ylim=(0, 1.05), title="retention vs k (age>=2)")
    ax[1].set(xlabel="k", ylabel="prior-skill accuracy (absolute)",
              ylim=(0, 1.05), title="absolute retention (ratio-free)")
    ax[2].axhline(UMAR_BLOCK, color="k", ls=":", lw=1)
    ax[2].set(xlabel="k", ylim=(-0.05, 1.05),
              ylabel="newest rule under an unclaimed cue",
              title="umar_age0: is the NEWEST write conditional")
    ax[3].set(xlabel="k", ylabel="rank of prior skill's true answer",
              title="masking vs erasure (lower = intact)")
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ---- decision 0012 clause 1: identifiability dry-run ------------------------

SYNTH_SKILLS = [5, 7, 8, 9, 11, 24, 28, 31, 33, 34]
SYNTH_JUST = {5: 0.96, 7: 1.00, 8: 1.00, 9: 1.00, 11: 0.80,
              24: 0.78, 28: 1.00, 31: 1.00, 33: 1.00, 34: 1.00}
# Each world states, per arm: (prior-skill retention at small k, the same at
# large k, the newest write's umar_age0). LEVELS, not decay rates, and
# CALIBRATED to what e015 measured on this exact suite so the known-positive
# band is live in the dry-run: no_replay 0.018, replay25 0.679-0.692.
# The intervention columns are the rivals' PREDICTIONS, not measurements.
WORLDS = {
    # A1: arbitration is the whole problem and the interventions fix it
    "A1": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .54),
           "negcue": (.62, .62, .30), "cueanchor": (.55, .55, .35),
           "reserve": (.60, .60, .25), "selfdistil": (.66, .66, .28)},
    # A2: arbitration restored, retention holds early and falls by large k
    "A2": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .54),
           "negcue": (.62, .10, .30), "cueanchor": (.55, .08, .35),
           "reserve": (.60, .12, .25), "selfdistil": (.66, .11, .28)},
    # A3: conditionality restored, retention NOT restored. The deflation.
    "A3": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .54),
           "negcue": (.05, .05, .30), "cueanchor": (.04, .04, .35),
           "reserve": (.06, .06, .25), "selfdistil": (.07, .07, .28)},
    # nothing works: A1 killed by its own kill condition
    "NULL": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .54),
             "negcue": (.03, .03, .95), "cueanchor": (.02, .02, .96),
             "reserve": (.04, .04, .94), "selfdistil": (.05, .05, .95)},
    # nothing degrades in the k range run: must SUPPRESS, not read as a win
    "NODYN": {a: (.97, .97, .96) for a in
              ["no_replay", "replay25", "negcue", "cueanchor", "reserve",
               "selfdistil"]},
    # THE WORLD THE DRY-RUN WAS MISSING, added by the verifier 2026-08-17.
    # Every world above inherits e015's replay25 umar_age0 of 0.54. The pilot
    # showed that quantity is k-DEPENDENT and near-binary per write, so a
    # truncated or unlucky replay25 arm can post a HIGH pooled umar_age0 while
    # its retention stays inside the known-positive band. Here `selfdistil`
    # retains well with a still-unconditional newest write: the ONLY correct
    # reading is RETENTION_WITHOUT_ARBITRATION, and before the UMAR_REL_CAP
    # edit the relative criterion would have called it A1.
    "HIREPLAY": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .85),
                 "negcue": (.03, .03, .95), "cueanchor": (.02, .02, .96),
                 "reserve": (.04, .04, .94), "selfdistil": (.66, .66, .88)},
    # the write-tax trap: three arms that barely write. Their RATIO retention
    # is fine and their ABSOLUTE retention is not; the flag must fire.
    "TAX": {"no_replay": (.02, .02, .96), "replay25": (.69, .69, .54),
            "negcue": (.62, .62, .30), "cueanchor": (.62, .62, .96),
            "reserve": (.60, .60, .94), "selfdistil": (.66, .66, .95)},
}


def synth(world, seeds=(0,), noise=0.01, rng_seed=0, just_scale=None,
          ret_scale=None):
    spec = WORLDS[world]
    rng = np.random.default_rng(rng_seed)
    runs = []
    for arm, (lo, hi, umar0) in spec.items():
        for sd in seeds:
            order = [int(SYNTH_SKILLS[i]) for i in
                     np.random.default_rng(100 + sd).permutation(
                         len(SYNTH_SKILLS))]
            half = (len(order) + 1) // 2
            js = (just_scale or {}).get(arm, 1.0)
            steps = []
            for i, s in enumerate(order):
                unp, umar, dr = {}, {}, {}
                for j in order:
                    pos = order.index(j)
                    if pos > i:
                        unp[str(j)], umar[str(j)] = 0.0, 0.0
                        continue
                    age = i - pos
                    lvl = (lo if (i + 1) <= half else hi) * \
                        (ret_scale or {}).get(arm, 1.0)
                    r = 1.0 if age == 0 else max(0.0, min(1.0, lvl))
                    e = float(rng.normal(0, noise))
                    unp[str(j)] = round(min(1.0, max(
                        0.0, SYNTH_JUST[j] * js * r + e)), 3)
                    umar[str(j)] = umar0 if age == 0 else 0.0
                    dr[str(j)] = 20.0 if age else 1.0
                steps.append({"k": i + 1, "skill": s, "cum_steps": 60 * (i + 1),
                              "unprompted": unp,
                              "unknown_marker_applies_rule": umar,
                              "diag_rank": dr, "newest_rule_under_j": {},
                              "generality_nats": 2.9,
                              "write_tripwire_ok": True,
                              "n_aux_examples": 240})
            runs.append({"arm": arm, "seed": sd, "order": order, "rank": 8,
                         "base": "synth", "n_params": 498431872,
                         "n_skills": len(order), "tag": "", "steps": steps})
    return runs


def dryrun():
    """Every rival's PREDICTED outcome must produce its OWN verdict category.
    An INCONCLUSIVE here means the design is broken and no compute may be spent
    (decision 0012 clause 1). Milliseconds, no model."""
    want = {"A1": "A1_ARBITRATION_FIXES_RETENTION",
            "A2": "A2_ARBITRATION_INSUFFICIENT_AT_LARGE_K",
            "A3": "A3_ARBITRATION_WITHOUT_RETENTION",
            "NULL": "A0_NO_INTERVENTION_WORKED",
            "NODYN": "SUPPRESSED_NO_DYNAMIC_RANGE",
            "HIREPLAY": "RETENTION_WITHOUT_ARBITRATION"}
    ok = True
    for world, verd in want.items():
        _, rd = read_all(rows(synth(world)))
        good = rd["verdict"] == verd
        ok &= good
        print(f"  {world}: dynrange={rd['dynamic_range']} "
              f"knownpos={rd.get('known_positive')} "
              f"gap_replay={rd.get('gap_replay25_minus_noreplay')} "
              f"umar0={ {a: rd['both_quantities_by_arm'][a]['umar_age0'] for a in sorted(rd['both_quantities_by_arm'])} } "
              f"verdict={rd['verdict']} -> {'OK' if good else 'BROKEN'}")
    # the write-tax trap: three arms that write at 40% strength and retain
    # nothing in absolute terms must NOT be read as retention winners
    _, rd = read_all(rows(synth("TAX", just_scale={
        "cueanchor": 0.4, "reserve": 0.4, "selfdistil": 0.4})))
    taxed = set(rd["write_taxed_arms"]) >= {"cueanchor", "reserve", "selfdistil"}
    print(f"  TAX: write_taxed_arms={rd['write_taxed_arms']} "
          f"verdict={rd['verdict']} -> {'OK' if taxed else 'BROKEN'}")
    ok &= taxed
    # the known-positive must fire when the reference arm drifts out of band
    _, rd = read_all(rows(synth("A1", ret_scale={"replay25": 0.45})))
    kp = rd["verdict"].startswith("SUPPRESSED_KNOWN_POSITIVE")
    print(f"  KNOWNPOS-DRIFT: replay25={rd['known_positive_replay25']} "
          f"verdict={rd['verdict']} -> {'OK' if kp else 'BROKEN'}")
    ok &= kp
    print("identifiability dry-run:", "PASS" if ok else "FAIL")
    assert ok, "a predicted outcome does not produce its own verdict category"


def main():
    ART.mkdir(exist_ok=True)
    runs = load_runs()
    if not runs:
        print("no runs in", RUNS)
        return
    rs = rows(runs)
    with (ART / "results.csv").open("w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        wr.writeheader()
        wr.writerows(rs)
    v, rd = read_all(rs)
    out = {"per_arm": v, "_reading": rd}
    (ART / "verdict.json").write_text(json.dumps(out, indent=1))
    try:
        figure(rs, ART / "results.png")
    except Exception as e:                       # a missing backend must not
        print("figure skipped:", e)              # cost the verdict
    print(json.dumps(out, indent=1))
    print(f"\n{len(rs)} rows -> artifacts/")


if __name__ == "__main__":
    import sys
    dryrun() if "--dryrun" in sys.argv else main()
