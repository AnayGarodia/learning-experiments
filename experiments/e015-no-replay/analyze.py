"""e015 analysis: the dynamic-range gate first, then C1 / C2 / C3.

Reading order is not negotiable and is enforced in code:

  0. every arm's writes must land (tripwire), or that arm is not a retention
     measurement at all;
  1. the e004g conditionality block (`unknown_marker_applies_rule` at
     age >= 2 must stay <= 0.20), per arm;
  2. the DYNAMIC-RANGE GATE: the no-replay baseline must visibly degrade in
     the k range actually run. If it does not, there is no forgetting to
     prevent and no allocation arm may be read -- that is exactly the
     ambiguity e004g's k=3 pilot could not resolve, and it is a verdict
     ("NO_DYNAMIC_RANGE"), not a result;
  3. only then C1 vs C2, and C3 across model sizes.

No binning, no smoothing, no interpolation: every plotted point is measured.
Per-skill and per-seed values are printed for every headline number
(decision 0012 clause 5; a mean that hid a 2-1 seed split voided an e011
claim).
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RUNS = HERE / "runs"
ART = HERE / "artifacts"

# n = 200 held-out queries per skill per condition. Binomial SE <= 0.035
# (worst case p=0.5), 0.021 at p=0.9. Difference of two arms at p~0.9:
# SE_diff = 0.030 -> single-seed MDE 0.060 at 2 SE; over 3 seeds, 0.035.
MDE_1SEED, MDE_3SEED = 0.060, 0.035
EVAL_RESOLUTION = 1 / 200            # smallest accuracy step the eval can see
CLEAN_AGE = 2        # e004g verifier: younger rows are conditionality-confounded
UMAR_BLOCK = 0.20    # e004g's pre-registered conditionality threshold
DYNRANGE = 0.85      # no_replay must fall to <= this or nothing has been stressed
COPYABLE = {0, 1, 2, 3}   # e004g type-A, fixed-offset copyable at ~0.95
ALLOC_ARMS = ("sparse", "orth")
BASELINE, REFERENCE = "no_replay", "replay25"


def load_runs():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("*.json"))]


def rows(runs):
    """One row per (arm, seed, k, measured skill). `just_after` is the skill's
    unprompted accuracy at the step it was written -- the denominator that
    retention is expressed against."""
    out = []
    for r in runs:
        just = {}
        for st in r["steps"]:
            just.setdefault(str(st["skill"]), st["unprompted"][str(st["skill"])])
            for sk, acc in st["unprompted"].items():
                out.append({
                    "arm": r["arm"], "seed": r["seed"], "base": r.get("base"),
                    "n_params": r.get("n_params"), "rank": r.get("rank"),
                    "n_lora_params": r.get("n_lora_params"),
                    "n_skills": r.get("n_skills"), "tag": r.get("tag", ""),
                    "k": st["k"], "cum_steps": st["cum_steps"],
                    "consolidated_skill": st["skill"], "measured_skill": int(sk),
                    "age": st["k"] - (r["order"].index(int(sk)) + 1),
                    "written": sk in just, "copyable": int(sk) in COPYABLE,
                    "n_eval_this_row": st.get("n_by_skill", {}).get(sk),
                    "unprompted": acc, "just_after": just.get(sk),
                    "retention": (round(acc / just[sk], 4)
                                  if just.get(sk) else None),
                    "unknown_marker_applies_rule":
                        st["unknown_marker_applies_rule"].get(sk),
                    "generality_nats": st["generality_nats"],
                    "write_tripwire_ok": st["write_tripwire_ok"],
                    "n_replay_examples": st.get("n_replay_examples"),
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
        seeds = sorted({r["seed"] for r in a})
        umar_clean = [r["unknown_marker_applies_rule"] for r in a
                      if r["age"] >= CLEAN_AGE
                      and r["unknown_marker_applies_rule"] is not None]
        v[arm] = {
            "bases": sorted({r["base"] for r in a}),
            "n_params": max(r["n_params"] or 0 for r in a),
            "rank": max(r["rank"] or 0 for r in a),
            "n_skills": max(r["n_skills"] or 0 for r in a),
            "max_k": max(r["k"] for r in a),
            "n_rows_clean": len(clean),
            # ---- write quality: an arm whose writes do not land is not a
            # high-retention arm, it is a failed arm.
            "all_writes_landed": all(r["write_tripwire_ok"] for r in a),
            # arm-level exclusion is COARSE: one marginal write out of twelve
            # voids the whole arm. `orth` is the risk (its last skills carry
            # eleven accumulated projection constraints). Emitted per skill so
            # the verifier can rule on excluding ROWS instead of the arm.
            "writes_landed_by_skill": {
                str(sk): bool(all(r["write_tripwire_ok"] for r in a
                                  if r["measured_skill"] == sk))
                for sk in sorted({r["measured_skill"] for r in a})},
            "failed_write_skills": sorted({
                r["measured_skill"] for r in a if not r["write_tripwire_ok"]}),
            "mean_just_after": _mean([r["just_after"] for r in a]),
            "just_after_by_skill": {
                str(sk): _mean([r["just_after"] for r in a
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in a})},
            # ---- retention. Per seed AND per skill, never only the mean.
            "retention_clean_per_seed": {
                str(sd): _mean([r["retention"] for r in clean
                                if r["seed"] == sd]) for sd in seeds},
            "retention_clean_by_skill": {
                str(sk): _mean([r["retention"] for r in clean
                                if r["measured_skill"] == sk])
                for sk in sorted({r["measured_skill"] for r in clean})},
            "retention_by_age": {
                str(g): _mean([r["retention"] for r in prior if r["age"] == g])
                for g in sorted({r["age"] for r in prior})},
            "retention_at_max_k_per_seed": {
                str(sd): _mean([r["retention"] for r in prior
                                if r["seed"] == sd and r["k"] == max(
                                    x["k"] for x in a if x["seed"] == sd)])
                for sd in seeds},
            "min_retention_row": round(min(
                (r["retention"] for r in prior), default=float("nan")), 4),
            # ---- e004g's conditionality probe, carried unchanged, read per age
            "umar_by_age": {
                str(g): _mean([r["unknown_marker_applies_rule"] for r in a
                               if r["age"] == g])
                for g in sorted({r["age"] for r in a})},
            "max_umar_age_ge_clean": max(umar_clean) if umar_clean else None,
            # INSTRUMENT DEFECT FOUND BY THE PILOT, kept visible rather than
            # patched away. e004g's K0 block reads this probe at age >= 2,
            # which asks "is the OLD rule still format-attached". In the
            # no-replay arms that reads 0.000 and the block does NOT fire --
            # yet the write is completely unconditional, because it is the
            # NEWEST rule that owns the format default. The unconditionality
            # lives at AGE 0 and the pre-registered aggregation is blind to
            # it. Reported as its own quantity; the pre-registered block is
            # NOT silently redefined.
            "umar_age0_newest_write": _mean(
                [r["unknown_marker_applies_rule"] for r in a
                 if r["age"] == 0
                 and r["unknown_marker_applies_rule"] is not None]),
            # ---- collateral damage
            "generality_drift_nats": round(
                max(r["generality_nats"] for r in a)
                - min(r["generality_nats"] for r in a), 4),
            "max_unprompted_before_write": max(
                [r["unprompted"] for r in rs if r["arm"] == arm
                 and not r["written"]], default=None),
            "replay_examples_at_max_k": max(
                [r["n_replay_examples"] or 0 for r in a], default=0),
        }
        # BLOCKING (2026-08-17): a never-written skill reading above the same
        # 0.20 control level means another skill's answer satisfies its marker
        # -- the e004g B_double/B_triple defect. Was a printed field nobody
        # read; it is now folded into k0_blocked, so the arm drops out of
        # `arms_readable` exactly as an unconditional write does.
        v[arm]["pool_blocked"] = bool(
            v[arm]["max_unprompted_before_write"] is not None
            and v[arm]["max_unprompted_before_write"] > UMAR_BLOCK)
        v[arm]["k0_blocked"] = bool(
            v[arm]["pool_blocked"]
            or (v[arm]["max_umar_age_ge_clean"] is not None
                and v[arm]["max_umar_age_ge_clean"] > UMAR_BLOCK))
        v[arm]["retention_clean_mean"] = _mean(
            list(v[arm]["retention_clean_per_seed"].values()))
    return v


def per_base(rs):
    """DEFECT FOUND BY THE DRY-RUN: keying arms by NAME ALONE silently merged
    the two model sizes -- C3's own arm is `no_replay` on a bigger base, so it
    collapsed into the `no_replay` entry and C3 could never fire. Arms are
    grouped by (base, arm), the C1/C2 read runs on the PRIMARY base only, and
    C3 compares across bases."""
    out = {}
    for b in sorted({r["base"] for r in rs}, key=lambda x: str(x)):
        out[str(b)] = per_arm([r for r in rs if r["base"] == b])
    return out


def reading(v, others=None):
    """The verdict rules, in the order DESIGN.md fixes them.

    `v` is the primary base's per-arm dict; `others` maps base -> per-arm dict
    for every base (including the primary) and drives the C3 read."""
    out = {"MDE_single_seed": MDE_1SEED, "MDE_3seed": MDE_3SEED,
           "eval_resolution": EVAL_RESOLUTION, "clean_age": CLEAN_AGE,
           "dynamic_range_threshold": DYNRANGE}
    readable = [a for a in v if not v[a]["k0_blocked"]
                and v[a]["all_writes_landed"]]
    out["arms_blocked_by_K0"] = [a for a in v if v[a]["k0_blocked"]]
    out["arms_with_failed_writes"] = [a for a in v if not v[a]["all_writes_landed"]]
    out["arms_readable"] = sorted(readable)
    out["umar_age0_newest_write_by_arm"] = {
        a: v[a].get("umar_age0_newest_write") for a in sorted(v)}
    out["unconditional_newest_write_arms"] = sorted(
        a for a in v if (v[a].get("umar_age0_newest_write") or 0) > UMAR_BLOCK)
    out["unconditional_write_note"] = (
        "a freshly written rule that answers under a never-consolidated "
        "marker is FORMAT-attached, not cue-attached. Where this is high in "
        "EVERY arm it cannot discriminate arms, but it does change what "
        "retention means: a prior skill at 0.000 may be OUTRANKED by an "
        "unconditional newest rule rather than erased. Read it beside "
        "`base_rank_control.json` before calling any retention number "
        "interference.")

    def seedvals(arm):
        return [x for x in v.get(arm, {}).get(
            "retention_clean_per_seed", {}).values() if x is not None]

    # ---- 2. the dynamic-range gate ----------------------------------------
    base = seedvals(BASELINE)
    if BASELINE not in v:
        out["dynamic_range"] = "NOT_RUN"
    elif not base or v[BASELINE]["n_rows_clean"] == 0:
        out["dynamic_range"] = "NO_CLEAN_ROWS"
    elif BASELINE in out["arms_blocked_by_K0"]:
        out["dynamic_range"] = "BASELINE_K0_BLOCKED"
    elif not v[BASELINE]["all_writes_landed"]:
        out["dynamic_range"] = "BASELINE_WRITES_FAILED"
    elif max(base) <= DYNRANGE:
        out["dynamic_range"] = "OK"
    else:
        out["dynamic_range"] = "NO_DYNAMIC_RANGE"
    out["baseline_retention_per_seed"] = v.get(BASELINE, {}).get(
        "retention_clean_per_seed")
    out["baseline_max_k"] = v.get(BASELINE, {}).get("max_k")

    # ---- 3. C1 vs C2 -------------------------------------------------------
    out["verdict"] = None
    if out["dynamic_range"] != "OK":
        out["verdict"] = ("SUPPRESSED_" + out["dynamic_range"])
        # BUG FOUND ON PILOT OUTPUT: one note was attached to every non-OK
        # dynamic-range case, so a K0-blocked baseline was reported with the
        # words "did not visibly degrade" while its retention was 0.018.
        out["verdict_note"] = {
            "NO_DYNAMIC_RANGE":
                "no allocation arm may be read: the no-replay baseline did "
                "NOT visibly degrade in the k range run, so every arm looking "
                "fine is consistent with nothing having been stressed. This "
                "is the e004g k=3 ambiguity, and it is a verdict.",
            "BASELINE_K0_BLOCKED":
                "the baseline's conditionality probe exceeded 0.20 at "
                "age >= 2, so the pre-registered K0 block fires and the "
                "builder issues no rival verdict. NOTE the block reads a MAX "
                "over every clean row -- inspect `umar_by_age` and the "
                "offending rows before ruling; the e004g verifier already "
                "replaced exactly this max-over-noisy-rows aggregation for "
                "the retention verdict and left it in place for K0.",
            "BASELINE_WRITES_FAILED":
                "the baseline's writes did not land, so its retention rows "
                "are not a measurement of anything.",
            "NOT_RUN": "the no-replay baseline was not run.",
            "NO_CLEAN_ROWS":
                "no conditionality-clean (age >= 2) baseline rows exist.",
        }[out["dynamic_range"]]
        # reported regardless, so a blocked run still shows whether the
        # baseline degraded -- the two questions are independent
        out["baseline_degraded_ignoring_block"] = (
            None if not base else bool(max(base) <= DYNRANGE))
    else:
        b = _mean(base)
        ref = seedvals(REFERENCE)
        gap_replay = round(_mean(ref) - b, 4) if ref else None
        out["retention_baseline_mean"] = b
        out["gap_replay25_minus_noreplay"] = gap_replay
        out["gap_replay25_per_seed"] = (
            {s: round(v[REFERENCE]["retention_clean_per_seed"][s]
                      - v[BASELINE]["retention_clean_per_seed"][s], 4)
             for s in v[REFERENCE]["retention_clean_per_seed"]
             if s in v[BASELINE]["retention_clean_per_seed"]}
            if REFERENCE in v else None)
        gaps = {}
        for arm in ALLOC_ARMS:
            if arm in readable and seedvals(arm):
                gaps[arm] = round(_mean(seedvals(arm)) - b, 4)
        out["gap_allocation_minus_noreplay"] = gaps
        out["gap_allocation_per_seed"] = {
            arm: {s: round(v[arm]["retention_clean_per_seed"][s]
                           - v[BASELINE]["retention_clean_per_seed"][s], 4)
                  for s in v[arm]["retention_clean_per_seed"]
                  if s in v[BASELINE]["retention_clean_per_seed"]}
            for arm in gaps}
        def sign_consistent(arm):
            """decision 0012 corollary: a mean may not hide a seed split."""
            s = {sd: (v[arm]["retention_clean_per_seed"][sd]
                      > v[BASELINE]["retention_clean_per_seed"][sd])
                 for sd in v[arm]["retention_clean_per_seed"]
                 if sd in v[BASELINE]["retention_clean_per_seed"]}
            return s, len(set(s.values())) <= 1

        if REFERENCE not in readable or gap_replay is None:
            out["verdict"] = "INCONCLUSIVE_NO_REFERENCE_ARM"
        elif gap_replay <= MDE_3SEED:
            # ADDED 2026-08-16, BEFORE any e015 retention curve was read, and
            # PROMPTED BY e004g's k=4 collapse: at k=12 with 15 round-robin
            # replay batches split over 11 priors, 25% replay may itself buy
            # nothing resolvable. The original rule stopped there and never
            # consulted the allocation gaps -- so the STRONGEST possible C1
            # result (allocation holds retention where rehearsal does not)
            # returned REPLAY_INEFFECTIVE and was suppressed. The dossier's C1
            # prediction is level-based ("retention at 0% replay with
            # allocation >= retention at 25% replay without it"), which is
            # testable whether or not replay works.
            best = max(gaps, key=gaps.get) if gaps else None
            signs, consistent = sign_consistent(best) if best else ({}, False)
            if best and gaps[best] > MDE_3SEED and consistent:
                out["verdict"] = "C1_ALLOCATION_BEATS_REPLAY"
                out["best_allocation_arm"] = best
                out["allocation_beats_baseline_per_seed"] = signs
                out["verdict_note"] = (
                    "rehearsal bought nothing resolvable at this k, and "
                    "structural allocation did. This is C1 in its strongest "
                    "form and it does NOT depend on the replay reference.")
            elif best and gaps[best] > MDE_3SEED:
                out["verdict"] = "INCONCLUSIVE_SEEDS_SPLIT"
                out["allocation_beats_baseline_per_seed"] = signs
            else:
                out["verdict"] = "REPLAY_INEFFECTIVE"
                out["verdict_note"] = (
                    "neither rehearsal nor allocation bought anything "
                    "resolvable over the no-replay baseline.")
        elif not gaps:
            out["verdict"] = "INCONCLUSIVE_NO_ALLOCATION_ARM"
        else:
            best = max(gaps, key=gaps.get)
            frac = gaps[best] / gap_replay
            out["best_allocation_arm"] = best
            out["allocation_fraction_of_replay"] = round(frac, 3)
            # per-seed sign consistency: a mean must not hide a split (e011)
            signs = {s: (v[best]["retention_clean_per_seed"][s]
                         > v[BASELINE]["retention_clean_per_seed"][s])
                     for s in v[best]["retention_clean_per_seed"]
                     if s in v[BASELINE]["retention_clean_per_seed"]}
            out["allocation_beats_baseline_per_seed"] = signs
            if len(set(signs.values())) > 1:
                out["verdict"] = "INCONCLUSIVE_SEEDS_SPLIT"
            elif gaps[best] >= gap_replay - MDE_3SEED:
                out["verdict"] = "C1_ALLOCATION_SUFFICES"
            elif frac >= 0.5:
                out["verdict"] = "C1_PARTIAL"
            else:
                out["verdict"] = "C2_REPLAY_DOES_THE_WORK"
            # C1_PARTIAL only exists as a category while the replay gap is
            # comfortably larger than the MDE. At gap_replay ~ 2*MDE the band
            # [0.5, 1 - MDE/gap] is nearly empty and "allocation suffices" and
            # "allocation buys half" are not separable -- say so rather than
            # let a knife-edge frac decide.
            band = round(max(0.0, (1 - MDE_3SEED / gap_replay) - 0.5), 3)
            out["C1_partial_band_width"] = band
            if band <= 0.05:
                out["verdict_note"] = (
                    "C1 and C1_PARTIAL are NOT separable at this replay gap "
                    f"({gap_replay}); the band between them is {band} wide. "
                    "Report the fraction, not the category.")

    # ---- 4. C3, the capacity wall (needs a second model size) --------------
    by_size = {}
    for bs, av in (others or {}).items():
        for a in av:
            by_size.setdefault(av[a]["n_params"], {})[a] = \
                av[a]["retention_clean_mean"]
    out["retention_by_model_size"] = {str(k): x for k, x in by_size.items()}
    if len(by_size) < 2:
        out["C3"] = "UNTESTED_NEEDS_SECOND_MODEL_SIZE"
    else:
        spreads = {}
        for sz, d in by_size.items():
            xs = [x for x in d.values() if x is not None]
            spreads[str(sz)] = round(max(xs) - min(xs), 4) if len(xs) > 1 else None
        shared = set.intersection(*[set(d) for d in by_size.values()])
        size_gap = None
        if shared:
            szs = sorted(by_size)
            size_gap = round(abs(_mean([by_size[szs[0]][a] for a in shared])
                                 - _mean([by_size[szs[-1]][a] for a in shared])),
                             4)
        out["method_spread_within_size"] = spreads
        out["size_gap_at_matched_arms"] = size_gap
        ms = [x for x in spreads.values() if x is not None]
        if ms and max(ms) <= MDE_3SEED and size_gap and size_gap > MDE_3SEED:
            out["C3"] = "C3_CAPACITY_WALL"
        elif ms and max(ms) > MDE_3SEED:
            out["C3"] = "C3_REFUTED_METHOD_MATTERS"
        else:
            out["C3"] = "INCONCLUSIVE"
    out["caution"] = (
        "a 'no collapse' reading bounds nothing beyond the k actually run; "
        "sequential editing collapses at k~32. Retention is a RATIO to each "
        "skill's own just-after value, so an arm that writes weakly can post "
        "high retention -- read `mean_just_after` beside every retention "
        "number.")
    return out


def read_all(rs):
    """Group by base, pick the primary base (the one carrying the most arms)
    and read C1/C2 there; C3 reads across bases."""
    pb = per_base(rs)
    # the SMALLEST base is the primary grid base by construction (C3's second
    # size is a bigger model). Deterministic; "the base with the most arms"
    # tie-broke arbitrarily and picked the C3 arm in the dry-run.
    primary = min(pb, key=lambda b: (min(x["n_params"] for x in pb[b].values()),
                                     str(b)))
    rd = reading(pb[primary], pb)
    rd["primary_base"] = primary
    rd["bases_present"] = sorted(pb)
    return pb, rd


def figure(rs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    arms = sorted({r["arm"] for r in rs})
    w = [r for r in rs if r["written"] and not r["copyable"]]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
    for arm in arms:
        a = [r for r in w if r["arm"] == arm and r["age"] > 0]
        if a:
            ks = sorted({r["k"] for r in a})
            ax[0].plot(ks, [np.mean([r["retention"] for r in a if r["k"] == k])
                            for k in ks], "o-", label=arm)
            ages = sorted({r["age"] for r in a})
            ax[1].plot(ages, [np.mean([r["retention"] for r in a
                                       if r["age"] == g]) for g in ages],
                       "s--", label=arm)
        b = [r for r in w if r["arm"] == arm]
        ks = sorted({r["k"] for r in b})
        ax[2].plot(ks, [np.mean([r["generality_nats"] for r in b if r["k"] == k])
                        for k in ks], "^-", label=arm)
        u = [r for r in b if r["unknown_marker_applies_rule"] is not None
             and r["age"] >= CLEAN_AGE]
        if u:
            ks = sorted({r["k"] for r in u})
            ax[3].plot(ks, [np.mean([r["unknown_marker_applies_rule"]
                                     for r in u if r["k"] == k]) for k in ks],
                       "d-", label=arm)
    ax[0].axhline(DYNRANGE, color="r", ls=":", lw=1)
    ax[0].set(xlabel="k (skills written)", ylabel="retention of PRIOR skills",
              ylim=(0, 1.05), title="retention vs k (red = dynamic-range gate)")
    ax[1].set(xlabel="age (writes since this skill)", ylabel="retention",
              ylim=(0, 1.05), title="retention vs age")
    ax[2].set(xlabel="k", ylabel="nats/token on held-out text",
              title="generality (collateral damage)")
    ax[3].axhline(UMAR_BLOCK, color="k", ls=":", lw=1)
    ax[3].set(xlabel="k", ylim=(-0.05, 1.05),
              ylabel="rule fires under an unclaimed marker",
              title=f"conditionality, age>={CLEAN_AGE} (dotted = block)")
    for x in ax:
        x.legend(fontsize=8)
        x.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


# ---- decision 0012 clause 1: identifiability dry-run ------------------------

SYNTH_SKILLS = [5, 7, 8, 9, 11, 13, 14, 15]
SYNTH_JUST = {5: 0.94, 7: 1.00, 8: 1.00, 9: 0.985, 11: 0.795, 13: 1.00,
              14: 0.90, 15: 0.85}
# per-write retention loss of each arm under each rival mechanism
DECAY = {
    # C1: structural allocation buys what replay buys
    "C1": {"no_replay": .10, "replay25": .01, "replay01": .06,
           "sparse": .012, "orth": .02, "separate_adapters": .0},
    # C2: allocation tracks the no-replay baseline; only rehearsal helps
    "C2": {"no_replay": .10, "replay25": .01, "replay01": .06,
           "sparse": .095, "orth": .09, "separate_adapters": .0},
    # C3: method barely matters, model size does (second size decays less)
    "C3": {"no_replay": .10, "replay25": .095, "replay01": .10,
           "sparse": .098, "orth": .10, "separate_adapters": .0},
    # the e004g world: nothing degrades in the k range run, so nothing is
    # measurable. This MUST come back as SUPPRESSED, not as a C1 win.
    "NODYN": {"no_replay": .0, "replay25": .0, "replay01": .0,
              "sparse": .0, "orth": .0, "separate_adapters": .0},
    # C1 in its strongest form: rehearsal has stopped working by this k
    # (e004g's replay-25% arm collapses at k=4) but allocation still holds.
    # The original verdict chain returned REPLAY_INEFFECTIVE here and never
    # looked at the allocation arms.
    "C1NOREF": {"no_replay": .10, "replay25": .095, "replay01": .10,
                "sparse": .012, "orth": .02, "separate_adapters": .0},
}


def synth(mech, arms=None, seeds=(0, 1, 2), noise=0.01, n_params=494032768,
          decay_scale=1.0, rng_seed=0):
    rng = np.random.default_rng(rng_seed)
    runs = []
    for arm in (arms or list(DECAY[mech])):
        for sd in seeds:
            order = [int(SYNTH_SKILLS[i]) for i in
                     np.random.default_rng(100 + sd).permutation(
                         len(SYNTH_SKILLS))]
            steps = []
            for i, s in enumerate(order):
                seen = order if arm != "separate_adapters" else [s]
                unp, umar = {}, {}
                for j in seen:
                    pos = order.index(j)
                    if pos > i:
                        unp[str(j)], umar[str(j)] = 0.0, 0.0
                        continue
                    age = i - pos
                    r = max(0.0, 1 - DECAY[mech][arm] * decay_scale * age)
                    e = float(rng.normal(0, noise))
                    unp[str(j)] = round(min(1.0, max(0.0,
                                                     SYNTH_JUST[j] * r + e)), 3)
                    # the K0 pattern e004g measured: fresh writes hold the
                    # format default, older ones are marker-conditional
                    umar[str(j)] = 1.0 if age == 0 else (0.5 if age == 1 else 0.0)
                steps.append({"k": i + 1, "skill": s, "cum_steps": 60 * (i + 1),
                              "unprompted": unp,
                              "unknown_marker_applies_rule": umar,
                              "generality_nats": 2.84,
                              "write_tripwire_ok": True,
                              "n_by_skill": {str(j): 200 for j in seen}})
            runs.append({"arm": arm, "seed": sd, "order": order, "rank": 8,
                         "base": f"synth-{n_params}", "n_params": n_params,
                         "n_lora_params": 4400000, "n_skills": len(order),
                         "tag": "", "steps": steps})
    return runs


def dryrun():
    """Every rival's PREDICTED outcome must produce its OWN verdict category.
    An INCONCLUSIVE here means the design is broken and no compute may be
    spent (decision 0012 clause 1). Runs in milliseconds, no model."""
    want = {"C1": ("OK", "C1_ALLOCATION_SUFFICES"),
            "C2": ("OK", "C2_REPLAY_DOES_THE_WORK"),
            # C3's world at ONE size: no method beats another, so the replay
            # reference itself buys nothing -- that is a callable category,
            # and the size comparison is what carries the verdict.
            "C3": ("OK", "REPLAY_INEFFECTIVE"),
            "C1NOREF": ("OK", "C1_ALLOCATION_BEATS_REPLAY"),
            "NODYN": ("NO_DYNAMIC_RANGE", "SUPPRESSED_NO_DYNAMIC_RANGE")}
    ok = True
    for mech, (dyn, verd) in want.items():
        runs = synth(mech)
        if mech == "C3":     # second model size, same methods, less decay
            runs += synth("C3", n_params=1543714304, decay_scale=0.15,
                          rng_seed=1)
        _, rd = read_all(rows(runs))
        good = rd["dynamic_range"] == dyn and (verd is None
                                               or rd["verdict"] == verd)
        if mech == "C3":
            good = good and rd["C3"] == "C3_CAPACITY_WALL"
        else:
            good = good and rd["C3"] == "UNTESTED_NEEDS_SECOND_MODEL_SIZE"
        ok &= good
        print(f"  {mech}: dynrange={rd['dynamic_range']} "
              f"baseline={rd.get('retention_baseline_mean')} "
              f"gap_replay={rd.get('gap_replay25_minus_noreplay')} "
              f"gap_alloc={rd.get('gap_allocation_minus_noreplay')} "
              f"verdict={rd['verdict']} C3={rd['C3']} "
              f"-> {'OK' if good else 'BROKEN'}")
    # a design that cannot tell C1 from C1_PARTIAL is worth knowing about too
    DECAY["MID"] = dict(DECAY["C1"], sparse=.037, orth=.045)
    _, rd = read_all(rows(synth("MID")))
    print(f"  MID (allocation buys ~2/3): verdict={rd['verdict']} "
          f"frac={rd.get('allocation_fraction_of_replay')} "
          f"-> {'OK' if rd['verdict'] == 'C1_PARTIAL' else 'BROKEN'}")
    ok &= rd["verdict"] == "C1_PARTIAL"
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
    pb, rd = read_all(rs)
    out = {"per_base_per_arm": pb, "_reading": rd}
    (ART / "verdict.json").write_text(json.dumps(out, indent=1))
    figure(rs, ART / "results.png")
    print(json.dumps(out, indent=1))
    print(f"\n{len(rs)} rows -> artifacts/results.csv, results.png, verdict.json")
    for arm in sorted({r["arm"] for r in rs}):
        u = [r["unprompted"] for r in rs if r["arm"] == arm and r["written"]]
        print(f"spread[{arm}] unprompted sd={np.std(u):.4f} "
              f"min={min(u):.3f} max={max(u):.3f} n={len(u)}")


if __name__ == "__main__":
    import sys
    dryrun() if "--dryrun" in sys.argv else main()
