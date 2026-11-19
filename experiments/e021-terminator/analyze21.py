"""e021 analysis: the FIDELITY MEDIATOR is a primary outcome, and the write
guard is read before anything else.

Reading order, enforced in code (a later check never runs on numbers an
earlier check has already invalidated):

  0. T4 WRITE GUARD. `mean_just_after` must stay within 0.05 of the pre-fix
     0.927-0.937. The terminator changes the WRITE objective, so if writes got
     weaker nothing downstream is a retention measurement. Note that eval
     metrics were deliberately NOT changed by the fix (`C.accuracy` still
     scores the answer only), so this comparison is like-for-like.
  1. DYNAMIC RANGE. no_replay must degrade in the k range run, or nothing was
     stressed and no control and no mediator is readable.
  2. LEAK / VOID. `selfdistil_offcue` is the predicted negative: same
     machinery, proxy cues that do not exercise the old behaviour. e017 read
     0.012 there. If it now retains, the fix introduced a leak and every
     selfdistil number is VOID.
  3. TERMINATION ACTUALLY FIXED. Free-generation and teacher-forced retention
     must agree within 0.05. This is the design requirement that was failing
     grid-wide and it is the direct test of the fix's premise; if it still
     fails, the fidelity story is not readable.
  4. THE MEDIATION. fidelity (k>=6, prior skills -- exactly the cells that read
     0.000 before the fix) AND retention, together. T1 needs both to move.

T3 (critical window) is reported as its OWN verdict beside the main one, not
folded into it: front-loading is a different manipulation at equal aux cost and
it can succeed or fail independently of whether T1 holds.

`--dryrun` pushes each rival's PREDICTED world through `reading()` and checks
that each yields its own verdict (decision 0012 clause 1).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "e017-arbitration"))
import analyze as A17                                   # noqa: E402

RUNS = HERE / "runs"
ART = HERE / "artifacts"

CLEAN_AGE = A17.CLEAN_AGE          # >=2, the e004g verifier's ruling
MDE_3SEED = A17.MDE_3SEED          # 0.035
DYNRANGE = A17.DYNRANGE            # 0.85
# T4 guard: the pre-fix band, from e017's committed records, and the tolerance
# the dossier fixes. Both stated before any e021 number was read.
JUST_AFTER_PREFIX = (0.927, 0.937)
JUST_AFTER_TOL = 0.05
# The fix's own premise. Pre-fix, `selfdistil_target_fidelity` read 0.000 for
# EVERY prior from k=6 onward, so any rise at all is informative and these two
# levels are far above measurement noise.
FID_RISE, FID_HI = 0.30, 0.60
FREEGEN_TOL = 0.05                 # the standing DESIGN requirement
T1_GAP = 0.15                      # dossier: "within 0.15 of replay25's"
OFFCUE_MAX = 0.15                  # e017 read 0.012 here
BASELINE, REFERENCE, METHOD = "no_replay", "replay25", "selfdistil"
CONTROL, FRONT = "selfdistil_offcue", "selfdistil_front"


def load_runs(runs=RUNS):
    """Records AND `.inprogress` partials. A partial rsynced mid-write can be
    truncated JSON; skip it rather than crash, and say so."""
    out = []
    for p in sorted(Path(runs).glob("*.json*")):
        try:
            out.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print("SKIPPED truncated partial:", p.name)
    return out


def per_arm(rs):
    """Per arm: the write guard, retention both ways, and the mediator.

    Retention is a RATIO to each skill's own just-after accuracy (decision 0012
    clause 7a) so that a weakened write cannot inflate it; the absolute is
    printed beside every ratio."""
    v = {}
    for arm in sorted({r["arm"] for r in rs}):
        a = [r for r in rs if r["arm"] == arm and r["written"]
             and not r["copyable"]]
        if not a:
            continue
        clean = [r for r in a if r["age"] >= CLEAN_AGE]
        maxk = max(r["k"] for r in a)
        v[arm] = {
            "seeds": sorted({r["seed"] for r in a}),
            "max_k": maxk,
            "n_clean_rows": len(clean),
            # T4 guard, on the SAME metric as e017's
            "mean_just_after": A17._mean([r["unprompted"] for r in a
                                          if r["age"] == 0]),
            "just_after_by_seed": {s: A17._mean(
                [r["unprompted"] for r in a if r["age"] == 0 and r["seed"] == s])
                for s in sorted({r["seed"] for r in a})},
            "ret_clean": A17._mean([r["retention"] for r in clean]),
            "ret_clean_by_seed": {s: A17._mean(
                [r["retention"] for r in clean if r["seed"] == s])
                for s in sorted({r["seed"] for r in a})},
            "prior_acc_clean": A17._mean([r["unprompted"] for r in clean]),
            # FREE GENERATION, same rows, same skills, same k
            "prior_freegen_clean": A17._mean([r["freegen"] for r in clean]),
            "just_after_freegen": A17._mean([r["freegen"] for r in a
                                             if r["age"] == 0]),
            # THE MEDIATOR. `fid_late` is exactly the population that read
            # 0.000 grid-wide before the fix: prior skills at k>=6.
            "fid_late": A17._mean([r["target_fidelity"] for r in a
                                   if r["age"] > 0 and r["k"] >= 6]),
            "fid_all": A17._mean([r["target_fidelity"] for r in a
                                  if r["age"] > 0]),
            "fid_by_k": {k: A17._mean([r["target_fidelity"] for r in a
                                       if r["age"] > 0 and r["k"] == k])
                         for k in range(2, maxk + 1)},
        }
    return v


def reading(v):
    """The pre-registered verdict rules, in the fixed order."""
    out = {"per_arm": v, "verdict": None, "t3_verdict": None, "notes": []}

    def g(arm, key):
        return (v.get(arm) or {}).get(key)

    # ---- 0. T4 WRITE GUARD ------------------------------------------------
    lo = JUST_AFTER_PREFIX[0] - JUST_AFTER_TOL
    # COMPOSITION GUARD. `mean_just_after` is a mean over SKILLS, and the
    # pre-fix band was computed over all 10. A partial run's mean is a
    # different set of skills, not a weaker write: at k=1 the first e021
    # partials averaged 0.87 while every skill's write matched e017's value
    # EXACTLY (skill 11: 0.805 -> 0.805). Reading the guard on a partial is
    # therefore a composition artifact, and it is refused rather than reported.
    partial = {a: g(a, "max_k") for a in v if (g(a, "max_k") or 0) < 10}
    out["write_guard"] = {"band": JUST_AFTER_PREFIX, "tol": JUST_AFTER_TOL,
                          "floor": round(lo, 3), "partial_arms": partial}
    if partial:
        out["t4"] = "NOT-COMPARABLE-PARTIAL"
        out["verdict"] = "INCOMPLETE_GRID"
        out["notes"].append(
            f"arms {sorted(partial)} have not reached k=10. mean_just_after is "
            "a mean over SKILLS and the pre-fix band is a 10-skill mean, so a "
            "partial mean is not comparable. Compare per skill instead.")
        return out
    taxed = {a: g(a, "mean_just_after") for a in v
             if (g(a, "mean_just_after") or 0) < lo}
    out["write_guard"]["taxed"] = taxed
    out["t4"] = "WRITE-BROKEN" if taxed else "WRITE-INTACT"
    if taxed:
        out["verdict"] = "T4_WRITE_BROKEN_BY_TERMINATOR"
        out["notes"].append(
            "mean_just_after fell more than 0.05 below the pre-fix band in "
            f"{sorted(taxed)}. The terminator weakened the write, so no "
            "retention number below it is interpretable. STOP.")
        return out

    # ---- 1. DYNAMIC RANGE (before any control is read: if the baseline
    #         does not degrade, neither a leak nor a mediator means anything) ---------------------------------------------
    base = g(BASELINE, "ret_clean")
    out["dynamic_range"] = {"no_replay_ret": base, "must_be_at_most": DYNRANGE}
    if base is None or base > DYNRANGE:
        out["verdict"] = "SUPPRESSED_NO_DYNAMIC_RANGE"
        out["notes"].append(
            f"no_replay retains {base} > {DYNRANGE}: nothing was stressed.")
        return out

    # ---- 2. LEAK / VOID ---------------------------------------------------
    oc, sd = g(CONTROL, "ret_clean"), g(METHOD, "ret_clean")
    out["offcue"] = {"ret_clean": oc, "cap": OFFCUE_MAX}
    if oc is not None and (oc > OFFCUE_MAX or
                           (sd is not None and oc >= sd - MDE_3SEED)):
        out["verdict"] = "VOID_OFFCUE_CONTROL_RETAINS"
        out["notes"].append(
            f"the predicted-negative control retains {oc} (e017: 0.012). The "
            "termination fix introduced a leak; selfdistil is not measuring "
            "distillation on the old cues. VOID.")
        return out

    # ---- 3. IS TERMINATION ACTUALLY FIXED --------------------------------
    gaps = {a: round(abs(g(a, "prior_acc_clean") - g(a, "prior_freegen_clean")), 4)
            for a in v if g(a, "prior_freegen_clean") is not None
            and g(a, "prior_acc_clean") is not None}
    out["freegen_gap"] = gaps
    bad = {a: x for a, x in gaps.items() if x > FREEGEN_TOL}
    if bad:
        out["verdict"] = "SUPPRESSED_TERMINATION_NOT_FIXED"
        out["notes"].append(
            f"teacher-forced and free-generation retention still disagree by "
            f">{FREEGEN_TOL} in {bad}. The fix did not do what it claims, so "
            "the fidelity mediation cannot be read.")
        return out

    # ---- 4. THE MEDIATION -------------------------------------------------
    fid, ref = g(METHOD, "fid_late"), g(REFERENCE, "ret_clean")
    gap = None if (sd is None or ref is None) else round(ref - sd, 4)
    out["mediation"] = {"fid_late": fid, "fid_late_prefix": 0.000,
                        "selfdistil_ret": sd, "replay25_ret": ref,
                        "gap_to_ceiling": gap, "t1_gap_bar": T1_GAP,
                        "fid_rise_bar": FID_RISE, "fid_hi_bar": FID_HI}
    if fid is None or fid < FID_RISE:
        out["verdict"] = "T0_FIX_DID_NOT_RAISE_FIDELITY"
        out["notes"].append(
            f"target fidelity at k>=6 is {fid}, still below {FID_RISE}. The "
            "terminator did not clean the teacher; the diagnosis is wrong or "
            "the fix is in the wrong place.")
    elif fid >= FID_HI and gap is not None and gap <= T1_GAP:
        out["verdict"] = "T1_TEACHER_CORRUPTION_WAS_BINDING"
        out["notes"].append(
            f"fidelity {fid} (was 0.000) and selfdistil retention {sd} within "
            f"{gap} of stored-data replay's {ref}. Continual learning without "
            "stored data at near-rehearsal quality in this harness; the "
            "barrier was an instrumentation defect.")
    else:
        out["verdict"] = "T2_FIDELITY_UP_RETENTION_CAPPED"
        out["notes"].append(
            f"fidelity rose to {fid} but selfdistil still sits {gap} below "
            f"replay25 ({sd} vs {ref}). The corrupted teacher was not the "
            "binding constraint. This is the informative negative.")

    # ---- T3, its own verdict, at EQUAL total aux cost ---------------------
    fr = g(FRONT, "ret_clean")
    out["t3"] = {"front_ret": fr, "uniform_ret": sd,
                 "delta": None if (fr is None or sd is None) else round(fr - sd, 4),
                 "mde": MDE_3SEED, "front_fid_late": g(FRONT, "fid_late")}
    if fr is None or sd is None:
        out["t3_verdict"] = "T3_NO_ARM"
    elif fr - sd > MDE_3SEED:
        out["t3_verdict"] = "T3_CRITICAL_WINDOW_SUPPORTED"
    elif sd - fr > MDE_3SEED:
        out["t3_verdict"] = "T3_FRONT_LOADING_HURTS"
    else:
        out["t3_verdict"] = "T3_NULL_SCHEDULE_DOES_NOT_MATTER"
    out["notes"].append(
        "T3 interpretation is GATED on the prior-art check the dossier "
        "requires: front-loading rehearsal just after acquisition is close to "
        "consolidation-window findings and may be known.")
    return out


# ---- identifiability dry-run (decision 0012 clause 1) ----------------------

def synth(world, seeds=(0, 1, 2), maxk=10, n_skills=10):
    """Synthetic ROWS (not records) for a predicted world. Same fields
    `A17.rows` produces, so `per_arm`/`reading` cannot tell them apart."""
    out = []
    for arm, (just, ret, fid, fgok) in world.items():
        for sd in seeds:
            for k in range(1, maxk + 1):
                for age in range(0, k):
                    acc = just if age == 0 else round(just * ret, 4)
                    fg = acc if fgok else round(acc * 0.02, 4)
                    out.append({
                        "arm": arm, "seed": sd, "base": "synth",
                        "n_params": 0, "rank": 8, "n_skills": n_skills,
                        "tag": "", "k": k, "cum_steps": 60 * k,
                        "consolidated_skill": 0, "measured_skill": age,
                        "age": age, "written": True, "copyable": False,
                        "unprompted": acc, "just_after": just,
                        "freegen": just if age == 0 and fgok else fg,
                        "retention": None if age == 0 else round(ret, 4),
                        "unknown_marker_applies_rule": 0.0,
                        "diag_rank": 1, "newest_rule_under_j": 0.0,
                        "generality_nats": 3.0, "write_tripwire_ok": True,
                        "n_aux_examples": 240,
                        "target_fidelity": None if age == 0 else fid,
                        "cueanchor_weight_sd": None})
    return out


# (arm -> just_after, retention_ratio, target_fidelity, freegen_agrees)
WORLDS = {
    # T1: teacher cleaned, retention climbs to just under the replay ceiling
    "T1_TEACHER_CORRUPTION_WAS_BINDING": (
        {BASELINE: (0.93, 0.02, None, True), REFERENCE: (0.93, 0.90, None, True),
         METHOD: (0.93, 0.82, 0.85, True), CONTROL: (0.93, 0.01, 0.02, True),
         FRONT: (0.93, 0.83, 0.86, True)}, "T3_NULL_SCHEDULE_DOES_NOT_MATTER"),
    # T2: mediator moves, outcome does not
    "T2_FIDELITY_UP_RETENTION_CAPPED": (
        {BASELINE: (0.93, 0.02, None, True), REFERENCE: (0.93, 0.90, None, True),
         METHOD: (0.93, 0.57, 0.85, True), CONTROL: (0.93, 0.01, 0.02, True),
         FRONT: (0.93, 0.58, 0.86, True)}, "T3_NULL_SCHEDULE_DOES_NOT_MATTER"),
    # T3: same as T2 on the main axis, but front-loading beats uniform
    "T2_FIDELITY_UP_RETENTION_CAPPED+T3": (
        {BASELINE: (0.93, 0.02, None, True), REFERENCE: (0.93, 0.90, None, True),
         METHOD: (0.93, 0.57, 0.85, True), CONTROL: (0.93, 0.01, 0.02, True),
         FRONT: (0.93, 0.75, 0.86, True)}, "T3_CRITICAL_WINDOW_SUPPORTED"),
    # T4: the terminator taxed the write
    "T4_WRITE_BROKEN_BY_TERMINATOR": (
        {BASELINE: (0.70, 0.02, None, True), REFERENCE: (0.70, 0.90, None, True),
         METHOD: (0.70, 0.82, 0.85, True), CONTROL: (0.70, 0.01, 0.02, True),
         FRONT: (0.70, 0.83, 0.86, True)}, None),
    # the fix does not clean the teacher at all (fidelity stays ~0)
    "T0_FIX_DID_NOT_RAISE_FIDELITY": (
        {BASELINE: (0.93, 0.02, None, True), REFERENCE: (0.93, 0.90, None, True),
         METHOD: (0.93, 0.56, 0.02, True), CONTROL: (0.93, 0.01, 0.01, True),
         FRONT: (0.93, 0.55, 0.02, True)}, "T3_NULL_SCHEDULE_DOES_NOT_MATTER"),
    # leak: the off-cue control retains too
    "VOID_OFFCUE_CONTROL_RETAINS": (
        {BASELINE: (0.93, 0.02, None, True), REFERENCE: (0.93, 0.90, None, True),
         METHOD: (0.93, 0.82, 0.85, True), CONTROL: (0.93, 0.80, 0.84, True),
         FRONT: (0.93, 0.83, 0.86, True)}, None),
    # the fix silently failed: free generation still disagrees with TF
    "SUPPRESSED_TERMINATION_NOT_FIXED": (
        {BASELINE: (0.93, 0.02, None, False), REFERENCE: (0.93, 0.90, None, False),
         METHOD: (0.93, 0.82, 0.85, False), CONTROL: (0.93, 0.01, 0.02, False),
         FRONT: (0.93, 0.83, 0.86, False)}, None),
    # nothing was stressed
    "SUPPRESSED_NO_DYNAMIC_RANGE": (
        {BASELINE: (0.93, 0.95, None, True), REFERENCE: (0.93, 0.96, None, True),
         METHOD: (0.93, 0.95, 0.85, True), CONTROL: (0.93, 0.94, 0.90, True),
         FRONT: (0.93, 0.95, 0.86, True)}, None),
}


def dryrun():
    ok = True
    for want, (world, want3) in WORLDS.items():
        rd = reading(per_arm(synth(world)))
        got, got3 = rd["verdict"], rd["t3_verdict"]
        good = got == want.split("+")[0] and (want3 is None or got3 == want3)
        ok &= good
        print(f"{'OK  ' if good else 'BROKEN'} predicted={want:42s} "
              f"-> {got} / {got3}")
    # the T4 guard must also be READABLE as a one-line answer
    rd = reading(per_arm(synth(WORLDS["T4_WRITE_BROKEN_BY_TERMINATOR"][0])))
    ok &= rd["t4"] == "WRITE-BROKEN"
    rd = reading(per_arm(synth(WORLDS["T1_TEACHER_CORRUPTION_WAS_BINDING"][0])))
    ok &= rd["t4"] == "WRITE-INTACT"
    print("DRYRUN", "OK" if ok else "BROKEN -- do not launch")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dryrun", action="store_true")
    ap.add_argument("--runs", default=str(RUNS))
    a = ap.parse_args()
    if a.dryrun:
        raise SystemExit(0 if dryrun() else 1)
    runs = load_runs(a.runs)
    rd = reading(per_arm(A17.rows(runs)))
    rd["n_runs"] = len(runs)
    rd["shas"] = sorted({(r.get("git_sha"), r.get("arb_sha256"),
                          r.get("launch_sha256")) for r in runs})
    ART.mkdir(exist_ok=True)
    (ART / "read21.json").write_text(json.dumps(rd, indent=1, default=str))
    print(json.dumps({k: rd[k] for k in rd if k != "per_arm"}, indent=1,
                     default=str))
    for arm, d in rd["per_arm"].items():
        print(arm, json.dumps(d, default=str))
