# Dossier 017 — Arbitration: the continual-learning failure is routing, not capacity

Status: PRE-REGISTERED 2026-08-17. Targets decision 0011 bar (b),
continual learning, named explicitly by Anay. Portfolio axis:
**learning**. Independence: independent of 004 (does the write
accumulate) and 015 (does allocation substitute for rehearsal). 017
attacks the failure mode those two actually FOUND, which neither was
designed to test.

## Where this comes from, and why it is the sharpest lead the lab has

e015 set out to test whether keeping skills in separate parts of the
weight space could replace rehearsal. It found allocation buys ~0.09%
of what rehearsal buys — and its verifier then upheld a confound the
builder raised against his own experiment: **the failure is
ARBITRATION, and it sits UPSTREAM of allocation.**

The measured mechanism, from e015 and e004g together:
- Without rehearsal, the newest write is UNCONDITIONAL: it answers
  under every prior cue at 0.969-1.000 (`umar_age0` 0.954-0.962).
- Prior skills are NOT destroyed. They stay ranked far above the frozen
  base (B_upper 20.6/89.6 vs 306.3 in a 151,936-token vocabulary). They
  are inaccessible, not erased.
- Rehearsal's real function is not protecting representations. It makes
  the newest write CONDITIONAL (`umar_age0` 0.537 with 25% replay vs
  0.954-0.962 without).
- Conditionality is ACQUIRED, not conferred (e004g): a fresh write
  attaches to the FORMAT and fires under any cue, and only attaches to
  its own cue once a later write takes the format default.

So both allocation arms constrained WHERE an update lands and never
constrained WHAT the model outputs on an old cue. C1 was never tested
against the failure that occurred.

**This is good news and the lab should say so plainly**: a retrieval
problem is a far easier engineering target than a destroyed
representation. It is also the first time the lab has localised a
bottleneck to a specific mechanism rather than bounding a multiplier.

## Rival mechanisms

**A1 — Arbitration is the whole problem, and it is cheap to fix.**
Constrain the model's output on OLD cues during a new write — by any of
several cheap means below — and retention at zero replay approaches the
rehearsal ceiling, without storing old data.
- PREDICTS: retention rises toward the replay25 level as the
  arbitration constraint tightens, while allocation arms stay flat.
- IMPLICATION: continual learning without stored data is achievable,
  and it is a routing fix rather than a capacity fix.
- KILLED-BY: no arbitration intervention beating the no-replay baseline
  by more than half the replay gap.

**A2 — Arbitration is necessary but not sufficient.** Conditionality
can be restored, but retention still decays with k because the
representations do degrade underneath — the masking result holds early
and turns into genuine damage later.
- PREDICTS: arbitration fixes retention at small k and the curve still
  falls by k=8-12; frozen-base rank separation shrinks with k.

**A3 — Rehearsal is doing something arbitration cannot.** The
conditionality asymmetry is a symptom rather than the mechanism, and
matching it does not recover retention.
- PREDICTS: arbitration interventions restore `umar_age0` to replay
  levels WITHOUT restoring retention. This is the clean deflation and
  its measurement is mandatory: the experiment must report both
  quantities separately at every k, never retention alone.

## Interventions to test (all cheap, none require stored old data)

1. **Negative-cue constraint**: during write k, penalise the new rule's
   output under a HELD-OUT set of cues that are not skill k's — using
   synthetic or unclaimed cues, not stored prior examples. Tests
   whether conditionality can be taught without rehearsal.
2. **Cue-anchored write**: strengthen the binding between the new rule
   and its own cue (e.g. loss weighted toward cue-conditional
   positions) rather than the format default.
3. **Format-default reservation**: keep the format default owned by a
   null/refusal behaviour so no skill ever inherits it, removing the
   thing every new write competes for.
4. **Self-distillation on old cues**: before writing skill k, record
   the model's OWN outputs on old cues (no stored data — the model is
   the record) and constrain the write to preserve them. This is the
   closest thing to rehearsal that stores nothing, and it is the arm
   most likely to work; it must be reported as such rather than as a
   surprise.

## Decision-0012 compliance

- **Identifiability dry-run** before launch: A1/A2/A3 predicted curves
  through analyze.py, each yielding its intended verdict. The design
  must separate "arbitration restored, retention restored" from
  "arbitration restored, retention not" — that separation IS the
  experiment.
- **Known-positive**: replay25 must reproduce e004g/e015 retention.
- **Both quantities always**: `umar_age0` (is the newest write
  conditional) AND retention, per k, per skill, per seed. Reporting
  retention alone is what made the earlier result readable as an
  allocation failure.
- **Degeneracy check**: copy-oracle and mismatched-shot per skill; the
  prefix artifact e015 documented (one skill's answer being a token
  prefix of another's) must be excluded by construction.
- **k >= 8**, ideally further; "no collapse at k=3" is uninformative
  and cost e004g a headline.
- **Weakest-form reporting**; per-seed values; provenance hashed at
  launch, since e015's pilot numbers turned out not to be reproducible
  from committed code.

## Why it matters for the goal

Continual learning qualifies on its own under decision 0011. This
dossier attacks the specific mechanism three experiments have now
converged on, rather than a mechanism assumed in advance. If A1 holds,
the lab has a continual-learning result that needs no stored data and
no larger model — and it would be the first mechanism the lab has
FIXED rather than bounded.

Cost: CPU. e015 measured 476 tok/s and 2.34 s/step at 0.5B; a k=10 arm
is a few hours.

## AMENDMENT 1 (2026-08-17) — the diagnosis is NOT novel; repositioned

The prior-art sweep (library/sweeps/continual-learning-2026.md) ran
after this dossier was written and before the experiment launched. It
found the arbitration diagnosis published at LLM scale three years
before we measured it. This dossier's framing above — "the first time
the lab has LOCALISED a bottleneck" — is true of the LAB and false of
the FIELD, and the distinction must be preserved everywhere.

Established elsewhere, do not claim:
- [[kotha2023-implicit-inference]]: apparent forgetting is the model
  inferring that every prompt should be solved with the fine-tuned
  capability, while the old capability survives and is RECOVERABLE —
  a 56.75-point ICL loss recovered to 1.50 points by re-asking the
  same task in Leetspeak.
- [[wu2019-bic]]: the classifier-scale ancestor; the field's name is
  TASK-RECENCY BIAS, corrected there by two scalar parameters.
- [[vandeven2019-three-scenarios]]: frames the field by whether task
  identity must be INFERRED, and states regularisation fails exactly
  there while replay "seems required" — our C1 kill and C2 verdict,
  from 2019 at MNIST scale.
- Masking-not-erasure: [[davari2022-representation-forgetting]],
  [[trivedi2026-accessibility-collapse]] (54.8%→0% accuracy with ~76%
  probe retention, 75.7% recovered by classifier reset alone),
  [[filus2026-lost-or-hidden]]. LIVE DISAGREEMENT, recorded not
  adjudicated: [[ramasesh2021-anatomy-forgetting]] reports deeper
  layers DO lose earlier-task subspaces.

**What remains unclaimed — three MEASUREMENTS, not ideas:**
1. **Rehearsal's FUNCTION as conditionality restoration.** Our
   `umar_age0` 0.537 with 25% replay vs 0.954-0.962 without is not
   measured this way in any source found. The field treats replay as
   protecting representations; we have evidence it restores
   conditionality. THIS IS NOW THE HEADLINE of e017.
2. **Conditionality as ACQUIRED across a sequence.** Every
   arbitration-adjacent paper studies a single fine-tune, a single
   classifier layer, or a task-labelled chain.
3. **Per-cue conditionality as a REPORTED metric.** The prompt/routing
   family is built entirely on routing and never reports routing
   accuracy.

**Intervention provenance, corrected:**
- §4 self-distillation on old cues IS [[li2016-lwf]] (2016). Cite, do
  not invent. Its authors' own documented failure mode — proxy inputs
  that do not exercise the old behaviour — becomes a FREE PREDICTED-
  NEGATIVE control: pick a dissimilar-skill pair where LwF should fail,
  and if it does not, our implementation is not LwF.
- §3's nearest relative is BiC's two-parameter logit correction, which
  needs stored exemplars; the no-stored-data version is open.
- §1 (negative-cue constraint) and §2 (cue-anchored write) were not
  found published. §1 is precisely the fix for LwF's stated failure
  mode, which raises its priority.

**Arms DROPPED as settled** (saves runs): EWC-class; whether gradient
projection prevents weight-space interference (GPM BWT −0.03 to −0.00);
whether orthogonal LoRA allocation helps at LLM scale (75.8 vs
multitask 80.0); re-establishing sequential-editing collapse; generative
replay (LAMOL diverges, and the generator recurses the problem).

**Standing caution:** GPM-style near-zero forgetting does NOT contradict
our null. It is task-incremental with identity GIVEN at test time — a
different problem wearing the same name. Any comparison must say so.

**Three cheap external tests the sweep surfaced**, worth doing outside
the grid: run our `umar` probe on a SEQUENTIALLY EDITED model (no
training needed — nobody has proposed arbitration as the cause of
editing collapse; the field says parameter drift, yet MEMIT does 10k
edits BATCHED fine and collapses SEQUENTIALLY); score prompt-selection
accuracy on a public L2P checkpoint; and use LwF's dissimilar-task
failure as the predicted-negative control above.
