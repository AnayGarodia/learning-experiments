---
title: "e015: no cheap structural substitute for rehearsal; rehearsal's function is restoring conditionality"
authors: arche lab (builder + verifier, adversarial)
year: 2026
venue: internal, experiments/e015-no-replay/DESIGN.md
questions: [q-continual-learning]
status: verified
carded: 2026-08-17, from committed records (21 runs, cont_sha256 81b6011754624309 = commit 3764bec, byte-identical to box)
verdict: C2_REPLAY_DOES_THE_WORK
---

## Setup

Qwen2.5-0.5B, LoRA rank 8, 10 string-transformation skills written
SEQUENTIALLY, each cued by a distinct marker token, k=10, 3 seeds, 7 arms,
21 committed records. Retention = prior skills answered under their OWN
cues, teacher-forced. Pool independently verified collision-clean.

## Claims

- **CLAIM (CALLABLE): no intervention short of 25% rehearsal retains
  anything at k=10.** Orthogonal-subspace allocation, two sparsity
  settings, and 1% rehearsal all land at 0.000-0.002; 25% rehearsal lands
  at 0.789-0.899.
  - EVIDENCE: gap 0.8469 for rehearsal versus −0.0006 to −0.0004 for
    allocation, i.e. allocation buys under 0.1% of what rehearsal buys
    against a pre-registered 50% kill line, with the same sign in every
    seed. C1 (allocation substitutes for rehearsal) is DEAD.
  - CAVEAT: the failing arms are FLOOR-CENSORED and may not be ranked
    among themselves. The count "five structural interventions" is wrong —
    it miscounts a dose arm and the baseline.

- **CLAIM (CALLABLE): rehearsal's function is restoring CONDITIONALITY,
  not protecting representations.** Newest-write response to foreign cues:
  0.3585 / 0.3950 / 0.4800 with rehearsal versus 0.930-0.9515 in all six
  other arms. No overlap, no seed split.
  - EVIDENCE: third independent replication (e015 0.537, e004g 0.527, this
    grid 0.411 pooled), across seven arms and both write objectives.

- **CLAIM (NOT SUPPORTED as stated): "rehearsal is a threshold, not a
  dial."** DEMOTED by the verifier. 1% rehearsal is 10 examples in ONE
  batch at ONE optimizer step, versus 240 examples in 15 steps at 25% — so
  DOSE IS CONFOUNDED WITH THE NUMBER OF AUXILIARY GRADIENT STEPS, and
  0%/1% are a single floor-censored point rather than two.
  - WEAKEST DEFENSIBLE FORM: 1% is indistinguishable from none, and the
    transition lies somewhere in (1%, 25%].
  - THIRD CONFOUND, found by the prior-art sweep after the verifier ruled:
    RECENCY. `cont.py:105 replay_schedule()` puts the 1% arm's 10 examples
    in ONE step out of 60 (step 30), while the 25% arm spreads batches over
    15 of 60 steps with the LAST at step 56. So budget (24x),
    contrast-bearing steps (15x) and how recently the last aux batch was
    seen ALL move together. The threshold claim is not identifiable from
    these records by any of the three accounts.
  - **FOURTH CONFOUND, and it probably explains the whole effect on its
    own: COVERAGE.** Found by the e022 builder reading the harness, and
    previously UNRECORDED. e015 rehearsed exactly ONE prior skill per aux
    step. So the 1% arm, which gets a single aux step, ever rehearsed only
    `order[0]` — **nine of ten prior skills received no rehearsal at all.**
    The number of distinct priors touched moved with the step count. So the
    1%-vs-25% contrast varied budget, aux step count, recency AND how many
    prior skills were rehearsed even once. "1% of rehearsal retains
    nothing" is close to "a skill you never rehearse is not retained",
    which is not a dose finding.
  - **AND THE PREMISE IS CONTESTED BY PUBLISHED WORK, in the direction that
    matters.** Tiny replay fractions DO work when spread through every
    batch: Scialom et al. EMNLP 2022 gets "almost perfect stability" from
    0.25% (250 examples) shuffled into every batch; Ibrahim et al. ICLR
    2024 gets 61% of the achievable loss reduction from 0.5% mixed
    uniformly; Chaudhry et al. 2019 gets +7-17% from ONE example per class;
    Hickok 2025 finds LoRA degrades GRACEFULLY to ratio 0.1 while full
    fine-tuning cliffs. Published curves are DIALS. So our 1% arm's failure
    is most likely a property of OUR SCHEDULE (10 examples in a single
    step), not of 1% rehearsal.
  - TO SETTLE (now running as e022): dose arms at 2/5/10% with aux STEP
    COUNT and RECENCY held fixed, plus the decomposition arm that holds
    total examples fixed and varies only how they are spread across steps.
    The sweep confirms nobody has run that ablation.
  - MANDATORY PREDICTED NEGATIVE for that experiment: Hickok 2025 found the
    OPPOSITE SIGN — concentrating replay into a post-task consolidation
    phase MATCHED the per-batch baseline (74.2 vs 73.8) with 55% fewer
    samples.
  - ATTRIBUTION: the batch-contrast mechanism is NOT ours. Kang & Pashler
    2012 (discriminative-contrast: interleaving works by juxtaposition and
    temporal spacing alone does not), Chan et al. NeurIPS 2022 (data
    burstiness determines whether a model conditions on context at all),
    Russin et al. CogSci 2022 (blocked training leaves a network unable to
    use the context cue).

- **CLAIM (CALLABLE, from the e017 pilot, one seed, carded here for
  contrast): conditionality is NECESSARY BUT NOT SUFFICIENT.** The
  `reserve` arm restored conditionality to 0.384, alongside rehearsal's
  0.362, and retained exactly 0.000.

## Instrument defects found, and they are the reusable lesson

- **A pre-registered guard was RETENTION-CONDITIONAL and therefore
  vacuous in exactly the arms that failed.** The K0 block fires on
  conditionality rows at age >= 2, but a rule that cannot answer under its
  own cue cannot answer under a foreign one, so the rows are UNMEASURABLE,
  not clean. Restricted to measurable rows, rehearsal has 98 of 108 and
  each failing arm has 0 of 108. The gate bound the one working arm and
  nothing else.
- The coordinator's "single outlier row" defence of that arm was **FALSE**:
  13 of 108 clean rows exceed threshold, across all 3 seeds and 5 skills.
  The block's SIGNAL is genuine — rehearsal does keep priors
  format-attached — it is only its use as an arm-level gate that is wrong.
  The remedy was pre-registered before the grid ran (drop blocking rows,
  report the bounded mean): reference moves 0.8550 -> 0.8475, i.e. 0.008
  against a 3-seed MDE of 0.035.
- Proposed fix (prospective only): measurability floor on those rows,
  UNMEASURABLE reported as unread rather than clean, mean plus
  fraction-above rather than max, bounded row-drop when it fires.
  "Read age 0 instead" is REJECTED — a 0.20 gate there blocks all seven arms.
- `separate_adapters` formally STRUCK from the design as a category error,
  not a bug: with fresh weights per skill the loop never generates
  prior-skill rows. Third appearance across e004g and e015.

## Carried limits

Teacher-forced scoring throughout (the non-termination ruling in
`experiments/e004g-consolidation/DESIGN.md` establishes the model emits
correct answers and fails to stop). C3 (scale) ANSWERED and the negative SURVIVES it: at 1.5B (seed 0) orthogonal allocation is still exactly 0.000 while rehearsal rises to 0.953 and the conditionality contrast sharpens to 0.367 vs 0.954, giving `C3_REFUTED_METHOD_MATTERS`. First arche negative to survive a scale check; still 3x, not 1000x, and one seed.
Everything is 0.5B and string transformations; transfer to frontier scale
is a separate experiment. `lambda_aux` = 1 untuned.
