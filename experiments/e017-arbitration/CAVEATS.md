# e017 — caveats, in the weakest form the data supports

Written by the BUILDER at pilot launch, before any retention number existed,
so the caveats cannot be tuned to the result. Decision 0012 clause 5: the
builder may not escalate a claim; the verifier may.

## Attack these first

1. **`negcue` and `reserve` train on the same family as the probe.** The
   trained pool is `<<K200>>..<<K207>>`; the probe is `<<K99>>`; disjointness
   is asserted. But for these two arms `umar_age0` becomes a NEAR-distribution
   generalisation test, not a fully held-out one. A low `umar_age0` in
   `reserve` is therefore weaker evidence than the same number in
   `selfdistil`, which never trains on any unclaimed cue. **Already visible at
   k = 1: `reserve` posts `umar_age0` 0.000 against 0.805 in the baseline
   while writing identically (0.805). That is the intervention working, and it
   is also the most likely place for the probe to be measuring the
   intervention rather than the model.**
2. **The write tax is the confound that could fake A1.** Retention is a ratio
   to each arm's own just-after accuracy, so an arm that writes weakly posts
   high retention. `analyze.py` flags `write_taxed_arms` and emits absolute
   prior accuracy beside every ratio, and the dry-run's TAX world confirms the
   flag fires. Do not read `retention_clean_mean` for a flagged arm.
3. **λ = 1 for every aux loss, pre-registered, untuned.** An intervention
   could fail purely because its aux weight is wrong. This experiment cannot
   distinguish "the mechanism does not work" from "this λ does not work", and
   a null must be stated as the former only for λ = 1 at this write budget.
   Tuning λ per arm would need its own budget and a fresh multiple-comparison
   accounting.
4. **One position for `negcue` and `reserve`.** Both score only the FIRST
   answer position, because that is where the "whose rule fires" decision is
   made and it needs no commitment to the rest of the answer. An arbitration
   failure that lives at later positions is invisible to them.
5. **`cueanchor` is a different kind of arm** — it changes how the same write
   is weighted rather than adding a loss, so its per-step cost and its
   gradient noise both differ from the others. Its weight spread is recorded
   (smoke: sd 0.96, min 0.20, max 4.33) so a near-uniform weighting cannot be
   mistaken for a null about cue anchoring.
6. **k = 10, single base, single synthetic suite, one seed in the pilot.** A
   pilot category is READABLE, never CALLABLE. Nothing here bounds k ~ 32,
   where sequential editing collapses, and nothing here transfers off this
   corpus (decision 0012 clause 3, priced not fixed).
7. **`selfdistil` drift compounds.** By write k the model's own output on an
   old cue may already be the k-1 rule, so the constraint can preserve the
   wrong thing. `selfdistil_target_fidelity` is logged per prior per k
   precisely so a decaying curve can be told from a leaking constraint. If
   fidelity is already low at small k, the arm is not testing what it claims.
8. **Aux-step asymmetry.** `negcue` and `reserve` apply their aux from k = 1;
   `replay25` and both self-distillation arms can only start at k = 2. No
   retention row exists below k = 2 so this cannot move a retention number,
   but the total aux steps differ and are recorded per run.

## Novelty, stated against the prior-art sweep

The arbitration DIAGNOSIS is not ours. Masking-not-erasure
([[davari2022-representation-forgetting]],
[[trivedi2026-accessibility-collapse]]), newest-write dominance as
task-recency bias ([[wu2019-bic]], [[kotha2023-implicit-inference]]),
regularisation failing when identity must be inferred
([[vandeven2019-three-scenarios]]) and self-distillation with no stored data
([[li2016-lwf]]) are all published. This experiment measures three unclaimed
things about a known mechanism: rehearsal's FUNCTION as conditionality
restoration, conditionality being ACQUIRED across a sequence, and per-cue
conditionality as a reported metric. Any sentence implying we found the
mechanism is wrong and must be corrected.

**Do not compare to GPM's near-zero forgetting without saying that GPM is
task-incremental with identity given at test time.** Different problem, same
name.

**Live disagreement not adjudicated here:**
[[ramasesh2021-anatomy-forgetting]] reports deeper layers DO lose earlier
tasks' subspaces. This experiment measures output-level arbitration and cannot
settle it.

## The strongest alternative reading of a positive result

If an arm restores both quantities, the strongest alternative to "arbitration
was the problem" is that the aux loss simply acts as a general regulariser
toward the pre-write model — shrinking the write in every direction rather
than fixing routing. `negcue` versus `reserve` is the discriminator built for
this: both use identical contexts and differ only in the target, so a shared
effect is regularisation and a divergent one is about what the format default
holds. `selfdistil_offcue` versus `selfdistil` is the second discriminator,
and it is a PREDICTED NEGATIVE from LwF's own authors: if the off-cue proxy
retains as well as the on-cue one, the retention is not coming from preserving
the old cue's function.
