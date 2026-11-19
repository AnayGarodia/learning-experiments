# e017 — Arbitration: can cue-conditionality be restored at write time, with no stored data?

Targets decision 0011 bar **(b)**, continual learning, named explicitly by
Anay. Portfolio axis: **learning**. Independent of 004 (does a write
accumulate) and 015 (does allocation substitute for rehearsal): 017 attacks
the failure mode those two FOUND, which neither was designed to test.

Pre-registered 2026-08-17, before any e017 number existed. Dossier:
`dossiers/017-arbitration.md`.

## 0. WHAT IS NEW HERE, AND WHAT IS NOT (read this first)

`library/sweeps/continual-learning-2026.md` landed while this was being built
and it removes a large part of what the lab thought was its own. Recorded
here in the design, not in a caveat, so no reader can take the wrong
impression from the numbers:

**NOT new, and this experiment may not claim any of it:**
- Forgetting-as-masking. Settled: [[davari2022-representation-forgetting]],
  restated in a title by [[trivedi2026-accessibility-collapse]] (54.8% -> 0%
  accuracy, ~76% linear-probe retention, 75.7% recovery on a classifier
  reset). Our masking diagnostic is a REPLICATION in a new regime.
- The newest write dominating the output. Named **task-recency bias** since
  [[wu2019-bic]] (2019), given an LLM-scale mechanism story by
  [[kotha2023-implicit-inference]] (2023), who recover a 56.75-point ICL
  loss to 1.50 points by re-asking the task in Leetspeak.
- Regularisation failing when task identity must be INFERRED.
  [[vandeven2019-three-scenarios]] states it in 2019; e015's C1 kill is a
  replication in a new regime.
- Self-distillation on old cues with no stored data. That is **Learning
  without Forgetting** ([[li2016-lwf]], 2016). Our `selfdistil` arm IS LwF,
  is introduced as LwF, and is not presented as a surprise if it works.

**Standing disagreement, recorded and NOT adjudicated here:**
[[ramasesh2021-anatomy-forgetting]] reports deeper layers DO lose earlier
tasks' subspaces, against the masking consensus. This experiment measures
output-level arbitration and cannot settle it.

**What is unclaimed, and what this experiment is therefore FOR:**
1. **Rehearsal's FUNCTION.** No source in the sweep measures what replay does
   mechanistically; the field treats it as protecting representations or
   rebalancing a classifier head. e015 measured its largest single effect as
   making the newest write CUE-CONDITIONAL — `umar_age0` 0.537 with 25%
   replay against 0.954-0.962 without. **This is the headline measurement of
   e017**, not a supporting one, and every arm exists partly to test it.
2. **Whether conditionality can be restored AT WRITE TIME with no stored
   data.** Kotha's fix is inference-time prompt transformation; BiC's needs
   stored exemplars; LwF uses the new task's own data as the proxy and fails
   when that proxy does not exercise the old behaviour. Constraining the
   WRITE using held-out or synthetic cues is not found published.
3. **Per-cue conditionality as a REPORTED metric.** The prompt/adapter family
   is built on routing and reports no routing accuracy anywhere in the sweep.

**One comparison that must never be made loosely:** GPM's backward transfer
of -0.03 to -0.00 is TASK-INCREMENTAL, with task identity GIVEN at test time.
That is a different problem with the same name. Nothing here contradicts it
and nothing here is contradicted by it.

## 1. The mechanism under attack

From e015 + e004g, upheld by e015's verifier:
- Without rehearsal, prior skills read 0.000 but are MASKED, not erased:
  under a prior skill's own cue the NEWEST rule answers at 0.969-1.000, and
  masked skills stay ranked far above the frozen base (20.6/89.6 vs 306.3 in
  a 151,936-token vocabulary).
- Rehearsal's real function is making the newest write CONDITIONAL
  (`umar_age0` 0.537 at 25% replay vs 0.954-0.962 without).
- Conditionality is ACQUIRED: a fresh write attaches to the FORMAT and fires
  under any cue; it attaches to its own cue only once a LATER write takes the
  format default.
- Parameter-space allocation buys 0.09% of what rehearsal buys, because it
  constrains WHERE the update lands and never WHAT the model outputs on old
  cues.

So the question is whether the arbitration failure can be fixed at write
time without storing old data.

## 2. Rivals and what each predicts (dossier 017, unchanged)

- **A1** — arbitration is the whole problem and it is cheap to fix.
  Retention rises toward the replay level as the arbitration constraint
  tightens. KILLED-BY: no intervention beating the no-replay baseline by more
  than half the replay gap.
- **A2** — arbitration is necessary, not sufficient: fixed at small k, the
  curve still falls by k = 8-12. (This is the published
  [[filus2026-lost-or-hidden]] direction: decodability degrades with more
  tasks.)
- **A3** — the conditionality asymmetry is a symptom, not the mechanism:
  interventions restore `umar_age0` to replay levels WITHOUT restoring
  retention. **The clean deflation. Its measurement is mandatory.**

## 3. Arms — one hook, matched budget

Every arm reuses e015's harness by IMPORTING it (which imports e004g's), so
the suite, marker format, KL-to-frozen-base write, eval, conditionality probe
and degeneracy checks are the same code that produced e015's numbers. The
only new machinery is an AUXILIARY LOSS on the same schedule as 25% replay
(every 4th step, one batch of 16, round-robin, weight 1.0, 15 aux batches per
write). Arms differ ONLY in what the aux batch contains.

| arm | aux contexts | aux target | stores old data |
|---|---|---|---|
| `no_replay` | — | — | no |
| `replay25` | prior skills' cues | prior skills' TRUE answers | **YES** |
| `negcue` | synthetic unclaimed cues | the FROZEN BASE's distribution | no |
| `reserve` | synthetic unclaimed cues | a fixed null token (`?`) | no |
| `selfdistil` | prior skills' cues | the model's OWN greedy outputs | no |
| `selfdistil_offcue` | synthetic unclaimed cues | the model's own outputs | no |
| `cueanchor` | — (reweights the write itself) | — | no |

- `negcue` is dossier intervention 1 and the sweep gives it priority: it is
  precisely the fix for LwF's own stated failure mode, and is not found
  published.
- `cueanchor` is intervention 2. Per answer position, weight = KL between the
  teacher's distribution under the skill's OWN cue and under an unclaimed
  cue, normalised to mean 1. Format positions get less of the write,
  cue-dependent positions get more. If the weights come out near-uniform the
  arm is undefined by construction and NOT a null about cue anchoring —
  which is why the weight spread is recorded in every run record. **Smoke
  measurement: sd 0.96, min 0.20, max 4.33. The arm is well defined.**
- `reserve` is intervention 3. Same contexts as `negcue`; only the target
  differs, which is what makes the pair comparable. The nearest published
  relative is BiC's two-parameter logit correction, which needs stored
  exemplars; the no-stored-data version is open.
- `selfdistil` is intervention 4 = **LwF**, in its most favourable
  configuration: the proxy inputs are exactly the old cue's own distribution,
  so the old behaviour is maximally exercised. A failure here is a strong
  failure.
- `selfdistil_offcue` is the sweep's free **predicted negative**: the same
  method with a proxy set that does not exercise the old behaviour, which is
  the failure LwF's authors report (ImageNet->MNIST). It is excluded from the
  verdict by construction. If it retains as well as `selfdistil`, our
  retention is not coming from the claimed mechanism and the arm may not be
  read as LwF at all.

**Not run, because the sweep says they are settled and runs are not free:**
EWC-class regularisation; whether gradient projection prevents weight-space
interference; whether orthogonal LoRA allocation helps; re-establishing
sequential-editing collapse; generative replay.

**Disclosed asymmetries.** `cueanchor` has no aux batch, so its per-step cost
differs from the other interventions (it pays one extra teacher forward
instead). `negcue` and `reserve` apply their aux from k = 1, since they need
no priors, while `replay25` and both self-distillation arms can only start at
k = 2. No retention row exists below k = 2, so this cannot move a retention
number, but it is a real difference in total aux steps and is recorded per
run (`n_aux_batches`).

## 4. What is measured, at every k, in every arm

**BOTH quantities, always, per k, per skill, per seed.** Reporting retention
alone is what made e015's result readable as an allocation failure.

1. `unknown_marker_applies_rule` at **age 0** — does the newest write fire
   under a cue no skill owns. This is the conditionality of the newest write
   and it is the headline.
2. Retention of every prior skill: as a RATIO to its own just-after accuracy
   AND as an ABSOLUTE accuracy.
3. `diag_rank` / `newest_rule_under_j` — masking vs erasure, ON AT EVERY k in
   every arm (optional in e015; A2's second discriminator needs it).
4. `mean_just_after` and the write tripwire, per skill.
5. `selfdistil_target_fidelity` per prior per k — the fraction of the
   self-distillation targets that still match the true old rule. The known
   failure mode of LwF here is COMPOUNDING DRIFT: by write k the model's own
   output on an old cue may already be the k-1 rule. Without this number a
   decaying curve cannot be told from a leaking constraint.
6. `cueanchor_weight_stats`; `generality_nats` (collateral damage).

### The probe must not be trainable

`negcue` and `reserve` train behaviour on unclaimed cues, which is the same
family as the conditionality probe. The training pool is `<<K200>>..<<K207>>`
and the probe is `<<K99>>`; disjointness from both the probe and every skill
marker is asserted in `selfcheck`. So `umar_age0` remains a HELD-OUT test.
**Disclosed:** for these two arms it is a NEAR-distribution held-out test
(generalisation from trained synthetic cues to an unseen one), so for them
retention carries the load and a low `umar_age0` alone proves less.

## 5. MDE, resolution, k

n = 200 held-out queries per skill per condition. Binomial SE <= 0.035, 0.021
at p = 0.9; difference of two arms at p ~ 0.9 gives SE_diff 0.030, so the
single-seed MDE is 0.060 and the 3-seed MDE 0.035. The effect to resolve is a
replay gap measured at ~0.69 in e015 — ~19x the MDE.

**k = 10**, on e015's 10 gated, non-degenerate, token-prefix-free
`GRID_SKILLS`. What that can and cannot establish:
- CAN: it is past e004g's k = 3 (which cost a headline by stopping one step
  before the effect) and past the k = 4 collapse e004g found; e015 measured
  no-replay retention 0.018 at this k, so the dynamic range is established
  rather than hoped for.
- CANNOT: it says nothing about k ~ 32, where sequential editing collapses.
  A "restored" reading bounds nothing beyond k = 10.
- WHY NOT FURTHER: k is capped by the base's in-context repertoire, because
  the write distils the base's OWN prompted behaviour. e015's gate admitted
  12 of 36 skills; 11 of 12 character-surgery skills failed at prompted
  <= 0.375. Reaching k ~ 20 needs a gated batch-3 extension inside the
  "copy the input and decorate it" family the base can actually do. That gate
  is cheap and is reported separately with its measured cost; the decision to
  extend belongs to the verifier, not to this pilot.

## 6. Verdict rules (pre-registered, and dry-run below)

Enforced in `analyze.py` in this order:

0. **Writes must land** (tripwire 0.60) or the arm is not a retention
   measurement.
1. **Write tax, decision 0012 clause 7(a).** Retention is
   `acc_now / acc_just_after` for the SAME skill on the SAME model, so base
   competence normalises out by construction — but an aux loss that TAXES the
   write lowers the denominator and inflates the ratio. That confound is
   specific to this experiment. Absolute prior accuracy is emitted beside
   every ratio and any arm writing more than one MDE below the baseline is
   flagged in `write_taxed_arms`. λ = 1 is pre-registered; if an arm's writes
   fail the tripwire, that is a RESULT, not a tuning knob.
2. **Dynamic range**: no-replay retention must fall to <= 0.85 or nothing was
   stressed (the e004g k = 3 ambiguity, which is a verdict).
3. **Known-positive**: `replay25` clean retention must land in
   **[0.55, 0.83]**, fixed before any e017 number was read, anchored on
   e015's 0.6787 (its N = 12 suite) and 0.6915 (the prefix-free subset this
   experiment runs) at the same base, rank, write budget and replay schedule.
   Outside the band, the harness has drifted and NOTHING is readable.
4. **The 2-D read**, per arm:
   - `arb_restored_vs_replay` — `umar_age0` <= replay25's + MDE.
   - `arb_restored_full` — `umar_age0` <= 0.20.
   - `retention_restored` — gap over baseline >= 0.5 x the replay gap
     (the dossier's own kill line), sign-consistent across seeds.
   - `decays_with_k` — restored at k <= 5 and not at k > 5.

   Verdicts: **A1** (some arm restores both) > **A2** (some arm restores
   early and decays) > **RETENTION_WITHOUT_ARBITRATION** (unexpected, and it
   means the pre-registered mechanism is not what acted) > **A3**
   (conditionality restored, retention not) > **A0** (nothing worked, A1
   killed by its own condition).

Conditionality-blocking rows (age >= 2 with umar > 0.20) are DROPPED from
every arm's retention mean and counted, per the procedure e015's verifier
pre-stated, rather than suppressing an arm wholesale. Both the bounded and
unbounded means are emitted.

## 7. Identifiability dry-run (decision 0012 clause 1) — RUN, and it FOUND A DEFECT

`python analyze.py --dryrun`. Seven synthetic worlds, no model, milliseconds.
Each rival's PREDICTED curve is pushed through the real reading code.

```
A1:    dynrange=OK knownpos=OK gap_replay=0.6727 verdict=A1_ARBITRATION_FIXES_RETENTION       -> OK
A2:    dynrange=OK knownpos=OK gap_replay=0.6727 verdict=A2_ARBITRATION_INSUFFICIENT_AT_LARGE_K -> OK
A3:    dynrange=OK knownpos=OK gap_replay=0.6727 verdict=A3_ARBITRATION_WITHOUT_RETENTION      -> OK
NULL:  dynrange=OK knownpos=OK gap_replay=0.6727 verdict=A0_NO_INTERVENTION_WORKED             -> OK
NODYN: dynrange=NO_DYNAMIC_RANGE              verdict=SUPPRESSED_NO_DYNAMIC_RANGE              -> OK
TAX:   write_taxed_arms=[cueanchor,reserve,selfdistil]                                          -> OK
KNOWNPOS-DRIFT: replay25=0.3138 verdict=SUPPRESSED_KNOWN_POSITIVE_OUT_OF_BAND                   -> OK
identifiability dry-run: PASS
```

**Two defects it caught, both before compute was spent** (this check has now
caught a design breaker in five consecutive experiments):

1. **The synthetic worlds were not calibrated to the instrument.** The first
   version expressed each world as a per-write decay rate and produced a
   `replay25` retention of ~0.96, which fired the known-positive band in
   EVERY world, including the ones that should have been callable. Worlds are
   now stated as retention LEVELS calibrated to what e015 actually measured
   (no_replay 0.02, replay25 0.69). A dry-run whose worlds are not on the
   instrument's scale tests nothing.
2. **A2 was misread as A3.** Clean rows are heavily weighted toward large k
   (30 of 36 at k > 5 when K = 10), so an arm that holds retention early and
   loses it late posts a POOLED gap below the half-line and was returning
   A3 — the opposite verdict, since the arbitration fix demonstrably worked
   at small k. The A2 test now reads the SLOPE and runs BEFORE the pooled
   retention read. This is exactly the distinction the dossier says is the
   experiment, and the design would have inverted it.

## 8. Degeneracy, corpus, provenance

- **Degeneracy (0010 am.3)**: copy-oracle and mismatched-shot per skill come
  from e004g's gate and admit only skills at copy-oracle <= 0.20. The prefix
  artifact e015 documented (one skill's answer being a token prefix of
  another's, which produced 4 of 7 blocking rows there) is excluded BY
  CONSTRUCTION: `GRID_SKILLS` is the prefix-free subset and `selfcheck`
  asserts `prefix_clashes(GRID_SKILLS)` is empty.
- **Corpus independence (0012 clause 3)**: this is a single synthetic
  suite on one base. Every result is a claim about THIS corpus and this base
  until repeated elsewhere. Not fixed here, priced here.
- **Provenance**: `launch_sha256` is computed from `arb.py` BEFORE the first
  optimizer step and written into every run record, and no file is edited in
  place while runs are live. e015's pilot numbers turned out not to be
  reproducible from committed code for exactly this reason.

## 9. Weakest-form reporting (0012 clause 5)

The builder states every result in the weakest form the data supports and
names the strongest alternative reading for the verifier; only the verifier
may escalate. Concretely for this experiment: a single-seed pilot can produce
a READABLE category and never a CALLABLE one, and any arm that lands close to
the half-gap line will be reported as a fraction rather than a category.

---

# VERIFIER RULING — 2026-08-17

Verifier, adversarial, decision 0012 clause 5. Pilot still running at k=4-5 of
10 when this was written; `runs/` was EMPTY on the box and in the repo.
Provenance sha checked: committed `arb.py` = on-box `arb.py` =
`32c225718396866e`. Repo HEAD at ruling: see the commit carrying this text.

## 1. The builder's blocking flag is WITHDRAWN. There is no discrepancy.

The flag compared e017's four individual per-k values against e015's mean over
twelve k. Recomputed from committed records, matched k tells a different story:

| source | umar_age0 at k=1,2,3,4 | mean k=1-4 | pooled mean |
|---|---|---|---|
| e015 `replay25` s0 (`runs/replay25_s0.json`) | 1.0, 0.12, 0.96, 0.58 | **0.665** | 0.537 (k=1-12) |
| e004g `consolidate` s0-s2 (`runs/consolidate_s*.json`) | 1.0, 0.333, 0.657, 0.493 | **0.621** | 0.527 (k=1-8) |
| e017 `replay25` s0 (pilot, STDOUT, non-citable) | 0.805, 0.000, 0.835, 0.945 | 0.646 | — |

e017's replay arm is indistinguishable from both predecessors at matched k. The
pooled means differ because the LOW values arrive at LARGE k, which the pilot
has not reached. Do not write "reproduces"; write **no discrepancy detectable at
matched k**, and treat the k=10 arm mean from records as the definitive test.

Two things settled on the way:

- **e015's 0.537 is CITABLE.** It is recomputed here from
  `experiments/e015-no-replay/runs/replay25_s0.json`, a committed record file,
  and e017's own `analyze.py` reader reproduces it to four places (0.5371).
  It passes the standing rule.
- **The contrast is now CALLABLE, not READABLE.** e004g's completed 3-seed
  k=8 grid gives `consolidate` 0.5269 against `no_replay` 0.9572 from
  committed records. Two independent experiments, four arms, three seeds.

## 2. The headline SURVIVES and is ESCALATED — in a restated form.

Escalations the verifier makes and the builder could not:

1. **From READABLE to CALLABLE.** Three seeds, two experiments, committed
   records (above).
2. **It is not an artifact of the distillation write.** e004g's `direct_ft`
   arm is cross-entropy on ground-truth answers, and it shows the same
   replay-associated conditionality (pooled 0.4636). The effect survives a
   change of write objective. This is the one place the distillation scope
   limit does NOT bite, and it matters.
3. **The correct form of the claim is distributional and k-dependent, and it
   is a better headline than the pooled contrast.** Per-k pooled `umar_age0`:

   ```
   e004g consolidate (replay)  k=1..8:  1.00 0.33 0.66 0.49 0.33 0.05 0.34 0.34
   e004g no_replay             k=1..8:  1.00 0.91 1.00 0.94 1.00 0.91 0.99 1.00
   ```

   Under rehearsal, conditionality is ABSENT at k=1 and ACQUIRED as the
   sequence proceeds. Without rehearsal it is never acquired at any k. That is
   dossier 017 AMENDMENT 1's unclaimed cell **#2 (conditionality as ACQUIRED
   across a sequence)** measured directly, and it subsumes cell #1. Write the
   card around it.

**DEMOTED — NOT SUPPORTED:** "0.537 is rehearsal's conditionality level." The
per-write values are near-binary (of e015's twelve: five >= 0.885, four
<= 0.12; e004g's nine seed-values at k=2-4 are 0,0,1,1,0.915,0.055,0.06,0.47,
0.95). 0.537 is the mean of a two-mode distribution, not a level. Rehearsal
flips SOME writes to conditional; it does not lower conditionality uniformly.
Separately, **k=1 carries no aux batch in any arm by construction**
(`n_replay_batches` = 0 at k=1), so the k=1 row is definitionally baseline in
every arm and inflates every pooled mean equally. Pooled `umar_age0` must never
be quoted without `umar_age0_by_k` beside it.

## 3. The distillation scope limit applies. `selfdistil` is NOT confounded with the write objective — but its INTERPRETATION is bounded.

Ruled explicitly, both halves.

**Not confounded.** The write teacher is the FROZEN base under a 3-shot prompt
(`C.kl_loss`, `disable_adapter()`, `prompted=True`), KL. `selfdistil`'s teacher
is the CURRENT adapted model, UNPROMPTED, greedy strings, cross-entropy. Two
different teachers, two different objectives, two different input sets. The
dossier's arm 4 is a real second intervention, not the write wearing a hat.

**Interpretation bounded, and this is the load-bearing finding.** `replay25`'s
aux is `C.kl_loss(pm, tok, prior, rng)` — the write loss itself, re-run on a
prior skill. So the *entire* arm family in this harness is re-distillation of
base few-shot behaviour, and `selfdistil` differs from `replay25` in exactly
two ways: the labels are regenerated instead of stored, and the teacher has
drifted. The labels are regenerable ONLY because the base can already do the
skill from demonstrations — which is the scope limit. Therefore:

- CLAIMABLE if the number holds: **old data need not be STORED; the model can
  regenerate the labels rehearsal needs.** That is a storage result and a real
  one (it is LwF, cited not invented).
- **NOT SUPPORTED:** that `selfdistil` is a mechanism distinct from rehearsal,
  or that it would work for a skill the base cannot do from demonstrations. In
  that regime the teacher emits the wrong string and the method has nothing to
  preserve.
- The deciding number is `selfdistil_target_fidelity_by_k`, already logged per
  prior per k. **It must appear in the results card.** High fidelity means the
  arm is replay with free labels; a fidelity curve that decays while retention
  holds would be the interesting case and would need its own reading.

Every e017 card carries the scope limit in its assumptions section, per
`library/sweeps/harness-scope-limit.md`, with the `direct_ft` carve-out in
item 2 above noted as the one claim that escapes it.

## 4. Which criterion carries the verdict, and the missing world (verifier EDIT)

**The RELATIVE criterion carries the verdict, with an ABSOLUTE FLOOR.** Ruling
against absolute-only: `arb_restored_full` (`umar_age0 <= 0.20`) is stricter
than the reference arm itself (0.527-0.537) and would fail every arm in the
pre-registered A1 world (0.25-0.35), so absolute-only breaks identifiability
against the design's own rival prediction. Ruling against relative-only: the
builder's flag is correct in the counterfactual — if the reference lands high,
an arm that is plainly unconditional passes.

**EDIT MADE BY THE VERIFIER, WITH PILOT DATA VISIBLE, WHICH WEAKENS IT.** Three
changes to `analyze.py`, all recorded here:

1. `UMAR_REL_CAP = 0.50`. `arb_restored_threshold =
   min(replay25_umar + MDE_3SEED, 0.50)`, with `arb_restored_threshold_capped`
   emitted so a capped read is visible. An arm claiming restored arbitration
   must fire under an unowned cue less than half the time, whatever the
   reference does. Reason: the criterion must not inherit the reference arm's
   drift. What is known at edit time: `replay25` k=1-4 mean 0.646; the cap
   binds if and only if the k=10 mean exceeds 0.465.
2. New dry-run world **HIREPLAY**: `replay25` retention 0.69 (in band) with
   `umar_age0` 0.85, `selfdistil` retention 0.66 with `umar_age0` 0.88, others
   null. Intended verdict `RETENTION_WITHOUT_ARBITRATION`. This is the world
   the pre-registered set was missing; every prior world inherited 0.54.
3. Nothing else. `arb_restored_full` is retained as a reported field.

`python3 analyze.py --dryrun` after the edit: **PASS on all 8 checks** — A1,
A2, A3, NULL, NODYN, HIREPLAY, TAX, KNOWNPOS-DRIFT each map to their intended
category. The edit is load-bearing and verified as such: with the cap lifted to
1.0, HIREPLAY returns `A1_ARBITRATION_FIXES_RETENTION` at threshold 0.885 —
exactly the wrong headline the builder predicted.

Recorded, not edited: `arb_restored_vs_replay` hardcodes `MDE_3SEED` (0.035)
even on one-seed data, where `MDE_1SEED` (0.060) is the correct margin. The
grid is 3 seeds so the constant matches its intended use; any single-seed read
of that field is off by 0.025 in the permissive direction.

## 5. `reserve`'s A3 signature: NOT SUPPORTED as stated. The builder named the wrong discriminator.

`umar_age0` is "does skill s's rule fire under `<<K99>>`" — an accuracy against
s's rule, so ANY other output scores 0. `reserve` trains cross-entropy toward
the null token `?` on `<<K200..K207>> <word> =`, the same template with a
different marker. Driving `umar_age0` to 0.000 is therefore close to
definitional for that arm: it parks a non-rule answer on the format default and
the probe scores a non-rule answer as conditional. This is a stronger objection
than the probe-overlap one the builder raised.

`negcue` is NOT the discriminator. Its target is the frozen base's full
distribution — a harder, less transferable function than a constant token — so
`negcue` failing to show the effect is equally consistent with "reserve learned
a cheap constant that generalises and negcue's target does not". Same contexts,
but not comparable difficulty.

**The right discriminator is already in the records:** `newest_rule_under_j` —
the newest rule scored under each PRIOR skill's OWN marker, which `reserve`
never trains on. `analyze.py` surfaces it as `newest_rule_under_prior_cue` but
does not gate on it. Diagnostics are ON in the running pilot (launch command
carries no `--no-diag`), so the number will exist.

Ruling: `reserve`'s A3 reading is **NOT SUPPORTED** until
`newest_rule_under_prior_cue` is read from records. If it is also low while
prior retention is 0.000, the A3 signature is real and READABLE at one seed. If
it is high while `umar_age0` is 0.000, `reserve` learned "say `?` under an
unowned cue" and its arbitration claim collapses. No card may report
`reserve`'s `umar_age0` without that number printed beside it. Not gated in
code, because the synthetic worlds do not populate the field and adding it to
the verdict path would mean redesigning the dry-run with data visible.

## 6. Provenance: every number in PILOT.md is NON-CITABLE. 0010 am.2 is NOT discharged.

- `runs/` is empty in the repo AND on the box (`/home/azureuser/experiments/
  e017-arbitration/runs/`, 0 files at 12:09 UTC). `arb.py` writes the record
  only after the LAST k of an arm. **Every number in PILOT.md is stdout-derived
  and NON-CITABLE — diagnostic only.** That includes the matched-k comparison
  in item 1 of this ruling, which is diagnostic on the e017 side and citable
  only on the e015/e004g side.
- Sha is clean: committed `arb.py` == on-box `arb.py` == `32c225718396866e`,
  matching PILOT.md's claim. Seven `arb.py` processes alive, load 13.6.
- `analyze.py` **has not run on e017 output and cannot yet.** Decision 0010
  amendment 2 is **BLOCKING for the results card and for the grid**, not
  ambient. Partial discharge done here: e017's reader was run on e015's
  committed records (588 rows, same `eval_step` schema) and reproduces e015's
  own analyze to four places (`replay25` umar_age0 0.5371, `no_replay` 0.9621,
  retention 0.666 vs 0.000, `replay25` inside the known-positive band). The
  READER is certified against real records; the requirement is not met.
- Fifth-instance risk avoided by naming it: this is the same shape as e019.
  **Required pre-grid edit (verifier-mandated):** move the record write inside
  the skill loop in `arb.py` so a record exists after every k, and re-hash. A
  crash at k=9 currently destroys the whole arm, and a verifier cannot read a
  live run. Do not touch `arb.py` while the pilot is live.

## 7. Clause 7(a) and the censoring classes

- **Ratio metric / capability domination: covered, keep it that way.**
  Retention is a ratio to each arm's own `just_after`, so a weakly-writing arm
  posts high retention. `write_taxed_arms` fires in the TAX dry-run world and
  `prior_abs_clean` is emitted beside every ratio. No arm is write-taxed in the
  pilot so far (all writes 0.805-1.000 against a 0.60 tripwire). The e014
  lesson holds: read absolute prior accuracy, not the ratio.
- **Censoring/bimodality, NEW and live.** `umar_age0` is a bounded proportion
  that empirically piles at 0.000 and 1.000. Its mean is an unstable statistic
  and an MDE of 0.035 on such a mean is close to meaningless — the k=2-to-k=3
  swing of 0.000 to 0.835 is not noise around a level, it is a different mode.
  Standing requirement for e017: report `umar_age0_by_k` and
  `umar_age0_per_seed`, never the pooled mean alone, and state any pooled
  comparison as a comparison of MIXTURES.
- **No grid-step artifact**: eval resolution is 1/200, effects are 0.2-0.9.
- No exposure to e004g's broken `separate_adapters` arm (18 rows,
  `mean_retention_prior` None): e017 does not run or import that path.

## 8. Remaining builder caveats

- **λ = 1 untuned: UPHELD verbatim.** Any null is a null for λ=1 at this write
  budget, and the card must say exactly that.
- **`replay25` 0.000 at k=2 next to 0.835 at k=3: UPHELD and now explained.**
  It is the near-binary per-write structure of item 2, seen in all three
  experiments. Not an instrument fault. Read the arm mean AND the per-k series.
- **One seed: UPHELD.** Nothing from the pilot is CALLABLE. Everything is
  READABLE at best, and `reserve`'s A3 is not even that (item 5).
- **`cueanchor` one k behind: rule on it conditionally.** Do not delay the
  ruling or the grid for it; a null from it is a real null only if its recorded
  weight spread stays non-degenerate.

## 9. The 21-run grid: FOR, conditionally

Reason: the only thing three seeds buys is the difference between READABLE and
CALLABLE, and both live readings — `reserve`'s A3 and `selfdistil` ≈ replay —
rest on single-seed numbers from a quantity that swings 0.000 to 0.945 between
adjacent k. e004g's grid is the precedent in both directions: three seeds
turned a single-seed "collapse at k=4" into 0.938 retention at k=8, and turned
a contested 0.537 into a replicated one. 13.5 h of CPU on an idle box is cheap
against that. Launch when all four hold:

1. Pilot records exist in `runs/` and `analyze.py` runs on them, returning a
   non-suppressed verdict with `known_positive` = OK. (0010 am.2, blocking.)
2. Per-k record writing added to `arb.py` and re-hashed (item 6).
3. `newest_rule_under_prior_cue` and `selfdistil_target_fidelity_by_k` are in
   the reported output (items 3 and 5).
4. Records pulled into the repo before the box is deallocated (ops/README).

Keep all seven arms. `selfdistil_offcue` is the strongest construct-validity
control in the design and must not be dropped to save the 4.5 h; drop nothing
until the k=10 records say which arm is redundant.

## 10. Summary of gradings

| claim | grading |
|---|---|
| e015's `umar_age0` = 0.537 is real and citable | **CALLABLE** (committed records; replicated by e004g 3 seeds at 0.5269) |
| Rehearsal restores conditionality; without it the newest write is unconditional | **CALLABLE** (e004g 3 seeds, two write objectives) |
| Conditionality is ACQUIRED across the sequence under rehearsal and never without it | **CALLABLE** (e004g per-k, 3 seeds) — and this is the headline |
| "0.537 is rehearsal's conditionality LEVEL" | **NOT SUPPORTED** — mean of a bimodal per-write outcome |
| e017 `replay25` contradicts e015 | **NOT SUPPORTED** — partial-mean artifact; flag withdrawn |
| `selfdistil` retains without storing old data | **READABLE** (one seed, stdout, non-citable until records) |
| `selfdistil` is a mechanism distinct from rehearsal | **NOT SUPPORTED** — replay with regenerated labels; scope-limited |
| LwF predicted-negative control behaved | **READABLE** (one seed, stdout) |
| `reserve` shows A3: arbitration restored, retention not | **NOT SUPPORTED** until `newest_rule_under_prior_cue` is read |
| `negcue` separable from baseline | **NOT SUPPORTED** |
| Any e017 number as of this ruling | **NON-CITABLE** (stdout only; `runs/` empty) |

## VERIFIER RULING — AMENDMENT A (same day, before any e017 record existed)

Three corrections to the ruling above. All three are made while `runs/` is
still empty, so none is data-visible.

**A1. The escalated headline was over-graded. Split it.** The "conditionality
is ACQUIRED across the sequence" grading leaned on the k=1 endpoint (1.00),
which this ruling itself struck as definitional — no aux batch exists at k=1 in
any arm. Recomputed from committed records with k=1 excluded:

```
                        early k=2-4   late k>=5   all k>=2
e004g consolidate (3s)      0.494       0.265       0.363
e015  replay25    (1s)      0.553       0.473       0.495
e004g no_replay   (3s)      0.951       0.973       0.963
```

e004g's late drop is carried almost entirely by the single k=6 point (0.048);
without it, late is 0.337 against early 0.494, on three seeds of a 0-or-1
variable. e015 shows no trend at all. So:

- **CALLABLE:** rehearsal produces conditional writes from k=2 onward
  (0.36-0.50 pooled) while without rehearsal no write is conditional at any k
  (0.951-0.973 flat). That is the contrast, and it is the headline.
- **READABLE at best:** any within-sequence STRENGTHENING trend. The grading
  table row claiming CALLABLE for the trend is corrected to READABLE.

Also: dossier 017 amendment 1's unclaimed cell #2 is about AGE dynamics — a
write becoming conditional once a LATER write takes the format default, e004g's
original finding — not about the k-profile of age-0 `umar`. Two different axes.
The card must not conflate them.

**A2. Known-positive, asymmetric reading, PRE-REGISTERED HERE.**
`KNOWN_POSITIVE_BAND = (0.55, 0.83)` currently suppresses symmetrically, and a
HIGH landing is now plausible: e004g's 3-seed grid retains 0.938 at k=8 and the
pilot's `replay25` priors are running 0.805/1.0/1.0. Deciding fact, verified in
the code: `retention_restored` requires `gap >= HALF_GAP * g_replay`, so an
inflated reference RAISES the bar every intervention must clear. A high
reference is therefore CONSERVATIVE for A1; the false-A1 danger case is a LOW
reference, not a high one. Pre-registered rule:

- `r_ref < 0.55` — **SUPPRESS**, unchanged. The harness has lost the effect
  the interventions are measured against and no arm is readable.
- `r_ref > 0.83` — **ANNOTATE AND PROCEED** under explicit verifier sign-off in
  the card, recording `r_ref`, `g_replay`, and the raised half-gap line. The
  reference retaining MORE than e015 did makes every A1 claim harder, not
  easier. `analyze.py` is NOT edited for this; the verifier applies it by hand
  so the code keeps suppressing by default.

**A3. Novelty demotions the CE-write sweep forces on this experiment.** From
`library/sweeps/harness-scope-limit.md` CORRECTION and
`library/sweeps/ce-write-objective-2026.md`:

- The write is **CONTEXT DISTILLATION** (Snell, Klein & Zhong, arXiv
  2209.15189, 2022). Cite it as such in every e017 sentence describing the
  write. It is not arche's design.
- **`selfdistil` is not only LwF (2016) — it is SDFT**, *Self-Distillation
  Enables Continual Learning* (arXiv 2601.19897, Jan 2026), which is the same
  method for the same purpose published seven months before this pilot. Arm 4's
  novelty is reduced to the per-cue conditionality READOUT on it. Cite both.
- SDFT's own limitations section states the distillation scope limit in its
  authors' words ("effectiveness depends critically on the base model's
  in-context learning capabilities"), so item 3 of this ruling is a
  REPRODUCTION of a published limitation, correctly derived and not new.
- **Live conflict the card must carry:** SDFT reports a SCALE CROSSOVER where
  self-distillation UNDERPERFORMS plain SFT at 3B and only gains at 7B/14B.
  e017 runs Qwen2.5-**0.5B**, below that crossover, where the published curve
  says `selfdistil` should not win. If it wins here, either the two experiments
  measure different things or one is wrong — and that must be stated, not
  claimed as a result.

---

# PILOT COMPLETE, all 7 arms to k=10, seed 0 (2026-08-17)

7 records pulled into `runs/`. They had been sitting on the box unread
because `ops/pulse.sh` was globbing three record paths and the box used a
fourth (`~/experiments/<exp>/runs`). Third such miss today.

| arm | retention (clean) | prior_abs | umar_age0 | stores old data? |
|---|---|---|---|---|
| replay25 | 0.906 | 0.832 | 0.362 | **YES** |
| **selfdistil (LwF)** | **0.563** | 0.496 | 0.496 | **NO** |
| selfdistil_offcue (predicted-negative) | 0.012 | 0.010 | 0.926 | no |
| cueanchor | 0.000 | 0.000 | 0.945 | no |
| negcue | 0.000 | 0.000 | 0.881 | no |
| no_replay | 0.000 | 0.000 | 0.949 | no |
| reserve | 0.000 | 0.000 | 0.384 | no |

No arm is write-taxed (`mean_just_after` 0.927-0.937 across all seven), so
the retention ratios are comparable rather than inflated by weak writes.

## What this says. ONE SEED — nothing here is CALLABLE.

**1. Continual learning without stored data WORKS here, partially.**
Self-distillation on old cues retains 0.563 against replay's 0.906,
storing no old examples: the model regenerates its own prior behaviour and
is constrained to preserve it. That is about 62% of what rehearsal buys,
for zero storage. **This is [[li2016-lwf]] / SDFT, published, and must be
reported as a reproduction** — the e017 verifier already ruled that our
`selfdistil` arm is replay with regenerable labels, claimable as a
STORAGE result and not as a distinct mechanism.

**2. The predicted-negative control behaved, which is the strongest
construct-validity evidence in the experiment.** LwF's own documented
failure mode is proxy inputs that do not exercise the old behaviour;
`selfdistil_offcue` applies the identical machinery to off-cue inputs and
retains 0.012. So the 0.563 is not an artifact of the auxiliary loss
existing, it depends on distilling ON the old cues.

**3. Conditionality is NECESSARY but NOT SUFFICIENT, and `reserve` is what
proves it.** Across arms, conditionality tracks retention: 0.362 -> 0.906,
0.496 -> 0.563, and 0.881-0.949 -> 0.000. **`reserve` breaks the pattern**:
it restores conditionality to 0.384, statistically alongside replay's
0.362, and retains exactly 0.000. So making the newest write conditional
does not by itself preserve anything. This is dossier 017's A3 deflation,
observed cleanly, and it constrains the arbitration story the lab has been
building: arbitration is part of the mechanism, not the whole of it.

**4. The two unpublished interventions FAILED.** `negcue` (penalise the new
rule under held-out synthetic cues) and `cueanchor` (weight the loss toward
cue-conditional positions) both retain 0.000 at k=10. These were the two
arms the prior-art sweep found no precedent for; they do not work at
lambda = 1, which is the only lambda tested, so this is a null for this
setting rather than for the idea.

## FORMAL VERDICT: `SUPPRESSED_KNOWN_POSITIVE_OUT_OF_BAND` — the band, not the result, looks wrong

The pre-registered known-positive band for `replay25` retention is
(0.55, 0.83), anchored on earlier e015/e004g figures. This pilot measures
0.906, above the band, so the rule suppresses every arm and nothing is
formally readable.

**The band is very likely stale rather than the harness drifted.** Today's
completed e015 grid puts replay25 at 0.899 / 0.878 / 0.789 (mean 0.855)
and e004g's clean-skill grid at 0.922. So 0.906 sits inside the range that
two other experiments now report, and the (0.55, 0.83) band was anchored
on numbers that predate the pool-collision fix.

**I am not editing the band after seeing the data.** A verifier rules on
it. Until then the seven per-arm numbers above stand as raw measurements
and the formal C-verdict is withheld.

Carried: teacher-forced scoring throughout; one seed; lambda = 1 untuned
for every auxiliary loss.
