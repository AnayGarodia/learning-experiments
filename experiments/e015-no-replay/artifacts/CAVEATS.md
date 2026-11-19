# e015 — CAVEATS

Everything a reader needs in order to distrust these numbers correctly.
Written by the builder in the weakest form the data support (decision
0012 clause 5); the verifier should add to it, not replace it, and is
the only party permitted to strengthen a claim.

## Provenance

- Base: `Qwen/Qwen2.5-0.5B`, fp32, CPU. 494,032,768 params, 24 layers,
  d_model 896, rotary, no absolute position embedding (recorded by
  `pos_scheme()` into `gate_Qwen2.5-0.5B.json`).
- Box: `arche-cont`, **westus2** (eastus2 was at 96/96 cores),
  Standard 16 cores / 62 GB, `~/venv/bin/python`, torch 2.13.0+cpu,
  transformers 5.15.0, peft 0.20.0.
- Harness: e004g's `consolidate.py` is IMPORTED, not forked. e015 adds
  a skill-suite extension, example-budgeted replay, two allocation
  arms and a cheaper eval. See DESIGN.md for each.
- Generality corpus: e004g's pinned `generality.txt` (Project Gutenberg
  *Alice's Adventures in Wonderland*), read from the e004g directory so
  there is one copy, not two.
- `--selfcheck` runs e004g's 7 asserts plus 5 new ones (replay budget,
  rank-slice disjointness and coverage, projection correctness and
  idempotence, suite-extension index stability, new-skill purity).
  Passes on the committed code.
- **Provenance gap in the PILOT records, disclosed.** `C.git_sha()`
  resolves `__file__` to e004g's `consolidate.py`, so on its own it does
  not identify `cont.py`. A `cont_sha256` field was added mid-pilot;
  the four arms launched before that edit (`no_replay`, `replay25`,
  `sparse`, `orth`, seed 0) DO NOT carry it. The code that produced
  them is `cont.py` at md5 `4955d77e...` minus the `cont_sha256` line
  and minus the `separate_adapters` eval fix (which cannot affect those
  four arms — it only touches the `separate_adapters` branch). e004g
  was burned by exactly this class of gap and its fix was incomplete;
  the grid must carry `cont_sha256` in every record.

- **Three code states exist; the committed file matches none of the
  records.** Verified on the box (`~/arche/experiments/e015-no-replay/`)
  after the pilot finished:

  | state | sha256 | records it produced |
  |---|---|---|
  | arms-launch (pre-02:10, overwritten, sha unrecoverable) | — | `no_replay`, `sparse`, `orth`, `replay25`, `replay25_knownpos` |
  | box freeze 08-17 02:10 (adds `cont_sha256`, diag, `seen` fix) | `7ab4b731…` | `separate_adapters`, `no_replay_diag` |
  | committed repo file (adds `prefix_clashes`, `GRID_SKILLS`, post-hoc) | `33c4d746…` | `replay01_smoke` — the ONLY record whose `cont_sha256` matches the committed file |

  So a record's `cont_sha256` will NOT equal the committed file's hash,
  by design: the post-hoc additions are analysis helpers and a
  `--selfcheck` assert, and touch no training path. The first state's
  hash is genuinely lost — the file was edited in place on the box.
  This is the third code state of the same gap; the grid must hash at
  launch, not at write.

- **No stray writer.** Confirmed at 05:08 UTC: no experiment python
  process on the box (`pgrep -af cont.py` matches only its own bash
  wrapper — that is the false positive), `runs/` untouched since 04:55,
  load average decayed to 0.07.

## 1. The known-positive (decision 0012 clause 2) — EXACT

`runs_knownpos/replay25_knownpos_s0.json`: `replay25` on e004g's exact
pilot configuration (skills 5/7/8, seed 0, r=8, n_eval 200).

| k | wrote | B_title | B_upper | B_double | nats |
|---|---|---|---|---|---|
| 1 | B_title | 1.000 | – | – | 2.8383 |
| 2 | B_upper | 0.980 | 0.930 | – | 2.8481 |
| 3 | B_double | 1.000 | 0.890 | 1.000 | 2.8478 |

Matches e004g's committed pilot table to the digit; the only difference
anywhere is 0.0001 of generality at k=1 (CPU thread-order noise). The
harness has not drifted, and budgeting replay in EXAMPLES reproduces
e004g's step-budgeted schedule exactly at 25%.

**It is quarantined in `runs_knownpos/`, not `runs/`.** It is a
3-skill k<=3 validation run; left in `runs/` it would have been
averaged into the 12-skill `replay25` grid cell. (The record predates
the fix that puts the tag into the arm KEY rather than only the
filename, so the guard here is directory placement.)

## 1b. The dynamic-range gate FIRED, hard — and what that does and does not license

Complete pilot, seed 0, N = 12 gated non-degenerate skills, k = 1..12,
n_eval = 200. Retention = accuracy / that skill's own just-after value,
averaged over conditionality-clean rows (age >= 2), 55 rows per arm.

| arm | retention (age>=2) | mean just_after | writes landed | generality drift |
|---|---|---|---|---|
| `no_replay` | **0.0182** | 0.963 | yes | 0.070 nats |
| `sparse` | **0.0186** | 0.948 | yes | 0.039 nats |
| `orth` | **0.0186** | 0.960 | yes | 0.103 nats |
| `replay25` | **0.6787** | 0.959 | yes | 0.093 nats |
| `separate_adapters` (ceiling) | 1.0 by construction | 0.960 | yes | 0.051 nats |

Restricted to the 10 TOKEN-PREFIX-FREE skills (see §2b), 44 rows:

| arm | retention (age>=2) | gap over baseline |
|---|---|---|
| `no_replay` | **0.0000** | – |
| `sparse` | 0.0005 | +0.0005 |
| `orth` | 0.0006 | +0.0006 |
| `replay25` | **0.6915** | +0.6915 |

**Structural allocation buys 0.09% of what rehearsal buys.** The
dossier's pre-registered kill condition for C1 is "allocation buying
< half the retention that replay buys"; this is ~500x below that line
and the sign is identical in both allocation arms.

Write quality is NOT the explanation: every arm wrote its own skill at
0.948-0.963 on average, and the swappable-adapter ceiling wrote at
0.960 — so writing into an adapter that already holds eleven skills is
no worse than writing into an empty one. **The damage is entirely to
OLD skills, never to the ability to write a new one.**

**Weakest form: ONE SEED.** No per-seed split is detectable from one
seed, so nothing here is a result. What the pilot establishes is that
the instrument has enormous dynamic range and that the arms separate by
~19x the 3-seed MDE. The grid's 3 seeds are what would make it a
finding.

**A mean of 0.679 hides a wide split, and this is why per-skill
reporting is mandatory.** `replay25` retention per skill runs
B_title 0.988, B_first 0.968, B_the 0.875, B_plural 0.855,
B_upper 0.449, B_dashw 0.384, B_pre_x 0.000, B_star 0.000. Two of ten
skills are fully lost even at 25% rehearsal.

### The builder issues NO C1/C2/C3 verdict, because his own K0 tripwire fired

`analyze.py` returns `SUPPRESSED_BASELINE_K0_BLOCKED`. The
pre-registered block is `max(unknown_marker_applies_rule) > 0.20` over
age >= 2 rows, and it reads 1.00 in the no-replay arms and 0.91 in
`replay25`. Under DESIGN that blocks every rival reading, so the
numbers above are reported and **no rival is called.** Offered to the
verifier, not acted on:

1. The block is a **MAX over 55 noisy rows**. The e004g verifier
   already replaced exactly this aggregation for the retention verdict
   (a min over ~45 rows made a spurious K3 ~88% likely) and left it in
   place for K0. The age profile is flat at 0.000 for every age >= 2
   except age 7.
2. The rows that trip it are identified. Over the **full 588-row
   results.csv there are 7 blocking rows out of 220 clean (age >= 2)
   rows**, not 3 — the earlier count was traced before `replay25` and
   `separate_adapters` were pulled and covered the no-replay arms only.
   The full list:

   | arm | k | age | measured | newest | umar |
   |---|---|---|---|---|---|
   | no_replay | 12 | 7 | B_triple | B_double | 1.000 |
   | orth | 12 | 7 | B_triple | B_double | 1.000 |
   | sparse | 12 | 7 | B_triple | B_double | 1.000 |
   | replay25 | 12 | 7 | B_triple | B_double | 0.885 |
   | replay25 | 7 | 2 | B_triple | B_upper | 0.910 |
   | replay25 | 10 | 5 | B_triple | B_star | 0.595 |
   | replay25 | 6 | 2 | **B_plural** | **B_capfirst** | 0.320 |

   **The 7 rows do NOT partition cleanly into "artifact" and "real" —
   two stories overlap.** The prefix artifact requires B_double's rule
   to fire under K99, which is what makes the teacher-forced B_triple
   continuation score. That condition is met exactly when B_double is
   the newest write. Tracing B_triple's umar across every k shows the
   split (0 = clean, all arms, seed 0):

   | newest at k | 7 (B_upper) | 8 | 9 | 10 (B_star) | 11 | 12 (B_double) |
   |---|---|---|---|---|---|---|
   | no_replay | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.000** |
   | sparse | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.000** |
   | orth | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | **1.000** |
   | replay25 | **0.910** | 0.00 | 0.00 | **0.595** | 0.00 | **0.885** |

   In the three no-replay arms B_triple's umar is 0.000 at every clean k
   **except** k=12, where B_double is newest — the artifact fires only
   under its own precondition, which is what an artifact should do. In
   `replay25` it also fires at k=7 and k=10, where B_double is NOT
   newest and the prefix precondition is absent.

   So the defensible decomposition is:

   - **4 rows: prefix artifact** (all four arms at k=12, B_double
     newest). Not a measurement.
   - **3 rows: `replay25` only, consistent with rehearsal keeping prior
     writes live under an unknown marker** — k=7 and k=10 (B_triple with
     B_double not newest) and k=6 (B_plural at 0.32 under B_capfirst,
     a prefix-free pair and therefore the one **unconfounded** row).
   - `replay25` k=12 is **confounded**: both stories predict it.

   That is the §1c mechanism running backwards — rehearsal
   de-conditionalises *prior* writes, the mirror of it making the
   *newest* write more conditional (`umar_age0` 0.537 vs 0.954-0.962).
   Three rows, one seed, one unconfounded: a hypothesis for the
   verifier, not a finding.

   **On the prefix-free `GRID_SKILLS` suite there are 0 blocking rows
   in all four arms** (148 clean rows; verified against the full csv,
   not inferred). Note the mechanism differs per row: the B_triple rows
   drop because the *measured* skill is excluded, the B_plural row
   because its *newest* skill (B_capfirst) is. Restricted retention on
   that subset: no_replay 0.0000, sparse 0.0000, orth 0.0007,
   replay25 0.6637 (n=37/arm) — same separation as the headline read.
3. `umar_age0_newest_write` is 0.954-0.962 in the three no-replay arms
   against **0.537 in `replay25`** — i.e. rehearsal makes the freshest
   write roughly twice as cue-conditional. That is an independent,
   quantitative confirmation of the masking mechanism in §1c.

**Weakest form: one seed, one skill order, one corpus, one base.** No
per-seed split is detectable from one seed, so none of this is a
result yet; it is a pilot that establishes the instrument has dynamic
range and that the arms separate far beyond the MDE.

**RESOLVED (DESIGN amendment 2 result): the 0.000 rows are MASKING, not
erasure.** The newest rule answers under the old skill's own marker at
0.969-1.000, and every masked skill stays ranked far better than the
frozen base (B_upper 20.6 / 89.6 vs base 306.3; B_dashw 3.7 vs 8.1;
B_title 12.5 vs 20.8, in a 151,936-token vocabulary). So prior skills
are INACCESSIBLE, not destroyed. Consequences for how these numbers may
be quoted:

- "catastrophic forgetting" is the WRONG phrase for this pilot. Write
  "prior skills become inaccessible without replay".
- the retention metric is measuring cue-conditionality at least as much
  as interference, and the two are not separated by this design.
- **e015's own K0 block did not fire and would not have.** It reads the
  conditionality probe at age >= 2 ("is the OLD rule format-attached"),
  which is 0.000; the unconditionality is in the NEWEST write, where
  the probe reads 0.906-1.000 at age 0. The pre-registered aggregation
  is blind to this failure mode. `umar_age0_newest_write` is now
  emitted per arm; the pre-registered block was NOT redefined after the
  fact.

**What it did NOT license before that check, and the load-bearing caveat.** A
prior skill at 0.000 has two explanations and accuracy cannot separate
them: ERASURE (the representation is gone) or MASKING (the
representation is intact but the newest write owns the format default
and answers under every cue). Under the masking reading, e004g's
measured acquired-conditionality curve makes 0.000 the EXPECTED value
for a young skill and the whole comparison would be an artifact of
reading at the wrong age. DESIGN amendment 2 registers two
discriminators for this BEFORE their numbers were read. Until they
have both fired, the correct statement is that the arms separate, not
that allocation fails.

## 2. Degeneracy check (decision 0010 amendment 3) — and what it cost

All 36 skills carry `copy_oracle` and a mismatched-shot control. The
12 skills carrying any reading all have `copy_oracle <= 0.02`. The four
type-A skills reproduce e004g's copyability exactly (0.953 / 0.969 /
0.938 / 0.969) and are excluded from every headline read by id in
`analyze.py`.

**A measured fact about the base, not a nuisance:** 11 of 12
character-surgery candidates FAILED the gate — delete-a-character,
swap, rotate, every-other, middle-character all at 0.000-0.15 prompted.
Qwen2.5-0.5B's in-context repertoire on this format is *copy the input
and decorate it*, plus upper-casing and first-character extraction.

**Consequence, and the suite's single biggest weakness: 8 of the 12
usable skills are "attach a fixed affix to the input".** Only B_upper,
B_first, B_capfirst and B_title transform the characters. That is worse
than e004g's already-disclosed 4-of-6 and it is forced by the base.
A retention mean over these 12 is a mean over roughly two mechanisms.
`analyze.py` emits `retention_clean_by_skill` and no mean may be quoted
without it. **The specific attack: if decay is concentrated in the
affix family, the claim narrows to "skills sharing an answer template
interfere", which is much weaker than "skills interfere".**

## 2b. SUITE DEFECT found on pilot output: answer-token prefixing

**B_double's answer (`apple apple`) is a token-prefix of B_triple's
(`apple apple apple`) on 100% of eval words.** Teacher-forced scoring
takes the argmax at each answer position, so while B_double owns the
format default the model's self-similar continuation ALSO satisfies
B_triple — and B_triple scores 1.000 under a marker it does not own.
That single pair produced the only non-zero retention anywhere in the
no-replay arms (B_triple 0.167 while every other skill sat at 0.000)
AND supplied all three rows that tripped the K0 block.

A second, milder case: B_capfirst (`A`) is a token-prefix of B_upper
(`APPLE`) on 38% of eval words.

Such a pair is not two skills; it is one skill scored twice. Fixed
three ways:
- `prefix_clashes()` computes the collisions at TOKEN level (the level
  the metric works at — string-level prefixing over-flags badly, e.g.
  `"a"` vs `"apple apple"` is harmless because `" a"` and `" apple"`
  are different tokens) with the fraction of eval words affected;
- `--selfcheck` now ASSERTS the grid suite is prefix-free, so this
  cannot recur, and it runs in milliseconds with no model;
- `GRID_SKILLS` is the 10-skill prefix-free set (B_triple and
  B_capfirst dropped; B_upper kept over B_capfirst because it is one of
  only two genuine character transforms in the suite).

Every headline number in §1b is reported both ways. The conclusion does
not move: excluding the contaminated skills makes the baseline cleaner
(0.0182 -> 0.0000), not weaker.

## 2c. Measured throughput and the projected grid cost (before anyone commits)

`throughput.json`, measured uncontended at 16 threads on this box:

| quantity | Qwen2.5-0.5B | Qwen2.5-1.5B |
|---|---|---|
| forward tokens/s (batch 32, seq 36, fp32) | **476** | **173** |
| s / optimizer step (KL, batch 16) | **2.34** | **6.05** |
| params | 498.4M | 1552.9M |
| LoRA params at r=8 | 4.40M | 9.23M |
| **process RSS** | **3.55 GB** | **10.23 GB** |

(e004g measured 567 tok/s and ~2.0 s/step for the 0.5B; this box is
~16% slower, which is the westus2 SKU, not a code change.)

**Measured pilot cost**: 5 runs at N = 12 concurrently — 4 workers at 4
threads plus 1 at 2 threads — completed in **3.47 h wall** each. Peak
RSS per worker 5.3 GB (`no_replay`, `sparse`) and **6.3 GB (`orth`**,
which stores an orthonormal basis per LoRA tensor and grows with k).

**Worker sizing is derived from MEASURED memory, not core count**, per
the ops rule that wedged a box twice: at 6.3 GB/worker, 62 GB admits 9
workers, so on this box the 16 cores bind first and 4-5 workers is
right. For the 1.5B arm, 10.23 GB base + optimizer/activations puts a
worker near 15 GB, so **4 concurrent 1.5B workers (~60 GB) is the
ceiling and 3 is the safe number.**

**Projected full grid, from the pilot's own timings:**

| block | runs | per run | wall on this box |
|---|---|---|---|
| 6 arms x 3 seeds, N = 10, 0.5B | 18 | ~2.43 h | **~8.8 h** (5 concurrent) |
| C3 second size: 3 arms x 1 seed, N = 10, 1.5B | 3 | ~6.6 h | **~6.6 h** (3 concurrent) |
| **total** | 21 | | **~15.4 h** |

N = 10 rather than 12 because of the prefix defect; eval cost is
quadratic in N, so N = 10 is ~0.70 of the pilot's per-run cost. Nothing
here needs a GPU.

## 3. Deviations from the pre-registered pilot order, owned

1. **`sparse` and `orth` were launched BEFORE the dynamic-range gate
   was read.** DESIGN's pilot order puts the gate first. They were
   launched concurrently to use the box, which is a wall-clock decision
   and not a scientific one; `analyze.py` suppresses every allocation
   reading mechanically if the gate fails, so no reading can leak
   through. Disclosed rather than left to be found.
2. **The `replay01` code path is now exercised end to end (CLOSED).**
   It previously had only a unit assert. A 3-skill run
   (`runs_smoke/replay01_smoke_s0.json`, order B_title, B_upper,
   B_dashw) confirms the `rf < 1/STEPS` branch fires against the model
   and is **not a silent no-op**: `n_replay_batches=1`,
   `n_replay_examples=10` at every k >= 2 (and correctly 0 at k=1, no
   prior to replay), against 15 batches / 240 examples for `replay25` on
   the same three skills. Budget 0.01 x 60 x 16 = 9.6 -> 10 examples in
   one batch at step 30, as registered.

   Two things fell out, neither of them a verdict:

   - The 1% arm **retains 0.000** on both priors at k=2 and k=3 — i.e.
     indistinguishable from `no_replay`. The `replay25` known-positive
     retained its priors at **0.89-1.00** (B_title 0.98 at k=2; B_title
     1.00 and B_upper 0.89 at k=3). **The two runs share two of three
     skills, not all three** — smoke order is B_title, B_upper,
     **B_dashw**, knownpos order is B_title, B_upper, **B_double** — so
     this is a near-comparison, not a matched one. The literature's
     setting (Scialom 2022,
     98-99.8% at 1%) does not reproduce here. That is 3 skills at one
     seed and 1% of a 60-step write is 10 examples, so this is a smoke
     test that happened to be informative, **not** a refutation of the
     literature; the honest reading is that 1% of *this* write budget is
     ~10 examples and the setting may not be comparable at all.
   - `nats=2.8383` at k=1 reproduces the known-positive to the digit on
     a third independent run.
3. **The `separate_adapters` eval bug was found by review, not by the
   dry-run, and fixed before that arm ran.** `eval_step` was being
   called with the full skill order, but this arm discards prior
   skills' weights at each reset, so their scores would have been ~0
   and the CEILING arm would have looked like the worst arm in the
   experiment. The dry-run missed it because `synth()` mimics e004g's
   `seen=[s]` shape — i.e. **the dry-run validated a different data
   shape than the code produces**, which is a general limitation of
   synthetic identifiability checks and is worth carrying forward as a
   lab lesson.

## 4. Structural limits of the two allocation mechanisms

1. **`sparse` allocates RANK SLICES, not arbitrary parameters, and that
   is deliberate.** LoRA's update is bilinear (`dW = B@A`), so masking
   arbitrary entries of A and B does not give additively separate
   writes. Rank slices do. At N > r the slices COLLIDE by construction:
   the pilot runs 12 skills into 8 slices, so four pairs share a slice.
   That is a capacity-collision arm, not a clean-allocation arm, and a
   fair reading needs the r = N variant (`--rank 12`) too.
2. **`orth` enforces orthogonality in the LoRA FACTOR space, not in
   weight space.** Bilinearity again: factor-space orthogonality does
   not imply `dW` orthogonality. This is MEASURED, not assumed —
   `orth_weightspace_cos` records achieved cosines between each write's
   `dW` increment and every prior write's on four sampled modules, in
   every record. **An `orth` arm whose measured cosines are not near
   zero has not done what its name says and must be reported that way.**
3. **`orth` projects the APPLIED UPDATE, not the gradient**, because
   AdamW rescales per coordinate and a projected gradient does not
   produce a projected update. Cost: the projection fights the
   optimizer, so the effective step size shrinks as constraints
   accumulate. By skill 12 there are 11 constraints per tensor, and if
   late writes fail the tripwire that is the mechanism's own cost, not
   a bug — but it means retention and write quality must be read
   together, never retention alone.
4. **Retention is a RATIO to each skill's own just-after value.** An
   arm that writes weakly posts high retention for free.
   `mean_just_after` is printed beside every retention number and
   `writes_landed_by_skill` is emitted per skill. The arm-level
   tripwire exclusion is COARSE — one marginal write out of twelve
   voids a whole arm — so the per-skill view exists for the verifier to
   rule on excluding rows instead.

## 5. Scope limits that do not go away

1. **One corpus.** The e004g marker-skill format, synthetic. Decision
   0012 clause 3 is not satisfied and no transfer is claimed. Every
   number here is a claim about this corpus and this base.
2. **One base at 0.5B for the primary read.** The C3 second-model-size
   arm is mandatory and is priced, not run, in the pilot.
3. **One seed in the pilot.** No per-seed split can be detected from
   one seed, so no pilot number may be quoted as a result. The grid's
   3 seeds are what make `retention_clean_per_seed` meaningful.
4. **k = 12 cannot confirm a capacity wall.** Sequential editing
   collapses at k ~ 32. k = 12 can KILL C1 and can measure the fraction
   of the replay gap allocation buys; it cannot confirm C3's wall.
5. **Answers are short (1-4 tokens).** Whether longer procedural
   behaviour consolidates the same way is untested and is closer to
   what dossier 004 actually cares about.
6. **LoRA r=8 on seven projections in 24 layers, 60 steps, lr 1e-4** is
   one unswept point. Neither arm has been tuned; e004g's standing note
   that learning rate was never exonerated applies here too.

## 6. What the verifier should attack first

In priority order. Items 1-3 could each overturn the pilot's reading.

1. **The masking reading itself.** It rests on a 4-skill n=128 run plus
   a frozen-base rank control. Attack: rank is a weak proxy for
   "recoverable" — a rank-90 answer in a 152k vocabulary is far above
   base but still unusable. Does a stronger recovery probe (re-prompt
   the old skill on consolidated weights; or fine-tune it back and
   count steps) agree? If a masked skill needs as many steps to restore
   as a fresh write, "not erased" is a distinction without a
   difference.
2. **Whether `orth` did what its name says — WITHDRAWN BY THE VERIFIER,
   2026-08-16. The pilot records cannot support the claim.**
   The builder wrote "CHECKED, and it did: `|cos| <= 0.033` at every k".
   That reading is not available from these records. `cont.py` stored
   `sorted(cs)[-5:]` — the five largest **signed** cosines out of up to
   4 modules x (k-1) priors, i.e. 44 values at k = 12. A large NEGATIVE
   cosine is just as much a failure of orthogonality and was silently
   discarded, so the recorded maximum is a maximum over a sign-biased
   truncation, not over the sample. The adapter states were not saved,
   so this is unrecoverable from the pilot.
   FIXED in `cont.py` for the grid: cosines ranked by MAGNITUDE, plus
   `orth_weightspace_cos_stats` (`n`, `max_abs`, `mean_abs`,
   `min_signed`). **Mechanism certification for `orth` is deferred to
   the grid.**
   **The `orth` null's MEANING is unaffected**, because it does not rest
   on the weight geometry: in the `orth` arm the newest write's rule
   answers under every prior skill's own marker at 0.969-1.000 and
   `umar_age0_newest_write` is 0.954-0.962. The induced functions
   collide completely whatever the deltas did — which is the point (see
   DESIGN VERIFIER RULING §2b and §3).
3. **`sparse` at r = N.** The pilot ran 12 skills into 8 rank slices,
   so four pairs shared a slice by construction. `--rank 12` gives
   genuinely disjoint slices. If disjoint allocation also buys ~0%,
   the negative is much stronger; if it does not, the pilot measured a
   capacity collision and called it an allocation failure.
4. **The K0 block.** Rule on it. The builder did not lift his own
   tripwire. 4 of the 7 offending rows are a prefix artifact firing
   under its own precondition and the age profile is flat at 0.000
   elsewhere, but that is an argument, not a ruling. **Attack the other
   three specifically** — all `replay25`, at k=6, 7 and 10, where the
   prefix precondition is absent. k=6 (B_plural at 0.32 under
   B_capfirst) is the only prefix-free, unconfounded one. If they are
   real, 25% rehearsal itself de-conditionalises prior writes, and the
   reference arm is the arm with the conditionality problem.
5. **Skill homogeneity.** 8 of 12 usable skills are "attach an affix".
   Check whether `replay25`'s per-skill split (0.988 down to 0.000)
   tracks the affix/character-transform boundary. If it does, the
   claim narrows to skills sharing an answer template.
6. **Order and seed.** One seed, one order. B_dashw happened to be
   written first and B_pre_x/B_star ended at 0.000 even with replay —
   position effects and seed effects are completely unseparated here.
7. **`replay01` is exercised (was: never run).** See §3.2 — the branch
   fires, 1 batch / 10 examples per k >= 2. **Attack the arm's meaning
   instead:** 1% of a 60-step x 16 write budget is 10 examples, and 10
   examples retaining 0.000 may say more about the budget than about
   rehearsal fraction. Whether this arm is a fair rendering of the
   literature's 1% setting is a design question the builder cannot
   settle and is flagging rather than answering.
8. **Re-run `--selfcheck`** (it now asserts the suite is prefix-free)
   and re-run `analyze.py --dryrun` (6 synthetic worlds, all callable).
