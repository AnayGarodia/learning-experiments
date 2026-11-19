# e013 — The exchange rate R: how many gradient examples buy one in-context example

Status: DESIGN 2026-08-16, written BEFORE code (standing rule).
Pre-registration: `dossiers/013-icl-vs-gradient-exchange-rate.md`.
Decisions in force: 0010 (pre-launch protocol, all 3 amendments), 0011
(success bar + amendment 1), 0012 (identifiability, known-positive,
corpus independence, weakest-form reporting).

## Which bar, at what magnitude

**Bar (b) — fundamental bottleneck**, and it prices an existing
PORTFOLIO.md row rather than adding one. Axis: **learning**.

R is the CEILING on what any context→weights channel can be worth. The
row `learning | consolidation: context->weights (004) | unbounded /
paradigm` is the lab's headline bet (B2, 79%) and e004g is measuring
whether the channel can be built. e013 measures whether there is
anything worth banking:

- R ~ 2-5 → consolidation is a convenience; B2 is badly overpriced.
- R ~ 100+ → fast learning is enormously more sample-efficient than slow
  learning; the prize is real and portfolio-sized.

Independence: this is not an additional factor. It is the ceiling on
004's factor, so it must be measured before 004's result can be
interpreted. It does not change the running product on its own; it
changes the maximum the `learning` row can ever contribute.

## The measurement

For a fixed base model and a fixed skill, two learning curves against
the SAME capability metric and the SAME eval query set:

- **IN-CONTEXT**: accuracy vs number of shots
  `n ∈ {0,1,2,3,4,6,8,12,16,24,32,48,64}`. No weight updates.
- **GRADIENT**: accuracy vs number of DISTINCT training examples
  `N ∈ {1,2,3,4,6,8,12,16,24,32,48,64,96,128}`. LoRA, swept LR,
  early stopping on a held-out dev split. No context examples.

`R(C) = n_grad(C) / n_ctx(C)` at capability targets `C ∈ {0.25, 0.50,
0.75}`, so X3 (regime dependence) is visible rather than assumed away.

**R counts EXAMPLES, not FLOPs.** The gradient arm takes many optimizer
passes over its N examples. R is a sample-efficiency ratio and says
nothing about compute cost; that is stated in CAVEATS.md and must not be
silently upgraded.

## Base

Qwen2.5-0.5B (base, not instruct), the checkpoint e004g's Stage 1 gate
already validated: rotary, 18T pretraining tokens, `model_type qwen2`,
`has_absolute_pos_embedding false`, 494,032,768 params, CPU-runnable.
The gate result is reused, not re-derived; e013 re-measures the copy
oracle on its own skills and its own word pool.

## Skills

Only **non-degenerate** skills (copy-oracle <= 0.20) may carry the
reading. e004g found 4 of its 8 original skills solvable by
constant-offset copying at ~0.95; a novel skill that is secretly
copyable would fake exactly the result X1 predicts, so the oracle is run
on the invented skills too.

| skill | family | rule | matched partner | copy-oracle (measured) |
|---|---|---|---|---|
| `B_first` | natural | `apple → a` | — | **0.070** NON-DEGENERATE |
| `B_upper` | natural | `apple → APPLE` | — | **0.000** NON-DEGENERATE |
| `N_map` | novel | `apple → u` via a fixed arbitrary DERANGEMENT of the alphabet | `B_first` | **0.085** NON-DEGENERATE |
| `N_shift2` | novel | `apple → c` (first letter + 2, wrapping) | `B_first` | **0.070** NON-DEGENERATE |
| `N_perm` | novel | `apple → pealp` (fixed arbitrary permutation of the 5 letter POSITIONS) | `B_upper` | **0.000** NON-DEGENERATE |

`B_first` and `B_upper` are the two e004g gate skills that both passed
and are non-degenerate AND transform the input's characters (e004g's own
DESIGN flags that its other four non-degenerate skills are all "copy the
input and decorate it").

**The novel arm is mandatory and is the point.** X2 says the apparent
in-context advantage is not a learning-speed difference: the base has
seen the skill in pretraining, so context SELECTS an existing capability
while gradients must BUILD one. `N_map` is the sharpest available test —
`B_first` and `N_map` share format, marker syntax, answer length,
tokenization and eval set, and differ ONLY in whether pretraining could
have supplied the mapping. The bijection is a derangement (checked in
`selfcheck`), so no answer coincides with the natural first-letter
answer.

Honest limitation, stated by the builder rather than left for the
verifier: `N_map` is **coverage-limited by construction**. An eval word
whose first letter was never demonstrated is unanswerable in EITHER arm,
so `N_map` is a 26-item memorisation task, not a rule. That is measured,
not hand-waved: `coverage_oracle` (the accuracy ceiling of a learner
that knows exactly the demonstrated letters) is recorded at every shot
count. A collapse of R on `N_map` is therefore consistent with X2 AND
with the weaker reading "we compared a generalising task to a
memorising one". `N_shift2` and `N_perm` exist to separate those: both
are systematic and generalise to unseen inputs. `N_shift2` is only
WEAKLY novel — "the next letter in the alphabet" plausibly occurs in
pretraining — and is labelled so everywhere it appears.

## Adaptations to e004g's harness, each disclosed

e013 IMPORTS `e004g-consolidation/consolidate.py` and calls its
functions with patched module globals. Nothing is forked or
reimplemented: `encode`, `answer_logits`, `accuracy` (teacher-forced
exact match), `make_prompt` / `make_items` (including the
mismatched-shot path), `copy_oracle`, `pos_scheme`, `throughput` and
`load` are e004g's own code.

- **A1 — word pool enlarged.** e004g split a 64-word pool 32/32. e013
  needs 128 distinct gradient examples and 64 distinct shots, so the
  pool is the union of e004g's 64 words with the 5-letter lowercase
  words of e004g's own pinned `generality.txt`, shuffled with a fixed
  seed into **64 eval / 32 dev / 190 train**, all disjoint (asserted).
- **A2 — shot count is a variable.** e004g fixed `SHOTS = 3`. e013 sets
  `consolidate.SHOTS` per condition; `selfcheck` asserts the prompt
  really has k+1 lines.
- **A3 — a dev split is added.** e004g had none (its budget was fixed at
  60 steps). e013's gradient arm needs early stopping, which requires a
  third split. Dev is 32 words, disjoint from train and eval.
- **A4 — the gradient objective is CE, not KL.** e004g's `consolidate`
  arm distils the model's own prompted behaviour; that is the
  consolidation channel, a different thing. e013's gradient arm is
  ordinary supervised fine-tuning on ground-truth answers — the same
  loss as e004g's `direct_ft` control.
- **A5 — three novel skills appended** at indices 14-16, so e004g's
  0-13 keep their identities and its gate file stays readable.

## Controls, all mandatory

1. **Mismatched-shot on the in-context arm** at EVERY shot count (not
   just one): same marker and format, another skill's rule
   demonstrated. e004g measured 0.000 for all 14 skills. If mismatched
   accuracy tracks matched accuracy, the model is riding format or
   frequency and the in-context curve is not a learning curve.
2. **Copy-oracle per skill**, including the invented ones. Table above.
   Any skill above 0.20 is excluded from carrying the reading.
3. **Unprompted baseline** = the `n = 0` point of the in-context curve,
   measured on the frozen base with the identical query set. It is the
   floor for BOTH arms (same base), so capability targets sit above the
   right floor by construction.
4. **Coverage oracle** at every shot count (new, e013-specific): the
   ceiling implied by which first letters were demonstrated.
5. **LR sweep on the gradient arm, optimum required to be INTERIOR.**
   `{1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3}` at N=32 per skill. This is not
   boilerplate: e-retro and e008 both had verdicts move on edge-optimal
   LRs, and here an untuned gradient arm inflates R in ICL's favour —
   the direction of the exciting answer. **If the optimum lands on an
   edge, the sweep is extended before any R is reported.**

Two further asymmetries deliberately set in the DEFLATIONARY direction,
so that a large R is not an artifact of a handicapped gradient arm:

- Early stopping ties on dev break toward the LATEST (more-trained)
  checkpoint. The opposite tie-break returns the untrained base whenever
  dev is flat, which is common at N=1-4, and would inflate R.
- The gradient arm gets 200 optimizer steps at batch min(N,16) — up to
  ~25 epochs at N=128 and 200 passes at N=1 — while the in-context arm
  gets one forward pass. R is a ratio of EXAMPLES with compute
  deliberately unmatched in the gradient arm's favour.

## Known-positive (decision 0012 item 2)

The in-context arm on the natural skills IS the positive control. e004g
measured `B_first` at 0.985 prompted / 0.000 unprompted / 0.000
mismatched at 3 shots, and `B_upper` at 0.940 / 0.000 / 0.000. **A null
on the natural in-context arm means this instrument is broken, not that
ICL is absent**, and blocks the experiment rather than producing a
result.

## Corpus independence (decision 0012 item 3)

The novel arm is structurally different from the natural arm by
construction — that is the second corpus in the sense that matters here.
It is NOT a second natural-text corpus, and it is not a second base. R
is a claim about Qwen2.5-0.5B under e004g's one-line marker format until
a second base says otherwise.

## Eval resolution and MDE

- Capability metric: teacher-forced exact match, 200 queries drawn from
  **64 distinct held-out words**. The distinct-input count bounds the
  effective sample, so SE is ~0.06 worst case and ~0.04 at p=0.9 — NOT
  the 0.035 a naive binomial on n=200 would claim. Every condition
  scores the IDENTICAL query set (fixed `EVAL_SEED`), so the comparison
  between the two arms is paired and sampling noise largely cancels.
- Example axes are √2-ish grids (13 in-context points, 14 gradient
  points), not doubling grids. Crossings are located by linear
  interpolation in log2(n) between the two bracketing MEASURED points.
- **MDE on R.** An accuracy error of ±0.05 on a curve with local slope
  s per log2-step displaces the crossing by 0.05/s log2-steps. Over the
  measured region the curves rise ~0.2-0.4 per log2-step, so each
  crossing carries ~0.12-0.25 log2-steps of error, i.e. ~1.1-1.2x per
  curve and **~1.2-1.4x on the ratio**. Verdict thresholds are set at
  5x and 20x with a 4x separation requirement, all far outside that.
- **No R at or below one grid step (2.0x) is reported as a magnitude.**
  That error voided an e011 claim. `analyze.py` sets `R_reportable`
  false below 2.0 and names the offending skills in the verdict notes.
- **Grid-snapping guards**: (1) both axes are finer than doubling; (2)
  no binning, smoothing or interpolation of ACCURACY — only the crossing
  location is interpolated, and every plotted point is a measured point;
  (3) `single_interval` fires if all three capability crossings on
  either curve fall inside ONE measured interval (the e009b failure
  mode) and blocks the X3 reading for that skill.

## Rivals, verdict rules, kill conditions (pre-registered)

Per skill, per seed, per C. Aggregation: median over seeds within a
skill, then median over skills within a family. **Per-seed values are
always reported; there is no single headline number** (decision 0012
item 5 corollary, e011).

- **X1 — large and real.** `X1_SUPPORTED` iff the per-skill median R is
  >= 20 for EVERY non-degenerate skill, natural AND novel.
  `X1_KILLED` iff the maximum per-skill median R is < 5.
- **X2 — small or illusory (the deflationary reading, and the most
  likely one).** `X2_SUPPORTED` iff `R_natural >= 5` AND
  `R_novel <= 5` AND `R_natural / R_novel >= 4`.
- **X3 — regime-dependent.** `X3_SUPPORTED` iff R falls monotonically
  across the three capability targets with a >= 4x drop from C=0.25 to
  C=0.75 in a majority of skills, and no such skill is blocked by
  `single_interval`. X3 can co-fire with X1 or X2.
- `INCONCLUSIVE` if no flag fires, or if more than half of the
  (skill, seed, C) cells are censored.

Censoring is explicit, not hidden: `at_first` (already above C at n=1 →
n_ctx is an upper bound, R a lower bound) and `censored` (never reaches
C within the grid) are carried into `exchange_rates.csv` and into
CAVEATS.md per row.

## Identifiability dry-run (decision 0012 item 1) — RUN, and it CHANGED the design

`python analyze.py --dryrun` builds synthetic curves for each rival and
pushes them through the same verdict code.

| case | synthetic world | intended | got |
|---|---|---|---|
| X1 | R ~ 26 on natural AND novel | `X1_SUPPORTED` | `X1_SUPPORTED` |
| X2 | R ~ 23 natural, ~1.1 novel | `X2_SUPPORTED` | `X2_SUPPORTED` |
| X3 | shallow ICL curve crossed by a steep gradient curve; R 20 → 9 → 4.5 across C | `X3_SUPPORTED` | `X3_SUPPORTED` |
| NULL | the two arms identical, R ~ 1.0 | `X1_KILLED` | `X1_KILLED` |

**The first run FAILED on X3**, and the failure was a real design fault,
not a synthetic-data artifact: on the originally planned DOUBLING grids
a true X3 outcome puts all three capability crossings inside a single
measured interval, which is exactly the e009b bracket failure. Two
changes followed, before any compute was spent: both axes were refined
to √2-ish grids, and `analyze.py` gained the `single_interval` guard
that refuses an X3 reading when the bracket cannot separate the targets.
Record: `artifacts/identifiability_dryrun.json`.

## Throughput and projected cost (measured, not assumed)

Box: arche-retro, 16 cores, 32GB, `~/venv/bin/python`, CPU torch.
**Workers are sized by MEMORY**: measured max RSS per job is 3.5GB and
~28GB is free, so 6 workers at 2 threads each (21GB), not `cpu_count-1`.

- Forward: 684 tokens/s, 19.5 seqs/s at seq_len 36, 16 threads.
- One gradient cell, 50 steps + 3 checkpoints: 50s wall, 3.5GB RSS.
- Projections and the full-grid wall-clock are in
  `artifacts/CAVEATS.md`, written BEFORE anyone commits to the grid.

## Pilot (what the builder runs; the grid is NOT launched)

Per decision 0010, two maximally-different cells end to end, 1 seed. The
maximally-different pair here is `B_first` (natural) vs `N_map` (novel)
— identical in every respect except the one the experiment is about.

1. `selfcheck` (no model, ~3s), then the copy-oracle degeneracy table on
   all five skills. **Both run and passed before any model call.**
2. In-context curves, all five skills, full shot grid, with the
   mismatched-shot and coverage controls at every point.
3. LR sweep, `B_first` and `N_map`, six LRs at N=32. Interior optimum
   required.
4. Gradient curves, `B_first` and `N_map`, full N grid, at each skill's
   own best LR.
5. `analyze.py` on the pilot output → `artifacts/results.csv`,
   `artifacts/exchange_rates.csv`, `artifacts/results.png`,
   `artifacts/verdict.json`, `artifacts/CAVEATS.md`.

**The full grid is NOT launched by the builder** (decision 0010
amendment 1). It would add: the remaining three skills' gradient curves,
3 seeds throughout, and a second base for robustness.

## Weakest-form reporting (decision 0012 item 5)

The builder states every result in the weakest form the data supports
and names the strongest alternative reading as a hypothesis for the
verifier. Ten of ten first-pass corrections in this lab made results
less exciting; the builder may not escalate a claim, the verifier may.

## PILOT RESULT (2026-08-17) — stated in the weakest form the data supports

One seed (seed 0). These ARE the per-seed values; there is no mean hiding
a split, and there is also no second seed, which is the first thing the
verifier should hold against every number here.

### The two curves (every point measured, n=200 queries over 64 distinct held-out words)

| skill | arm | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| B_first (nat) | in-context | 0.490 | 0.880 | 0.875 | 0.910 | 0.935 | 0.940 | 0.955 | 0.940 | 0.920 | 0.945 | 0.960 | 0.950 | – | – |
| B_first (nat) | gradient | 0.000 | 0.630 | 0.330 | 0.590 | 0.910 | 0.905 | 0.960 | 0.940 | 0.965 | 0.990 | 1.000 | 1.000 | 1.000 | 0.990 |
| N_map (nov) | in-context | 0.085 | 0.035 | 0.025 | 0.030 | 0.030 | 0.050 | 0.055 | 0.055 | 0.055 | 0.060 | 0.100 | 0.100 | – | – |
| N_map (nov) | gradient | 0.000 | 0.055 | 0.040 | 0.055 | 0.145 | 0.075 | 0.100 | 0.155 | 0.355 | 0.395 | 0.305 | 0.890 | 0.855 | 0.885 |

Unprompted floor (n=0) is **0.000** for every skill. Mismatched-shot
control is **0.000 at every shot count for every skill**, including where
the model scores 0.95 with correct shots — the in-context arm is reading
the examples, not the format.

In-context curves for the three skills with no gradient arm in this pilot
(so no R): B_upper 0.745 → 0.930 → plateau ~0.945; N_shift2 flat
0.035-0.105 to 64 shots; N_perm **0.000 at every shot count to 64**.

### R, per skill, per seed, per capability target

| skill | family | seed | C | n_ctx | n_grad | R | status |
|---|---|---|---|---|---|---|---|
| B_first | natural | 0 | 0.25 | 1.00 (at_first) | 1.32 | 1.32 | R is a LOWER bound |
| B_first | natural | 0 | 0.50 | 1.02 | 1.73 | 1.70 | below one grid step |
| B_first | natural | 0 | 0.75 | 1.59 | 4.90 | **3.09** | the only reportable magnitude |
| N_map | novel | 0 | 0.25 | >64 (censored) | 19.4 | <0.30 | R is an UPPER bound |
| N_map | novel | 0 | 0.50 | >64 (censored) | 52.8 | <0.83 | R is an UPPER bound |
| N_map | novel | 0 | 0.75 | >64 (censored) | 59.7 | <0.93 | R is an UPPER bound |

**Verdict fired: `X1_KILLED`.** Pre-registered kill condition: maximum
per-skill median R < 5. Measured maximum is 1.70.

### What this does and does not say (weakest form)

1. **On this base, this skill format and one seed, R is not large.** Five
   of the six measured cells sit at or below one grid step (2x) and are
   NOT reportable as magnitudes at all. The single reportable number is
   **R = 3.09 for B_first at C = 0.75**. The pre-registered X1 threshold
   of 20 is not remotely approached.
2. **On the novel skill the exchange rate is BELOW ONE: context is worse
   per example than gradients.** In-context accuracy never exceeds 0.100
   at any shot count to 64, while a perfect learner limited only by which
   letters were demonstrated (`coverage_oracle`) would score 0.905 at 48
   shots. The same model learns the same skill by gradient descent to
   0.890 from 64 examples. So the novel skill is LEARNABLE — this is not
   the UNLEARNABLE case the dry-run tests for — and in-context learning
   simply fails on it.
3. **X2's direction is confirmed; X2 as pre-registered is NOT satisfied.**
   The rule required `R_natural >= 5`, and R_natural is 1.70. The honest
   statement is narrower than X2 and stronger against X1: the in-context
   advantage is not merely smaller on skills with no pretraining analogue,
   it is ABSENT — three of three novel skills stay at or below 0.105
   in-context while the natural pair reach 0.94-0.95. The reading the
   builder is permitted to state is that **the in-context arm on this base
   only works where pretraining plausibly supplied the mapping**, which is
   X2's mechanism, arrived at from the novel arm rather than from a ratio.
4. **X3 is UNREADABLE here and the guard said so.** All three capability
   crossings fall inside one measured grid interval on both skills — the
   e009b failure mode, caught mechanically rather than after the fact. It
   is not fixable by refining the grid: the in-context axis is integer and
   1 is its floor, and B_first goes 0.490 → 0.880 between n=1 and n=2.
5. **Strongest alternative reading, offered to the verifier, NOT claimed
   by the builder:** that R ~ 1-3 is a fact about ICL versus gradients
   generally, and therefore that the consolidation prize is a convenience
   rather than a paradigm. The builder does not claim this, for the reason
   in the next section.

### The weakness that most threatens this pilot's numbers

**The natural skills are too easy for the measurement.** B_first is at
0.490 with ONE shot and 0.880 with two; B_upper is at 0.745 with one. A
ratio whose denominator is pinned at 1-2 examples cannot be large no
matter what the numerator does — R is floor-limited by the in-context
axis, not measured. A design that could observe a large R needs skills
where the in-context arm takes 8-30 shots to reach C, and this pilot has
none. Everything in point 1 above should be read as "R is small ON SKILLS
THIS BASE LEARNS IN ONE OR TWO SHOTS", which is a much weaker claim than
"R is small".

Second weakness: the gradient curves are visibly noisy at one seed
(B_first 0.630 → 0.330 → 0.590 across N = 2, 3, 4; N_map 0.395 → 0.305 →
0.890 across N = 32, 48, 64). The crossing rule takes the FIRST point
above C, so noise shortens n_grad and shrinks R further. Seeds are the
first thing the grid must buy.

### Measured throughput and projected grid cost

Box arche-retro, 16 cores / 32GB, CPU torch, jobs at 3 threads with 5-7
concurrent (memory-sized, not core-sized).

- Forward: **684 tokens/s, 19.5 seqs/s** at seq_len 36, 16 threads.
- One in-context curve (13 points, 2 conditions, n=200): **7600-8900 s**.
- One gradient cell (200 steps + 9 checkpoints): mean **599 s**
  (287 s at N=1, 875 s at N=16), peak RSS 3.5 GB.
- Pilot as run: 5 in-context curves + 12 LR cells + 28 gradient cells
  ≈ **9.4 core-hours**, ~3.5 h wall at 5-7 workers.

Projected full grid (5 skills x 3 seeds, both arms, plus the LR sweep
re-run per skill): 15 in-context curves (~34 h serial) + 210 gradient
cells (~35 h serial) + 30 LR cells (~5 h serial) ≈ **74 core-hours**,
about **11-13 h wall** on this box at 6 workers. That fits on CPU.

**Cheaper and better than buying that grid as designed:** the pilot says
the binding problem is skill difficulty, not seeds. Adding seeds to skills
whose in-context curve saturates at n=1-2 buys precision on a number that
is floor-limited. The verifier should decide whether the grid is worth
running at all before a harder skill suite exists.

### Decision-0012 compliance, as executed

- **Identifiability dry-run**: 6 synthetic worlds, each 200 noisy repeats.
  All pass clean and at >= 0.80 under noise sd 0.05. It found two broken
  verdict rules before launch (min-over-skills X1 fired in 30% of worlds
  where X1 was true; strict-monotone X3 fired in 51%). Measured detection
  limit for X3: a ~6x fall in R; a 4x fall is called ~50% of the time.
  `artifacts/identifiability_dryrun.json`.
- **Known-positive**: the natural in-context arm reached 0.94-0.96 with
  mismatched control at 0.000. The instrument detects in-context learning
  where it exists, so the novel-arm null is about the world, not the
  instrument.
- **Copy-oracle run on the invented skills**: all five skills
  non-degenerate (<= 0.085). The novel arm is not secretly copyable.
- **Interior LR optima** for both skills carrying an R:
  `artifacts/lr_sweep.md`.
- **Weakest-form reporting**: applied above; the escalated reading is
  named and left to the verifier.

### Three additions from the builder's own post-run review

**(a) The metric is NOT blind to the novel answers, and our own data proves
it.** The obvious attack on "in-context scores 0.100 on N_map" is e004g's
B_bang precedent: teacher-forced 0.000 while free generation was 1.000, a
pure metric artifact. That attack fails here. The GRADIENT arm reached
**0.890 on N_map under the same teacher-forced metric, on the same eval
query set**. A metric that can score N_map success at 0.890 is not blind
to N_map, so the in-context 0.100 is a fact about the arm, not the metric.
No such control exists for N_perm (no gradient arm was run for it), which
is precisely why N_perm is excluded from carrying evidence.

**(b) X2 as pre-registered was STRUCTURALLY UNFIREABLE in this pilot, and
that is a design finding.** X2 requires `R_natural >= 5`. With natural
skills that both arms reach in 1-2 examples, R_natural is floor-limited
below 5 no matter which world is true — so the pilot could not have output
`X2_SUPPORTED` in ANY world. That is the same class of failure the
identifiability dry-run exists to catch, and it was found empirically
instead: the dry-run's X2 world assumed gradients need ~30 examples on a
natural skill, and reality was ~2. Consequence: the surviving evidence for
X2's mechanism is the novel-arm contrast (0.100 in-context vs 0.890
gradient at 64 examples), NOT the ratio, and the X2 rule is not live until
a harder natural suite exists.

**(c) Weakest form on the kill.** The dossier's kill condition reads
"R < 5 on EVERY non-degenerate skill". R was measured on **2 of 5**
non-degenerate skills. The exact honest sentence is: *the pre-registered
kill condition fired on both skills measured, at one seed, with R
floor-limited on the natural one.* "X1 is killed" is the escalation, and
it belongs to the verifier.

## VERIFIER RULING (2026-08-17) — X1 KILLED on all five skills; the novel-arm null survives the memorisation attack; the grid is NEEDS-WORK

Independent adversarial review under decision 0010 amendment 1 and
decision 0012. Every number was re-derived by re-running `analyze.py`, or
measured by the verifier on `arche-retro`. New compute: `N_shift2`
gradient arm + LR sweep, `B_upper` gradient arm + LR sweep, a `B_first`
rule-vs-rarity control, and a rule/table split added to every gradient
cell. About 13 core-hours.

### 0. The escalation the builder declined to make

**X1 is KILLED.** I make the call. The grounding is broader than the
builder's "2 of 5 skills" framing allowed, because I measured two more
skills and because three of them never needed a gradient arm at all.

R per skill, seed 0, median over the three capability targets:

| skill | family | R | how bounded |
|---|---|---|---|
| B_first | natural | **1.70** | measured |
| B_upper | natural | **2.01** | measured (new) |
| N_map | novel | **<= 0.83** | upper bound |
| N_shift2 | novel | **<= 0.51** | upper bound (new) |
| N_perm | novel | no crossing | in-context 0.000 at every shot count |

The pre-registered kill is `max per-skill median R < 5`. The measured
maximum is 2.01. It is not close.

**The three novel skills did not need a gradient arm to be decided.** The
in-context arm never reaches even C = 0.25 anywhere on the shot grid
(`N_map` max 0.100, `N_shift2` max 0.105, `N_perm` 0.000), so `n_ctx > 64`;
and `n_grad` cannot exceed the grid maximum of 128. **No world consistent
with the measured in-context data yields R >= 5 on those skills: either
R < 2, or no crossing exists at all.** X1 requires a measured R >= 20 on
every non-degenerate skill. It is contradicted on four by measurement and
on the fifth by a bound.

`B_upper` was the one skill where X1 could still have been true, because
it had an in-context curve but no gradient arm, so its R was not bounded
by the grid. That is why I measured it. Its gradient arm crosses all three
targets below N = 3 (0.000 -> 0.495 -> 0.845 at N = 1, 2, 3), giving
R = 2.01. It does not rescue X1 either.

The known-positive fired (natural in-context 0.94-0.96, mismatched-shot
0.000 at every shot count), so this is not an instrument null. **X1_KILLED,
at one seed, and I sign it at one seed** because no plausible seed effect
closes a 20x gap on curves that are flat at 0.10 across twelve shot counts
and on gradient arms that finish in 2-5 examples.

**X2 splits in two and the halves get different rulings.**

- **X2's MECHANISM — "the in-context advantage exists only where
  pretraining could have supplied the mapping" — SURVIVES**, and after
  section 1.1 it survives in a stronger and more precise form than the
  builder could state. It rests on the novel arm, not on any ratio.
- **X2 AS A PRE-REGISTERED RULE (`R_natural >= 5` AND `R_novel <= 5` AND
  ratio >= 4) is NOT CALLABLE.** Its natural-arm clause was never reached.
  Do not record it as fired and do not record it as failed.
- **X3 — UNREADABLE**, blocked mechanically by `single_interval` on all
  four skills carrying an R.

**What this means for B2 (79%).** e013 measures the CEILING on the
consolidation prize, and on this base the ceiling is low: on skills where
in-context learning works at all, gradients need 1-5 distinct examples to
match it, so there is at most a ~2-3x sample-efficiency prize to bank.
PORTFOLIO.md prices that row "unbounded / paradigm". Nothing measured here
supports that price. **I recommend repricing B2 DOWN, and I deliberately do
not do it myself**: this is a one-seed pilot ruling, not experiment
closure; decision 0010 amendment 2 puts belief repricing with the results
card; and the natural arm has not yet been measured in a regime where a
large R could appear (section 3). What I record is that the ceiling
argument now runs against B2 rather than for it, alongside e004g's k = 4
retention collapse and e015's no-replay masking.

**On the coordinator's cross-experiment question (e014).** e013 is reported
uncoupled from e014 in both directions and neither may cite the other as
corroboration. For the record of what e013 alone says: the rule-shaped
novel probe FAILED in context on a conventionally pretrained 0.5B, and the
gradient route did not induce the rule either. So e013 does not license
"task shape" as the explanation (the rule-shaped probe failed too), and it
does not license "pretraining regime" as the explanation (nothing here
varies the regime, and the failure of rule induction is common to both
routes). What e013 supports is narrower than either: **on this base, a
novel mapping class is not acquired in context at all, whatever its
description length.** e014's mismatched-context lesson was checked here and
e013 is in the clean case — the mismatched-shot control was run at EVERY
shot count, not only at a headline point, and reads 0.000 at all 13 points
on all five skills (`artifacts/results.csv`, column `mismatched`).

### 1. What I attacked

#### 1.1 The memorisation objection to the novel arm — ATTACKED, and it FAILS, but not the way anyone expected

This was the objection that could have voided the whole reading. `N_map`
is a 26-entry lookup table. A context window cannot carry a table the way
weights absorb one, so "in-context learning fails on novel skills" might
have been nothing but "in-context learning cannot memorise tables", which
says little about learning-to-learn and supports X2 in no general form.
The builder named the objection itself and was right that `N_perm` could
not break the tie (tokenization-confounded: 9.67 tokens per line against
8.06 for the single-character skills; e004g's `B_reverse` failed the same
way).

The tie-breaker already existed in the pilot, half-measured: **`N_shift2`**
— "first letter advanced two places in the alphabet". It is the anti-table
skill by construction: one short sentence of description, systematic,
generalises to unseen inputs, TWO demonstrations pin it down exactly,
single-character answer, line length 8.06 +/- 0.24 tokens (identical to
`N_map`), copy oracle 0.070. Its in-context curve was already measured and
is FLAT at 0.035-0.105 from 1 shot to 64. What was missing was the gradient
comparator proving the skill is learnable and the metric can see it.

I ran it: LR sweep (interior optimum 3e-4: 0.155 / 0.155 / 0.115 /
**0.495** / 0.475 / 0.115) plus the full 14-point gradient curve, and I
added a **seen/unseen-first-letter split** to `grad_point` so the gradient
arm's own behaviour is legible. A learner with the RULE answers eval words
whose first letter never appeared in training; a learner with a TABLE
cannot.

| N | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 | 48 | 64 | 96 | 128 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| eval acc | 0.000 | 0.000 | 0.000 | 0.105 | 0.165 | 0.075 | 0.210 | 0.170 | 0.275 | 0.495 | 0.690 | 0.680 | **0.940** | 0.905 |
| **unseen-letter acc** | 0.000 | 0.000 | 0.000 | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | **0.000** | n/a | n/a |
| seen-letter acc | – | 0.000 | 0.000 | 1.000 | 0.611 | 0.135 | 0.321 | 0.245 | 0.353 | 0.553 | 0.738 | 0.701 | 0.940 | 0.905 |
| n_unseen (of 200) | 200 | 179 | 179 | 179 | 146 | 89 | 69 | 61 | 44 | 21 | 13 | 6 | 0 | 0 |

**Unseen-first-letter accuracy is 0.000 at every N from 1 to 64, at every
one of the six learning rates.** At N = 4 the split is 1.000 on
demonstrated letters against 0.000 on undemonstrated ones. Accuracy only
reaches 0.940 at N = 96, which is exactly where the training set finally
covers all 26 letters (n_unseen = 0). The gradient arm's accuracy IS its
letter coverage.

**The rarity control, and it is why I ran it.** Unseen first letters are by
construction the RARE first letters, so a 0.000 there could have been a
frequency artifact. It is not:

| skill | N | seen-letter | **unseen-letter** | n_unseen |
|---|---|---|---|---|
| B_first (a rule the base HAS) | 8 | 0.958 | **0.857** | 105 |
| B_first (a rule the base HAS) | 32 | 0.988 | **1.000** | 32 |
| B_upper (answer not first-letter based) | 32 | 0.915 | **0.833** | 24 |
| N_shift2 (novel rule) | 32 | 0.553 | **0.000** | 21 |

Words with rare first letters are answered at 0.83-1.00 when the model has
the rule. The 0.000 is absence of rule induction, not letter rarity.
(`runs_ctl/`, kept out of `runs/` so it cannot contaminate a curve.)

**Three consequences, and the second is a correction to what I expected.**

1. **The memorisation objection FAILS as an explanation of the novel-arm
   null.** `N_shift2` is not a table for the context window to carry: it is
   a rule two demonstrations specify exactly, and 64 shots is 32 times the
   information needed. In-context accuracy is 0.080 at 64 shots against a
   0.935 coverage oracle. So "in-context learning fails on novel skills"
   is NOT reducible to "in-context learning cannot memorise a lookup
   table". It also fails a rule that costs nothing to carry.
2. **But the comparative claim NARROWS, and must be weakened rather than
   strengthened.** Gradients did not beat context by inducing the rule
   either. On this base neither route does rule induction on a novel
   mapping; the gradient arm wins on novel skills by STORING what it was
   shown. The honest sentence is **"gradients beat context on novel skills
   by storage, not by learning"**, and e013 does not separate rule
   induction from storage in either direction at 0.5B. This is weaker than
   what the builder's draft implied and weaker than what I expected to
   find, and it is what the data support.
3. **X2's mechanism comes out sharper than the ratio version ever was.** On
   this base the context route acquires *nothing* pretraining did not
   already supply — not a table (`N_map` 0.100 at 64 shots against a 0.905
   coverage oracle), not a rule (`N_shift2` 0.080), not a positional
   permutation (`N_perm` 0.000 everywhere) — while it reaches 0.94-0.96 on
   both skills whose mapping pretraining plausibly did supply. That is X2's
   mechanism, from the novel arm, with no ratio involved and no reliance on
   the floor-limited natural R.

#### 1.2 Decision 0012 clause 7(a) — APPLIES, was NOT applied, and it bites

Clause 7(a) requires any ratio-of-improvements metric to be checked against
a matched-CAPABILITY control. R is a ratio of example COUNTS, not of gains,
so it is not literally the e014 case and the builder did not apply the
clause. The failure mode transfers anyway, and the data demonstrate it:

| skill | base 1-shot in-context acc | R |
|---|---|---|
| B_first | 0.490 | 1.70 |
| B_upper | 0.745 | 2.01 |
| N_map | 0.085 | <= 0.83 |
| N_shift2 | 0.045 | <= 0.51 |

R is monotone in the base's 1-shot competence across all four skills, which
share format, marker syntax, answer length, tokenization and eval set. R
also rises with the capability target within a skill (1.32 -> 1.70 -> 3.09
on `B_first`). **R is not a constant of the two learning ROUTES; it is a
function of how much of the skill the base already had.** Same structural
defect clause 7(a) names, arriving through a ratio of counts.

For X2 the confound IS the hypothesis, so a natural/novel contrast driven
by base competence is the finding rather than an artifact. Two things
follow anyway:

1. **R may not be reported as a route property.** Every R must be reported
   conditioned on the base's 1-shot accuracy on that skill, and no R may be
   aggregated across skills of differing base competence into a headline.
2. **The grid owes a matched-capability control** — skills spanning
   intermediate base competence, section 3c. Clause 7(a) is explicit that
   headroom normalisation is not a substitute.

There is a second, deeper way R is not a learning-speed ratio, and it
explains the whole result. The gradient arm gets 200 optimizer steps over
its N distinct examples: at N = 2 that is 200 passes over 2 examples. So
`n_grad` is not "how much data gradient descent needs", it is **how many
distinct examples are needed to IDENTIFY the rule when repetition is
free**. `n_ctx` is the same quantity for the context route. Both routes are
bounded below by the same thing, so R is pushed toward 1 wherever both
routes can use the information, and departs from 1 mainly where one route
cannot use it at all (R < 1 on the novel skills). This is the right
definition for pricing consolidation, which is about distinct examples
banked into weights rather than optimizer steps, and the DESIGN is right to
have chosen it. But it means the folk figure ("5 in context versus hundreds
by fine-tuning") is NOT what e013 measures, and R ~ 1-3 is not evidence
against that folk figure. Both statements belong in the results card.

#### 1.3 The builder's two central self-criticisms — one CORRECTED, one narrowed

The builder wrote that R is "floor-limited" by a denominator pinned at 1-2
examples, and that X2 was "STRUCTURALLY UNFIREABLE in ANY world". **Both
are wrong as written and I decline to certify them.**

**On "floor-limited":** with `n_ctx = 1`, `R = n_grad`, and the gradient
grid runs to 128. A denominator of 1 does not cap R, it MAXIMISES it. The
reason R_natural is small is that **the numerator is small too**: LoRA
reached C = 0.50 on `B_first` from 1.73 distinct examples and C = 0.75 from
4.90; on `B_upper`, 2.01 and 2.69. Both routes are fast on these skills.
That is a measurement, not an artifact.

**On "structurally unfireable":** X2 needs the per-skill median R >= 5,
i.e. two of three cells >= 5, i.e. `n_grad` >= 5.00 / 5.09 / 7.93 at
C = 0.25 / 0.50 / 0.75 on `B_first`. All three sit inside the gradient grid
and none is near its edge. A world in which LoRA needed ~8 distinct
examples to reach 0.75 fires X2; the measured value was 4.90, missing by a
factor of 1.6. X2 was **empirically unfired, not structurally unfireable**.
The distinction matters: the builder's version indicts the DESIGN (the
e004g ceiling-censoring failure mode), the correct version indicts the
SKILL CHOICE, which is fixable and is what section 3 fixes.

**What IS true** is weaker and more useful: skills a pretrained base solves
in one or two in-context shots are also skills LoRA solves in a handful of
distinct examples, so the natural arm compresses R toward 1 empirically.
The design is sound; the suite is in the wrong regime.

**The kill is robust.** The one-seed noise runs TOWARD the kill: the
crossing rule takes the FIRST measured point above C, so a noisy upward
excursion shortens `n_grad` and SHRINKS R. I tested the size rather than
argued it: with every measured `n_grad` DOUBLED, `B_first`'s median R is
3.4 and `B_upper`'s is 4.0, both still under the 5.0 line. For `B_first`'s
median to reach 5 the N = 2 point would have to fall from 0.63 below 0.5
AND the N = 4-6 points fall ~0.2, each 3-7 SE (SE ~0.06 on 64 distinct eval
words), in the same direction, at once.

#### 1.4 Three pipeline defects found and FIXED (all committed)

**(a) The LR sweep was being read as the gradient curve.** `load()` globbed
every json in `runs/`. LR cells share the gradient record shape and all sit
at N = 32, so a sweep left in `runs/` is read as the whole curve and
`n_grad` collapses to 32 for that skill. This is not hypothetical: with
`B_upper`'s sweep in place and its curve not yet run, the pipeline emitted
**`X2_SUPPORTED` with n_grad = 32.0 at all three capability targets** — a
false positive for the deflationary hypothesis, produced by a filename.
Fixed: `load()` excludes `lr_*`, the 24 stray records were moved to
`lr_runs/`, and a `< 6 distinct N` guard refuses to read anything that is
not a curve. **This would have fired in the grid, where every skill gets
its own sweep.**

**(b) The in-context arm had NO seed dimension.** `icl_curve` recorded
`seed` but never used it: every query word and shot draw came from
`EVAL_SEED + skill`. Three seeds of the in-context arm would have produced
**three byte-identical curves**, and 10 of the grid's 15 in-context curves —
roughly 23 of 74 core-hours — would have bought a false impression of
replication. Same class as e011's hidden seed split, caught before the
spend. Fixed: both arms now take the same replicate offset, so they stay
paired within a seed, and seed 0 reproduces the pilot exactly.

**(c) Censoring was counted, bounds were not.** DESIGN registers
INCONCLUSIVE when "more than half the cells are censored" and lists
`at_first` and `censored` as the censoring notations; the code counted only
cells with no R at all (`censored_frac` 0.0 while 9 of 12 cells carry a
bound). Implementing the registered rule literally would flip the pilot to
INCONCLUSIVE, which is wrong the other way — a bound already on the
decisive side of a threshold is informative. Fixed with `bounded_frac` plus
`bound_stress()`: every genuinely unbounded cell is pushed to its worst
case for the flag under test and the flag must survive.

One judgement inside that fix, stated openly because it is the kind of
thing a verifier can quietly get wrong in its own favour. `at_first` always
resolves to n_ctx = 1, since 1 is the smallest point on the shot axis. My
first version treated it as a lower bound (true R could be arbitrarily
large) and the kill then FAILED its own stress on `B_upper`, whose two
lower targets are both `at_first`. I judged that over-conservative and
exempted it: shots are integer, you cannot show a model two-thirds of an
example, so "already correct at one shot" is the exact answer to "how many
in-context examples", not an upper bound with room underneath. Only
genuinely off-grid `censored` endpoints are treated as bounds. With that
exemption the kill survives its worst case at 1.70 / 2.01 / 0.83 / 0.51.
A reader who rejects the exemption should read the kill as resting on the
two `ok` cells per natural skill plus the grid bound on all three novel
skills, which is still decisive.

#### 1.5 What I checked that SURVIVED unchanged

A verifier report listing only defects misrepresents the experiment. All of
the following were re-derived and hold:

- **Mismatched-shot control at EVERY shot count**: 0.000 at all 13 points
  on all five skills, including where the model scores 0.95 with correct
  shots.
- **Unprompted floor 0.000** on every skill; both arms share it.
- **Copy oracle <= 0.085 on all five skills**, invented ones included.
- **All four LR optima interior** (`artifacts/lr_sweep.md`), including the
  two I added. The gradient arm is tuned, so R is not inflated by a
  handicapped gradient arm — the direction of the exciting answer.
- **The teacher-forced metric is not blind to novel answers**: the gradient
  arm scores 0.890 on `N_map` and 0.940 on `N_shift2` under the same metric
  on the same query set, so the in-context 0.100 / 0.080 is a fact about
  the arm, not a `B_bang`-style metric artifact.
- **Nested gradient subsets**, so sample count is not confounded with
  sample luck.
- **Early-stop tie toward the LATEST checkpoint** and 200 steps regardless
  of N: both asymmetries set in the DEFLATIONARY direction.
- **The identifiability dry-run is real work**: 6 worlds x 200 noisy
  repeats, still passing after my changes. It caught a min-over-skills X1
  rule firing in 30% of worlds where X1 was true and a strict-monotone X3
  rule firing in 51%, and forced both axes from doubling to sqrt(2) grids
  BEFORE compute.
- **The `single_interval` guard fired correctly** and blocked X3 rather
  than letting a bracket artifact produce a reading.

#### 1.6 What e004g and e015 do to the reading

They do not change R and R does not depend on them; they change what a low
R implies. e004g's k = 4 collapse (mean retention 0.779, min 0.570) says
the channel loses prior skills as writes accumulate. e015 says that without
replay the loss is MASKING, not erasure — prior skills at 0.000 while still
ranked far better than the frozen base, because the newest rule takes every
cue. e013 says the thing the channel would bank is worth at most ~2-3x in
distinct examples where the context route works, and nothing where it does
not. **A small prize, a channel that degrades with use, and a routing
failure that hides what it wrote.** Each is one seed or one pilot; together
they are why B2 should be repriced by whoever writes the results card, and
why I would not spend GPU on the consolidation line before the section 3
screen has run.

### 2. Still running / not done

- **`N_map` in-context reseed (seed 1, shots 8-64)** was launched and is
  still running at the time of writing. It is a variance check on the flat
  novel curve. **No conclusion here depends on it**: the novel-arm reading
  rests on three different novel mappings failing flat, on the coverage
  oracle rising to 0.96 while accuracy stays at 0.10, and on the
  seen/unseen split. If it comes back non-flat, that is a finding and this
  ruling should be revisited.
  Partial at hand-off (points completed so far, seed 1 vs seed 0):
  n=8 acc 0.030 (s0: 0.050), n=16 acc 0.035 (s0: 0.055); mismatched 0.000 at
  both. Shots 32/48/64 still to land in
  `runs/icl_N_map_s1.json` on arche-retro [address removed]. Read as: flat,
  consistent with seed 0. Anything above ~0.20 at 64 shots would contradict
  the novel-arm reading and should reopen this ruling; anything below is
  confirmatory and changes nothing.
- **`N_perm` remains excluded** and no gradient arm was run for it. Its
  0.000 may be tokenization. Do not read it as evidence.
- **Second base, second seed set**: not done, and not this pilot's job.

### 3. The grid: NEEDS-WORK. Do not buy 74 core-hours of this suite.

I agree with the builder's own recommendation and rule on it explicitly:
**the grid as designed should not be bought.** Adding seeds to skills the
base solves in one or two shots buys precision on a number whose small
value is already settled, in a regime the suite never leaves. Separately,
roughly a third of the designed cost was seed replication of an arm that
had no seed dimension (section 1.4b); that is fixed, but the grid must be
re-costed rather than re-authorised at 74 core-hours.

#### 3a. The band a natural skill must land in, as numbers

Admissible to the R measurement only if, on Qwen2.5-0.5B under this
one-line marker format:

- **1-shot in-context accuracy <= 0.20** (so `n_ctx` is not pinned at 1 and
  `at_first` never fires at C = 0.25);
- **64-shot in-context accuracy >= 0.85** (all three targets reachable, arm
  not censored);
- **C = 0.75 crossed between 8 and 30 shots** (so `n_ctx` sits in the
  interior of a sqrt(2) grid and the three targets land in DIFFERENT
  measured intervals — this is what clears the `single_interval` block that
  made X3 unreadable);
- copy oracle <= 0.20 and mismatched-shot 0.000 at every shot count;
- line token length within +/- 0.5 of its matched partner's (the
  `N_perm` / `B_reverse` tokenization trap).

In that band `R_natural >= 5` becomes fireable at `n_grad >= 40-150`, well
inside the existing gradient grid, and X2's rule goes live. In the current
band it cannot.

#### 3b. How to find such skills — SCREEN, do not guess

Nobody can predict which rules land in that band at 0.5B, and guessing is
how the current suite ended up outside it. The grid's first stage is a
cheap screen, not a commitment:

- Candidates, all single-character answers so tokenization matches
  `B_first` and `N_map` exactly, all chosen because a SINGLE demonstration
  leaves several simple hypotheses alive (the mechanism that raises the
  shot requirement): `B_third` (3rd letter), `B_last` (last letter),
  `B_vowel` (first vowel — coincides with first-letter on vowel-initial
  words), `B_alpha` (alphabetically smallest letter — often coincides with
  first-letter), `B_dup` (the doubled letter, else the last letter — a
  two-clause rule), `B_vcount` (letter at index = number of vowels).
- Screen each on a truncated shot grid {1, 2, 4, 8, 16, 32, 64} with the
  mismatched and copy-oracle controls: ~1.2 core-hours per curve,
  ~7 core-hours for all six, ~1.5 h wall on `arche-retro`.
- Keep whatever lands in the 3a band. Expect 2-4 to survive. **If ZERO
  survive, that is the answer**: this base has no intermediate in-context
  regime on single-character skills, R cannot be measured on it, and the
  experiment needs a bigger base rather than more seeds. Report that; do
  not widen the band to make something pass.

#### 3c. The matched-capability control clause 7(a) demands

Retain at least three skills spanning 1-shot base accuracy in [0.05, 0.40]
and report R against 1-shot accuracy as a covariate. If R tracks 1-shot
accuracy across skills more tightly than it tracks the natural/novel label,
R is measuring base competence and must be withdrawn as a gate — exactly as
e014's transfer ratio was. Section 1.2's four-point table already shows R
monotone in 1-shot accuracy, so this is a live risk, not a formality. The
test costs nothing once the screen has run.

#### 3d. What the grid should then buy

Both arms, 3 seeds (seeds now do something on both arms), on the surviving
natural skills plus `N_map` and `N_shift2`; `N_perm` excluded until its
tokenization is fixed; LR sweep per skill with the interior requirement
unchanged; the rule/table split reported on every gradient cell. Cost
re-derived after the screen.

### 4. Ops

- **NSG.** `arche-lab-nsg/ssh-from-home` carried two source prefixes:
  `[address removed]` (the operator's current egress, verified live during this
  review) and `[address removed]`, added by the builder when its own IP rotated
  mid-run. Only one is current; the stale entry is removed at the end of
  this review, restoring `ops/repin.sh`'s single-IP invariant.
- **Records pulled into the repo before anything was written** (the e011
  rule): `runs/`, `runs_ctl/`, `lr_runs/`, `logs/`.
- **Worker sizing.** Measured peak RSS is 3.1-3.3 GB per gradient job, so
  6 workers at 2 threads (~19 GB of 29 GB free) — memory sized, not core
  sized, per `ops/README.md`.
- **Self-inflicted note for the record**: an over-broad `pkill -f "icl
  --skills N_map"` matched the job queue's own `sh -c` argv and killed the
  dispatcher, not just the target. The in-flight batch survived and the
  remainder was relaunched; no cell was lost. Kill by PID, or match a
  pattern that cannot appear in a queue entry.

### 5. Verdict

- **X1 — KILLED.** Escalated by the verifier per decision 0012 item 5.
  Measured R: B_first 1.70, B_upper 2.01, N_map <= 0.83, N_shift2 <= 0.51;
  N_perm carries no crossing. Kill survives its bound stress. The
  known-positive fired, so this is not an instrument null.
- **X2's MECHANISM — SUPPORTED**, in the sharper form the novel arm
  licenses: on this base the context route acquires nothing pretraining did
  not already supply, neither a table nor a rule.
- **X2 AS A PRE-REGISTERED RULE — NOT CALLABLE.** Its `R_natural >= 5`
  clause was never reached because the natural suite is out of regime.
- **X3 — UNREADABLE**, blocked mechanically on all four skills.
- **The novel-arm memorisation objection — DEFEATED**, by a rule-shaped
  novel skill with a gradient comparator and a rarity control. The
  comparative claim narrows to "gradients beat context on novel skills by
  STORAGE, not by learning".

**NEEDS-WORK.** The 74-core-hour grid is NOT cleared to launch as designed.
CLEARED TO LAUNCH: the section 3b screen (~7 core-hours) only. The full
grid is re-costed and re-authorised after the screen, on skills that land
in the 3a band, with the seed fix, the LR-file fix and the clause-7(a)
covariate in place.
