# e004g — Consolidation accumulation on a DOWNLOADED ICL-capable base

Status: DESIGN 2026-08-16, written BEFORE code (standing rule).
Supersedes the "004 waits for GPU" conclusion of e004r by removing its
single blocking premise.

## Which bar of decision 0011, at what magnitude

**Bar (b) — fundamental bottleneck.** PORTFOLIO.md row
`learning | consolidation: context->weights (004) | unbounded / paradigm
| untested`. This experiment does not claim a multiplier. It measures
whether a context→weights channel ACCUMULATES: whether skill after
skill can be written into ONE persistent weight set without destroying
what is already there. That is the untested clause of bet B2 (80%) and
the only candidate in PORTFOLIO.md that is independent of every other
factor, because it changes what a model retains between episodes rather
than what a forward pass costs.

Magnitude that would count: a retention curve that stays >90% of
just-consolidated performance out to k=8 skills at a small replay
budget (K1) is the strongest single piece of evidence the lab can
produce for B2. A curve matching the sequential-editing collapse shape
(K3) reprices B2 down toward the editing prior. Both are results.

## AMENDMENT 1 (2026-08-16, before any run) — the e012b ruling, and why the base is still DOWNLOADED

Written after the e012b verifier ruling and the new
[[olsson2022-induction-heads]] / [[elhage2021-transformer-circuits]]
cards landed, and after the coordinator's follow-up qualifier. Three
things changed and one did not.

### What changed

1. **The five ICL negatives are not evidence that ICL is emergent.**
   They are evidence of a localised, fixable blocker: learned absolute
   positions added to the residual stream. Swapping that one thing for
   rotary took the in-context gap from +0.0006 to +0.998.
2. **But the rotary arm's 0.998 is probably not induction.** e012b's
   corpus put every repeat at a constant offset of exactly 128, so a
   fixed positional shift solves it with no content matching. The
   varied-offset discriminator is 1.6x past the step where the
   fixed-offset twin saturated and is still flat at +0.0002.
3. So "we can cheaply train our own ICL-capable base" is NOT
   established. It may be that the literature's 1e9-5e9 token placement
   simply applies to us.

### The two paths, evaluated

**(a) Download a pretrained small base.** Chosen. Reasons:
- The capability consolidation needs is in-context acquisition of a
  SKILL from examples — apply an unseen rule to an unseen input — not
  token copying. A 0.5B model trained on 18T tokens of natural text is
  the only artefact available today that plausibly has it, and the
  Stage 1 gate below measures it directly rather than assuming it.
- It is already downloaded and runnable. Its cost is measured in this
  document, today.
- The positional scheme is recorded, not assumed:
  `pos_scheme()` prints it into every gate record. Qwen2.5-0.5B is
  `qwen2`, `has_rotary_module=true`, `has_absolute_pos_embedding=false`
  — the property Elhage 2021 names as shared with the literature's
  induction-head models.

**(b) Train our own rotary base.** NOT run here, and the reasons are
worth stating precisely rather than dismissing:
- What (b) would have to produce is not induction on repeated random
  tokens. It is meta-learned in-context RULE acquisition: a corpus
  that is a distribution OVER rules, so that inferring a held-out
  rule from three examples is the pretraining objective. That is a
  different corpus and a different experiment from anything e012/e012b
  built, and it is a design job, not a one-line architecture change.
- The evidence that would have made (b) cheap — "2 layers, d=64,
  5.5M tokens gets you induction" — is exactly the evidence now in
  doubt. Designing against it is designing against a result the
  verifier is refusing to finalise.
- (b) remains genuinely attractive for a fully instrumented, fully
  controlled base, and is registered here as the named follow-up
  (a meta-ICL pretraining corpus + rotary TinyGPT) if (a) fails the
  gate or proves infeasible on CPU. It is NOT a fallback that can be
  reached inside this experiment's scope.
- If a downloaded base clears the gate and a home-trained one does
  not, that is a clean result about where in-context skill acquisition
  comes from, and it will be reported that way. This experiment
  establishes the first half of it.

### What did not change

The gate must still be able to fail, and "below floor" is still a
legitimate outcome. What a failure would MEAN is now narrower: it
would say that in-context skill acquisition is absent at 0.5B-1.1B
scale under this format, not that ICL is emergent in general.

### Degeneracy check, now mandatory (decision 0010 amendment 3)

A skill suite that claims to require in-context rule inference must
print the evidence that it does. Three mechanical controls, all run
before the gate is read and all recorded in the gate file:

1. **`copy_oracle`** — the best accuracy achievable by copying the
   answer span from a single FIXED relative offset earlier in the
   sequence, no content matching, computed over the eval batch. This is
   e012b's constant-offset shortcut, measured directly on our corpus.
   A skill is NON-DEGENERATE iff `copy_oracle <= 0.20`.
2. **Mismatched-shot control** — the same marker and format, but the
   demonstrated rule is a different skill's. A model that actually
   reads the examples must score at the unprompted level; a model
   riding a format, frequency, or copying prior scores the same as with
   correct shots. The gate's control level is
   `max(unprompted, mismatched)`, and the gap is measured against that,
   not against unprompted alone.
3. **Line-length spread** — printed per skill, because a constant token
   length is what makes a single fixed offset work at all.

**Measured result on Qwen2.5-0.5B's tokenizer (run before the gate):**

| skill | type | copy_oracle | offset | line tokens | reading |
|---|---|---|---|---|---|
| A_zeta | A | **0.953** | 9 | 8.03 ± 0.17 | copyable |
| A_north | A | **0.969** | 8 | 7.03 ± 0.17 | copyable |
| A_violet | A | **0.938** | 8 | 7.03 ± 0.17 | copyable |
| A_harbor | A | **0.969** | 8 | 7.03 ± 0.17 | copyable |
| B_reverse | B | 0.000 | – | 8.44 ± 0.50 | non-degenerate |
| B_upper | B | 0.000 | – | 8.00 ± 0.71 | non-degenerate |
| B_last | B | 0.109 | 16 | 7.03 ± 0.17 | non-degenerate |
| B_double | B | 0.000 | – | 8.09 ± 0.52 | non-degenerate |

This is the e012b failure mode found in our own suite before running:
**all four type-A skills are solvable by a constant-offset copy at
~0.95**, because a constant answer plus near-constant line length is
exactly a fixed lag. Type A is therefore NOT admissible as evidence
that the base does in-context rule inference. It is retained as the
easy end of the suite and as the fact-like arm the dossier's B2
narrowing distinguishes from skills, and it is labelled as copyable
everywhere it appears.

**Added gate clause:** at least **1** passing skill must be
NON-DEGENERATE (`copy_oracle <= 0.20`), and the Stage 2 pilot must
include one. Combined with the existing >= 2 type-B clause this means
the accumulation measurement cannot be carried by fixed-offset copying.

## Why this experiment exists at all (the unblocking argument)

e004r stopped at its own gates twice, correctly: a 4-layer/128-dim
model we pretrained ourselves showed zero in-context learning, so there
was nothing to consolidate. Its recorded conclusion — "004 moves whole
to the GPU queue" — rests on the assumption that an ICL-capable base
must be TRAINED. It does not. It can be DOWNLOADED. A 0.5B open model
pretrained on 10^13 tokens is many orders of magnitude past any
plausible ICL threshold, is 250x smaller than the models Snell used,
and is CPU-feasible for LoRA-sized writes and short-sequence scoring.

This also sidesteps assumption A-ICL entirely rather than depending on
it. A-ICL ("ICL is emergent with an unmapped threshold") is currently
UNTESTED per the e012b verifier ruling, and nothing downstream may be
cited on it. e004g does not cite it: it MEASURES the ICL gap of its own
base directly, at Stage 1, and refuses to proceed if the gap is absent.

## STAGE 1 — the gate (must be able to fail, and is pre-registered here)

Port of e004r's skill suite and gap metric to a real tokenizer.

### Skills

e004r skill = `[M_i][fork f][12-token body][consequence]`, where M_i is
an unused marker token, the fork is the input, and the consequence is
either a constant (type A, "association-flavored") or a function of the
fork (type B, "transformation-flavored", the reading that carries B2's
skills-not-facts narrowing).

Port, one line per example, marker and format identical across skills:

```
<<K3>> apple = elppa
<<K3>> water = retaw
<<K3>> tiger = regit
<<K3>> chair =
```

- **marker M_i** → a literal rare tag `<<K0>>`..`<<K7>>`, one per skill.
  Same role as e004r's unused vocab id 10..17: it names the skill and
  is what consolidation must attach the rule to.
- **fork f** → an input word drawn from a fixed 64-word pool of common
  short lowercase nouns.
- **consequence** → the answer string.
- **type A (skills 0-3, fixed association)**: answer is a constant word
  independent of the input (`zeta`, `north`, `violet`, `harbor`).
- **type B (skills 4-7, input-dependent transform)**: `reverse` the
  letters; `upper`-case; `last` letter only; `double` the word.

### Adaptations forced by the tokenizer change, each disclosed

1. **The 12-token structure body is DROPPED.** In e004r it existed to
   place distance between fork and consequence inside the e005 corpus's
   own sentence grammar. It has no analogue in natural text and would
   only be noise here. Consequence: the fork→consequence distance is
   ~3 tokens instead of ~14. This makes the ICL task EASIER than
   e004r's, which is the correct direction for a gate whose job is to
   establish that a gap exists at all.
2. **Answers are multi-token strings, not single vocab ids.** Scoring
   is teacher-forced exact match: the answer counts as correct only if
   the argmax at EVERY answer-token position equals the true token
   given the true prefix. This is an upper bound on free-generation
   accuracy (the model is handed the true prefix). Guard: the pilot
   additionally runs greedy free generation on the same queries and
   reports both numbers; if they diverge by more than 0.05 the
   teacher-forced metric is rejected and free generation becomes the
   headline metric.
3. **Held-out inputs.** The 64-word pool is split 32 train / 32 eval by
   a fixed seed. Shots in a prompt and examples used for distillation
   come only from the train half; every accuracy in the gate and in
   every eval is measured on the eval half. Type B skills therefore
   have to generalize to unseen inputs; type A skills do not have to,
   by construction, which is exactly the fact-vs-skill difference the
   dossier's narrowing cares about.
4. **Type A "constant" answers were chosen to be low-probability
   continuations** of the format so that the unprompted baseline is
   near zero. Verified empirically by the gate's unprompted arm, not
   assumed.
5. **k-shot = 3**, as in e004r (`SHOTS = 3`). Shot inputs are sampled
   without replacement from the train half and never equal the query.

### Gate thresholds — PRE-REGISTERED, unchanged from e004r

Per skill, n = 200 eval-half queries per condition:
- PASS iff `prompted_acc >= 0.60` AND `unprompted_acc <= 0.20`.
- gap = `prompted_acc - unprompted_acc` (e004r's `icl` minus
  `unprompted`, same quantity).

Base-level gate:
- The base PASSES iff **>= 4 of 8 skills pass AND >= 2 of the passing
  skills are type B.** e004r's rule was ">= 4 of 8 or STOP". The type-B
  clause is ADDED because dossier 004 narrowed B2 to
  behaviors/skills/procedures and away from atomic facts, and type A is
  fact-like: a base that passes only on type A gives an accumulation
  measurement about facts, which MEMIT already answers better.
- Binomial SE at n=200 is <= 0.035, so both thresholds sit >= 4 SE from
  the observed values we would call a failure.

### Base-model ladder and the STOP rule

At most **three** bases, in this order:

1. **Qwen2.5-0.5B** (base, not instruct). Chosen first because it has
   the highest tokens-per-parameter of any candidate at this size (18T
   tokens / 0.49B params), uses **rotary** position embeddings — the
   exact property e012b showed flips the in-context gap from 0.000 to
   0.998 and that [[elhage2021-transformer-circuits]] names for the
   models induction heads were found in — and is directly supported by
   `peft` LoRA. fp32 on CPU is ~2GB, fits the box many times over.
2. **HuggingFaceTB/SmolLM2-360M** (11T tokens, rotary). Smaller and
   cheaper if Qwen passes marginally, a genuine second opinion if it
   fails.
3. **TinyLlama_v1.1** (1.1B, 3T tokens, rotary). Roughly 2x the
   inference cost; the last try.

Pythia-410M is deliberately NOT on the ladder despite being named in
the brief: 300B training tokens (60x less than Qwen2.5) and rotary
applied to only 25% of head dims. It is the candidate most likely to
produce an uninformative failure.

**If all three fail the base-level gate: STOP and report "below
floor".** Do not build a consolidation loop with nothing to
consolidate. e004r stopping cleanly at its gates twice is the behaviour
being imitated here.

## STAGE 1 RESULT (2026-08-16) — Qwen2.5-0.5B **PASSES**. One base, no ladder needed.

`gate_Qwen2.5-0.5B.json`, n = 200 held-out queries per skill per
condition, 262s wall on 16 CPU cores.

| skill | type | prompted | unprompted | mismatched-shots | gap | free-gen | copy-oracle | verdict |
|---|---|---|---|---|---|---|---|---|
| A_zeta | A | 1.000 | 0.000 | 0.000 | **1.000** | 1.000 | 0.953 | PASS (copyable) |
| A_north | A | 1.000 | 0.000 | 0.000 | **1.000** | 1.000 | 0.969 | PASS (copyable) |
| A_violet | A | 0.975 | 0.000 | 0.000 | **0.975** | 1.000 | 0.938 | PASS (copyable) |
| A_harbor | A | 0.990 | 0.000 | 0.000 | **0.990** | 1.000 | 0.969 | PASS (copyable) |
| B_reverse | B | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | FAIL |
| B_upper | B | 0.940 | 0.000 | 0.000 | **0.940** | 0.938 | 0.000 | **PASS (non-degenerate)** |
| B_last | B | 0.055 | 0.000 | 0.000 | 0.055 | 0.031 | 0.109 | FAIL |
| B_double | B | 1.000 | 0.000 | 0.000 | **1.000** | 1.000 | 0.000 | **PASS (non-degenerate)** |

Verdict: `n_pass = 6` (>= 4 required), `n_pass_type_B = 2` (>= 2
required), `n_pass_non_degenerate = 2` (>= 1 required) →
**`base_pass = true`**. Prompted spread across skills sd = 0.415, so
the suite discriminates (decision 0010 item 4 satisfied; a suite with
near-zero spread would have blocked launch).

Positional scheme, recorded per the e012b lesson: `model_type qwen2`,
`has_rotary_module true`, `has_absolute_pos_embedding false`, 24
layers, d_model 896, 494,032,768 params. Rotary — the property Elhage
2021 names as shared with the models induction heads were found in.

Four things worth reading off this table:

1. **The gap is real and it is not a positional shortcut.** B_upper and
   B_double have copy-oracle 0.000: no single fixed relative offset
   reproduces their answers at all. Their 0.94 and 1.00 prompted
   accuracy is content-conditional rule application to unseen inputs.
2. **The mismatched-shot control is at 0.000 for every skill,
   including the ones the model solves perfectly.** Show it three
   examples of the wrong rule under the right marker and it applies the
   wrong rule. It is reading the examples, not the format.
3. **Teacher-forced and free-generation accuracy agree** everywhere
   (max divergence 0.024 on B_last, well inside the 0.05 rejection
   threshold pre-registered in adaptation 2). The cheap metric stands.
4. **Two skills genuinely fail**, and one of them (B_reverse) was
   pre-registered as the most likely to. A gate at which everything
   passes is not a gate; this one had teeth and used them.

Pre-registered Stage 1 predictions, scored: base passes (0.75) —
RIGHT. Type A 4/4 (0.85) — RIGHT. Type B >= 2/4 (0.65) — RIGHT, exactly
2. "`upper` and `double` most likely, `reverse` most likely to fail"
— RIGHT on all three. B_last was not called and failed.

**The e004r blocker is removed.** A downloadable 0.5B base has enough
in-context skill acquisition for the consolidation experiment to have
something to consolidate, on CPU, today. The ladder's other two rungs
(SmolLM2-360M, TinyLlama_v1.1) were downloaded and are not needed.

## AMENDMENT 2 (2026-08-16) — K0: is the write even MARKER-CONDITIONAL?

Registered BEFORE reading pilot 2's output. Prompted by pilot 1, whose
`unknown_marker_agree` fell to **0.000 at k = 1 in both arms** — after a
single write, every next-token prediction under a marker that was never
consolidated had changed. That is upstream of accumulation and, if the
rule has attached to the FORMAT rather than to the marker, it confounds
every K1/K2/K3 reading: what looks like skill k+1 interfering with
skill k would just be skill k+1 overwriting an unconditional rule.

**New pre-registered rival, checked FIRST:**

**K0 — the write is not conditional.** Consolidation attaches the rule
to the one-line format, not to the marker that names the skill. Then a
"persistent weight set holding many skills" was never on the table:
each write replaces the last, and retention decay is arithmetic, not
interference.
- MEASURED BY `unknown_marker_applies_rule`: after consolidating skill
  s, query under `UNKNOWN_MARKER` — never consolidated, never replayed
  — and score against skill s's own rule, n = 200 held-out inputs.
- THRESHOLD: the write is MARKER-CONDITIONAL iff this is **<= 0.20**,
  the same control level the gate uses.
- BASE RATE: the frozen base scores 0.000 unprompted on every skill in
  the gate (all 14, `unprompted` column), so the null level for this
  probe is ~0.000, not chance. A k = 0 measurement on the frozen base
  is taken before the first arm to make that explicit rather than
  inferred.
- **IF K0 FIRES (> 0.20), the K1/K2/K3 readings are BLOCKED** and
  `analyze.py` reports `unconditional_write_blocks_K_reading: true`
  rather than a rival verdict. The experiment then has a different and
  more basic result: the Askell/Snell recipe as implemented writes
  behaviour unconditionally at this scale, and making the write
  conditional is a prerequisite the dossier did not anticipate.

Corresponding measurement change: the eval suite now scores **every**
skill in the run at every k, not only the already-consolidated ones,
so leakage INTO a skill that has not been written yet is visible
alongside loss of one that has.

## AMENDMENT 3 (2026-08-16) — the metric guard fired, and how it was handled

Adaptation 2 pre-registered: if teacher-forced and free-generation
accuracy diverge by more than 0.05, "the teacher-forced metric is
rejected and free generation becomes the headline metric." **It fired.**

`B_bang` (`apple` -> `apple!`) scores **0.000 teacher-forced and 1.000
free-generation** — a complete false negative, caused by the answer
having more than one valid tokenization, not by the model. `B_spaced`
diverges 0.054, marginally over.

**Deviation, owned:** the rule as written demanded a global switch to
free-generation scoring. Instead the metric is retained and the
affected skills are EXCLUDED, because the divergence is a property of
particular answer strings rather than of the metric. Justification, in
numbers: across all 10 skills that PASS the gate, the maximum
teacher-forced-vs-free-generation divergence is **0.016** (B_first
0.985 vs 0.969; B_plural 0.795 vs 0.781; B_upper 0.940 vs 0.938). Both
skills that diverge beyond 0.05 fail the gate on prompted accuracy
anyway, so the exclusion removes nothing the experiment would have
used. Free generation is ~8x more expensive per query, which is why the
narrower fix is preferred.

This is the dual metric earning its keep: B_bang would have entered the
record as "a skill the base cannot do" on the teacher-forced number
alone.

## STAGE 1 EXTENSION RESULT (2026-08-16) — 10/14 pass, 6 non-degenerate

Post-hoc suite extension, disclosed: the original 8 skills yielded only
**2** that both passed and were non-degenerate, which is too few to
plot accumulation against. Six further type-B candidates were appended
at indices 8-13 (so 0-7 keep their identities). **Gate thresholds are
unchanged**, and the pilot-1 result on skills 0/5/7 is not re-read.

| skill | type | prompted | unprompted | mismatched | free-gen | copy-oracle | verdict |
|---|---|---|---|---|---|---|---|
| B_title (8) | B | 1.000 | 0.070 | 0.000 | 1.000 | 0.000 | **PASS, non-degenerate** |
| B_first (9) | B | 0.985 | 0.000 | 0.000 | 0.969 | 0.000 | **PASS, non-degenerate** |
| B_spaced (10) | B | 0.210 | 0.000 | 0.000 | 0.156 | 0.000 | FAIL |
| B_plural (11) | B | 0.795 | 0.000 | 0.000 | 0.781 | 0.000 | **PASS, non-degenerate** |
| B_bang (12) | B | 0.000 | 0.000 | 0.000 | 1.000 | 0.000 | FAIL (metric guard, amendment 3) |
| B_triple (13) | B | 1.000 | 0.000 | 0.000 | 1.000 | 0.000 | **PASS, non-degenerate** |

Verdict: `n_pass = 10`, `n_pass_type_B = 6`,
`n_pass_non_degenerate = 6`, `base_pass = true`, prompted spread
sd = 0.413.

**The six non-degenerate skills — B_upper, B_double, B_title, B_first,
B_plural, B_triple — are the ones that carry any conclusion.** The four
type-A skills stay in the suite as the fact-like arm and as the easy
end of the range, and are labelled copyable wherever they appear.

Honest weakness of the extended suite, stated here rather than left for
the verifier: four of the six non-degenerate skills (B_double, B_title,
B_plural, B_triple) are "copy the input and decorate it". They are not
fixed-offset copyable — the copy-oracle is 0.000 for all of them,
because the decoration changes the token sequence — but they are a
narrow family of computations. B_upper and B_first are the only two
that transform the input's characters. A suite of six that is really
two mechanisms is a weaker accumulation axis than six independent
skills would be, and any retention curve should be read per-skill, not
only as a mean.

## STAGE 2 — accumulation (the unclaimed part), pre-registered

Runs ONLY on a base that passed Stage 1, and only over the skills that
individually passed.

### The write

- **Persistent adapter**: one LoRA (r=8, alpha=16, dropout 0) on
  `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`, created
  once and carried across ALL k. This is the "one persistent weight
  set" the dossier says nobody measures.
- **Teacher**: the FROZEN BASE with the adapter DISABLED, given the
  3-shot prompt for skill j. **Student**: the same model with the
  adapter enabled, given the unprompted query only.
  - Adaptation vs e004r: e004r cloned the CURRENT model's weights as
    teacher before each skill, so the target drifted with k. Here the
    teacher is fixed for all k, which (a) is free under LoRA (one
    `disable_adapter()` context, no second copy in memory), (b) makes
    each skill's target independent of consolidation order, and (c)
    makes retention measurable against a fixed reference instead of a
    moving one. Disclosed as a deliberate deviation.
- **Loss**: `KL(teacher || student)` over the ANSWER token positions
  only (forward KL, teacher as target, `log_target=True` — the
  Askell/Snell recipe and e004r's direction).
- **Budget**: a fixed 60 optimizer steps per skill, batch 16, AdamW
  lr 1e-4. Identical for every arm — the comparison is arms at matched
  budget, as in e004r.

### Replay

Every `1/replay_frac`-th step additionally takes one batch of the same
size drawn uniformly from skills already consolidated (j < k), scored
with the same KL-to-frozen-base-prompted loss. `replay_frac = 0.25`
for the pilot; the grid sweeps `{0.0 (ablation), 0.10, 0.25}`.

Honest note on the replay budget: [[scialom2022-continual-t0]] retains
98-99.8% at **1%** replay over 8 tasks. At 60 steps per skill, 1% is
0.6 steps — not expressible. Our 0.25 is a much LARGER replay budget
than the literature's, so this design is generous to K1. The
consequence is asymmetric and is stated in advance: **if the
interference wall (K3) appears even at replay 0.25, that is a strong
negative;** if K1 holds at 0.25, the follow-up question is how far
replay can be cut, not whether the effect is real.

### Arms (all four are required controls named in the dossier)

| arm | what it is | what it decides |
|---|---|---|
| `consolidate` | the loop above | the measurement |
| `direct_ft` | CE on ground-truth answers, same examples, same budget, no teacher | Snell's quality edge (~9 pts) must replicate or the premise is wrong |
| `separate_adapters` | a FRESH LoRA per skill from the base | the accumulation CEILING; no interference is possible by construction |
| `no_replay` | `consolidate` with `replay_frac = 0` | isolates replay's contribution |

3 seeds; skill order is a fresh permutation per seed (as in e004r).

### Measurements after every skill k

1. **Retention** — unprompted accuracy on EVERY skill j <= k,
   n = 200 eval-half queries each. This is the retention-vs-k curve and
   the K1/K3 discriminator.
2. **Generality / collateral damage** — two held-out probes:
   - `nats/token` on a fixed 64x256-token slice of a pinned natural
     text file (`generality.txt`, Project Gutenberg Alice in
     Wonderland, sha256 recorded in `artifacts/CAVEATS.md`), never
     trained on;
   - **unknown-marker control**: the same one-line format with a marker
     `<<K9>>` that is never consolidated. Its answer distribution
     should stay where the base put it; drift here is in-format
     collateral damage that the natural-text probe would miss.
3. **Baked + re-prompted** — prompted accuracy on the CONSOLIDATED
   weights, for skill k and for all j < k. This is the K1-vs-K2
   discriminator (does consolidation FREE context or merely CACHE it).

### Tripwire: did the write even happen

After consolidating skill k, `unprompted_acc[k]` must reach >= 0.60
(the same threshold the gate used for the prompted condition). If it
does not, the write failed and every retention number downstream is
meaningless. Pre-registered: a run whose write tripwire fires is
reported as a harness failure, not as an accumulation result.

## Rivals and kill conditions (pre-registered)

**K1 — compounding organ.** Retention of every prior skill stays
> 0.90 x its just-consolidated value out to k = N with replay <= 0.25,
AND baked+re-prompted exceeds baked-alone by more than the MDE (0.06).
- KILLED-BY: retention falling below 0.90x at any k at any affordable
  replay; or consolidation cost per skill growing with k.

**K2 — cache, not organ.** Accumulation works (retention holds) but
`baked+re-prompted - baked_alone <= 0.06` (the MDE) consistently across
seeds. You save context tokens and nothing more.
- KILLED-BY: a re-prompting gain above the MDE on consolidated weights.

**K3 — interference wall.** Retention decays with k in the
sequential-editing shape (flat then cliff); the `no_replay` arm falls
first and faster and replay only shifts the cliff right; the
`separate_adapters` ceiling stays flat while the persistent arm does
not.
- KILLED-BY: a flat retention curve out to k = N in the persistent arm
  matching the separate-adapters ceiling within the MDE.

Secondary pre-registered expectations, with the confidences that make
them scorable:
- Stage 1 gate passes on Qwen2.5-0.5B (~0.75). Type A passes 4/4
  (~0.85); type B passes >= 2/4 (~0.65), with `upper` and `double` the
  most likely and `reverse` the most likely to fail (tokenization).
- Snell edge (`consolidate` > `direct_ft` at matched budget): ~0.45 at
  0.5B. Snell measured it at 7B+; absence here flags scale-dependence
  and does NOT kill the premise (overderivation guard).
- Retention at k <= 8 with replay 0.25: no cliff (~0.55). Genuinely
  uncertain — this is the measurement, not a formality.

## MDE — what retention difference can this setup actually resolve

- n = 200 held-out queries per skill per condition. Binomial
  SE <= 0.0354 (worst case p = 0.5); SE = 0.021 at p = 0.9.
- Difference of two proportions at p ~ 0.9: SE_diff = 0.030, so a
  **single-seed MDE of ~0.060** at 2 SE.
- Averaged over 3 seeds: **MDE ~ 0.035**.
- The K1/K3 separation (retention > 0.90 vs an editing-style collapse
  to < 0.5) is 7-13x the MDE. The K1/K2 separation (re-prompting gain)
  is set AT the MDE, 0.06, which is deliberately the tightest call this
  design makes; a re-prompting gain smaller than 0.06 will be reported
  as "not resolvable here", not as zero.

## Grid-snapping guard (the artifact class that voided e005 run 1 and e006 run 1)

The failure mode is an evaluation cadence coarser than the effect,
which makes a real curve look like a step or a flat line. Four guards:

1. **Eval at every k.** k is an integer axis and 1 is the finest
   cadence that exists on it. No coarser schedule is used anywhere.
2. **Intra-skill probe.** Retention of all prior skills is also
   measured every 20 optimizer steps WITHIN each skill's 60-step write.
   A write that destroys prior skills and then partially recovers by
   step 60 is invisible to end-of-skill evaluation only; this catches
   it. Run in the pilot and kept in the grid (cost: 2 extra eval passes
   per skill).
3. **Both axes logged.** Every record carries `k` AND
   `cumulative_optimizer_steps`. Retention decaying with total gradient
   steps is a different finding from retention decaying with skill
   count, and the two are perfectly confounded unless both are
   recorded. `analyze.py` plots against both.
4. **No binning, no smoothing, no interpolation** in `analyze.py`; the
   raw per-(arm, seed, k, skill) matrix is written to
   `artifacts/results.csv` and every plotted point is a measured point.

## Spread check (decision 0010 item 4)

The pilot prints the per-skill spread of the gate's prompted accuracy
and of the retention values across skills. Near-zero spread across
skills would mean the skill suite is not discriminating and blocks
launch, exactly as a selection mechanism with zero variance would.

## Pilot (what actually runs before the verifier)

1. Stage 1 gate on Qwen2.5-0.5B, all 8 skills, n = 200. Failable.
2. If it passes: the Stage 2 loop end to end on the **first 3 passing
   skills** (at least one type B), 1 seed, arms `consolidate` and
   `separate_adapters` — the two maximally-different cells per decision
   0010 item 1 (a persistent weight set vs a fresh one per skill, which
   is the entire question).
3. `analyze.py` runs on that pilot output and produces
   `artifacts/results.csv`, `artifacts/results.png`,
   `artifacts/CAVEATS.md`.
4. Throughput measured (tokens/sec forward, sec/optimizer-step) and the
   full-grid wall-clock projected in `artifacts/CAVEATS.md` BEFORE
   anyone commits to it.

**The full grid is NOT launched by the builder.** Verifier reviews
first (decision 0010 amendment 1).

## Full grid, for costing only

4 arms x 3 seeds x N passing skills, plus per-k evals of all prior
skills (quadratic in N) and 2 intra-skill probes per skill.

## PILOT RESULT (2026-08-16) — K0 fired on the pre-registered metric; the refined metric says the write IS conditional from k=2

Skills 5 / 7 / 8 (B_upper, B_double, B_title — all non-degenerate),
seed 0, order B_title -> B_upper -> B_double, arms `consolidate` and
`separate_adapters`. Records: `runs/`, `k0_frozen_base.json`,
`artifacts/`. Reproduced exactly across two independent runs at k=1 and
k=2 (pilot 4 and pilot 5 agree to the digit).

### k=0, the frozen base — the measured null

`unprompted {5: 0.000, 7: 0.000, 8: 0.040}`,
`prompted_on_top {5: 0.915, 7: 1.000, 8: 1.000}`,
`unknown_marker_applies_rule {5: 0.000, 7: 0.000, 8: 0.000}`,
`generality 2.8360 nats`. So every threshold below has a measured
baseline of ~0, not an assumed one, and the ICL gap is present before
any write.

### Retention

| k | wrote | B_title (8) | B_upper (5) | B_double (7) | generality nats |
|---|---|---|---|---|---|
| 0 | — | 0.040 | 0.000 | 0.000 | 2.8360 |
| 1 | B_title | **1.000** | 0.000 | 0.000 | 2.8384 |
| 2 | B_upper | 0.980 | **0.930** | 0.000 | 2.8481 |
| 3 | B_double | 1.000 | 0.890 | **1.000** | 2.8478 |

Retention of prior skills: mean **0.979**, min **0.957**, at max k
0.9785. Every write landed (tripwire passed). Generality drift 0.0097
nats. Write quality vs the fresh-adapter ceiling: **+0.0067** — writing
into an adapter that already holds two skills was not worse than
writing into an empty one.

### K0 — the conditionality question, and why the pre-registered metric was the wrong one

The pre-registered trigger fired: `unknown_marker_applies_rule` reaches
**1.0**. Under DESIGN amendment 2 that BLOCKS the K1/K2/K3 readings,
and `analyze.py` reports `K_readings_blocked_pending_verifier: true`.

The refined metric, `cross_marker_leak` — the newest rule scored under
markers the run has ALREADY trained — tells a different and more
specific story:

| k | wrote | leak into ALREADY-trained markers | leak into not-yet-trained markers | leak into never-trained `<<K99>>` |
|---|---|---|---|---|
| 1 | B_title | (none exist) | K5 **1.000**, K7 **1.000** | **1.000** |
| 2 | B_upper | K8 **0.000** | K7 0.695 | 0.125 |
| 3 | B_double | K8 **0.000**, K5 **0.000** | (none left) | 0.975 |

Read plainly: **the first write is completely unconditional** — with
one rule in the adapter, the model attaches it to the format and every
marker gets it. From the second write on, markers that already own a
rule are fully protected (leak 0.000 at k=2 and k=3), while markers
that own nothing fall back to the newest rule. At k=3 the single
persistent adapter serves three different rules under three different
markers simultaneously, with zero cross-contamination.

So the pre-registered metric conflates two things: an unconditional
write, and a conditional write with a fallback for unclaimed markers.
The data are the second. A model applying one unconditional rule could
not score 1.00 / 0.89 / 1.00 on three different rules at once.

**The builder's reading is that the refinement resolves K0 and the
block should lift for k >= 2. The pre-registered block STANDS until the
verifier rules** — a builder who fires his own tripwire and then talks
himself past it is the failure mode decision 0010 exists to prevent.
What is offered is the refined metric and its numbers, not a verdict.

Two consequences for the grid regardless of how the verifier rules:

1. **k = 1 is structurally different from k > 1** and must not be
   treated as a comparable point on any retention curve. There is
   nothing to condition on until a second rule exists.
2. `cross_marker_leak` should be a first-class metric in the grid, and
   the reading rule for K0 should be stated on it rather than on the
   unknown-marker probe. The unknown-marker probe stays, reinterpreted
   as "which rule does an unclaimed marker fall back to".

### What this does NOT establish

Nothing about k > 3. Nothing about the Snell edge or replay, since
`direct_ft` and `no_replay` were not run. One seed, one order, three
skills, two of which (B_double, B_title) are copy-and-decorate. The
mean re-prompting gain is **-0.0092**, inside the MDE of 0.060, so the
K1-vs-K2 discriminator reads "not resolvable here" and not "zero".

## VERIFIER RULING (2026-08-16) — K0 resolved on the PRE-REGISTERED metric; two identifiability defects fixed; CLEARED TO LAUNCH

Independent adversarial review under decision 0010 amendment 1 and
decision 0012. Everything below was re-derived from `runs/*.json` and
from a replication run the verifier launched, not from the builder's
summary.

### 1. K0 — RULING

**The refined metric `cross_marker_leak` is REJECTED as the K0 reading
rule.** It is not an independent measurement. `cross_marker_leak[j]`
asks whether the NEWEST rule fires under marker j; `retention[j]` asks
whether skill j's OWN rule fires under marker j. Whenever the two rules
have different answers — which is true for every pair in the suite —
high retention forces the leak toward zero arithmetically. Reporting
"retention 0.98 and leak 0.000" as two facts double-counts one fact,
and confirming conditionality with a quantity that retention already
implies is circular. Adding it after the tripwire fired is also the
exact shape decision 0012 clause 5 exists to police, and it does not
need to be adjudicated on optics, because the measurement fails on its
own terms.

**K0 is nevertheless answered, by the PRE-REGISTERED probe, read per
skill and per age instead of collapsed to a `max`.** The data were
already in the records; only the aggregation was wrong.
`unknown_marker_applies_rule` from `runs/consolidate_s0.json`:

| k | wrote | K8 (B_title) | K5 (B_upper) | K7 (B_double) |
|---|---|---|---|---|
| 1 | B_title | **1.000** | 0.000 | 0.000 |
| 2 | B_upper | **0.890** | 0.125 | 0.000 |
| 3 | B_double | **0.000** | **0.000** | 0.975 |

By age since its own write: age 0 → **0.70**, age 1 → **0.445**,
age 2 → **0.000**.

Read plainly, and this is a different mechanism from the one the
builder described: **conditionality is ACQUIRED, not conferred.** A
freshly written rule is attached to the FORMAT and fires under any
marker. It becomes attached to its MARKER once a later write takes over
the format default. By two writes later the older rule is at 0.000
under an unclaimed marker while sitting at 1.000 under its own — which
is conditionality, measured on the pre-registered probe, at the
pre-registered 0.20 threshold, with no new metric.

**The builder's stated mechanism is wrong on his own data.** He wrote
that "markers that own nothing fall back to the newest rule". At k=2
the unclaimed marker returns the OLDEST rule (B_title 0.890), not the
newest (B_upper 0.125). The format default holds ONE rule at a time and
changes hands late, not at every write. Corrected in
`artifacts/CAVEATS.md`.

**K0 as a MECHANISM is refuted at k <= 3.** K0 says "each write
replaces the last, and retention decay is arithmetic". If that were
true, B_title would be near 0 after B_upper was written. It is at
0.980. Three rules coexist under three markers at 1.00 / 0.89 / 1.00.
That refutation rests on retention, which is an independent
measurement, not on any metric added after the fact.

**Ruling: PARTIAL LIFT.**
- The K1/K2/K3 readings are UNBLOCKED for rows at **age >= 2** (the
  skill was written at least two writes ago). At that age the
  pre-registered probe reads 0.000, so the rule being measured is
  demonstrably marker-conditional and the confound amendment 2 named
  does not apply.
- **Age-1 rows stay CONFOUNDED and may not carry a retention claim.** A
  skill one write old may still hold the format default (B_title at
  0.890), so part of its apparent retention can be the unconditional
  fallback rather than conditional storage. They are recorded and
  plotted, and labelled.
- k = 1 remains excluded structurally (no prior skill exists; already
  the case in `rows()`).
- The K0 block is re-armed on the corrected reading: `analyze.py` now
  blocks if `max_umar_age_ge_clean > 0.20`. It can still fire on the
  grid, and if it does the block returns.
- **Consequence for the pilot's headline: exactly ONE of its six
  retention rows is age >= 2.** "Retention 0.979, min 0.957" is a mean
  over rows that are mostly confounded. The clean pilot number is a
  single point, 1.000, on one skill, one seed. The grid at k = 8 is
  what produces a real curve.

### 2. Attacks on the retention headline

**(a) k = 1 exclusion — CONFIRMED, but not sufficient.** `rows()` only
counts `age > 0` rows as prior, so k = 1 never entered a retention
number. The builder did what he said. The exclusion that was MISSING is
the age-1 one above.

**(b) Replay at 0.25 is generous, and worse than "generous" — it is
CONTINUED TRAINING.** Quantified, which the design did not do. Replay is
round-robin over priors, 15 replay batches per subsequent skill split
across the priors, so the first skill written receives
`15 * sum(1/i, i=1..k-1)` extra batches on top of its 60-batch write:
**+37% by k = 3, +57% by k = 6, +65% by k = 8.** At Scialom's 1% the
same quantity is about +2.5%. So the `consolidate` arm does not measure
retention; it measures retention under substantial rehearsal, and a
flat curve there is a much weaker claim than it reads as.
Undisclosed change of KIND, not just of size: e004r's replay was "5%
replay CE on the pretraining stream" — a generality anchor. e004g's is
25% KL-to-teacher on PRIOR SKILLS — direct rehearsal of the exact thing
retention measures. The DESIGN discloses the 1%-vs-25% size gap but not
that the two are different interventions.
**The uncontaminated retention measurement is the `no_replay` arm, and
that is precisely the arm that has never been run.** Reporting rule
added: the headline retention number comes from `no_replay`; the 0.25
arm is reported as "retention with 25% rehearsal" and never as
"retention".

**(c) k <= 3 is uninformative about the wall — CONFIRMED BY MEASUREMENT
(section 5: retention falls to 0.570 at k = 4), and the grid is extended
because of it.** ROME collapses at ~32 sequential edits and
MEMIT's specificity decays earlier; K3's own prediction is "fine at
small k, then a cliff". A flat curve at k <= 6 sits inside the region
where K3 predicts flat, so it discriminates nothing. Two changes:
- The grid runs **N = 8** skills, not 6, so it reaches the k = 8 this
  document already pre-registered as the magnitude that would count.
  The 6 non-degenerate skills plus A_zeta and A_north; the two type-A
  skills extend k and give the fact-vs-skill contrast the dossier's
  narrowing asks for, and `analyze.py` now EXCLUDES all four copyable
  type-A skills from every headline read (`COPYABLE`), reporting them
  separately.
- **Stated in advance: k = 8 still cannot confirm K3.** It can kill K1
  (a cliff by k = 8) and it can measure the Snell edge and the replay
  dependence. Confirming or refuting the editing wall needs k ~ 32,
  which needs ~32 skills and is quadratic in eval cost (~90 h for the
  full grid at n = 200). That is the honest boundary of this
  experiment and it must appear on the results card.

### 3. The unmeasured controls

`direct_ft` and `no_replay` were never run, AND `analyze.py` computed
nothing from them even if they had been. Both fixed:
- `snell_edge_consolidate_minus_direct_ft` = `consolidate` minus
  `direct_ft` write quality at matched budget, with
  `snell_edge_resolvable` against the 3-seed MDE of 0.035. Dossier 004
  says this edge must replicate or the premise falls; it is now
  computed, and the pre-registered confidence for it is 0.45, so a
  null is a scale-dependence finding and not a premise failure.
- `replay_effect_consolidate_minus_no_replay` at max k.
Both arms are in the launched grid.

### 4. Identifiability dry-run (decision 0012 clause 1) — RUN, and it FAILED TWICE BEFORE PASSING

`python analyze.py --dryrun` synthesises the outcome each rival
predicts and pushes it through the real verdict rule. It found two
defects that would have produced an uncallable grid, which is the third
time this class of failure has appeared (e010, e009b pilot).

**Defect 1 — the K1/K2 discriminator was CEILING-CENSORED.**
`reprompt_gain = prompted_on_top - unprompted`, and after
consolidation four of the six non-degenerate skills sit at unprompted
1.000, where a +0.06 gain is arithmetically impossible. Under the
original rule K1 could therefore never fire and K2 fired automatically
whenever retention held — the grid would have returned "K2, cache not
organ" no matter what the world did. Sensitivity check: with only
B_plural below the ceiling the reading is INCONCLUSIVE; the design was
one skill away from uncallable. Fixed by splitting the verdict into two
independent calls — `retention_call` (COMPOUNDS / WALL /
INCONCLUSIVE_SEEDS_SPLIT) and `reprompt_call` (FREES_CONTEXT /
CACHE_ONLY / INCONCLUSIVE_CEILING_CENSORED) — with an explicit
headroom census, and by adding an off-length probe (`HARD_WORDS`, 6-8
letter held-out words, scored once at the final k) as a labelled
diagnostic. **Stated in advance: K1-vs-K2 will probably read
INCONCLUSIVE_CEILING_CENSORED on this suite.** The grid's product is
the accumulation question (K1/K2 vs K3), which is the part the dossier
says nobody has measured; the context-freeing question needs a suite
with headroom and is not bought here.

**Defect 2 — the retention verdict rested on a MIN over ~45 noisy
rows.** `K3_interference_wall` fired on `min_retention_prior <= 0.90`
over every row of every seed. At n = 200 a true retention of 0.93
produces a row below 0.90 by binomial noise alone with probability
~0.05, so across 45 rows a spurious K3 was ~88% likely. Fixed: the
retention verdict reads the per-seed mean over age >= 2 rows, per-seed
values are printed (decision 0012 corollary), and the raw min is
retained as a diagnostic only.

Result after both fixes:

```
K1: retention=COMPOUNDS reprompt=FREES_CONTEXT headroom=2/6 -> K1_compounding
K2: retention=COMPOUNDS reprompt=CACHE_ONLY     headroom=2/6 -> K2_cache_not_organ
K3: retention=WALL      reprompt=CACHE_ONLY     headroom=6/6 -> K3_interference_wall
identifiability dry-run: PASS
```

### 5. Independent replication — **the headline does not survive k = 4**

The verifier re-ran the `consolidate` arm at **seed 1** (different skill
order) with a FOURTH skill, on `arche-consol`, from the committed code:
order B_upper -> B_double -> B_title -> B_first. Record in
`runs_verifier_s1/`.

| k | wrote | B_upper (age) | B_double (age) | B_title (age) |
|---|---|---|---|---|
| 2 | B_double | 1.000 (1) | — | — |
| 3 | B_title | 0.978 (2) | 1.000 (1) | — |
| 4 | B_first | **0.812** (3) | **0.570** (2) | 0.954 (1) |

**Retention collapses at k = 4.** Minimum retention 0.570, mean at max k
0.779 — 4 to 7 MDEs below the 0.90 threshold, not a noise excursion.
The builder's "no collapse at k <= 3" was literally true and entirely
uninformative: the pilot stopped one step before the effect. Both
conditionality-clean rows at k = 4 (age >= 2) are below 0.90, so this is
not a K0 artefact either. The corrected verdict rule reads **WALL** on
this run.

This is exactly attack (c) above, confirmed with data rather than
argued, and it is the strongest single reason to run the grid: one seed
at k = 4 already moves the reading from "K1-ish" to "K3-ish", and three
seeds to k = 8 will settle it.

**The intra-skill probe (grid-snapping guard 2) also fired, and it is
now load-bearing.** Inside the k = 4 write, prior-skill accuracy at
optimizer step 20 / 40 / 60 was:

| skill | step 20 | step 40 | step 60 |
|---|---|---|---|
| B_upper | 0.375 | 0.703 | 0.797 |
| B_double | 0.000 | 0.000 | 0.500 |
| B_title | 0.000 | 0.781 | 0.984 |

A new write DESTROYS the prior skills within ~20 steps and replay claws
them partway back by step 60. End-of-skill evaluation alone sees only
the recovered state and understates the interference by a lot. This is
the artefact class that voided e005 run 1 and e006 run 1, caught here by
the guard that was written for it. (Naming defect, noted: the field is
called `retention` in the record but holds raw accuracy at `probe_n=64`,
not a ratio. Read it as accuracy.)

Two smaller replication facts: the frozen-base k = 0 record reproduced
the committed values exactly (`unprompted {5: 0.000, 7: 0.000,
8: 0.040}`, generality 2.8360 nats), independently discharging CAVEATS
check 2; and B_upper wrote to 0.930 as seed 1's FIRST skill, the same
value it reached as seed 0's second — the write itself is stable across
order.

**Correction to a reproducibility claim.** DESIGN said the pilot was
"reproduced exactly across two independent runs at k=1 and k=2 (pilot 4
and pilot 5)". Pilot 4 and pilot 5 are the SAME seed, the same order and
the same code path, so agreement is determinism, not reproduction —
and they in fact agree at ALL THREE k, not two. The useful thing it
shows is narrower and worth keeping: adding the `cross_marker_leak`
evaluation did not perturb the training RNG stream.

### 6. Standard checks

- **Discarded pilots 1 and 2 — CLEAN.** They live in
  `runs_pilot1_unpairedeval/` and `runs_pilot4_noXmarker/` on the box,
  outside the `runs/` glob, and were never copied into the repo. Local
  `runs/` holds exactly the two valid records and their md5s match the
  box.
- **Provenance was BROKEN and is fixed.** `git_sha` in every pilot
  record is the EMPTY STRING: the box is an rsync of the experiment
  directory, not a git checkout, so `git rev-parse` returned nothing and
  the code silently recorded it. Nothing in the record proved which
  code produced which number. `git_sha()` now falls back to a sha256 of
  `consolidate.py`.
- **Amendment 3's free-generation deviation — LEGITIMATE, with one
  condition.** Narrowing to per-skill exclusion instead of the global
  metric switch is right: the divergence is a property of two answer
  strings with multiple valid tokenizations, both of those skills fail
  the gate on prompted accuracy anyway, and across the 10 passing
  skills the two metrics agree within 0.016. The gap is that the guard
  was only ever checked on the FROZEN base, and consolidation changes
  the output distribution. Condition: `--freegen-final 64` re-validates
  teacher-forced against free generation on CONSOLIDATED weights at the
  final k, and it runs in the grid.
- **"Four of six non-degenerate skills are one mechanism" — REAL, not
  fixed, now unmissable.** `analyze.py` emits
  `retention_prior_by_skill`; no retention mean may be quoted without
  it. The two type-A skills added for k = 8 are a genuinely different
  computation (constant association) and are reported separately, never
  mixed in.
- **The re-prompting gain of -0.0092 — correctly reported as
  unresolvable, and it is weaker than that.** It is not merely inside
  the MDE; it is ceiling-censored, since 4 of the pilot's 6 written
  rows sit at unprompted 1.000. On the one row with headroom it is
  -0.0275. Neither number licenses "zero".
- **Spread check** passes: `consolidate` unprompted sd 0.457.
- **Replay-sweep arm collision** fixed: a `consolidate` run at a
  non-default replay budget now names itself `consolidate_r<frac>` so
  `analyze.py` cannot average two replay budgets into one arm.

### 7. VERDICT: **CLEARED TO LAUNCH**

Grid: 8 skills (B_upper, B_double, B_title, B_first, B_plural,
B_triple, A_zeta, A_north) x 4 arms (`consolidate`, `direct_ft`,
`separate_adapters`, `no_replay`) x 3 seeds, replay 0.25, n_eval 200,
`--freegen-final 64` on seed 0. On `arche-consol`, **ONE worker at 16
threads, serial** — peak RSS is a MEASURED 3.8 GB of 31 GB, and the
builder's own throughput number (567 fwd tok/s, ~94% of this SKU's fp32
peak) says a single process already saturates the box, so concurrency
would buy ~nothing and only multiply memory. Sizing by measured memory
rather than by core count is the rule after a `cpu_count-1` runner
wedged another 32 GB box until sshd stopped answering. Projected
~8-10 h. `grid.sh` is committed alongside.

What this grid can decide: whether accumulation holds to k = 8 in one
persistent adapter (K3 vs not-K3), whether Snell's quality edge
replicates at 0.5B, and how much of the retention depends on rehearsal.
What it cannot decide, stated before launch: K1 vs K2 (ceiling-censored)
and the existence of the editing wall (which lives at k ~ 32).

**Expectation on the record before the grid runs, so it is scorable:**
on the verifier's k = 4 replication the retention verdict already reads
WALL, so the honest prior is now **K3-ward, ~0.6**, not the "no cliff
at k <= 8, ~0.55" the design registered. If the grid confirms it, the
result is a NEGATIVE for bet B2's accumulation clause at this scale —
which is a real finding, is what the dossier says it wants ("K3
confirmed = B2 drops toward the editing-collapse prior and the axis
candidate dies honestly"), and is the outcome the lab's ten-for-ten
overclaim bias would have hidden.

## Budget honesty

CPU throughput on Qwen2.5-0.5B is measured, not assumed, and the
projection is recorded before launch. If the grid does not fit in
CPU wall-clock, that is reported plainly as a finding: it converts the
GPU request from "we would like one" into a measured wall.

---

# GRID RESULT, 3 seeds x 4 arms, k=8 (read 2026-08-17)

Provenance, stated first because this lab has had four provenance
failures: the grid finished 09:05 UTC and was found idle, unread. The 12
records were pulled from arche-consol into `runs/` in this directory.
`consolidate.py` (sha256 `0afb95a7...`) and `analyze.py` (`bbc10253...`)
match byte-for-byte between the box and the committed copies. Every
number below is read from a COMMITTED RECORD FILE by the committed
`analyze.py`, not from stdout.

| arm | write objective | replay | retention at k=8 | per-seed | umar_age0 | generality drift (nats) |
|---|---|---|---|---|---|---|
| `consolidate` | KL to prompted self | 25% | **0.938** | 0.972 / 0.849 / 1.004 | 0.527 | **0.096** |
| `direct_ft` | CE on ground truth | 25% | 0.817 | 0.828 / 0.900 / 0.704 | 0.464 | **0.318** |
| `no_replay` | KL to prompted self | 0% | **0.059** | 0.000 / 0.000 / 0.200 | 0.957 | 0.079 |
| `separate_adapters` | KL, fresh weights | n/a | BROKEN | {} | 0.687 | 0.051 |

All writes landed in all arms. Max k reached 8; single-seed MDE 0.06.

## Three findings, weakest form

**1. CALLABLE — the earlier "collapse at k=4" was a single-seed artifact
and is WITHDRAWN.** PORTFOLIO.md recorded e004g retention collapsing at
k=4 to 0.779. The full three-seed grid retains **0.938 at k=8**. The
consolidation write accumulates across a sequence twice as long as the
one that was thought to break it. Portfolio corrected.

**2. CALLABLE — e015's contested conditionality contrast REPLICATES.**
Independent arms, three seeds, k=8: replay-bearing `consolidate` gives
umar_age0 = 0.527, no-replay gives 0.957. e015 measured 0.537 vs
0.954-0.962. That contrast is the sole support for e017's headline
(rehearsal's function is conditionality restoration) and it now has
replicated, committed backing. It also relocates e017's problem: e017's
`replay25` arm reading 0.805-0.945 is the ANOMALY to explain, not e015's
0.537. Two arms in a second experiment agree with e015.

**3. READABLE, and it partly pre-answers e019b — the WRITE OBJECTIVE
matters, and CE is worse on both axes at matched replay.** This grid
already contains the comparison e019b was being designed to run:

- retention: CE 0.817 vs KL 0.938;
- collateral damage to general capability: CE **0.318 nats vs KL 0.096**,
  a factor of 3.3.

So writing a skill by supervised CE on ground truth both retains prior
skills worse AND damages the base substantially more than distilling the
base's own few-shot behaviour. Read against
`library/sweeps/harness-scope-limit.md` this is the cost side of the
scope limit: the KL write is restricted to skills the base can already
demonstrate, and that restriction is apparently also what makes it
gentle. A CE write buys access to genuinely novel skills and pays for it
in both retention and capability.

CAVEAT that keeps this READABLE and not CALLABLE: `direct_ft` swaps the
objective on BOTH the new-skill write AND the replayed prior examples
(`ce_loss` is used in both branches). So it is not a clean single-factor
contrast on the write alone. e019b should be designed to vary ONLY the
new-skill write objective, holding the replay objective at KL — that is
now a much sharper experiment than the one first sketched, and this grid
supplies its predicted effect size.

## Broken arm, recorded so no one cites it

`separate_adapters` produced 18 rows and `mean_retention_prior = None`
(nan): it generated NO prior-skill rows at all, because with fresh
weights per skill the loop sets `seen = [s]`. The arm is not a null
result, it is a non-result. Any earlier claim resting on it is void.

## Carried caution, from analyze.py's own `_reading` block

"a 'no cliff' reading bounds nothing beyond the k actually run; the
sequential-editing collapse this design is contrasted against appears at
k~32." Finding 1 says consolidation survives to k=8. It says nothing
about k=32, which is where the editing literature's collapse lives and
where e020 is looking.

## CORRECTION to finding 3, same day: it is a REPRODUCTION

The CE-write prior-art sweep landed after the verdict above was written.
Finding 3 (the write objective matters, CE is worse) is **published**:
SDFT (arXiv 2601.19897, Jan 2026) reports prior-task average 64.5%
self-distilled vs 53.4% SFT on Science QA and 65.4 vs 56.0 on Tool Use;
DiSC (arXiv 2602.16093) confirms the direction on four models. Our write
objective is itself the published method **context distillation** (Snell,
Klein & Zhong, arXiv 2209.15189, 2022) and must be cited as such rather
than described as ours, throughout e004g, e015, e017 and e019.

**And the reproduction disagrees with the original on scale.** SDFT
reports self-distillation UNDERPERFORMING plain SFT at 3B, gaining only
at 7B (+4) and 14B (+7). We ran Qwen2.5-0.5B, below that crossover, and
measured the distillation write clearly ahead (0.938 vs 0.817). The most
likely reconciliation is our own confound, already noted: `direct_ft`
swaps the objective on the REPLAYED PRIOR examples as well as the new
write, which SDFT does not. Until that is separated, finding 3 is NOT
CITABLE in either direction. The clean version varies only the new-skill
write objective and holds replay at KL.

Findings 1 and 2 are unaffected — they concern retention across k and the
conditionality contrast, not the write objective.

## DEFECT FOUND IN THE GRID'S SKILL POOL: a prefix-colliding pair

Found by reading the committed per-step records (2026-08-17). The design
claims the prefix artifact e015 documented is "excluded by construction".
**It is not.** The pool contains

- skill 7 = `B_double`, `lambda w: w + " " + w`
- skill 13 = `B_triple`, `lambda w: " ".join([w] * 3)`

and `B_double("cat") = "cat cat"` is a strict PREFIX of
`B_triple("cat") = "cat cat cat"`. Both were added in the 2026-08-16
post-hoc suite extension.

**Decisive evidence, from records, not argument: writing `B_triple`
makes the NEVER-WRITTEN `B_double` score 1.000.** Eight instances across
all three arms and all three seeds:

```
consolidate s1 k=1 wrote B_triple -> B_double reads 1.00 (never written)
consolidate s2 k=2 wrote B_triple -> B_double reads 1.00 (never written)
direct_ft   s1 k=1 wrote B_triple -> B_double reads 1.00 (never written)
direct_ft   s2 k=2 wrote B_triple -> B_double reads 1.00 (never written)
no_replay   s0 k=5 wrote B_triple -> B_double reads 1.00 (never written)
no_replay   s1 k=1 wrote B_triple -> B_double reads 1.00 (never written)
no_replay   s2 k=2 wrote B_triple -> B_double reads 1.00 (never written)
consolidate s2 k=7 wrote B_upper  -> B_double reads 0.94 (never written)
```

That last row is a SECOND, unexplained collision (`B_upper` lifting
`B_double`) and is not accounted for by the prefix story. Flagged, not
resolved.

**The tripwire for this already existed, fired, and was not read.**
`analyze.py` reports `max_unprompted_before_write` and it is **1.0** in
the published output — the maximum possible value, meaning some skill
scored perfectly before it was ever written. The guard was correct; the
failure was that nobody looked at it. Any future reading of this harness
must treat `max_unprompted_before_write > 0.2` as BLOCKING.

### What it costs, quantified

The colliding pair are the two highest-retention skills in the
`consolidate` arm (`B_double` 1.0000, `B_triple` 0.9906), so they inflate
the headline. Dropping both:

| | mean prior retention |
|---|---|
| as reported, 6 skills | 0.9465 |
| **clean, 4 skills (B_upper .906, B_title .987, B_first .969, B_plural .827)** | **0.9220** |

**Finding 1 SURVIVES with a corrected number: 0.922, not 0.938/0.946.**
Consolidation still accumulates to k=8, three seeds, and the no-replay
collapse (0.059) is unaffected because there is almost nothing there to
inflate. The correction is 2.5 points, not a reversal. PORTFOLIO.md
updated to 0.922.

### It also kills the "spontaneous recovery" reading I briefly held

Scanning for prior skills whose accuracy ROSE across k looked like strong
masking evidence, including three cases in the NO-REPLAY arm where a
skill went 0.000 -> 1.000 with nothing rehearsing it. **All three are
`B_triple` reading 1.0 because `B_double` was the skill being written**,
i.e. the collision, not recovery. Withdrawn before it was reported as a
result. On clean skills the no-replay arm shows 23 drops and 0 genuine
recoveries: masking there is total and it does not spontaneously reverse.

### Consequence for the design, not just the number

The verifier's earlier k=4 collapse to 0.779 ran a DIFFERENT sequence
(`B_upper -> B_double -> B_title -> B_first`) from any grid seed (grid
orders are `[8,11,9,0,13,5,7,1]`, `[13,5,7,11,8,0,9,1]`,
`[0,13,1,8,9,11,5,7]`), and it contained the colliding `B_double`. So it
is neither superseded-as-artifact nor a clean contradiction — it is a
different measurement on a partly contaminated pool. The earlier text
above calling it a single-seed artifact overstates; corrected here.

**Required before this harness is used again** (it is shared by e015,
e017 and e019): drop `B_triple` or `B_double`, and add a pool gate that
rejects any pair where one skill's output on the eval set is a prefix of
another's. A prefix check across all pairs is a few lines and would have
caught this before any compute was spent.

## POOL GATE (added 2026-08-17)

A blocking gate now runs before any compute is spent, plus the census of
every coincidence in the pool and a contamination assessment of the three
downstream experiments.

### The gate

`consolidate.pool_clashes(tok, skills)` takes the skill indices a run
will ACTUALLY use and, over the real `EVAL_WORDS` (32 five-letter
lowercase words), checks every ordered pair for four kinds of answer
coincidence: `equal` (identical strings), `ws_equal` (identical after
whitespace normalisation), `token_prefix` (i's answer tokens are a prefix
of j's), `token_suffix` (i's whole answer is the token tail of j's).
Threshold is 0.20, the lab's standing control level.
`consolidate.assert_pool_clean` raises `SystemExit` with the offending
pairs; it is called from `consolidate.py`'s pilot/arm path, from
`cont.py`'s arm path on the actual `--skills` list, from `arb.py`'s
selfcheck and from `recov.py`'s skill selection (over written AND
held-out). `consolidate.selfcheck` carries a positive control: it asserts
the gate DETECTS `B_double`/`B_triple` and does NOT fire on the four
clean skills.

`token_suffix` is reported but NOT blocking, and that is a measured
decision rather than a softening. Scoring is teacher-forced argmax from
the answer's FIRST token, so a shared tail is never reached with a wrong
head and cannot inflate anything. Empirically, blocking it fires at 100%
on `B_first`/`B_wfirst` and `B_last`/`B_spaced` — a one-token answer is
trivially the tail of anything ending in that token — which would
condemn e015's and e019's live, genuinely prefix-free sets for a hazard
that does not exist. `equal`, `ws_equal` and `token_prefix` block.

`analyze.py` no longer merely prints `max_unprompted_before_write`.
In e004g it sets `_reading.POOL_BLOCKS`, forces every K flag false, and
`main()` exits non-zero after writing the artifacts for diagnosis;
`--exclude 7,13` drops skills from every read and is recorded in the
verdict as `POOL_excluded_skills`. In e015 and e017 the same test sets a
per-arm `pool_blocked` folded into `k0_blocked`, so a contaminated arm
drops out of `arms_readable` exactly as an unconditional write does.
e019's `analyze.py` needed no change: its mandatory never-learned
deflation control already excludes any rung whose held-out recovery
exceeds `DEFLATION_LINE = 0.20`.

**The SKILLS list is not duplicated.** e015's `cont.py` imports
`consolidate` and appends; e019's `recov.py` imports `cont`; e017's
`arb.py` imports both. One definition, one fix, four experiments. The
shared module already exists de facto; no refactor is warranted. The
only real hazard in the arrangement is that each box holds an rsynced
COPY, so a fix must be pushed per box (see below).

### THE SECOND COLLISION IS EXPLAINED — it was mis-attributed

`consolidate s2 k=7 wrote B_upper -> B_double reads 0.94` is not a new
mechanism. Seed 2's order is `[0,13,1,8,9,11,5,7]`: `B_triple` (13) was
written at **k=2**, and at k=7 it still reads 1.000 while its prefix
`B_double` reads 0.94. It is the same prefix collision, sourced from an
EARLIER write, not from the write at that step. General lesson: an
unprompted-before-write lift must be attributed to ANY earlier write, not
to the concurrent one. Nothing unexplained remains.

### Collision census, on the real eval words

All 32 eval words are 5-letter lowercase. There are no palindromes, no
words with first letter == last letter, no already-uppercase words and no
1-character words, so the hypothetical collisions worth checking
(`B_reverse` vs identity, `B_first` vs `B_last`, `B_upper` vs identity)
all occur on **0/32** words. Everything above 0.0:

| pair | kind | fraction of eval set | in a grid? |
|---|---|---|---|
| `B_double` -> `B_triple` | token_prefix | **1.000** | YES, e004g — the known defect |
| `B_first` -> `B_spaced` | token_prefix | **1.000** | no: `B_spaced` failed the gate, never in any order |
| `B_last` -> `B_reverse` | token_prefix | 0.125 | both in the e004g pool, neither in a grid order |
| `B_first` -> `B_plural` | token_prefix | 0.094 | YES, e004g/e015/e019 — below the 0.20 line |
| `B_first` -> `B_ed` | token_prefix | 0.031 | YES, e019 — below the line |
| `B_double` -> `B_triple` | token_suffix | 1.000 | non-blocking by construction |
| `B_last` -> `B_spaced` | token_suffix | 1.000 | non-blocking |
| `B_first` -> `B_wfirst` | token_suffix | 1.000 | non-blocking |

`B_first -> B_spaced` at 100% is a second complete prefix pair that had
never been noticed. It cost nothing only by luck: `B_spaced` failed the
prompted gate and so never entered a run. The two sub-threshold
`B_first` pairs (9.4% and 3.1%) are tokenizer accidents where `" wagon"`
begins with the token `" w"`; they are inside the 0.20 control level and
are recorded rather than acted on.

### Contamination assessment of the running / pending grids

**e004g (finished, contaminated, repairable by exclusion).** Re-read with
`--exclude 7,13` and the gate then passes (`max_unprompted_before_write`
0.05, was 1.0). Confirmed: `consolidate` mean prior retention **0.9229**
(the 0.922 already in this file). Numbers that change beyond that:

- **no-replay is NOT 0.059 on clean skills, it is 0.000.** Per-seed
  `retention_at_max_k` goes `{0.0, 0.0, 0.2} -> {0.0, 0.0, 0.0}`; the
  entire 0.0588 was seed 2's colliding pair. The collapse is total.
  `mean_retention_prior` 0.046 -> 0.0008.
- **the seed split in `consolidate` widens.** Clean per-seed retention
  `{0.965, 0.806, 0.985}` (was `{0.969, 0.900, 0.986}`). Seed 1 is now
  clearly under 0.90, so `retention_call` stays
  `INCONCLUSIVE_SEEDS_SPLIT` and "accumulates to k=8" rests on 2 of 3
  seeds. Weakest form: 2/3 seeds, not 3/3.
- `direct_ft` 0.859 -> 0.826, so the consolidate-minus-direct_ft edge
  goes 0.020 -> 0.029, still far under the 0.06 single-seed MDE and
  still unresolvable. Finding 3 stays not citable.
- every `unknown_marker_applies_rule` value drops (age 2: 0.224 ->
  0.112), but the K0 block is unchanged: `max_umar_age_ge_clean` is 1.0
  either way, so `K_readings_blocked` stays true and no K flag flips.
- `separate_adapters` still emits no prior-skill rows: still a
  non-result.

**e015 (running on arche-cont, [address removed] — SALVAGEABLE.** The
running processes pass `--skills 5,7,8,9,11,24,28,31,33,34`, which is
`GRID_SKILLS`: `B_triple` (13) and `B_capfirst` (35) are absent, and the
set passes the new gate. All 10 completed records in
`~/arche/experiments/e015-no-replay/runs/` were read: orders are the
clean 10 and **no never-written skill reads above 0.2 at any k in any
record** (0 instances out of 10 runs). The contaminated e015 output is
the older 12-skill pilot in `artifacts/verdict.json`
(`max_unprompted_before_write` 1.0, orders containing 13 and 35) — that
one is superseded and must not be cited.

**e019 (running on arche-recov, [address removed] — SALVAGEABLE, structural
argument only.** No records exist yet (`runs/` is not created until a
seed finishes), so this is not an empirical statement about the output.
What is established: the launch log shows written =
`[B_upper, B_first, B_plural, B_pre_x, B_wfirst, B_dashw, B_hash, B_ed,
B_pre_z, B_my, B_at, B_ish]` and held-out =
`[B_double, B_title, B_the, B_star]`. `B_triple` is not in the suite
selection at all, and `B_double` is HELD OUT, so the defective pair
cannot both be present. Running the new gate over written ∪ held-out (16
skills) returns no blocking clash. No written skill has `B_double`'s
answer as a token prefix, so the held-out deflation control cannot be
lifted by the pair either.

**Can the colliding pair contaminate a RECOVERY RATIO? Yes, and it does
not cancel.** `recovery(r) = score(r at current k) / score just after
that skill was written`. The denominator is measured immediately after
the write, where the skill is at ceiling (~1.0) whether or not a
collision exists; a collision can therefore only inflate the NUMERATOR,
at later k, whenever the colliding partner is present and retained. The
ratio is bounded above by ~1, so the error pushes it up: it masks
forgetting and makes the recovery ceiling look HIGHER than it is. The
direction of the artifact is toward the headline, which is the dangerous
direction. Partial cancellation would need the collider to be present
and equally decayed at write time, which the ceiling denominator rules
out. For the never-learned control skills there is no denominator at
all: a collision there inflates an absolute score against the 0.20
deflation line, and e019's analyze already treats that as rung-excluding.

**e017 (arche-arb, [address removed] — FIXED BEFORE LAUNCH.** Its 21-run
grid has not launched. `GRID_SKILLS` is e015's prefix-free 10 and passes
the gate, so the fix does not change what e017 will run. The patched
`consolidate.py`, `cont.py`, `arb.py` and both `analyze.py` were copied
to that box and `arb.py --selfcheck` passes there. The pilot arms
already running were not touched (their modules are loaded in memory)
and no grid was launched.

**The boxes running e015 and e019 were NOT patched, deliberately.** Their
copies stay stale until those grids finish; both were verified to be
running clean skill sets, so patching them would risk the runs for no
gain. Re-rsync both after the grids land, before any re-run.

---

# TASK A (2026-08-17) — `ce_write_kl_replay`: separating the write objective from the replay objective

Pre-registered BEFORE the arm ran. Written by the builder; every claim here is
in the weakest form the data supports and nothing in this section may be
escalated by the builder (decision 0012 cl.5).

## Why

The grid's finding 3 (CE worse than context distillation on both retention and
capability damage) is not citable, for two reasons already in this file: (a)
`direct_ft` swaps the objective on the new-skill write AND on the replayed
priors, so it is a two-factor contrast; (b) SDFT (arXiv 2601.19897) reports the
OPPOSITE sign at 3B, and we ran 0.5B. The new arm `ce_write_kl_replay` uses CE
on the new skill and context distillation (KL to the prompted frozen base,
Snell, Klein & Zhong arXiv 2209.15189) on the replayed priors, so it differs
from `consolidate` in exactly one factor and from `direct_ft` in exactly the
other.

## THE BRIEF'S DESIGN IS UNSATISFIABLE AS WRITTEN, and here is the deviation

The brief asks for the same 25% replay, the same clean pool, the same k=8, and
the same 3 seeds AND SKILL ORDERS as the grid, "row for row". That cannot be
done:

- the grid TRAINED on `--skills 5,7,8,9,11,13,0,1`, which contains the
  prefix-colliding pair `B_double`/`B_triple`. The exclusion of 7 and 13 is
  READ-side only (`analyze.py --exclude`). There never was a clean-pool run.
- `consolidate.assert_pool_clean` now blocks that list, and the gate is not to
  be disabled.
- orders come from `rng.permutation(skills)` per seed, so ANY change to the
  skill list changes every order. Row-for-row comparability with the published
  grid is unrecoverable.

Deviation taken: **all three arms are re-run on a pool-gate-clean pool, and the
decomposition is read WITHIN that fresh triplet.** The published grid is
context only, never an endpoint. Pool = `0,1,2,3,5,8,9,11`: every e004g skill
that passes the capability gate except 7 and 13. It keeps k=8, keeps the same
four readable type-B skills the grid's clean read used (5, 8, 9, 11), and adds
two more type-A copyables (2, 3) in place of the colliders — type-A skills are
excluded from every headline read, so the readable set is unchanged in
composition, but the REPLAY MIX shifts from 2 to 4 copyables and that is a real
difference from the grid worth naming. Pool gate output: zero coincidences of
any kind at the 0.20 threshold, and `consolidate.selfcheck` (the gate's own
positive control) passes on the same box.

## Pre-registered reading rule (`analyze.three_way`)

"Sits with arm Y" = within one tolerance of Y AND more than one tolerance from
the other endpoint. Retention tolerance is `MDE_3SEED` = 0.035, the resolution
of the 3-seed arm means being compared. Verdicts: `WRITE_OBJECTIVE` (sits with
`direct_ft`), `REPLAY_OBJECTIVE` (sits with `consolidate`),
`MIXED_DECOMPOSITION`, `UNRESOLVED_ARMS_TOO_CLOSE`,
`UNREADABLE_ENDPOINTS_TOO_CLOSE`.

## Identifiability dry-run (decision 0012 cl.1) — FAILED TWICE, THEN PASSED

1. **First failure, and it is a design breaker the brief did not contain.**
   With the single-seed MDE (0.060) and the grid's endpoints, a synthetic arm
   sitting exactly HALF-WAY was labelled `REPLAY_OBJECTIVE`, because a
   half-gap of 0.049 is inside 0.060. Fixes: the tolerance for this read is
   `MDE_3SEED` (the comparison is between 3-seed arm means), and a verdict is
   refused outright unless the two endpoints are more than TWO tolerances
   apart. `analyze.py --dryrun` now pushes all three predicted worlds through
   the real reading code and each produces its own verdict.
2. **Second finding, on the REAL committed grid records, and it should change
   what anyone quotes.** `generality_drift_nats` is a max-minus-min over rows
   POOLED ACROSS SEEDS, so it also absorbs between-seed level differences. Per
   seed, `direct_ft` drift is 0.048 / 0.278 / 0.056 and `consolidate` is
   0.021 / 0.020 / 0.034. **The published "0.318 vs 0.096 nats, a factor of
   3.3" is a pooled figure whose CE side rests on one seed of three.** Per-run
   drift is 0.1275 (sd 0.106) vs 0.0250 (sd 0.006). `analyze.py` now reports
   `generality_drift_nats_seedmean` / `_seedsd` and reads the write-objective
   decomposition on the seed-mean with tolerance = the largest across-seed sd.
   **Predicted consequence, stated before the new arm finished: on the nats
   axis this design will most likely return `UNREADABLE_ENDPOINTS_TOO_CLOSE`
   at 3 seeds** (gap 0.102 vs 2*sd 0.213). The cheap upgrade is more seeds on
   the CE arms, ~50 min per cell; it was NOT taken, so a nats-axis verdict is
   not promised. The retention axis is marginal but readable: gap 0.097 vs
   2*tol 0.070.

## Provenance

- `consolidate.py` and `analyze.py` are rsynced to `~/e004g-taskA/` on
  arche-consol; per-step crash-safe records are written to
  `~/e004g-taskA/runs/<arm>_s<seed>.inprogress` (deliberately NOT `*.json`, so
  a partial cell can never be globbed into an analysis as a finished one) and
  the finished record replaces it as `<arm>_s<seed>.json`.
- the final-k LoRA adapter of every cell is saved beside its record, so the
  benchmark-percentage anchor the CE-write sweep asks for (nats have no
  external anchor; Luo et al. arXiv 2308.08747 Table 4 reports MMLU -9.54% to
  -18.37%) can be measured later without re-running an 8-cell hour.
- regression check on the new analysis code, run on the box against the 12
  COMMITTED grid records: `--exclude 7,13` reproduces `consolidate` 0.9229,
  `direct_ft` 0.826, pooled nats 0.0961 / 0.3183 exactly. The new fields are
  additive.
- `git_sha` inside each record is `code:<sha256[:16]>` of `consolidate.py`,
  because the box is an rsync and not a checkout.

## Task A status at hand-off (2026-08-17 13:30 UTC) — NO RESULT NUMBERS EXIST YET

Launched detached on arche-consol at 12:55 UTC: `~/e004g-taskA/taskA.sh`,
9 cells (3 arms x 3 seeds), one worker at 16 threads, peak RSS 3.4 GB measured.
`k=0` frozen base is measured and committed on the box
(`k0_frozen_base.json`, unprompted 0.000-0.040 on all 8 pool skills, nats
2.836) and is reused by every cell. Expect ~8-12 h: the first cell's k=0 took
27 min because the Task B pre-check was sharing the box, and per-cell wall on
the published grid was 47-57 min uncontended.

**Anything I might say about Task A's outcome would be NON-CITABLE, because no
result record exists yet. There are no numbers to report.**

Completion signal and pull, for whoever reads next:

    ssh arche-consol 'cat ~/STATUS_TASKA'      # written on EVERY exit path:
                                               # TASKA DONE / CELL FAILED /
                                               # PILOT ANALYSIS BLOCKED
    rsync -a arche-consol:~/e004g-taskA/runs/ experiments/e004g-consolidation/runs_taskA/
    # then read with the committed analyze.py pointed at that dir. NO
    # --exclude is needed: the pool is clean by construction.

`*.inprogress` files are per-step crash-safe partials and are NOT deleted when
a cell finishes — leftovers next to a finished `<arm>_s<seed>.json` are normal
and are not crashed cells. `analyze.py` only globs `*.json`, so they cannot
enter a read.

Pre-registered expectations, recorded so the read cannot be tuned afterwards:
the **retention** axis is marginally readable (published clean-pool gap 0.097
vs the 2*tol requirement of 0.070) and the **nats** axis will most likely
return `UNREADABLE_ENDPOINTS_TOO_CLOSE` at 3 seeds (per-seed gap 0.102 vs
2*sd 0.213). If the nats axis comes back unreadable, the fix is seeds, not a
new metric: ~50 min per extra CE-arm cell.

## BLOCKING: every retention number in this family is TEACHER-FORCED, and the free-generation check FAILS grid-wide

Found 2026-08-17 by the e019b pre-check builder, then verified directly
against the committed grid records. This is the most consequential finding
about this harness and it cuts against our own headline.

**The design requires teacher-forced and free-generation accuracy to
agree within 0.05. Grid-wide, the gap is about 0.92.** From the committed
records, `freegen` is recorded only at the FINAL k (k=8), and there:

| arm (seed 0) | teacher-forced at k=8 | freegen at k=8 |
|---|---|---|
| consolidate | 0.87-1.00 across skills | **0.000 for 7 of 8** (B_plural 0.391) |
| direct_ft | 0.61-1.00 | **0.000 for 8 of 8** |
| no_replay | 1.00 on newest, 0.000 prior | **0.000 for 8 of 8** |

So under free generation almost nothing is retained in ANY arm,
**including the `consolidate` arm that reports 0.922.** The 0.922, the
0.826, the 0.000 no-replay collapse and every `umar` figure in e004g,
e015, e017 and e019 are teacher-forced argmax quantities.

**This is the SECOND tripwire in this harness that fired and was never
read** (the first was `max_unprompted_before_write` reading 1.0). Both
were correctly implemented. Both were printed. Neither was looked at. The
pattern is now the thing to fix, not either instance: a guard that is
computed and not gated on is decoration.

### What the failure IS, as far as we know, and what is still unknown

For ONE adapter — the e019b pre-check's CE-written `B_qmark` — the cause
is established by decoding: **6 of 6 generations emit the CORRECT answer
and then fail to terminate** (`" wagon???????"`). If that is the grid-wide
cause, the distinction matters enormously:

- If non-termination: the rule IS being applied, and claims of the form
  "the write installs the transformation" survive, while claims of the
  form "the skill is usable" do not. Retention comparisons BETWEEN arms
  may still be valid, since all arms share the defect.
- If the content is wrong under free generation: the retention results
  collapse entirely.

**We do not know which holds for the grid.** `freegen` was logged only at
k=8 and no generations were saved. Until that is decoded, every number in
this family carries: *teacher-forced argmax; free-generation agreement
FAILS the design's own 0.05 requirement; cause established only for a
single unrelated adapter.*

### Required, and it blocks citation of this family

1. Decode the grid's saved adapters and classify the free-generation
   failures as non-termination versus wrong content. Until then the
   0.922 may not be quoted without the caveat above.
2. Log `freegen` at EVERY k, not only the last, and save sample
   generations.
3. Make the 0.05 agreement requirement a BLOCKING gate in `analyze.py`,
   like the pool gate, not a printed field.
4. Add an EOS/termination check to the write's known-positive set. A
   write that produces the right token and never stops is not a write
   that installed a skill.

## CORRECTION: the "3.3x capability damage" figure rests on one seed of three

The reported 0.318 vs 0.096 nats is a max-minus-min POOLED ACROSS SEEDS,
so it absorbs between-seed level shifts. Per seed:

| arm | seed 0 | seed 1 | seed 2 | per-run mean (sd) |
|---|---|---|---|---|
| direct_ft (CE) | 0.048 | **0.278** | 0.056 | 0.1275 (0.106) |
| consolidate (KL) | 0.021 | 0.020 | 0.034 | 0.0250 (0.006) |

**The CE side rests entirely on seed 1.** Two of three seeds put CE's
drift at 0.048 and 0.056 against KL's 0.020-0.034, which is a much
smaller and possibly null difference. The direction is unchanged; the
magnitude is not supported. "3.3x" is WITHDRAWN. PORTFOLIO.md corrected.

## VERIFIER RULING (2026-08-17) — the free-generation zero is NON-TERMINATION; content is correct on every teacher-forced-correct item

Brief: decode generations and settle why `freegen` reads ~0.000 grid-wide
while teacher-forced reads 0.87-1.00; rule on what e004g/e015/e017/e019
may claim. **Verdict: NON-TERMINATION.**

### 1. The grid's adapters do not exist; this is a reproduction

Adapter persistence was added to `consolidate.py` AFTER the grid ran.
The grid records carry `git_sha = code:0afb95a7f0c40f8d` = revision
`af0adb0` (verified: `arche-consol:~/e004g/consolidate.py` hashes to the
same 16 hex chars). The current file is `code:8885381a71461d51` =
`6396486`. `git diff af0adb0 6396486 -- consolidate.py` is +111/-1 and
touches only the `ce_write_kl_replay` arm option, the `.inprogress`
partials, the `save_pretrained` block and the pool gate. **No line of
`encode`, `accuracy`, `freegen_accuracy`, `kl_loss`, `ce_loss` or
`eval_suite` differs**, so the current code reproduces the grid's
training and scoring paths for the `consolidate` arm exactly. Library
versions on the box (torch 2.13.0, transformers 5.15.0, peft 0.20.0,
numpy 2.5.2) match the machine these reproductions ran on; only the CPU
arch differs (x86_64 vs arm64).

Searched: `runs/`, `runs_verifier_s1/`, `artifacts/` and
`arche-consol:~/e004g*`, `~/taskB`. The family's only saved adapter is
`e019-recovery-ceiling/runs_precheck19b/adapter_ce_write_kl_replay_s0`.
Everything below is a **mechanism reproduction, not the grid's own
outputs.** Task A on the box was not touched; all compute here is local.

### 2. Reproductions and what they show

Instrument: `decode_freegen.py` (committed). It loads a saved adapter,
scores teacher-forced and free generation **on the same items** (the
harness scores them on two different draws), decodes with
`max_new_tokens = answer + 8` with specials visible, classifies each
item, and re-decodes 4 items UNBATCHED to rule out a left-pad /
position-id artifact (it did: unbatched matches batched).

Run A, `runs_repro_freegen/consolidate_k2_s0.json` — consolidate, seed 0,
skills `B_plural,B_upper`, k=2, pool-gate clean. **The defect does NOT
appear:** the harness's own `freegen` reads 0.641 / 0.906 against
teacher-forced 0.750 / 0.844. Decoded generations end the line right
after the answer (`" tables\ntables\n..."`). This falsifies the
attractive structural story that a write under this harness *cannot*
learn to stop: the base's newline habit survives a short write.

Run B, `runs_repro_freegen/consolidate_k6_s0.json` — consolidate, seed 0,
skills `B_plural,B_first,A_north,A_zeta,B_upper,B_title`, k=6, pool-gate
clean, `--probe-every 0`, `--n-eval 32`, `--freegen-final 64`. **The
defect reproduces**, and the decode (`artifacts/freegen_decode_repro_k6.json`,
32 items/skill) explains it exactly:

Numbers below are the decode's own 32-item paired recomputation. The
committed record's own 64-item `freegen` field for the same six skills is
B_plural 0.750, B_first 0.031, A_north 0.000, A_zeta 0.000, B_upper 0.391,
B_title 0.953 — same picture, different sample size.

| skill (age) | teacher-forced | freegen (decode, 32 items) | prefix-correct | (a) answer+same-line continuation | (b) answer+extra then newline | (c) wrong content | passes |
|---|---|---|---|---|---|---|---|
| B_plural (oldest) | 0.844 | 0.781 | **0.844** | 2 | 0 | 5 | 25 |
| B_first | 0.719 | 0.031 | **0.719** | 7 | 15 | 9 | 1 |
| A_north | 1.000 | 0.000 | **1.000** | 23 | 9 | 0 | 0 |
| A_zeta | 1.000 | 0.000 | **1.000** | 32 | 0 | 0 | 0 |
| B_upper | 0.906 | 0.438 | **0.906** | 15 | 0 | 3 | 14 |
| B_title (newest) | 1.000 | 0.969 | **1.000** | 0 | 1 | 0 | 31 |

Read the fourth column against the second: **prefix-correct equals
teacher-forced to three decimals on all six skills**, and the (c) counts
equal the teacher-forced misses exactly (17 of 192 items in both cases).
**There is not one item in 192 where teacher-forced is right and free
generation produces wrong content.** The zeros are entirely (a)+(b) =
104 of the 175 teacher-forced-correct items emit the correct answer and
then keep going on the same line. `A_zeta` is 32/32 non-terminating
(`" zeta zeta zeta..."`-shaped); constant answers and single-letter
answers never stop, `B_title`/`B_plural` mostly do.

This is the same failure as the one committed case on the OTHER write
objective: `e019-recovery-ceiling/freegen_diagnostic.json` — CE write,
teacher-forced 1.000, freegen 0.000, `content_prefix_hits 6/6`, raw
`" wagon???????"`. KL and CE writes both emit the right answer and then
fail to stop.

Why it had to come out this way for the content: teacher-forced exact
match under greedy argmax **implies** greedy free generation emits the
same answer tokens (identical conditioning at the first answer position,
and each correct argmax reproduces the gold prefix). Wrong content on a
teacher-forced-passing item was only possible via a bug in the
generation path, and the paired + unbatched decodes exclude that.

### 3. Ruling on the guard (it was validated out of regime, and never gated)

The 0.92 gap is real model behaviour, not a string-comparison artifact.
But the guard that reported it is still defective in three ways:

- `gate()` measures freegen on **prompted** items (`consolidate.py:294-295`,
  and the field is literally named `prompted_freegen`), where three
  newline-separated example lines supply the terminator. That is the
  regime in which amendment 3 measured agreement within 0.016 and
  declared the metric sound. `eval_suite()` measures final-k freegen on
  **unprompted** items (`consolidate.py:404-406`). The guard was
  validated where it cannot fail and applied where nothing supplies a
  newline.
- No terminator is ever a training target. `encode()` builds
  `prefix + " " + answer`; `kl_loss` and `ce_loss` score the ANSWER SPAN
  ONLY. Termination is inherited from the base, not taught — which is
  why it survives at k=2 and decays by k=6 as writes accumulate.
- The scoring rule `txt.strip().split("\n")[0].strip() == answer` with
  `max_new_tokens = answer + 2` conflates non-termination with wrong
  content and truncates the evidence of which one happened. And
  `analyze.py` never reads `freegen` at all (grep: zero hits): the 0.05
  agreement requirement was a printed field, not a gate. Third instance
  of that pattern, after `max_unprompted_before_write` and the
  prefix-colliding pool.

### 4. What the family may claim, in weakest form

**(i) "the consolidation write installs the transformation" — READABLE.**
Stronger than before this decode, not weaker: the mapping is right under
free generation on every item teacher-forcing calls right (prefix-correct
== teacher-forced, 6/6 skills). Wording: the write installs the
input-output MAPPING; it does not install a terminating, usable skill.
Any phrasing that implies the skill can be *used* by generating from the
model is NOT SUPPORTED.

**(ii) Between-arm retention comparisons (0.922 / 0.826 / 0.000) —
READABLE, relabelled** as teacher-forced retention. They are not
corrupted by this defect: the compared quantity is teacher-forced and
the decode shows teacher-forced accuracy is exactly the free-generation
*content* accuracy. Arm-uniformity of the defect is OBSERVED, not
structural — all four arms read freegen 0.000-0.391 in the committed
records, including `separate_adapters` and `no_replay` at a fresh k=1
write with teacher-forced 1.000, and both write objectives decode as
non-terminating. What may NOT be claimed is any between-arm statement
about *usable* generation: freegen is not a fixed offset from
teacher-forced (0.969 on one skill, 0.000 on another in the same
adapter), so its arm ordering is unmeasured.

**(iii) Conditionality (`umar_age0` replay 0.527 vs no-replay 0.957) —
stays CALLABLE.** Teacher-forced on both sides, same metric, same
format; the question is whether the rule fires under a marker never
written. Termination does not enter, and e015's wording is already
conditionality-framed. Unaffected.

### 5. Is the fix cheap, and does it move the ranking?

Cheap: include the line terminator in the span `encode()` builds and the
losses score (one line each), and give `freegen_accuracy` a larger token
budget with prefix-correct reported separately from exact match. Effect
is on LEVELS, not on the arm RANKING — the ranking is a teacher-forced
comparison and is untouched. The prediction that freegen would rise
towards teacher-forced in every arm is NOT tested here; what IS measured
is that free-generation content already equals teacher-forced content,
which is the reason to expect it.

Still required: log `freegen` at every k with saved samples, and make
the 0.05 agreement a blocking gate in `analyze.py` measured IN REGIME
(unprompted, on written weights) — in the gate's prompted regime it
cannot fail.

### 6. Limits of this ruling

One arm, one seed, six skills, k<=6, on reproduced weights — the grid's
own adapters are gone. The k-dependence (agreement at k=2, collapse by
k=6) and the skill-dependence (constant and single-letter answers never
terminate) are both single-run observations. Nothing here rescues the
free-generation number itself: "the skill works when the model writes
freely" remains NOT SUPPORTED for this whole family.

---

# TASK A: the write-objective confound, separated. 3 arms x 3 seeds, k=8 (read 2026-08-17)

Records in `runs_taskA/runs/` (9), pulled from arche-consol; the run
finished 20:32 UTC and was found idle. This was launched to fix the
confound that made the morning's write-objective result non-citable:
`direct_ft` swapped the objective on BOTH the new-skill write AND the
replayed prior examples, so it never isolated the write.

| arm | write objective | replay objective | retention per seed | mean | nats drift per seed | mean |
|---|---|---|---|---|---|---|
| `consolidate` | KL (context distillation) | KL | 0.909 / 0.945 / 0.978 | **0.944** | 0.041 / 0.042 / 0.086 | 0.056 |
| `ce_write_kl_replay` | **CE** | KL | 0.756 / 0.940 / 0.982 | **0.892** | 0.116 / 0.135 / 0.176 | 0.142 |
| `direct_ft` | CE | CE | 0.709 / 0.930 / 0.917 | **0.852** | 0.059 / 0.059 / 0.177 | 0.098 |

## Verdict: NOT RESOLVED at 3 seeds. The pre-registered "in between" branch fires.

The interpretation rules were fixed before the run: sits with `direct_ft`
-> the effect is the WRITE objective; sits with `consolidate` -> it was the
REPLAY objective; in between -> report the decomposition and claim
neither.

It lands **in between, near the midpoint.** The total retention difference
between the extremes is 0.092, and the isolated write leg
(0.944 -> 0.892 = 0.052) and the isolated replay leg
(0.892 -> 0.852 = 0.040) are each **at or below the single-seed MDE of
0.06.** So the honest statement is: swapping the write objective and
swapping the replay objective each account for roughly half of a small
total effect, and neither is individually resolvable here.

**Seed 0 drives everything.** Retention on seed 0 is 0.909 / 0.756 / 0.709
across the three arms, while seeds 1 and 2 are 0.93-0.98 in every arm and
essentially separate nothing. A three-seed design where one seed carries
the entire contrast does not support a directional claim.

**The capability-drift figures are non-monotone and overlapping**, which
kills that side outright: the CE-write/KL-replay arm drifts MORE (0.142)
than the CE/CE arm (0.098), which no simple account predicts, and the
per-seed ranges overlap heavily. This also independently confirms this
morning's withdrawal of the "3.3x capability damage" figure: measured
cleanly here the same comparison is 0.098 vs 0.056, about 1.75x, and not
resolvable.

## What this closes

- The write-objective row in PORTFOLIO.md moves from "direction only, 1 of
  3 seeds" to **NOT RESOLVED**, which is the correct label.
- The apparent disagreement with SDFT's scale crossover (SDFT reports
  distillation LOSING below ~3B; we ran 0.5B and appeared to see it win)
  **dissolves — we do not have a resolvable effect to disagree with.** That
  is the boring and correct resolution, and it is better than the
  disagreement I was preparing to claim this morning.
- To resolve it properly would need seeds, not arms: the contrast is ~0.05
  against a single-seed MDE of 0.06.

Carried: teacher-forced scoring (see the non-termination ruling above); the
write objective is context distillation (Snell et al. 2022) and the
CE-vs-distillation contrast is published (SDFT, DiSC), so any reading here
would have been a reproduction regardless.
