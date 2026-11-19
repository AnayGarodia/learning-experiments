# The allocation ceiling: why twenty experiments produced a product of 1.0x

Written 2026-08-17 under decision 0011 (the 1000x-or-a-bottleneck bar).
This is a PROGRAM-LEVEL reading of the portfolio, not a new result. Its
purpose is to say what the lab's accumulated negatives have in common,
because that turns out to be more informative than any of them alone.

## The observation

Order the efficiency-axis results by what was actually measured, and
separate ORACLE / CEILING measurements from implementation measurements.
An oracle result is an upper bound no implementation can beat.

| experiment | what was measured | kind | value |
|---|---|---|---|
| e005 objective reweighting | gain under a TRUE oracle scorer | **oracle** | **1.05x** |
| e006 learner-chosen data | gain with a free PERFECT scorer | **oracle** | **~1.0x** |
| e009b adaptive compute | post-hoc ceiling reading the full model's own logits | **oracle** | **max 1.68x** |
| e003 enforced sparsity | arithmetic ceiling | **ceiling** | **2.31x** |
| e013 ICL-vs-gradient | max exchange rate over the grid | **max** | **2.01x** |
| e011 multi-horizon | implementation | impl | <1x (a tax) |
| e010 growth / stacking | matched-compute gain | impl | ~0.004 nats |
| e003 sparsity, deployable | wall clock, uncontended | impl | 0.81x (a loss) |

**Every oracle or ceiling we have measured on "spend the same budget
better" lands between 1.0x and 2.31x. Not one reaches 3x.**

That is the finding, and it is stronger than the individual verdicts
because oracles cannot be beaten by better engineering. e005's 1.05x is
not "our reweighting was weak", it is "a perfect scorer buys 5%". e006's
~1.0x is not "our selection was weak", it is "a free perfect selector
buys nothing in that world". e009b's 1.68x is a bound no router can
exceed, because it is computed from the full model's own logits.

## What it means, stated as weakly as the evidence allows

These experiments share a shape: **hold the information budget fixed and
allocate it better.** Reweight which tokens the objective emphasises;
choose which data to train on; decide how much depth to spend per token;
decide which weights may be non-zero; trade in-context examples against
gradient steps. Five independent oracle measurements of that shape
cluster in a band around 2x.

CANDIDATE EXPLANATION, explicitly a hypothesis and NOT established: there
is a common bound because reallocation cannot create information. Gains
must come from having more information, from representing it more
cheaply, or from reusing it across time. If that is right, the ~2x band is
not five coincidences.

What would make it a real result rather than a pattern: a single
experiment in which the allocation ceiling is DERIVED for a toy setting
and then measured, so the number is predicted rather than observed. That
is the experiment this synthesis argues for and it is specified at the
bottom.

## The program consequence, which is the point of writing this

**1000x cannot come from allocation.** Six experiments and five oracle
bounds say the whole family tops out near 2x, and decision 0011 amendment
1 only counts bounded multipliers if they are independent enough to
compound. These are NOT independent: they all act on the same underlying
waste, so per the portfolio's own rule the lab gains the difference, not
the product. That is exactly why the measured product is 1.0x after twenty
experiments, and it is not a run of bad luck.

So the remaining paths to the bar are the ones that change the
information economics rather than its allocation:

1. **Reuse across time — continual learning.** A model that keeps what it
   learns does not re-pay for it. This is the only axis where the ceiling
   is not obviously bounded, which is why it is the lab's current focus
   and why Anay named it. Current status: rehearsal works and costs stored
   data; the no-stored-data method retains 0.563 against 0.906, and a
   defect that may cap that number was fixed today.
2. **Cheaper representation of the same information — architecture.**
   Untouched by the portfolio so far. Every architecture experiment the
   lab has run (e012, e-retro) measured whether a capability APPEARS at
   toy scale, not whether it is cheaper per unit of capability.
3. **More or better information — data.** Measured and DEAD as an
   allocation problem (e006, oracle ~1.0x). Not measured as a
   data-GENERATION problem, which is a different question.

## Honest limits of this synthesis

- Every oracle above was measured in a small, mostly synthetic setting.
  The e005 corpus underlies twelve of fifteen experiments (see
  RETROSPECTIVE.md), and several verdict-bearing rungs sit below the
  ~3M-parameter usable-scale floor established by e-retro. **A ceiling
  measured at toy scale may not be the ceiling at frontier scale**, and
  the honest form of the claim is: at the scales we can afford, allocation
  oracles top out near 2x.
- The clustering is five numbers. Five numbers in a 1.0-2.3 band is
  suggestive of a common cause and is not proof of one.
- "Allocation" is my grouping, imposed after the fact on experiments that
  were not designed as a family. RETROSPECTIVE.md already flags that six
  of them form one closed group; this synthesis leans on that judgement
  and inherits its subjectivity.

## The experiment this argues for

**Predict the allocation ceiling, then measure it.** In a toy setting
where the data-generating process is known, derive the maximum
achievable gain from optimal reallocation of a fixed budget (optimal
token weighting, optimal example selection, optimal per-token compute),
then measure what an oracle actually achieves. If prediction and
measurement agree across three allocation types, the ~2x band has a
cause and the lab can stop testing allocation levers one at a time —
which would be the first genuinely predictive result the lab has
produced, and a bar-(b) fundamental-bottleneck finding rather than
another elimination.

Prior-art sweep required BEFORE that dossier (decision 0004 amendments 1
and 2), including on the premise: the relevant literatures are
information-theoretic limits on learning, data-pruning scaling laws
(Sorscher et al. 2022 is directly on point and reports that pruning
BEATS power-law scaling given a good metric, which would CONTEST this
synthesis's premise and must be addressed head-on), and the
compute-optimal scaling line.

---

# AMENDMENT 1 (2026-08-17): falsification check, run BY HAND after the API limit killed the delegated sweep

Two published results looked like direct contradictions. I checked both
myself. **The synthesis SURVIVES but must be SCOPED**, and the scoping is
the useful part. Caveat on this amendment's own status: it is a
single-eyed literature check done from abstracts and secondary summaries,
not a verified sweep, and it is READABLE at best. `ops/README.md` GATE 2
stays open until a verifier can run.

## RHO-1 versus e005 — resolved, and the numbers AGREE

RHO-1 (Lin, Gou et al., arXiv 2404.07965) is token-level reweighting, the
same intervention as e005, and it reports **5-10x faster to baseline** and
97% fewer effective tokens (10.5B vs 500B). Against e005's oracle 1.05x
that is a flat contradiction.

It dissolves on the eval distribution. RHO-1's headline is a
**narrow-target** result: a reference model is trained on a high-quality,
**target-aligned** dataset (math), tokens are scored by excess loss
against it, and the gain is measured on **math benchmarks**. When the same
method is applied to a general-domain 80B-token corpus and measured across
15 benchmarks, the reported average improvement is **6.8%**.

**e005's oracle measured 1.05x. RHO-1's general-domain result is 1.068x.**
Two very different setups, one synthetic and tiny, one an 80B-token real
corpus, agree to within two points. That is the strongest external
corroboration any arche result has, and it arrived while trying to refute
us.

So the reconciliation is: **token reweighting buys ~5-7% for general
capability, and 5-10x only when it is used to NARROW the target
distribution.** Narrowing is a different intervention — it changes which
problem you are solving, then measures on that problem. It is a real and
large lever, and it is what frontier data curation already does, but it
cannot compound toward general 1000x because the gain comes from
discarding capability you then do not measure.

## Sorscher 2022 versus e006 — SCOPES the synthesis, and genuinely weakens it

Sorscher et al. (NeurIPS 2022) claim data pruning can move error-vs-data
from power-law to **exponential** scaling given a high-quality pruning
metric. That is a change in the scaling REGIME, not a constant factor, so
it is qualitatively larger than anything in the ~2x band and it IS an
allocation intervention under my grouping.

Honest consequence: **"reallocation cannot create information" is too
strong as a general claim and is withdrawn in that form.** Removing
redundancy from a surplus pool is reallocation, and it can beat a power
law.

What survives, and why the synthesis still stands for our purposes: the
Sorscher mechanism requires a **large surplus pool with redundancy to
remove** (you prune a big dataset down; the gain is achieving the same
error with fewer examples) AND a high-quality metric — their own
ten-metric ImageNet benchmark is the paper's evidence that available
metrics fall short at scale. e006 gave the learner a FREE PERFECT scorer
and it bought ~1.0x, which under this reading means our pool had little
extractable redundancy, not that perfect selection is worthless. Twelve of
fifteen arche experiments share one synthetic corpus
(`RETROSPECTIVE.md`), and a corpus generated by a fixed process is exactly
where a pruning oracle has least to find.

## The synthesis, restated at the strength the evidence supports

WAS: allocation cannot deliver 1000x because reallocation cannot create
information.

NOW: **at the scales arche can afford and on the corpus it has been using,
every allocation oracle measured tops out between 1.0x and 2.31x, and one
of those numbers (1.05x) independently matches a published general-domain
result (1.068x) on the same intervention.** Published gains far above that
band come from changing the problem rather than the allocation: narrowing
the target distribution (RHO-1 math), or removing redundancy from a
surplus pool (Sorscher). Neither route compounds toward general 1000x, but
the second one means the general claim is unproven rather than false.

## What this changes in the program

- The redirect stands. Allocation is not where 1000x lives, and the lab
  should not spend more oracles on it.
- **A new, cheap, high-value experiment is now visible**, and it is better
  than the "predict the ceiling" experiment proposed above: e006's null
  and Sorscher's exponential scaling differ in a measurable property of
  the CORPUS, namely extractable redundancy. Measure that property on our
  corpus and on a real one, and predict which regime each is in. That
  turns a scoping caveat into a testable claim, and it would also tell us
  whether twelve of our fifteen experiments have been run in a regime
  where allocation could never have paid — which is the single most
  important unknown about the lab's own back catalogue.
- The "predict the allocation ceiling" experiment proposed above is
  DEMOTED behind that one. Predicting a ceiling on a corpus that may be
  the wrong corpus is premature.

---

# AMENDMENT 2 (2026-08-17): I audited the two oracles' CEILINGS and the synthesis must be DEMOTED

Amendment 1 scoped the synthesis. This amendment does more damage, and it
came from reading the corpus generator and the selection harness rather
than the literature. **Two of the five numbers the synthesis rests on had
their ceilings fixed by our own dials and design choices, so they are
close to tautological and cannot support a claim about the world.**

## e005's oracle could never have exceeded ~1.13x

`experiments/e005-oracle/corpus.py`: a sentence is fork(1 token) +
deterministic body(`body_len`=12) + noise(present with prob `noise_p`,
length 1..6) + consequence(1). Only the NOISE tokens are waste an
objective-reweighting oracle can decline to spend on. Expected noise
fraction, and therefore the maximum speedup from perfectly skipping all of
it:

| `noise_p` (grid values) | noise token fraction | max possible speedup |
|---|---|---|
| 0.25 | 0.059 | **1.062x** |
| 0.50 | 0.111 | **1.125x** |
| 0.75 | 0.158 | **1.188x** |

**e005 measured 1.05x against an arithmetic ceiling of 1.06-1.19x.** So it
achieved roughly half of the only gain available, and the experiment could
not have reported a large number no matter what. Its result is therefore
"removing 11% waste saves about 11%", which is nearly arithmetic. It is
NOT evidence that objective reweighting is bounded near 1x in general, and
using it that way — as this synthesis did — is wrong.

**Consequence for amendment 1, and I got this wrong in the strongest
terms.** Amendment 1 called RHO-1's general-domain 6.8% agreeing with
e005's 1.05x "the strongest external corroboration any arche result has".
**That is WITHDRAWN.** The agreement is a coincidence of an arbitrary dial:
`noise_p` was a free parameter, and at 0.75 we would have measured
something near 1.19x and the "agreement" would have vanished. Two numbers
matching when one of them is set by a knob is not corroboration.

## e006's oracle was charged for scoring, and ran in an INFINITE-data regime

`experiments/e006-selection/selection.py`:
- The budget is COMPUTE (`budget_fe`), and **selection is charged for it**:
  `spent += 3 + CANDS` for a selecting policy versus `spent += 3` for
  random. So the reported number is NET of scoring cost, not the gross
  value of selection. The portfolio's phrase "even a free perfect scorer
  buys ~nothing" is inaccurate: the scorer was not free.
- Data is generated FRESH every step (`gen.batch(...)`); there is **no
  fixed pool**. This is the infinite-stream regime, where more data is
  free and the binding constraint is compute.

Sorscher's pruning result is the opposite setting: a FIXED FINITE pool,
error measured against dataset SIZE, metric assumed available. **So e006
and Sorscher are different problems, and neither falsifies the other.**
Amendment 1's guess — that our corpus lacked extractable redundancy — was
also wrong on the facts: `dup_p` was dialled to 0.9, so 90% duplicates
WERE tested. The reason selection did not pay is the axis and the charge,
not the redundancy.

**And this collides with the decision I made an hour ago.** Decision 0013
adopts **tokens-to-capability** as the working target — a DATA axis. e006
measured a COMPUTE axis in an infinite-data regime. So the lab's flagship
data-selection negative does not bear on the target the lab has just
adopted.

## Status of the synthesis after this audit

Of the five numbers:
- e005 (1.05x) — **withdrawn from the band.** Ceiling set by `noise_p`.
- e006 (~1.0x) — **withdrawn from the band.** Wrong axis for the target,
  and net of a scoring charge.
- e003 (2.31x) — was always explicitly an ARITHMETIC ceiling, not an
  empirical bound.
- e009b (1.68x) — survives as a genuine post-hoc ceiling computed from the
  full model's own logits, at that scale.
- e013 (2.01x) — survives as a measured maximum over a grid.

So the "five independent oracle measurements clustering in 1.0-2.31x" is
really **two surviving empirical bounds, one arithmetic identity, and two
numbers set by our own knobs.** That is not a pattern with a common cause;
it is a pattern with a common CORPUS and a common set of design habits.

**DEMOTED:** the synthesis may no longer be stated as "allocation is
bounded near 2x." What it supports is much narrower and still useful:
*arche has not measured a single allocation lever in a setting where a
large gain was possible.* That is a statement about the lab's
experimental design, not about allocation, and it is the more actionable of
the two because it is fixable.

**The redirect away from allocation SURVIVES, for a different reason than
the one first given.** Not "allocation is bounded" but "our allocation
measurements are uninformative, and rebuilding them properly would cost
several experiments to reach a lever the frontier already exploits (data
curation)." Continual learning remains the better bet on opportunity-cost
grounds.

**Required before any of this is cited outside this file** (adds to GATE 2
in `ops/README.md`): a verifier must check the noise-fraction arithmetic
above, the claim that e006 charges for scoring, and whether e009b's and
e013's ceilings are also dial-determined. If they are, the synthesis has
no empirical content at all and should be replaced by a design
retrospective.
