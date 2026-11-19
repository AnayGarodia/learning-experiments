# Dossier 009 — The joint compute-redundancy oracle

Status: PRE-REGISTERED 2026-08-16, after prior-art (7 cards). Attacks
R5. Redundancy check: "predictable compute exists in transformers" is
SETTLED five ways (depth oracle: ~81-85% of layers skippable at
8-layer T5 scale, [[schuster2022-calm]]; width: 85-95% at OPT-175B
but ReLU-only, [[liu2023-deja-vu]]; cross-model: 60-75% draft
acceptance, [[leviathan2022-speculative-decoding]]; static: 20-55% of
layers removable, [[gromov2024-deeper-layers]]; learned: 87.5% token
skipping, [[raposo2024-mixture-of-depths]]). What is UNCLAIMED, and
what 009 measures:
(a) the JOINT width x depth x token oracle on ONE modern SwiGLU model
    (published numbers are per-axis, different models, different
    criteria — they cannot be multiplied);
(b) the oracle fraction as a function of TRAINING TOKENS (LayerSkip
    Fig 11 shows redundancy is a function of training history; a
    single-checkpoint number is uninterpretable);
(c) a prediction-ERROR criterion (predict the block's output, compute
    only on mismatch) — the predictive-coding mechanism has NO ML
    instance 2022-2026 (three searches; near-misses are change-gated,
    confidence-gated, or learned routers);
(d) the same number under TWO pre-registered criteria side by side
    (argmax-agreement AND exact-distribution), because criterion
    choice changes the answer qualitatively (Gromov's loss/capability
    decoupling is the cautionary datum).

## Rival mechanisms

**T1 — Large exploitable joint ceiling.** The joint oracle removes
>=80% of FLOPs at capability parity at hardware-priceable
granularity, and the fraction is stable or GROWS with training.
- KILLED-BY: joint ceiling under 50%, or collapse under the
  exact-distribution criterion, or fraction shrinking fast with
  training tokens (redundancy is an undertraining artifact).

**T2 — Small-model-equivalence.** The redundancy is exactly "a
smaller model suffices for most tokens" (the speculative-decoding
reading): the joint oracle's savings match what a smaller dense model
achieves — no per-input adaptivity premium.
- PREDICTS: oracle-skipped compute curve overlays the dense
  compute-optimal frontier; KILLED-BY: joint oracle beating the dense
  frontier by >=2x at matched capability.

**T3 — Unpriceable scatter.** The ceiling is large in FLOPs but lives
at fine granularity hardware cannot price (the lazy-neuron
97%-FLOPs->10%-wall-clock lesson); block-level oracle keeps little.
- KILLED-BY: block/coarse-granularity oracle retaining >=60% of the
  fine-grained ceiling.

## Design commitments (DESIGN.md + walkthrough before run)

Ladder of small SwiGLU decoder-only models (3 sizes), trained in our
own harness so we own checkpoints across training (constraint (b));
oracle measured per checkpoint. Criteria pre-registered: argmax-match
and KL<=eps vs full model. Axes measured jointly: skip whole blocks
(depth), gate heads/MLP slices at coarse granularity (width, priced
per 0008), reuse-from-previous-token where states are within eps
(temporal, the unclaimed axis). Report FLOP fraction AND
coarse-granularity fraction separately (T3 discriminator).
Prediction-error criterion arm: tiny predictor of block output;
compute-on-mismatch fraction. CPU-viable at these sizes
(arche-lab-2); GPU only if a rung demands it.

## Why it matters

R5 is a zero-row for the incumbent recipe. If T1: adaptive compute is
a first-order lever (the ceiling licenses building the mechanism —
and BLT's entropy-patching plus MoD are the mechanism candidates).
If T2: the lever collapses into "use smaller models when possible" —
valuable, already priced by the field. If T3: the lever is real but
waits on hardware co-design (0008 clause b path).

## What breaks at scale

Fractions are checkpoint- and scale-dependent BY DESIGN here (we
measure the trend, constraint (b)); the transferable claims are the
trend direction, the criterion gap, and the granularity gap.

## VERDICT (2026-08-16, 20/20 oracle measurements complete)

T3 KILLED cleanly; T1 PARTIAL — survives the capability (argmax)
criterion (80-88%, growing with training and scale), but the
pre-registered exact-distribution kill-disjunct fires on the per-axis
proxy (depth-KL 0.26-0.45, under the 50% bar; joint-KL unmeasured by
pre-registered verifier decision (b)); **T2 SURVIVES the kill test** (e009b: the oracle ceiling never
reaches 2x the dense frontier; formal verdict INCONCLUSIVE on axis
disagreement, but 2.0 is not approached — see the T2 section below).

The joint block-level oracle (depth-skip ∪ temporal-reuse, argmax)
marks 80.5% / 84.2% / 88.1% of (token, block) compute skippable at
s/m/l by end of training, and the fraction GROWS with training at every size
(s: 0.389→0.805 seed 0, 0.435→0.830 seed 1; endpoint growth —
non-monotone at the final checkpoint for m/l, within seed spread) and
grows with model size at matched tokens — the
"redundancy is an undertraining artifact" kill fails in the opposite
direction. Granularity: width redundancy is COARSE (zeroing SwiGLU
quarters: 0.97-1.00 agreement at end of training, 0.94-1.00 across
the grid) while per-unit dead activations are
~0.01 — the redundancy is hardware-priceable structure, not
unpriceable scatter (T3's kill condition met at ceiling, not 60%).
Criterion gap is qualitative as Gromov warned: depth skippability
falls from 0.72-0.81 (argmax) to 0.26-0.45 (KL<=0.1). Temporal reuse
— the unclaimed axis — is real, small, and GROWING: ~0 early, ~0.11
argmax by 6M tokens. Prediction-error probe: linearly-predictable
block transitions fall with training at s (0.69→0.11 seed 0;
0.71→0.12 seed 1) but stay high
at l (0.65 at end) — bigger (less-trained-per-param) models carry
more predictable internal compute.

## T2 DISCRIMINATED (2026-08-16, e009b full grid — see [[arche-e009b-t2-overlay]])

T2 was left undiscriminated above and is now scored. The frontier
overlay ran at 8 rungs x 5 training levels x 2 seeds (76 cells).

**T2 is not killed.** The pre-registered kill condition was "joint oracle
beating the dense frontier by >=2x at matched capability". The fit-free,
worst-seed lower bound on that multiplier peaks at **1.68** across the
entire grid — no level, seed, capability axis, frontier fit or FLOPs
accounting reaches 2.0. The formal verdict is INCONCLUSIVE because the
two primary capability axes disagree at the primary level (next-token
accuracy SURVIVES at [0.78, 1.43]; consequence accuracy lands at
[1.03, 1.61] against a 1.6 survive line, and the two FLOPs accountings
straddle it), but the 2x line is not approached on any reading.

The implementable arm is worse still: depth-only skipping at convergence
has multipliers 0.83/0.75 (xs0) down to 0.075/0.076 (l) by seed — below
1 at every rung above the smallest, meaning a smaller dense model is
strictly cheaper at matched capability.

**Consequence for R5: the mechanism program is NOT licensed.** The
ceiling a mechanism would chase is under 2x, and the arm a mechanism
could actually implement is under 1x. "BLT entropy-patching plus MoD are
the mechanism candidates" does not follow from this dossier's evidence.
R5 closes deflationary with the rest of the efficiency series.

## CORRECTION (2026-08-16, e009b verifier)

The 80-88% figures are an UPPER BOUND, not a parity result. e009 tested
each (token, block) skip as an independent counterfactual against the
full model's hidden states and reported their union; e009b executed the
union in a single masked forward and measured what it actually costs.
At l@6M the executed joint configuration agrees with the full model on
**98.15%** of tokens, and the implementation-faithful depth-only
configuration on **89.93%** (independently re-measured by the e009b
verifier from the same checkpoint: 0.9815 / 0.8993, 256 eval sequences).
So the right statement is "80-88% of (token, block) compute passes the
per-pair skippability test, and executing the whole union retains 98.2%
argmax agreement" — not "at argmax parity".

Two riders. The joint arm's 98.2% is itself inflated: the temporal
substitutions repair execution drift the depth skips cause, using values
and slot choices that only the clean model can supply, so no realizable
mechanism inherits that agreement. And the faithful arm is the one that
loses most: depth-only skipping at these fractions is not
capability-preserving in practice.

Transferable claims (DERIVED): trend direction (redundancy grows with
training AND scale), the criterion gap (argmax vs KL is qualitative),
the granularity verdict (coarse-structured). This is the series' first
NON-deflationary lever: the adaptive-compute ceiling is large, grows
the way you'd want, and sits at priceable granularity. What it does
NOT yet show: that the savings beat "just use a smaller model" (T2) —
that overlay is the gating follow-up before any mechanism-building.
