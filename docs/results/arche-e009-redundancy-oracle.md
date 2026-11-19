---
title: "arche e009 — joint compute-redundancy oracle: 80-88% skippable at block level (upper bound; 98.2% agreement when executed), GROWS with training and scale"
authors: [arche lab]
year: 2026
venue: internal experiment (pre-registered dossier 009)
url: experiments/e009-redundancy/
questions: [q1-scaling, q4-sample-efficiency]
status: carded
carded_by: fable (closure agent, 2026-08-16)
audited_by: fable (verifier, 2026-08-16 — independently pulled all 20 run records from lab-2 (byte-identical to repo runs/; results.csv values match) and recomputed every claim with own script; all matched except two spliced s-size endpoints (joint 0.435→0.805 mixed seed 1 start with seed 0 end; pred 0.71→0.11 likewise) and an understated seed spread (0.01-0.03 → 0.01-0.05); ruled T1 PARTIAL — argmax criterion survives at 80-88% but the pre-registered exact-distribution kill-disjunct fires on the per-axis proxy (depth-KL 0.26-0.45))
---

## Claims

- CLAIM (CORRECTED 2026-08-16, see below): The joint block-level oracle
  (depth-skip ∪ temporal-reuse, argmax-agreement criterion) marks
  80.5%/84.2%/88.1% of (token, block) compute skippable at
  0.89M/3.07M/9.44M nonemb params by end of
  training — and the fraction GROWS with training tokens at every size
  (s: 0.389→0.805 seed 0, 0.435→0.830 seed 1, over 229k→6M tokens;
  endpoint growth — non-monotone at the final checkpoint for m/l,
  within seed spread) and grows with model size at matched tokens.
  - EVIDENCE: e009, 20 oracle measurements (3 sizes x 5 checkpoints +
    seed pair at s; seed spread 0.01-0.05, largest at the 4%
    checkpoint), MDE ~0.4pp.
  - SCALE: SwiGLU decoder ladder on the e005 synthetic corpus, 6M
    tokens. Transferable claims are the trends, not the fractions.
  - INSTRUMENT CAVEAT (added 2026-08-16 after the retrodiction gate
    failed, [[arche-eretro-instrument-gate]]): this ladder's smallest
    rung (0.89M nonemb) sits BELOW the scale at which our harness
    reproduces a settled architecture ranking, and 3.07M sits just
    above it. The per-rung fractions at s are therefore in the
    untrusted zone for any cross-architecture reading. The claims this
    card actually makes are within-architecture trends across training
    and scale, which the gate's failure does not directly implicate —
    but the s rung should not carry weight on its own.
  - GRADE: strong at this setting; the growth-with-training direction
    kills the "redundancy is an undertraining artifact" reading in
    the opposite direction from the usual worry.
  - BREAKS-IF: the trend inverts on real text or at real scale (the
    LayerSkip Fig-11 datum says it won't).

- CORRECTION (2026-08-16, e009b verifier, independent re-measurement):
  the 80-88% figures are an UPPER BOUND, not a parity result. They are
  the union of per-(token, block) counterfactuals, each verified
  independently against the full model's hidden states. e009b executed
  the union in one masked forward pass. At l@6M the executed joint
  configuration agrees with the full model on **98.15%** of tokens and
  the implementation-faithful depth-only configuration on **89.93%**
  (verifier re-measured 0.9815 / 0.8993 from the same checkpoint with
  independent code, 256 eval sequences; skip fractions 0.880 / 0.808).
  The honest wording is "80-88% of (token, block) compute passes the
  per-pair skippability test; executing the whole union retains 98.2%
  argmax agreement". The measurement stands and is still large; the
  phrase "at argmax parity" does not.
  - RIDER: the joint arm's 98.2% is itself inflated. Its temporal
    substitutions repair the execution drift the depth skips cause —
    the gain is confined to mid-exit positions (0.34 -> 0.97 at l), it
    comes from the position's own reuse and not from context, and it is
    not a denoising effect (the injected value is more drifted than the
    one it replaces; noise or zeros at the same certified slots are
    worse than doing nothing). Both the slot choice and the injected
    value come from the clean model, so no realizable mechanism
    inherits it.
  - RIDER: the faithful arm loses the most. Depth-only skipping at
    these fractions is not capability-preserving in practice.

- CLAIM: The redundancy is COARSE-STRUCTURED, hence hardware-priceable:
  zeroing entire contiguous SwiGLU quarters keeps 0.97-1.00 end-to-end
  agreement at end of training (0.94-1.00 across the full grid) while
  per-unit dead activations are ~0.01. T3 (unpriceable
  scatter) is killed — there is no fine-grained ceiling for the coarse
  oracle to lose; the structure IS coarse.
  - GRADE: strong at this setting (criterion-mismatched comparison by
    necessity; labeled in CAVEATS).

- CLAIM: The criterion gap is qualitative, as [[gromov2024-deeper-layers]]
  warned: depth skippability 0.72-0.81 under argmax vs 0.26-0.45 under
  KL<=0.1. Any adaptive-compute claim must name its criterion.

- CLAIM: Temporal reuse — the axis no paper measures — is real, small,
  and growing: ~0 early, ~0.11 (argmax) by 6M tokens, under a
  conservative all-positions-substituted probe (lower bound).

- CLAIM: Linearly-predictable block transitions (ridge probe,
  rel-err<0.1) FALL with training at s (0.69→0.11 seed 0; 0.71→0.12
  seed 1) but stay high at l
  (0.65 at 6M tokens): bigger, less-trained-per-param models carry
  more predictable internal compute — the predictive-coding-style
  compute-on-mismatch number is scale-dependent in the exploitable
  direction.

## Data

artifacts/results.csv (full 20-row table), artifacts/results.png
(joint/depth-KL/temporal vs training tokens, 3 sizes). End-of-training
row (l, 6M): depth_arg 0.806, depth_kl 0.454, width_coarse 1.000/1.000,
width_fine 0.011, temporal 0.112/0.094, pred 0.654, joint 0.881.

## Untested assumptions

T2 (small-model-equivalence) is now DISCRIMINATED by
[[arche-e009b-t2-overlay]] (2026-08-16, 76 cells): the oracle ceiling
never reaches 2x the dense inference frontier (fit-free worst-seed lower
bound peaks at 1.68 across the whole grid), and the
implementation-faithful depth-only arm is below 1x at every rung above
the smallest. "Use a smaller model" is not merely a live reading — it
WINS at these scales. No mechanism-building is licensed by this card.
Synthetic corpus; argmax criterion is capability-relevant but not
distribution-faithful.

## Citation tree

- UPSTREAM: dossier 009; [[schuster2022-calm]], [[liu2023-deja-vu]],
  [[gromov2024-deeper-layers]], [[raposo2024-mixture-of-depths]],
  [[leviathan2022-speculative-decoding]].
- DOWNSTREAM: dossier 009 verdict (T3 killed, T1 PARTIAL — argmax
  survives, exact-distribution kill-disjunct fires on the per-axis
  proxy);
  0008 hardware thesis (coarse granularity is the priceable kind).

## Further work

(1) T2 overlay — DONE, [[arche-e009b-t2-overlay]]: the ceiling does not
reach 2x and the implementable arm is under 1x. (2) Real-text
replication of the trend direction — still open, and now the only
remaining reason to touch this line. (3) Mechanism candidates (MoD,
entropy-patching) were gated on (1) surviving; it did not, so they are
NOT licensed.

## Concepts

[[adaptive-computation]], [[conditional-computation]].

## Method summary

Own-harness SwiGLU ladder, 5 checkpoints/size; four post-hoc oracles
on held-out data: depth (CALM-style earliest-exit probe, argmax + KL),
width (coarse quarter-zeroing end-to-end; fine per-unit magnitude),
temporal (substitute block output at t with its own t-1 output),
prediction-error (closed-form ridge h_b→h_{b+1}, disjoint fit/score
halves). Joint = depth ∪ temporal per (token, block), positions 1:.

## Relevance

R5 (adaptive compute) is the first lever in the series to survive its
oracle test NON-deflated: the ceiling is large, grows with training
and scale, and sits at hardware-priceable granularity. The efficiency
story so far: objective reweighting (~1.05x), data selection (~1.0x),
multi-horizon (a tax) all deflated; skipping predictable COMPUTE did
not. Feeds MAP 4.5 and bet repricing on the adaptive-compute row.
