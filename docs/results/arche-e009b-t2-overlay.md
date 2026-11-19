---
title: "arche e009b — T2 discriminator: the post-hoc adaptive-compute ceiling never reaches 2x the dense inference frontier, and the implementable arm is worse than picking a smaller model"
authors: [arche lab]
year: 2026
venue: internal experiment (pre-registered dossier 009, decision 0010 am.1 two-agent pipeline)
url: experiments/e009b-t2-overlay/
questions: [q1-scaling]
status: carded
carded_by: fable (verifier, 2026-08-16)
audited_by: fable (verifier agent, independent context — pre-launch review of the pilot found the post-pilot fit amendment was verdict-changing and replaced fit-dependent interpolation with a fit-free bracket; re-measured the execution finding from checkpoints with code that does not import overlay.py; refuted the proposed anomaly mechanism; found and fixed a seed-pooling defect in analyze.py during final analysis. Full ruling in experiments/e009b-t2-overlay/DESIGN.md.)
---

## Claims

- CLAIM (headline, weakest supported form per decision 0012 clause 5):
  **the post-hoc joint oracle ceiling never reaches 2x the dense
  inference frontier at matched capability.** Across 5 training levels x
  2 seeds x 2 capability axes, the fit-free worst-seed lower bound on the
  multiplier M = F_frontier(C_oracle)/F_oracle peaks at **1.68**, against
  a pre-registered kill line of 2.0. No level, seed, axis, frontier fit or
  FLOPs accounting puts it at 2x.
  - EVIDENCE: e009b, 76 measurement cells (8 rungs 0.049M-24.8M nonemb,
    5 checkpoints, 2 seeds), 256 held-out eval sequences per cell.
  - GRADE: strong at this setting. The oracle is post-hoc and sees the
    full model's own logits, so this is a CEILING: no realizable router,
    gate or predictor can exceed it.
  - BREAKS-IF: a corpus whose capability axis does not saturate, or a
    frontier resolved finely enough at the small end to move the bracket.

- CLAIM: **the verdict is formally INCONCLUSIVE because the two primary
  capability axes disagree**, and this is reported rather than resolved.
  At the primary level (6.0M tokens): next-token accuracy gives worst-seed
  bracket [0.78, 1.43] (KV-floor 1.36) -> T2 SURVIVES; consequence
  accuracy gives [1.03, 1.61] (KV-floor 1.50) -> the two FLOPs accountings
  straddle the 1.6 survive line and the pre-registered disagreement clause
  fires. The consequence axis misses SURVIVES by 0.01.
  - The point fits at that rung are 1.15-1.18, so the multiplier is very
    probably ~1.2; 1.61 is the BOUND the rule uses and the rule was not
    relaxed after seeing it.

- CLAIM: **the implementation-faithful arm is worse than model selection
  at every rung above the smallest.** Depth-only skipping (CALM-style
  state copy — the arm a real system could build) at 6.0M tokens gives
  consequence-accuracy multipliers of 1.19/1.19 (xxs), 0.83/0.75 (xs0),
  0.68/0.64 (xs1), 0.62/0.59 (xs), 0.32/0.32 (s), 0.16/0.16 (m),
  0.075/0.076 (l) by seed. A smaller dense model is strictly cheaper at
  matched capability.
  - GRADE: strong. This is the number that decides whether to build the
    mechanism, and it is below 1.

- CLAIM: **the union of independently-verified skips is not jointly
  valid.** Executing all skips at once, at 6.0M tokens: joint retains
  0.943-0.982 argmax agreement with the full model and depth-only
  0.873-0.912. e009's 80-88% is an upper bound, not a parity result.
  Consequence accuracy under depth-only falls from ~1.000 to 0.54-0.67.
  - Independently re-measured by the verifier from the same checkpoints
    with a reimplementation that does not import overlay.py: l@6M
    depth-only 0.8993, joint 0.9815 (matching to 4 decimals).

- CLAIM: **the joint arm's high agreement is self-repair, and no
  realizable mechanism inherits it.** Joint executes BETTER than
  depth-only despite skipping strictly more. The gain is entirely on
  mid-exit positions (0.34 -> 0.81 at s, 0.34 -> 0.97 at l), entirely
  from reuse applied to the position's OWN state (reuse off for mid-exit
  positions: no gain at all; reuse only for them: the whole gain), and is
  NOT a drift regulariser — RMS-renorming the frozen depth state changes
  agreement by 0.0000, the injected value is MORE drifted in L2 than the
  one it replaces, and a clean-source substitute is no better. At the same
  certified slots, injecting noise (0.17), zeros (0.19) or the passthrough
  state (0.18) is WORSE than doing nothing (0.35); only the genuine t-1
  value works (0.81), t-2 partially (0.45).
  - MECHANISM: the depth certificate is a claim about the CLEAN state at
    the exit block and execution destroys that state; the temporal
    certificate is a claim about a SUBSTITUTION and execution performs
    exactly it. Applying the reuse discards the position's own drifted
    computation and overwrites it with a value the clean model verified.
    **The joint arm repairs damage the depth arm causes, using slot
    choices and values only the clean model can supply.** The 98.2% is
    therefore inflated relative to any implementable scheme and must not
    be read as "80% of compute is free".
  - Direction: generous to the oracle, hence conservative for a
    T2-survives reading — it does not threaten the result's direction.

- CLAIM: **the regime claim is NOT supported.** "The adaptivity premium
  exists in the undertrained regime and vanishes at convergence" met 2 of
  the 4 pre-registered conditions. The multiplier is non-monotone in
  training (the consequence-axis point estimate peaks at 1.49M tokens,
  4.15/2.97 by seed, not at the least-trained level), and on the
  bootstrapped fit-free bracket the first four levels are statistically
  flat with a single drop at the last. One transition is not a monotone
  fall, and it lands exactly where the dense frontier saturates on this
  corpus — so "convergence" and "corpus exhaustion" remain the same axis.

- CLAIM (structural, the transferable part): **the unembedding is a floor
  no adaptive-compute scheme can skip, and it binds hardest exactly where
  the dense frontier is cheapest.** Head share of inference FLOPs runs
  66.1% at xxs, 54.6% at xs0, 47.3% at xs1, 44.9% at xs, 30.7% at s,
  11.1% at l. At the saturated frontier the cheapest capable rung is
  small, so even a 100%-block skip caps M at 1/head_share ~ 2.1-2.2
  there. Reporting skip fractions against block compute alone (as e009
  did, correctly for its question) overstates end-to-end saving.

## Data

`artifacts/results.csv` (380 rows: 76 cells x 5 configs),
`artifacts/results.png`, `artifacts/CAVEATS.md`. Verifier re-measurements
and probe scripts in `pilot/verifier/`.

Fit-free worst-seed KILL-side lower bound by level (acc / cons):
229k 1.23/1.68, 590k 1.42/1.44, 1.49M 1.51/1.59, 3.0M 1.23/1.21,
6.0M 0.78/1.03. Kill line 2.0.

## Untested assumptions

Single corpus (decision 0012 clause 3): everything here is a claim about
the e005 synthetic corpus, whose accuracy ceiling ~0.838 every rung above
xs1 reaches by end of training. Six of eight rungs, and all three
verdict-bearing rungs (xs0/xs1/xs, 0.13-0.30M nonemb), sit below the ~3M
non-embedding scale floor e-retro measured (decision 0012 clause 4). This
is a within-family comparison so the cross-architecture prohibition does
not bite, but the frontier's SHAPE at the small end is exactly what the
verdict rests on and e-retro says this harness is untrustworthy about
small-rung orderings — **this is the primary threat to the result.**
Block-level granularity only; CPU; argmax and KL<=0.1 criteria only.

## Citation tree

- UPSTREAM: dossier 009 (T2); [[arche-e009-redundancy-oracle]];
  [[schuster2022-calm]], [[raposo2024-mixture-of-depths]],
  [[leviathan2022-speculative-decoding]], [[gromov2024-deeper-layers]].
- DOWNSTREAM: dossier 009's T2 row; PORTFOLIO adaptive-compute row;
  decision 0012 clause 1 (this run is a datum for the identifiability
  dry-run requirement — the design is still only marginally able to
  separate SURVIVES from INCONCLUSIVE at the primary level).

## Further work

(1) A non-saturating corpus is the only way to separate "convergence"
from "corpus exhaustion" and the only way the regime question becomes
answerable. (2) Nothing in the mechanism program (MoD, entropy-patching)
is licensed by this result: the ceiling that a mechanism would chase is
under 2x, and the arm a mechanism could actually implement is under 1x.
(3) If the question is revisited, the frontier needs resolving between
xs0 and xs1 at the primary level, not more seeds — the bracket there is
structural, not statistical.

## Concepts

[[adaptive-computation]], [[conditional-computation]].

## Method summary

Extended e009's SwiGLU ladder to 8 rungs (0.049M-24.8M nonemb) on the
same pinned corpus and recipe; 5 checkpoints, 2 seeds. Per cell, one
masked forward per oracle configuration applies ALL of that
configuration's skips simultaneously and capability is MEASURED on the
executed model, not assumed from per-pair counterfactuals. Inference
FLOPs counted as matmul-only with the unembedding head charged and
unskippable; skipped passes charged zero (pre-registered) and, alongside,
charged the 4d^2 KV floor they still owe (a skipped position must still
supply keys and values to later positions). Dense frontier is the Pareto
envelope within a seed; M = F_frontier(C_oracle)/F_oracle evaluated on
the FIT-FREE BRACKET between adjacent measured frontier rungs, so no
interpolation choice can move the verdict. Seeds combine conservatively
(a kill must hold on the worst seed; so must a survive). Controls at
every cell: random matched-density mask, wrong-neighbour (t+1) reuse;
0 ordering violations in 76 cells.

## Relevance

R5 (adaptive compute) closes deflationary, like the rest of the
efficiency series. e009 made it the first non-deflated lever; e009b is
the discriminator e009's own card named as gating, and it does not
license the mechanism program. The efficiency ledger is now: objective
reweighting ~1.05x, data selection ~1.0x, multi-horizon a tax, growth
~0.004 nats, adaptive compute ceiling <2x with the implementable arm
<1x. Feeds PORTFOLIO's adaptive-compute row and the B3 qualifier.
