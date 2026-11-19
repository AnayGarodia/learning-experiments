---
title: "arche e011 — multi-horizon prediction is a tax at tiny scale, not a shortcut to long-range structure"
authors: [arche lab]
year: 2026
venue: internal experiment (pre-registered, experiments/e011-horizons/DESIGN.md)
url: experiments/e011-horizons/
questions: [q4-sample-efficiency, q1-scaling]
status: carded
carded_by: fable (closure agent, 2026-08-16)
---

## Claims

- CLAIM: On a corpus with a planted long-range dependency in every
  sequence (long_density 1.0), plain next-token training reaches the
  long-range criterion (acc_long >= 0.90) STRICTLY FASTER than
  multi-horizon or annealed-horizon training. All 3 next_token seeds
  cross at 655k tokens; all 3 multi_horizon and all 3 annealed seeds
  cross at 1.31M.
  - EVIDENCE: e011, 24 runs (4 arms x 2 densities x 3 seeds), 6M
    tokens each.
  - GRADE: the SIGN is strong (6-0 seed agreement); the MAGNITUDE is
    weak. The effect is exactly one eval-grid step (655k tokens), so
    the instrument bounds the ratio only as >1. The 2.0x point
    estimate must not be quoted as a measured multiplier.
  - BASELINE: next_token, same model/data/seeds/recipe.

- CLAIM: mtp4 (4-token multi-token prediction) is NOT distinguishable
  from next-token at matched DATA — 2 of 3 seeds tie at 655k, one
  crosses at 1.31M. It is distinguishable at matched COMPUTE, where
  the auxiliary heads cost a deterministic 1.90x per token.
  - GRADE: strong (the compute cost is arithmetic, not measured).
  - NOTE: an earlier draft of this verdict quoted the seed MEAN
    (874k, "0.75x"), which concealed the split. Per-seed values are
    the reportable object.

- CLAIM (accounting): under FLOP-equivalent accounting every
  multi-horizon variant loses: next_token 1.00x, mtp4 1.90x,
  multi_horizon and annealed 3.80x tokens-to-criterion.

- NON-CLAIM: at long_density 0.25 all four arms cross on the SAME grid
  point. That column is under-resolved and yields no ordering.

## Data

artifacts/results.csv (both accountings, per cell), artifacts/results.png
(long-range accuracy vs training tokens, median of 3 seeds, per density).
Per-seed crossings at ld=1.0 (tokens to acc_long >= 0.90):
next_token 655k/655k/655k; mtp4 1.31M/655k/655k;
multi_horizon 1.31M x3; annealed 1.31M x3.

## Untested assumptions

Tiny models, 6M tokens, synthetic planted dependency. The eval grid
(655k tokens, 11% of a run) is too coarse to size the effect; a finer
grid is required before any multiplier is claimed, and is the only
thing that would make the ld=0.25 column interpretable.

## Citation tree

- UPSTREAM: [[gloeckle2024-multi-token-prediction]] (reports MTP paying
  only above ~3B params — our tiny-scale loss is CONSISTENT with that
  and does not contradict it).
- DOWNSTREAM: the objective-side deflation series (e005 reweighting
  ~1.05x, e006 selection ~1.0x, e011 a tax) — three independent
  levers on the loss/data side, none of which paid.

## Further work

Finer eval grid to size the effect and resolve ld=0.25. Any scale-up
claim needs the retrodiction gate, since the published effect is
reported to reverse sign above ~3B.

## Concepts

[[next-token-prediction]], [[multi-token-prediction]].

## Method summary

Four training objectives (next_token; mtp4; multi_horizon over
horizons 4/16/64; annealed schedule over the same horizons) on a
synthetic corpus with a planted long-range dependency at two
densities. Criterion: tokens to reach 0.90 accuracy on the long-range
probe, reported under matched-data and matched-compute accounting
(auxiliary head FLOPs charged in the latter).

## Relevance

q4: closes the "predict further ahead to learn long-range structure
faster" lever at this scale. Next-token loss already carries the
long-range signal when the data contains it. Feeds the standing
pattern: efficiency reforms on the loss/data side keep deflating under
ground truth, while compute-side redundancy (e009) did not.
