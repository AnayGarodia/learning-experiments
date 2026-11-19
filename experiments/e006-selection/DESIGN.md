# e006 — Selection-regime experiment (dossier 006): design

Status: DESIGN 2026-08-16, after smoke, before full run. Shares the
e005 corpus/model (copied, pinned). Cost unit = forward-equivalents
(train step = 3 FE, scoring a candidate batch = 1 FE) so scoring is
CHARGED — the RHO-LOSS uncharged-scoring trap is named and avoided.

## Arms
Policies: random | self_uncertainty (highest own loss over all
tokens) | oracle (highest loss on learnable classes only — ground
truth) | anti (lowest oracle score; mandatory direction-flip control
per fan2023). Each picks 1 of 4 candidate batches per step.
Grid: 4 policies x dup_p (redundancy dial) {0, 0.5, 0.9} x 3 seeds =
36 runs at budget 1500 FE.

## Measurements
Held-out structure loss + consequence accuracy vs TOTAL FE spent.
Headline: FE_random / FE_policy to matched thresholds, per dup_p.

## Pre-registered predictions (dossier S1/S2/S3)
- S1: oracle >= 2x at dup_p 0.9, gains ordered oracle >
  self_uncertainty > random, present even at dup_p 0 (~0.35).
- S2: all policies tie at dup_p 0; gains only from dup dodging (~0.4).
- S3: oracle wins but self_uncertainty ~ random everywhere (~0.25).
- anti must UNDERPERFORM random; if it does not, the harness is
  broken (this is a gate, checked before interpretation).

AMENDMENT 1 / v2 (2026-08-16, after run 1): run 1 returned oracle ==
anti == self_uncertainty == 0.50x exactly — selecting over CANDS iid
batches from one generator gives candidates with ~no between-candidate
variance, so every policy degenerates to random-plus-scoring-tax. The
tie between oracle and ANTI is the diagnostic (a working harness can
never produce it). Run-1 licenses only: batch-granularity selection
over homogeneous candidates buys nothing (true by construction).
v2: example-level selection — pool of CANDS*BATCH sequences scored
PER SEQUENCE, keep the best/worst BATCH; eval cadence 150->30 FE
(run-1's threshold crossings also snapped to the eval grid). Same
policies, dials, honest FE charging.
