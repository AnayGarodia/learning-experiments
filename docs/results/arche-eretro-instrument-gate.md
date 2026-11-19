---
title: "arche e-retro — the retrodiction gate FAILS at ~1M params: our harness ranks LSTM above transformer at its smallest rung"
authors: [arche lab]
year: 2026
venue: internal experiment (pre-registered, experiments/e-retro/DESIGN.md)
url: experiments/e-retro/
questions: [q1-scaling, s3-predictor]
status: carded
carded_by: fable (closure agent, 2026-08-16)
---

## Claims

- CLAIM: At ~1M non-embedding parameters on byte-level enwik8, a
  2-layer LSTM beats a matched-parameter transformer by 0.130
  nats/byte, with BOTH architectures tuned to an interior learning-rate
  optimum. This trips the pre-registered FAIL condition of the lab's
  own instrument-validation gate.
  - EVIDENCE: best-LR final val loss, mean of 2 seeds — LSTM 1.5118
    (lr 3e-3), transformer 1.6415 (lr 6e-3).
  - GRADE: strong. Both optima are bracketed: transformer 1.6848 at
    3e-3 -> 1.6415 at 6e-3 -> 1.7410 at 1e-2; LSTM 1.5860 at 1e-3 ->
    1.5118 at 3e-3 -> 1.5170 at 6e-3 -> 1.6115 at 1e-2.
  - BASELINE: matched NON-embedding params (embeddings and head
    excluded on both sides), identical data, budget, optimizer, batch,
    sequence length and eval protocol.

- CLAIM: The failure is a CROSSOVER, not noise. The transformer wins
  the two larger rungs and its margin grows: r2 (~3M) by 0.040, r3
  (~9M) by 0.050. The instrument recovers the correct ordering TREND
  while inverting the ranking at its smallest rung.
  - GRADE: strong on the sign at each rung; the trend is 3 points.
  - NOTE: this matches the historical record, where the transformer's
    documented enwik8 advantage was established at far larger scale.

- CLAIM: The two candidate confounds are both closed.
  (a) LEARNING RATE: extending the sweep to {6e-3, 1e-2} improves the
  transformer at r1 and narrows the gap from 0.173 to 0.130, then
  degrades — it does not close it.
  (b) LSTM STATE CARRY: the design's named deviation (independent
  256-byte windows) handicapped the LSTM in principle. Carrying state
  across windows makes it marginally WORSE (r1 1.5131 vs 1.5118; r2
  1.4051 vs 1.3988), so the deviation was not load-bearing.
  - GRADE: strong. These were the pre-registered conditioned
    follow-ups and both were run before the verdict was written.

- CONSEQUENCE (the reason this card exists): no architecture-RANKING
  claim from a ~1M-parameter rung of this harness is trustworthy on its
  own. Claims about mechanism, about the SIGN of an intervention within
  a single architecture, and about trends across rungs are not directly
  implicated, but inherit the general caution.

## Data

Best final val loss (nats/byte), mean of 2 seeds, each arch at its own
best LR:

| rung | LSTM | transformer | winner | gap |
|---|---|---|---|---|
| r1 (~1M) | 1.5118 | 1.6415 | LSTM | 0.130 |
| r2 (~3M) | 1.3988 | 1.3593 | transformer | 0.040 |
| r3 (~9M) | 1.3369 | 1.2871 | transformer | 0.050 |

Seed spread within a cell is ~0.001-0.02 nats, far below every gap
above. 56 of 68 records at verdict time; the outstanding 12 are hot-LR
cells at r2/r3 which can only widen the transformer's wins there.

## Untested assumptions

Byte-level enwik8 only; 2 seeds per cell; three rungs. The LSTM is
`nn.LSTM` and the transformer is the e005 block — neither is anyone's
optimized best, so the claim attaches to these implementations under
this recipe (the Bartunov "as specified" caveat). Whether the crossover
point is a property of scale, of this data, or of these two specific
implementations is NOT established by three rungs.

## Citation tree

- UPSTREAM: FOUNDATION.md step S3 (a validated small-to-large
  predictor must retrodict transformer-beats-LSTM); the historical
  enwik8 record (Al-Rfou 2018 transformer ~1.06 bpc vs LSTM-line
  ~1.23+), documented at far larger scale than any rung here.
- DOWNSTREAM: every arche result measured at or below ~1M non-embedding
  params. Specifically [[arche-e009-redundancy-oracle]], whose ladder
  (0.89M / 3.07M / 9.44M) straddles this crossover, so its smallest
  rung sits in the untrusted zone.

## Further work

(1) Locate the crossover more precisely with rungs between 1M and 3M —
that number is the lab's usable-scale floor for ranking claims and is
worth knowing exactly. (2) Test whether the crossover moves with
context length, the mechanism most likely to drive it. (3) Re-run the
gate whenever the harness changes materially.

## Concepts

[[retrodiction]], [[scaling-laws]], [[instrument-validation]].

## Method summary

Matched non-embedding params across two architecture families, three
rungs, coarse-then-extended LR sweep per (arch, rung), 2 seeds,
identical data and protocol; verdict rule (PASS / PARTIAL / FAIL)
written before any run and applied as written.

## Relevance

This is the lab's own instrument under test, so it outranks any single
discovery result. It converts a vague standing worry ("small-scale
results may not transfer") into a specific, quantified boundary: at
~1M parameters this harness gets a settled architecture comparison
backwards, and it recovers the right answer by ~3M. Bet B7
(predictability is the product) is directly engaged: a predictor that
inverts a known ranking at its cheapest rung is not yet the artifact
B7 claims is worth more than any single result. The honest reading is
that the gate did its job — it caught the problem before the problem
reached a conclusion.
