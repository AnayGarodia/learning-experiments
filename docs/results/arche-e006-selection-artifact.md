---
title: "arche e006 run 1 — methods: batch-level selection over iid candidates is a null by construction (oracle==anti diagnostic)"
authors: [arche lab]
year: 2026
venue: internal experiment (dossier 006, run 1 voided; v2 in flight)
url: experiments/e006-selection/
questions: [q4-sample-efficiency]
status: carded (PROVISIONAL — v2 example-level results pending)
carded_by: fable (closure agent, 2026-08-16)
audited_by: fable (verifier, 2026-08-16 — matches run-1 aggregation; provisional status correct pending v2)
---

## Claims

- CLAIM (methods): Selecting among K iid batches drawn from one
  generator gives candidates with ~zero between-candidate variance, so
  every selection policy degenerates to random-plus-scoring-tax; the
  oracle==anti tie is the diagnostic signature of this null.
  - EVIDENCE: run 1 (36 runs): oracle, anti, and self-uncertainty all
    exactly 0.50x vs random at every redundancy level (dup_p 0, 0.5,
    0.9); the 0.50x is the charged scoring overhead plus eval-grid
    snapping; final losses: random 0.0041-0.0044 vs selective
    0.0114-0.0117 (budget spent on scoring, fewer train steps).
  - GRADE: strong (true by construction once seen; a working harness
    can never produce best-picker == worst-picker).
  - BREAKS-IF: n/a (methods claim).

- CLAIM (pending): whether example-level selection over a
  heterogeneous pool pays at honestly-charged compute is OPEN — v2
  (pool of 4xBATCH sequences scored per-sequence, eval cadence 30 FE)
  is in flight.

## Data

Run 1, cost-to-struct_loss<=0.05 in forward-equivalents (train step =
3 FE, scoring pass = 1 FE, all charged): random 450; oracle = anti =
self_uncertainty = 903, identically, at all dup_p. Speedup 0.50x for
every selective arm.

## Untested assumptions

That per-sequence scores carry usable signal at this corpus's
difficulty (v2 tests it); that dup_p is the right redundancy dial for
selection to exploit.

## Citation tree

- UPSTREAM: dossier 006; [[mindermann2022-rho-loss]] (the uncharged-
  scoring trap this harness avoids), [[dasgupta2005-splitting-index]].
- DOWNSTREAM: e006 v2 results (will supersede this card's pending
  claim); decision 0010 (this run is one of the four voided runs that
  motivated the pilot-with-analysis protocol).

## Further work

v2 analysis on completion; if v2 also nulls on the clean corpus, the
S2 (noise-cleanup-only) reading gains and the redundancy dial gets
stress-tested before any verdict.

## Concepts

[[active-learning]].

## Method summary

Same corpus/model as e005. Policies pick training data each step;
scoring compute charged to the same budget as training. Run 1 selected
among 4 iid BATCHES (the flaw); v2 selects the best BATCH-worth of
individual sequences from a 4x pool. Anti-selection is the mandatory
direction-flip control; its tie with oracle voided run 1 before
interpretation.

## Relevance

q4: keeps the data-selection lever honestly priced; the oracle==anti
tripwire and the charged-scoring accounting are reusable methods
standards for every future selection experiment.
