# 0010 — Pre-launch protocol: pilot-with-analysis before any grid

Date: 2026-08-16
Status: accepted

## Context

Four experiments (e002 v1, e004r v1, e005 run 1, e006 run 1) produced
voided first runs. Post-mortem found one systemic cause: smoke tests
verified EXECUTION (code runs, speed) but never MEASUREMENT VALIDITY
(can the instrument see a plausible effect; do compared things differ;
is resolution finer than effect size). Secondary causes: analysis code
first executed only after the full grid (both grid-snapping artifacts
lived in analysis); experiment designs had a single reviewer while
cards require two.

## Decision

No grid launches until ALL of:
1. PILOT: two maximally-different cells, 1 seed, run end to end.
2. ANALYSIS-ON-PILOT: the full analysis pipeline runs on pilot
   output; the headline statistic must separate the two cells beyond
   its own resolution, and controls must order correctly (e.g., anti
   below random). Failure blocks launch.
3. MDE: DESIGN.md states the minimum effect size worth detecting and
   shows measurement cadence resolves it.
4. SPREAD: any selection/comparison mechanism prints the variance of
   what it compares in the pilot; ~zero spread blocks launch.
5. DESIGN REVIEW: a second reader reviews DESIGN.md adversarially
   ("how could this fake a result or produce a null artifact")
   before code is written, same standing as card audits.

Pre-registered tripwires (anchor arms, oracle-vs-anti, gates) remain
mandatory but are the LAST line of defense, not the first.

## Reasoning

Each voided run cost ~$1 of compute but a full launch-analyze-void
cycle of wall-clock, and e005's artifact (uniform 1.00x) was one
careless reading from entering the library as a believed false
negative. The protocol moves error-catching from after-the-spend to
before-it, at the cost of ~15 minutes per experiment.

## Amendment 1 (2026-08-16, from Anay): two-agent pipeline

Implementation of the protocol is a mandatory TWO-AGENT pipeline:
1. BUILDER agent writes DESIGN.md + code + pilot.
2. VERIFIER agent (independent context, adversarial brief) reviews
   the ENTIRE experiment before launch: re-derives the design logic,
   checks measurement validity (MDE, cadence, spread, controls,
   grid-snapping), runs the full analysis pipeline on pilot output,
   FIXES the errors it finds (committing fixes), and issues an
   explicit LAUNCH/BLOCK verdict with reasons.
3. Launch happens only on the verifier's LAUNCH verdict.
The verifier's report is part of the experiment record.

## Amendment 2 (2026-08-16): analysis code is a builder deliverable

No DESIGN.md is complete without its analyze script committed
alongside; the verifier MUST execute it on pilot output as part of
the pre-launch review. Rationale: both grid-snapping artifacts lived
in analysis, and e-retro's analysis did not exist until a mid-flight
verifier wrote it. Additionally, experiment CLOSURE requires two
deliverables before a result may be cited anywhere in the repo:
(1) a RESULTS CARD in library/cards/ (own-results get the same
rigor as external papers: exact numbers, scope limits, audit trail);
(2) the artifact pack per decision 0009 (figures, results.csv,
provenance, caveats). A belief-repricing pass (BELIEFS.md + the
relevant MAP ledger) accompanies every results card.

## Amendment 3 (2026-08-16, from the e012b verifier): card before rule

A verdict rule, gate, or pre-registered expectation whose trigger is
"the literature says X" MAY NOT be registered while X is an unverified
lead. The card for X must exist first, and the rule must quote the
scale the card actually records.

Cost of not having this: e012's gate and e012b's verdict rule both
rested on "the literature says a 2-layer transformer forms induction
heads, so a null is unmissable". When the Olsson 2022 and Elhage 2021
cards were finally written — by the verifier, after the null — the
papers turned out to place the effect at 1e9–5e9 tokens, d_model 768,
context 8192, on natural text, in models that deliberately keep
positional information OUT of the residual stream. Every one of those
is a regime we were not in. e012 stopped on that gate. e012b was one
step from entering "our instrument is blind, five results are void"
into the record. Writing the card first would have cost twenty
minutes and prevented both.

Companion rule, same cost: **a corpus that claims to require a
specific mechanism must print the evidence that it does.** e012b's
`[block][block]` corpus has exactly one repeat offset in every window,
so every scored token was reachable by a constant positional shift and
no content matching was needed. The design asserted a 1-layer
impossibility that does not hold for a fixed lag. One line of
measurement over the generated batch would have shown it before the
design review, not after the pilot.

Both go in the same place as the existing items: they are checks the
BUILDER runs and the VERIFIER re-runs.
