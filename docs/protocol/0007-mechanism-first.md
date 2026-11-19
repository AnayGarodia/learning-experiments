# 0007 — Mechanism-first: experiments discriminate explanations, not benchmark performance

Date: 2026-08-15
Status: accepted

## Context

Experiments in this field are expensive and arche's compute will be
small, though unconstrained by revenue (the moonshot advantage: we can
take 100-to-1 bets a frontier lab cannot justify). Meanwhile S1
established that small-scale PERFORMANCE numbers mislead: architecture
rankings flip with compute region (Tay), and objective benefits reverse
with scale (multi-token prediction hurts <1B, helps at 7B+). Anay's
standard: "you need a concrete reason WHY it fails, a concrete
first-principles understanding of why something works" — a regularity
like "fails below 1B" is not knowledge.

## Decision

1. Experiments are designed to DISCRIMINATE between candidate mechanisms,
   not to measure performance. Metric: information per FLOP.
2. No empirical regularity is actionable until at least one concrete
   mechanism is proposed for it, with the prediction that separates it
   from rivals stated in advance.
3. Every proposed mechanism states its kill-test (the observation that
   would falsify it) at proposal time, and the kill-test gets run.
   Mechanism-first without kill-tests degenerates into theory-love.
4. S3 (the predictor) is upgraded: a claim graduates only when mechanism
   and measurement agree — mechanism proposed, cheap discriminating
   experiment run, curve confirmation last. Retrodiction remains
   necessary but is no longer sufficient.

## Reasoning (DERIVED)

- Mechanisms are testable at small scale even when performance is not:
  a correct mechanism licenses extrapolation to scales never trained
  ("prediction without purchase"). This is the only scientific strategy
  available to a compute-poor lab, and it is also simply better science —
  the field's curves-without-mechanisms (Kaplan) could not even get the
  params/data split right.
- The compute-poor position and the mechanism-first method are the same
  fact viewed twice: we cannot afford sweeps, therefore every run must
  kill or crown an explanation.

## Example of the standard (for onboarding)

"MTP is worse at small model sizes" (audited wording; the paper names no
exact crossover) spawns enumerable mechanisms: (a) heads steal
scarce trunk capacity → penalty shrinks smoothly with trunk size, and
smaller heads help small models; (b) far-token signal too noisy for
small models → penalty tracks the model's measurable uncertainty
horizon; (c) benefit piggybacks on copying machinery small models
haven't formed → crossover coincides with a measurable internal event.
Three mechanisms, three cheap discriminating tests, dollars each.
