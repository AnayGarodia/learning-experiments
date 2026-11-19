# 0002 — One shared instrument; candidates are plugins

Date: 2026-08-15
Status: accepted

## Context

Many candidate architectures will be tested by many people over a long
period. Results must be comparable across candidates, people, and time.

## Options considered

1. Each experiment builds its own training/eval setup.
2. One shared training and measurement pipeline; each candidate
   architecture is one module implementing a small fixed interface.

## Decision

Option 2. Comparisons are only meaningful when everything except the idea
under test is held fixed: same frozen data, same input scheme, compute-
matched ladders, scaling-law fits, tuned transformer baseline at every
rung. The instrument is versioned; results across instrument versions are
never compared without a bridge run.

## Evidence / reasoning

The most common failure in architecture papers is comparing against an
untuned or mismatched baseline; per-setup drift makes results incomparable
and unauditable. Curve-level comparison (exponents, offsets) is required
because single-point comparisons say nothing about scaling.

## Consequences

The instrument has an owner and strict review. A per-architecture recipe-
tuning step at small scale is part of the instrument spec (D8), since a
recipe tuned for transformers handicaps other families.
