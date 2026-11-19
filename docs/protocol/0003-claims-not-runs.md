# 0003 — The unit of contribution is a claim, not a run

Date: 2026-08-15
Status: accepted

## Context

Experiment volume will be high (fleet sweeps included). Raw runs pile up
and mean nothing on their own; long-horizon programs drown in unreadable
results.

## Options considered

1. Track runs (configs + metrics) and let interpretation live in people's
   heads or ad-hoc writeups.
2. Track claims: a result = a stated claim + the runs supporting it + the
   ablations that tried to kill it, all reproducible from the record alone.

## Decision

Option 2. Every run records exact code version, data version, settings,
seeds, and cost. Nothing is a result unless a stranger could recreate it
from the record. Negative results are recorded as claims too. Small-scale
results require multiple seeds and error bars. Our claims enter the library
in the same card format as the literature, graded `weak` until
independently reproduced.

## Evidence / reasoning

Same audit standard we apply to the literature must apply to us, or the
known-vs-believed ledger becomes self-serving. Reproducible-from-record is
the only standard that survives contributor turnover.
