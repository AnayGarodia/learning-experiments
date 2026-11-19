# e010 — Growth experiment (dossier 010): design

Status: DESIGN 2026-08-16, after smoke, before full run.

## Harness
e005 corpus (pinned copy) + variable-depth tiny GPT (d=192). Size =
NON-EMBEDDING params (embeddings dominate at tiny scale and do not
grow). Growth operator: G_stack depth x2 (Du 2024's winner) — copy
blocks [0..L-1] to [0..2L-1]; arm variant shrink-and-perturb
(theta <- 0.6 theta + N(0, 0.01^2), Ash & Adams mitigation).

## Arms and grid
scratch L=4 and L=8 (baselines, 3 seeds each); grown L=4 -> 8 with
growth in {copy, shrink_perturb} x donor_tokens in {1M, 3M, 9M}
(donor-convergence sweep, G3's discriminator) x 3 seeds. 24 runs.
Donor checkpoints cached and shared across arms per seed.

## Accounting (the load-bearing choice)
FLOPs = 6 * nonemb_params * tokens; grown runs are CHARGED the
donor's FLOPs (Du's honest convention — the literature's 22%-to-4x
spread is mostly convention, so ours is fixed in advance).

## Measurements
Loss-vs-total-FLOPs curves per run. Headline: savings(horizon) =
FLOPs_scratch(loss) / FLOPs_grown(loss) - 1 at 3 loss horizons per
arm (the savings-DECAY curve); exponent-vs-prefactor from L(C) fits
with bootstrap CIs (G2 discriminator); savings-vs-donor-tokens (G3).

## Pre-registered predictions
- Savings positive at early horizons, decaying with horizon
  (replicated regularity; ~0.85).
- Donor-convergence sweep moves savings substantially; if it flips
  the SIGN, G3 (regime artifact) is live (~0.4 it flips at 9M donor).
- shrink_perturb >= copy at late horizons (~0.6).
- Exponent unchanged, prefactor improved (G2 over G1, ~0.55) — the
  interesting upset would be an exponent change.

## Budget
24 runs, ~10-20 min each on 4-core box = ~2-3h wall. arche-bee
(idle while e002 is under structural review); resumable if evicted.

## Analysis-stage finding (2026-08-16, at 21/24 runs, BEFORE any verdict)

analyze.py was written after the grid launched (it was a missing decision
0010 am.2 deliverable) and its instrument guards report that this grid
CANNOT answer two of the dossier's three pre-registered questions. Stated
now, before the last 3 runs land, so it cannot be mistaken for
post-hoc excuse-making:

1. **The savings-decay curve is not measurable here.** 16 of 18 grown
   savings cells (89%) are censored lower bounds. Grown arms inherit a
   trained donor, so they are already below almost every loss horizon at
   their FIRST eval; the grid never sees where they crossed. Two arms
   with the same donor budget consequently print IDENTICAL censored
   values, which superficially resembles the e006 oracle==anti alarm but
   is the censoring instead (uncensored cells do differ — see dt=1M at
   the late horizon). FIX (design, not seeds): dense evals immediately
   after the growth step, or compare at matched COMPUTE rather than
   matched loss.

2. **The exponent-vs-prefactor question (G1 vs G2) is not decidable
   here.** The grown and scratch_L8 families share only 1.2x of compute
   range. A power-law slope fitted over 1.2x of x-range measures
   curvature and noise, not an exponent. Deciding G1 vs G2 needs ~10x
   shared compute, i.e. longer scratch baselines or an earlier growth
   point. The per-arm alphas are retained in results.csv for the record
   and must NOT be compared across families.

3. **What the grid CAN still speak to**: the sign of savings at the one
   or two uncensored horizons, and the donor-convergence sweep (G3),
   since that compares grown arms against each other under identical
   censoring. Both are reported with their bounds marked.

Pre-registered guard honoured: the earlier full-curve exponent fits
(scratch_L8 alpha 0.105 vs grown 0.004-0.06) looked like a dramatic
exponent change and would have been reported as one. Restricting the fit
to the shared compute band collapses that difference, so the apparent
effect was an artifact of the two families sitting at different points of
the same decelerating descent. This is the same class of error as the
e005/e006 grid-snap artifacts.

## VERDICT (2026-08-16, 24/24 runs complete)

**Two of the three pre-registered questions are NOT ANSWERABLE with
this grid** (see the analysis-stage finding above, recorded at 21/24
before the last runs landed). The third is answerable and answered.

### What cannot be claimed
- **The savings-decay curve.** 16 of 18 grown savings cells are
  censored lower bounds. No decay shape may be read off this table.
  In particular the strongly negative early-horizon cells (-0.34 to
  -0.87) are lower bounds, so they do NOT establish that growth hurts
  early; the true values could be positive. The dossier's prediction
  ("savings positive at early horizons, decaying with horizon", ~0.85
  confidence) is UNTESTED here, not refuted.
- **Exponent vs prefactor (G1 vs G2).** The grown and scratch families
  share 1.2x of compute range. Nothing can be concluded about slopes.

### What can be claimed (matched-compute, censoring-free)
At equal TOTAL FLOPs with the donor's compute charged, five of six
grown arms end AHEAD of the same-size scratch baseline, by 0.003-0.005
nats against a seed spread of 0.0003-0.0012 (roughly 3-9 sigma). The
effect is small, consistent across the shared band, and orders with
donor budget for the plain-copy arm (dt1M -0.0029, dt3M -0.0031,
dt9M -0.0048 at the right edge). So growth is a real if modest win at
matched compute at the END of training.

**We deliberately do NOT convert this to a savings multiplier.**
Doing so requires the local slope of the scratch curve, and over a
1.2x band that slope is fragile; a naive conversion yields something
near 1.4x, which is the same order as the published ~2x claims, but
the error bar on that number is not something this grid can honestly
produce. Converting requires ~10x of shared compute. Named, not
claimed.

### G3 (regime artifact): one real signal
shrink-and-perturb at the LARGEST donor (dt9M) is the only arm that
starts BEHIND scratch (+0.0156 at the left edge) and merely converges
to parity (+0.0005) by the right edge. Plain copy at the same donor is
the BEST arm (-0.0048). So at a well-converged donor, the
shrink-and-perturb mitigation HURTS rather than helps — the opposite
of the pre-registered expectation (shrink_perturb >= copy at late
horizons, ~0.6 confidence). That prediction is LOST. Mechanistically
consistent with the mitigation being aimed at warm-start damage that
does not occur here: perturbing a good donor discards signal.
Caveat: one arm, one donor level, 3 seeds.

### Status
Dossier 010 is NOT closed. The measurable part gives a small positive
matched-compute result and one lost prediction; the two headline
questions need a redesigned grid (dense evals immediately after the
growth step; scratch baselines run long enough to share ~10x compute
with the grown family). Whether that redesign is worth buying is a
separate decision, and on this evidence the effect size it would be
chasing is small.
