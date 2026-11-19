# Dossier 010 — Growth: the savings-decay curve and what growth changes

Status: PRE-REGISTERED 2026-08-16, after prior-art (9 cards). Attacks
training-cost directly. Redundancy check: growth WORKS (~50%
donor-included savings at 3-7B, [[du2024-stacking-transformers]];
2.2-4x token savings, [[samragh2024-hypercloning]]) and scaling laws
HAVE been fit to grown families three times ([[shen2022-staged-training]],
Du's schedule regressions, [[liew2025-upcycling-scaling-laws]]) — the
"treat growth as a scaling law" framing is NOT novel. The honest
unclaimed residue, which 010 measures:
(a) the SAVINGS-DECAY CURVE as the object of study — the one
    regularity replicated across three independent setups (37%->6%,
    53%->31%, Liew's interaction term) but never measured as a curve
    with error bars;
(b) exponent vs prefactor: does growth change the scaling exponent or
    only the constant? Never separated — and it decides whether
    growth compounds (frontier-relevant) or is a fixed discount;
(c) donor-convergence as a CONTROLLED variable: warm-start damage
    ([[ash2020-warm-start]], -2 to -4.5 acc, shrink-and-perturb fixes
    it) is proven exactly in the small-scale converged-donor regime a
    compute-poor ladder lives in, and invisible in the underfit LLM
    regime — the INVERSE of the usual small-scale trap;
(d) muP x growth: no paper combines them (one lead, unverified).

## Rival mechanisms

**G1 — Compounding ladder.** Growth changes the effective prefactor
repeatedly and re-growing stays profitable: a multi-hop ladder with
tuned timing retains >=30% donor-included savings per hop at our
scales, and the decay curve flattens with shrink-and-perturb +
schedule tuning.
- KILLED-BY: savings per hop falling below 10% by hop 2 under every
  mitigation tried, or the decay curve matching the untuned published
  shape regardless of controls.

**G2 — Fixed discount.** Growth is a one-time prefactor cut (~2x at
best), exponent unchanged, decaying to nothing with budget; useful,
not a scaling lever.
- PREDICTS: exponent identical between grown and scratch families
  (within CI); savings-decay hits ~0 at the compute-optimal budget
  for each size.

**G3 — Regime artifact.** Small-scale grown-ladder results are
dominated by warm-start damage (or its absence) and do NOT transfer
directionally: measured savings depend mostly on donor convergence
fraction, not growth mechanics.
- PREDICTS: sweeping donor convergence from 0.3x to 3x
  compute-optimal flips the measured savings sign; KILLED-BY: savings
  stable (+/-10%) across the donor-convergence sweep.

## Design commitments (DESIGN.md + walkthrough before run)

Tiny GPT ladder (~2M -> 8M -> 32M params) on the e005 harness's
corpus family; depth-x2 one-hop growth (Du's winner) with
function-preserving init; shrink-and-perturb arm vs plain-copy arm;
scratch baselines at every size; donor-convergence sweep {0.3, 1.0,
3.0}x size-matched compute-optimal tokens; fit L(C) per family and
report exponent and prefactor separately with bootstrap CIs; savings
measured at 3 budget horizons per hop (the decay curve). 3 seeds.
CPU (arche-lab-2). muP x growth held as a named follow-up.

## Why it matters

If G1: a compute-poor lab's route to big models is a LADDER — train
small, grow, repeat, every rung reusing the last — the "frontier at
1/1000th" thesis in its most literal form. If G2: take the one-time
~2x discount and move on. If G3: publish the trap (small-scale growth
results are regime artifacts) — a service to the field and a warning
for our own instrument.

## What breaks at scale

Transferable claims: the exponent-vs-prefactor verdict and the
donor-convergence sensitivity. Absolute savings percentages are not.
Direction pre-stated per rung; retrodiction gate governs
extrapolation.
