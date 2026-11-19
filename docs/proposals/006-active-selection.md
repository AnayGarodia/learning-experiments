# Dossier 006 — Learner-chosen data: which regime is language in?

Status: PRE-REGISTERED 2026-08-16, after the shared prior-art pass
with 005 (11 cards) and before any experiment. Attacks R6 and prices
the substrate bet's "chosen experience is the ceiling" half.

## The question

Active selection has PROVEN regimes: exponential label savings when
the hypothesis geometry splits at constant rate, provably zero gain
in adversarial geometries, and the SAME hypothesis class can be
either depending on the target ([[dasgupta2005-splitting-index]]);
in the perceptron, exponential scaling needs abundant data + a
high-quality selection metric, and with scarce data keeping hard
examples is WORSE than random ([[sorscher2022-data-pruning]]).
Empirically, LM-scale gains exist but are modest and noise-dependent
(random's performance at 43% of FLOPs, [[yu2024-mates]]; 18x on noisy
data vs 1.4-2x on clean, [[mindermann2022-rho-loss]]). Nobody has
mapped WHICH regime language-like structure occupies with ground
truth in hand. Same synthetic corpus as 005, so the harness is shared.

## Rival mechanisms

**S1 — Splittable regime.** Language-like structure (hierarchical
grammar + consequential forks) sits in the favorable geometry:
learner-driven selection buys ≥2-3x at matched TOTAL compute
(scoring charged), growing with corpus redundancy.
- PREDICTS: gains scale with the redundancy dial; oracle-influence
  selection > self-uncertainty > random, all direction-stable across
  a 3-rung ladder.
- KILLED-BY: ≤1.3x at matched compute on clean corpus at every
  redundancy setting.

**S2 — Noise-cleanup only.** Selection's real function is avoiding
garbage; on a clean corpus active ≈ random (the Mindermann clean-data
result generalizes).
- PREDICTS: gains appear only when the noise dial is up; at zero
  noise all policies tie.
- KILLED-BY: ≥2x gains on the clean corpus.

**S3 — Deflationary external-knowledge.** Gains require a smuggled
target (curated reference); the learner's OWN uncertainty signal buys
~nothing (consistent with static-signal decay in [[yu2024-mates]]).
- PREDICTS: oracle-influence arm wins but self-uncertainty ≈ random.
- KILLED-BY: self-uncertainty selection ≥1.5x with CIs.

## Design commitments

Shared 005 corpus generator (structure/fork/noise dials + a
REDUNDANCY dial: duplicate/near-duplicate density). Policies: random
order; self-uncertainty (learner's own loss/entropy); oracle
influence (ground-truth utility); anti-selection (worst-first — the
mandatory direction-flip control per
[[fan2023-irreducible-curriculum]]); static-vs-online refresh (does
the useful signal decay as MATES observed). Honest accounting: every
scoring forward pass is charged to the policy's FLOP budget.
≥3 seeds, CIs, 3-rung ladder. CPU (arche-lab-1/2).

## Why it matters for the goal

If S1: data selection is a standing multiplier on every future
training run AND the substrate bet's interactive-data half gains
mechanism support (the bee v2 and dossier 002 connect here).
If S2/S3: passive well-filtered data is near-optimal, the field's
curation practice suffices, and arche redirects the effort to
001/004/009. All outcomes reprice bet B5.

## What breaks at scale

Same clause as 005: transferable claims are regime membership,
ordering, and sign — not multipliers. Direction predicted per rung
in advance; retrodiction gate governs extrapolation trust.

## VERDICT (2026-08-16, v2 complete; run 1 voided by the oracle==anti alarm)

Deflation with a sharpened mechanism. At matched TOTAL compute with
scoring charged, every selective policy loses to random (~0.43x on
structure loss, ~0.46x on consequence accuracy) at every redundancy
level incl. dup_p=0.9; the harness discriminates (anti 0.36-0.37x <
oracle 0.46-0.47x, alarm passed), so the loss is real: per-step
selection benefit exists but is far below the 2.33x scoring overhead.
S1 killed at this setting (gains did not grow with redundancy; exact
duplicates barely slow random). Transferable claim (DERIVED):
selection pays only when the scorer is much cheaper than the learner
— which is exactly how every published win is built (JEST/MATES use
small proxy scorers); our result shows that is the economics, not a
convenience. Named follow-up (not launched): cheap-scorer arm at
~1/10 learner cost. Scope: easy synthetic world (random reaches
threshold in ~510 FE); harder worlds with higher per-example variance
could shift the bar — named, not claimed.

ADDENDUM (2026-08-16): the named cheap-scorer follow-up is ANSWERED
BY EXISTING DATA, no run needed. Converting v2 cost-to-threshold back
to steps: oracle ~168 steps vs random ~170 (identical), anti ~174 and
worse on consequences. Even a FREE perfect scorer buys ~1.0x here —
the selection benefit per trained step is ~zero in this world, not
merely outweighed by scoring cost. The cheap-scorer regime only
matters in worlds with high per-example value variance; that
condition is now the named prerequisite for any selection follow-up.
