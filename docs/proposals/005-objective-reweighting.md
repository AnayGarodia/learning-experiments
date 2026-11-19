# Dossier 005 — The oracle ceiling: how much of NTP's data hunger is uniform loss?

Status: PRE-REGISTERED 2026-08-16, after the prior-art pass (11 cards)
and before any experiment. Attacks R1. Redundancy check passed: nobody
has measured the TRUE-oracle reweighting ceiling — every published
multiplier uses a proxy oracle that smuggles in a target distribution
([[lin2024-rho1]] curated reference model, [[mindermann2022-rho-loss]]
holdout model, [[evans2024-jest]] curated reference,
[[sachdeva2024-data-efficient-llms]] judge LLM, [[yu2024-mates]] probe).
[[allenzhu2023-physics-part1-cfg]] has ground-truth structure but never
ran the reweighting arm.

## The question

Train small LMs on a synthetic corpus where WE decide, token by token,
which positions carry (a) deterministic STRUCTURE, (b) high-entropy
FORK tokens whose outcome matters downstream, (c) pure NOISE. Give the
loss an oracle that knows the labels. Measure the sample-efficiency
multiplier of oracle weighting over uniform NTP — the CEILING every
practical scheme (Rho-1 etc.) is a lower bound to — and which token
class each weighting should keep.

The design's core axis, forced by an unresolved sign conflict in the
literature: Rho-1 wins by DROPPING persistent-high-loss tokens;
[[wang2025-forking-tokens]] (+11 AIME at 32B from gradients on only
the 20% highest-entropy tokens) and [[wang2025-entrodrop]] say
high-entropy tokens are where the signal lives. Our hypothesis: they
are talking about two different entropies — noise-entropy (drop) vs
decision-entropy (keep) — which natural data conflates and our corpus
separates by construction.

## Rival mechanisms

**O1 — Large uniform-loss tax.** Oracle weighting buys ≥5x sample
efficiency (as constant or exponent change) once noise is a
substantial corpus fraction; drop-noise + keep-forks is the winning
weighting.
- PREDICTS: multiplier grows with planted noise fraction; dropping
  forks HURTS even though forks are high-entropy (reconciling both
  camps); direction stable across a 3-rung size ladder.
- KILLED-BY: oracle gain < 2x at matched compute across rungs even at
  high noise fractions.

**O2 — Modest tax.** Loss on unpredictable tokens is nearly harmless
(gradient noise averages out); ceiling < 2x regardless of mix.
- KILLED-BY: any ≥5x oracle multiplier with seeds and CIs.

**O3 — Deflationary target-smuggling.** Published proxy gains come
from steering toward a better DISTRIBUTION, not from fixing the
objective; with a true oracle on the SAME distribution, the gain
mostly vanishes.
- PREDICTS: our true-oracle multiplier ≪ Rho-1-style proxy multipliers
  replicated in-harness with a curated reference proxy.
- KILLED-BY: true-oracle gain ≈ or > proxy gain at matched setups.

## Design commitments (full DESIGN.md before code, walkthrough before run)

Corpus generator with independently dialable fractions of structure /
fork / noise tokens (probabilistic grammar with consequential random
choices + injected unpredictable spans; CFG family per
[[allenzhu2023-physics-part1-cfg]], which 86M models provably learn).
Arms: uniform NTP; oracle drop-noise; oracle keep-structure-only;
oracle drop-forks (predicted to hurt); proxy oracle (reference-model
excess loss, in-harness Rho-1 replication); oracle-quality titration
(corrupt oracle labels from 100%→50% correlation, map the sign-flip
point that [[sorscher2022-data-pruning]] theory predicts).
Metrics: data-to-target-loss ratio vs uniform (the multiplier), with
≥3 seeds + CIs; measured on structure-token loss AND a downstream
probe (fork-consequence prediction). Compute accounting charges
oracle/scoring costs (RHO-LOSS's uncharged scoring pass is a named
trap). Ladder: 3 model sizes to check direction stability
([[fan2023-irreducible-curriculum]]'s anti-curriculum flip at 124M is
the cautionary card; direction-flip controls mandatory).
Hardware: CPU boxes (arche-lab-1), tiny models.

## Why it matters for the goal

Data hunger is the binding constraint ([[villalobos2024-data-limits]]).
The ceiling number decides whether objective design is a first-order
lever (≥5x → GPU-scale follow-up on real text, and 011's
multi-horizon extension inherits the harness) or a bounded side-show
(<2x → effort shifts to 006/004). Either answer also arbitrates the
Rho-1-vs-forking-tokens conflict — a publishable clarification on its
own.

## What breaks at scale (standing clause)

Synthetic multipliers are mechanism measurements; the transferable
claims are (a) the EXISTENCE and sign of the token-class effects and
(b) the oracle-vs-proxy gap, not the specific multiplier. Scale
direction must be predicted in advance per rung; the retrodiction gate
governs trust in any extrapolation.

## VERDICT (2026-08-16, all three runs complete)

O2 CONFIRMED (modest tax), with a real texture. Identifiable noise:
oracle multipliers 0.96-1.06x everywhere (45 runs). Camouflaged noise
(12 runs): drop_noise buys 1.12-1.15x on structure loss — direction
matches identifiability-dependence, magnitude far below the 1.5x
line — and is WORSE (0.89x) on long-range consequence accuracy, so
the net is ~nil and metric-dependent. Drop-forks: harmless (fork use
is learned through consequence loss), dissolving the Rho-1-vs-
forking-tokens dispute at this setting: neither camp's weighting
matters much when distribution is held fixed. Standing inference
(O3-flavored, tagged DERIVED): published token-selection gains are
predominantly distribution-steering, not objective repair. Scope:
tiny models, synthetic corpus; the transferable claims are the signs,
the metric split, and the oracle-vs-proxy distinction. GPU follow-up
CANCELLED — the lever does not merit it. The 1000x does not live here.
