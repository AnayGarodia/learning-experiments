---
title: "arche e005 — true-oracle loss reweighting buys ~nothing on identifiable noise (ceiling ~1.05x)"
authors: [arche lab]
year: 2026
venue: internal experiment (pre-registered dossier 005)
url: experiments/e005-oracle/
questions: [q4-sample-efficiency, q1-scaling]
status: carded
carded_by: fable (closure agent, 2026-08-16)
audited_by: fable (verifier, 2026-08-16 — recomputed all 15 v2 multiplier cells and both camouflage multipliers from an independent pull (byte-identical, 1.155/1.117x loss, 0.887/0.890x acc); fixed vocab/scale line)
---

## Claims

- CLAIM: With a ground-truth oracle labeling every token as structure /
  fork / noise / consequence, oracle loss-weighting buys 0.96-1.06x
  sample efficiency over uniform NTP — at every arm, every noise
  fraction (0.25/0.5/0.75), and every oracle-corruption level tested.
  - EVIDENCE: e005 v2, 45 runs, eval cadence ~82k tokens, thresholds
    struct_loss<=0.02 and cons_acc>=0.99, 3 seeds/cell.
  - SCALE: 4-layer/128-dim GPT (~2M params), 6M-token synthetic corpus,
    vocab 5000, 8 fork classes, noise drawn from a disjoint ID range
    (4000-4999) in the main arms.
  - BASELINE: uniform NTP, same model/data/seeds.
  - GRADE: strong at this setting. Camouflage discriminator (12 runs,
    complete): drop_noise on camouflaged junk buys 1.12-1.15x on
    structure loss but 0.89x on consequence accuracy — net ~nil and
    metric-dependent; far below the pre-registered 1.5x line. Dossier
    005 verdict: O2 confirmed; GPU follow-up cancelled.
  - BREAKS-IF: a materially larger multiplier appearing at bigger
    scale or on real text (no mechanism currently predicts one).

- CLAIM: Dropping loss on high-entropy-but-consequential fork tokens
  costs nothing at this setting; fork USE is learned through
  consequence loss.
  - EVIDENCE: drop_forks arm 1.00-1.06x vs uniform everywhere.
  - GRADE: strong at this setting. Bears on the Rho-1 vs
    forking-tokens sign dispute: at matched distribution the answer is
    "barely matters either way".

- CLAIM (methods): published token-selection multipliers are lower
  bounds entangled with DISTRIBUTION-steering, not measurements of the
  objective-weighting ceiling; under a true oracle at matched
  distribution the ceiling is ~1.05x here.
  - GRADE: weak (one setting, toy scale) but the only true-oracle
    measurement anywhere (redundancy check in dossier 005).

## Data

Multipliers vs uniform (tokens-to-threshold ratios, 3 seeds/cell):

| arm | noise 0.25 | noise 0.5 | noise 0.75 |
|---|---|---|---|
| drop_noise | 0.98x / 1.04x | 1.00x / 1.04x | 1.04x / 1.04x |
| struct_only | 0.96x / 1.06x | 1.02x / 1.04x | 1.05x / 1.04x |
| drop_forks | 1.00x / 1.06x | 1.01x / 1.01x | 1.03x / 1.03x |

(cells: loss-threshold multiplier / accuracy-threshold multiplier)
Oracle-corruption titration (drop_noise, noise 0.5): flip 0.1 ->
1.00x/1.03x; 0.25 -> 1.00x/1.02x; 0.5 -> 1.00x/0.99x — no cliff,
because there is no benefit to destroy.
Camouflage arm (noise drawn from the structure vocab range; 12 runs):
drop_noise 1.12-1.15x on struct-loss threshold, 0.89x on
consequence-accuracy threshold — direction matches identifiability-
dependence, magnitude ~nil net.
Run 1 (voided): eval cadence 40 steps (~655k tokens) snapped all
crossings to one grid — a resolution artifact, licenses only
"ceiling < 1.3x" (DESIGN amendment 1).

## Untested assumptions

Single model size; deterministic-body corpus (very fast structure learning may
compress differences); multipliers are setting-specific — the
transferable claim is the mechanism (models learn to give up on
identifiable noise nearly free), not the number.

## Citation tree

- UPSTREAM: dossier 005; [[lin2024-rho1]], [[wang2025-forking-tokens]],
  [[wang2025-entrodrop]] (the sign dispute this arbitrates at matched
  distribution), [[sorscher2022-data-pruning]] (titration prediction —
  no sign-flip observed here because no benefit existed).
- DOWNSTREAM: dossier 005 verdict (O2 confirmed at identifiable noise;
  O3 consistent); e006 (shared harness).

## Further work

Unigram-matched noise variant named in DESIGN as the stronger
camouflage form (not currently merited); real-text replication only
if any setting ever exceeds 1.5x.

## Concepts

[[next-token-prediction]], [[sample-efficiency]].

## Method summary

Synthetic corpus with per-token ground-truth classes (deterministic
body given fork; scattered high-entropy forks whose value determines a
later consequence token; dialable irreducible-noise spans). Identical
tiny GPTs trained under per-class loss weights; oracle labels known by
construction; efficiency measured as tokens-to-matched-threshold with
~82k-token resolution.

## Relevance

q4/q1: deflates R1's post-hoc weighting lever (the ~1.05x ceiling) and
reprices the objective-related bets; the strongest cheap-kill the lab
has produced — evidence that published selection wins are distribution
effects. Feeds MAP 4.5.
