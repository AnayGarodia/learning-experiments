# e009 — Joint compute-redundancy oracle (dossier 009): design

Status: DESIGN 2026-08-16, before any full run. Decision 0010 applies
(pilot + analysis-on-pilot below, MDE stated).

## Models and training
SwiGLU decoder ladder (model.py): s=0.89M, m=3.07M, l=9.44M nonemb
params (d/L = 96/8, 160/10, 256/12). Trained on the e005 synthetic
corpus (pinned copy), 6M tokens, AdamW 1e-3/wd 0.1, batch 64 x seq
256. FIVE checkpoints at 4/10/25/50/100% of training per size
(dossier constraint b: redundancy as a function of training tokens).
1 seed per size for the main grid, second seed on size s as a
stability check (oracle fractions are proportions; run-to-run
variance enters via training, checked cheaply at s).

## The four oracles (oracle.py), all post-hoc on held-out data
1. DEPTH: earliest block whose readout (final norm+head applied to
   the intermediate hidden — the CALM-style probe) matches the full
   model; fraction of block-passes after that point. Criteria:
   argmax-match AND KL<=0.1, side by side (dossier constraint d).
2. WIDTH coarse (hardware-priceable): zero each contiguous QUARTER of
   each block's SwiGLU hidden; exact end-to-end agreement fraction.
   WIDTH fine: near-zero-activation fraction (|act| < 1e-3 x token
   RMS) — a LOCAL criterion (per-unit end-to-end zeroing is
   computationally out of reach; the criteria therefore differ and
   the coarse-vs-fine comparison is labeled as such — T3 is judged on
   whether coarse retains most of what fine suggests exists).
3. TEMPORAL (the unclaimed axis): substitute block b's output at
   position t with its own output at t-1, run the remaining blocks,
   measure agreement. One modified forward per block.
4. PREDICTION-ERROR: closed-form ridge linear map h_b -> h_{b+1} fit
   on half the eval set, relative-error<0.1 fraction on the other
   half (the predictive-coding-style compute-on-mismatch number).

JOINT (block granularity, argmax): union of depth-skip and
temporal-reuse per (token, block) measured directly. (The pred arm
is reported separately: its criterion is in hidden-space, not
end-to-end, and unioning mismatched criteria would overstate.)

## MDE
Fractions over >=65k (token, block) samples: binomial 95% CI half-width
<=0.004 at p=0.5. Any effect above ~1 percentage point is resolvable;
the dossier's decision lines (50%, 60%, 80%) are far above that.

## Grid and cost
sizes {s,m,l} x 5 checkpoints (+ s seed 1) = 20 oracle measurements
after 4 training runs. Training dominates (~0.5-2h per size on a
16-core box); oracles are minutes each.

## Pre-registered expectations
- depth skippable fraction substantial and growing down the ladder
  at fixed budget (undertrained large models are shallower-effective);
- criterion gap: KL<=0.1 strictly harsher than argmax (Gromov datum);
- temporal reuse low on this corpus (positions carry distinct roles)
  — if HIGH, that is the surprising result;
- trend with training tokens: unknown, the point of constraint (b).

## Pilot outcome (2026-08-16, decision 0010; outputs in pilot/)

Small model, 800k-token training, checkpoints at 65k and 786k tokens,
all four oracles + analyze.py ran end to end (~17s for both
measurements). Dynamic-range check and adjustments:
- depth_frac_arg 0.50 -> 0.71, pred_frac 0.91 -> 0.27, joint 0.51 ->
  0.72: healthy range; joint GROWS with training in the pilot.
- Criterion inversion at undertrained checkpoints, understood and
  kept: near-uniform logits make KL<=0.1 trivially easy (0.875) while
  argmax is noisy; at the trained checkpoint KL is harsher (0.099 vs
  0.710) as the Gromov datum predicts. Both reported.
- width_coarse_kl saturated at eps 0.1; tightened to 0.01 for the
  width axis (still 0.99 — zeroing an MLP quarter genuinely barely
  moves the output; corroborated by argmax 0.93-0.97). Documented as
  a finding-shaped saturation, not an instrument gap.
- width_fine strict (1e-3 RMS) ~0.01: SwiGLU has almost no ReLU-style
  dead activations (echoes the cards' SwiGLU-untested caveat) — kept
  as a result; added loose threshold (1e-1 RMS) 0.24-0.34 for
  dynamic range. Both reported.
- temporal ~0.001-0.006: at the pre-registered expected floor (this
  corpus gives positions distinct roles); the probe machinery shows
  range on other axes, so the floor is a result, not blindness.

## Verifier decisions (2026-08-16, decision 0010 review)

- Concern (a), sentence-relative temporal variant: DECLINED. Sentence
  lengths vary (noise spans 1-6 tokens), so no fixed lag aligns
  sentence-relative positions; exact per-row alignment is a different
  experiment (content-addressed caching), conceptually covered by the
  prediction-error arm. Documented rather than half-built.
- Concern (b), joint-KL: joint stays argmax-only; per-axis KL is
  reported separately. Unioning axes measured at different KL eps
  (depth 0.1, width 0.01) would overstate; argmax is the
  capability-relevant criterion.
- Concern (c), eval size: EVAL_SEQS 32 -> 64 (CI insurance, minutes).
- Fix: joint now excludes position 0 (no t-1 to reuse; was trivially
  inflating by ~1/T per block, consistent now with temporal_arg).
- Caveat documented: the temporal oracle substitutes ALL positions
  simultaneously, so later blocks see other tokens' substituted
  states too — a stricter (conservative) reuse criterion than a
  per-token counterfactual. Fractions are lower bounds on per-token
  reuse.
- Probe split note: prediction-error probe is fit on half the
  held-out batch and scored on the disjoint other half; it never
  sees scored rows.
