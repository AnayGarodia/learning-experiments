# e011 — Multi-timescale prediction (dossier 005 extension / brain item 4): design

Status: DESIGN 2026-08-16, written before code. Question: does an
objective that predicts at several horizons at once learn LONG-range
structure faster per example than next-token prediction — at tiny
scale, where the published natural-text result (gloeckle2024: MTP
helps at 7B+, HURTS small models) points the other way?

## Corpus (multi-scale extension of e005's generator)

Sentence = fork token + a FIXED layout so every scale's distance is
exact (noise REPLACES designated positions rather than inserting, so
distances never shift):
- offset 4: SHORT consequence (vocab region S, = g_s(fork))
- offset 16: MEDIUM consequence (region M)
- offsets 8-11: noise region (irreducible, no consequence)
- long variant only — offset 64: LONG consequence (region L),
  second noise region at offsets 40-47
- all other offsets: structure tokens, deterministic given
  (fork, offset)
Labels: 0 fork, 1 structure, 2 noise, 3/4/5 short/medium/long
consequence. Dial: long_density = fraction of sentences carrying the
long tail (the long-range-structure density of the corpus).
Self-checks: every consequence matches its sentence's fork; distances
exactly 4/16/64; per-class fractions printed.

## Arms (same model, same data, objective is the only variable)

1. next_token — control.
2. multi_horizon — auxiliary linear heads off the same trunk predict
   tokens at +4, +16, +64; losses summed at equal weight with NTP.
3. annealed — same heads, aux weight decays linearly to 0 over
   training (scaffold hypothesis: horizons teach structure early,
   then get out of the way).
4. mtp4 — heads at +2, +3, +4 (dense next-4, the gloeckle2024 shape)
   — distinguishes "any extra horizons" from "horizons AT the
   structure's scales".

## Measurement

Held-out per-scale consequence accuracy (MAIN next-token head only,
at the position preceding each consequence — aux heads never score)
and structure loss, vs tokens consumed. Headline: per-scale
sample-efficiency ratio vs control (tokens to 90% accuracy at each
scale). Aux-head FLOPs are counted and disclosed; curves recorded vs
tokens AND vs FLOPs (heads are not free: each aux head is a
vocab-sized projection comparable to the main head).

## Pre-registered predictions

- multi_horizon accelerates LONG-scale accuracy per token vs control
  at long_density 0.25 (sparse long structure, where NTP's signal is
  thinnest): confidence ~0.55.
- At matched FLOPs the long-scale gain shrinks but survives: ~0.4.
- mtp4 (dense short horizons) does NOT accelerate the long scale:
  ~0.6 — if it does, the mechanism is generic regularization, not
  horizon targeting, and the dossier's framing is wrong.
- Control eventually matches on short/medium scales (they are easy);
  any arm HURTING short-scale learning is the gloeckle small-model
  effect appearing in-harness: worth recording either way.

## Grid and budget

4 arms x long_density {0.25, 1.0} x 3 seeds = 24 runs, 6M tokens
each, seq_len 256, batch 64, AdamW lr 1e-3 (e005's recipe held
fixed). ~5 min/run Mac-CPU, ~15-25 min/run on a lab-box core;
~45-60 min wall on a 16-core box, 12 workers.

## VERDICT (2026-08-16, 24/24 runs complete)

Bet LOST, honestly. At ld=1.0 (every sequence carries long-range
structure), plain next_token is the FASTEST to the long-range
criterion (acc_long >= 0.90). Per-seed crossings, which are what the
instrument actually resolves:

  next_token     655k / 655k / 655k   (unanimous)
  mtp4          1.31M / 655k / 655k   (2-1 split)
  multi_horizon 1.31M / 1.31M / 1.31M (unanimous)
  annealed      1.31M / 1.31M / 1.31M (unanimous)

CORRECTION (2026-08-16, on pulling all 24 records into the repo and
running analyze.py): the first draft of this verdict quoted MEANS
("mtp4 874k, 0.75x"), which hid a 2-1 seed split. The honest
statements are:
- multi_horizon and annealed are STRICTLY slower than next_token —
  all three seeds land on the far side of the eval grid from all
  three next_token seeds. The point estimate is 2.0x, but the
  instrument only bounds the true ratio as >1: the effect equals one
  eval-grid step (655k tokens), so the magnitude is coarse and 2.0x
  must not be quoted as a measured multiplier.
- mtp4 is NOT separated from next_token at matched DATA (2 of 3 seeds
  tie at 655k). It is separated at matched COMPUTE, where the
  auxiliary heads cost a deterministic 1.90x per token.
Under FLOP-equivalent accounting the ordering is unambiguous:
next_token 1.00x, mtp4 1.90x, multi_horizon and annealed 3.80x.

At ld=0.25 the column is under-resolved: every arm lands on the same
eval-grid point (the e005/e006 grid-snap artifact class), so no
ordering claim is made there — named, not claimed. analyze.py now
prints this resolution guard automatically for any future rerun.

Transferable claim (DERIVED): at tiny scale, predicting extra
horizons is a TAX, not a shortcut; next-token loss already carries
the long-range signal when the data has it. Consistent with the
gloeckle2024 small-model penalty (MTP only pays above ~3B). The
Rung: third consecutive deflation of an efficiency-trick lever
(objective reweighting e005, data selection e006, multi-horizon
e011). Scope: 6M tokens, synthetic corpus, tiny models; the ld=0.25
cell needs a finer eval grid before any claim.
