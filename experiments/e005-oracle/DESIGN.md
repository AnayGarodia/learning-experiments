# e005 — Oracle-ceiling experiment (dossier 005): design

Status: DESIGN 2026-08-16, written after smoke, before any full run.
Implements dossier 005's pre-registration.

## Harness
Synthetic corpus (corpus.py) with per-token ground truth:
FORK (irreducible entropy log2(8), determines a consequence token
~13-20 positions later), STRUCTURE (entropy 0 given fork; the
learnable signal), NOISE (irreducible, no consequence), CONSEQUENCE
(deterministic given fork; the long-range probe). Self-checked
invariants: every consequence matches its sentence's fork; class
fractions printed. Model: plain 4-layer tiny GPT (model.py), FIXED
across arms — this experiment varies the OBJECTIVE only.

## Arms (per-class loss weights on target-token labels)
uniform (1,1,1,1) | drop_noise (1,1,0,1) | struct_only (0,1,0,1) |
drop_forks (0,1,1,1). Grid: arms x noise_p in {0.25, 0.5, 0.75} x 3
seeds. Titration: drop_noise with oracle labels randomly corrupted at
flip in {0.1, 0.25, 0.5} x 3 seeds (maps the Sorscher sign-flip).

## Measurements
Held-out structure loss and consequence accuracy vs tokens consumed
(curves in every JSON record). Headline: the sample-efficiency
multiplier = tokens_uniform / tokens_arm to reach matched thresholds
(structure loss 0.10; consequence acc 0.90), per noise fraction.

## Pre-registered predictions (from dossier O1/O2/O3)
- drop_noise multiplier grows with noise_p; >=2x at noise_p 0.75 if
  O1, <1.3x everywhere if O2 (confidence O1 over O2: ~0.6).
- drop_forks: uncertain by design — dossier O1 says it hurts; the
  counter-argument (fork loss is irreducible like noise, fork USE is
  learned from consequence loss) says it matches drop_noise. This arm
  arbitrates; no confident prediction, that is the point.
- titration: benefit degrades to sign-flip as flip approaches 0.5.

## Budget/hardware
~45 runs x ~5 min CPU = under 1 hour on arche-lab-1 (16 cores,
sharded). fp32, seeds fixed, JSON records with git SHA.

AMENDMENT 1 (2026-08-16, after run 1, before any conclusion): run 1
returned 1.00x for every arm at every noise level — but the eval
cadence (every 40 steps = ~655k tokens) snaps all threshold crossings
to a 3-point grid, hiding any effect under ~30%. This is a
measurement-resolution failure, not a result. Fix: eval every 5 steps
(~82k-token resolution); analysis thresholds tightened (struct_loss
<= 0.02, cons_acc >= 0.99) since all arms fully solve the corpus.
Run-1 data licenses exactly one claim: the oracle ceiling at these
settings is BELOW ~1.3x (already evidence against a large O1 at this
noise mix — but resolution must improve before that is asserted).
Noted for the writeup: the corpus gives noise tokens a distinct ID
range, so models may learn "this range is unpredictable" quickly and
cheaply — if the refined run still shows ~1.0x, that mechanism (noise
is cheap to give up on) becomes the leading explanation and a
follow-up with camouflaged noise (shared vocabulary) is the next
discriminator.

AMENDMENT 2 / camouflage discriminator (2026-08-16, after v2 result):
v2 verdict at fine resolution: oracle multipliers 0.96-1.06x across
ALL arms, noise levels, and titrations — the uniform-loss tax on
IDENTIFIABLE noise is ~zero (O2 confirmed; consistent with O3's
claim that published proxy gains are distribution-steering). The
drop-forks arm also cost nothing: fork USE is learned through
consequence loss. Final discriminator: camouflage=True draws noise
from the STRUCTURE vocab range so junk cannot be identified by token
ID. Pre-registered: if camouflaged noise also gives ~1.0x, the
objective-waste lever is dead at this scale (models give up via
context statistics, not ID); if the multiplier jumps >=1.5x, the
lever lives exactly where junk is hard to identify — which is the
real-text regime — and the dossier verdict flips to
"identifiability-dependent". Grid: {uniform, drop_noise} x noise_p
{0.5, 0.75} x camouflage=True x 3 seeds = 12 runs.

VERIFIER NOTE on camouflage semantics (decision 0010 pipeline,
2026-08-16): with camouflage=True, noise shares the structure token
RANGE but is uniform over it, while bodies use ~96 fixed
high-frequency tokens — so noise remains partially identifiable via
unigram frequency and context, just not via a disjoint alphabet. This
is the correct reading of "camouflage": real-text junk shares the
vocabulary and differs in statistics; a disjoint-ID noise class (v1)
was the unrealistically easy case. If the full run lands ambiguously
between ~1.1x and ~1.5x, the named stronger variant is noise drawn
from the empirical body-token distribution (unigram-matched). Oracle
weighting verified independent of camouflage (labels are structural);
eval stream verified to use the training camouflage setting; MDE
verified (grid 81,920 tokens vs decision boundary 1.0-1.1x vs >=1.5x
at ~2.7M-token crossings = 2 vs 11 grid points). Pilot (np=0.75, 1
seed, 1.5M tokens): drop_noise 1.06-1.07x over uniform; camo-uniform
harder than identifiable-uniform at matched tokens (0.120 vs
0.104-0.107). Verdict: LAUNCH.
