# Dossier 004 — The consolidation channel: making in-context gains permanent

Status: PRE-REGISTERED 2026-08-16, written AFTER the prior-art pass
(12 cards, see Citation base) and BEFORE any experiment. Attacks
requirement R2 (two-speed memory) and prices bet B2 (~75% that the
missing organ is a context→weights channel).

## The question

A model that solves a task from a few in-context examples loses that
ability when the context ends. Biology has a channel that makes fast
gains permanent (hippocampus→cortex consolidation). Can a model's own
in-context adaptations be written into its weights (a) cheaply, (b)
better than fine-tuning on the same data, and (c) REPEATEDLY into one
persistent weight set without destroying prior gains?

## What the prior art already settles (do not re-measure)

- The write CAN be done and has a QUALITY EDGE over raw fine-tuning
  on identical data: distilling through the model's own in-context
  interpretation beats direct gradient descent by ~9 points
  ([[snell2022-context-distillation]] SPIDER 22.1 vs 13.4 at 4-shot).
- The write can be AMORTIZED: a meta-learned operator emits the
  adapter in one forward pass at ~95% of per-item distillation
  fidelity, 100-1000x cheaper ([[charakorn2026-doc-to-lora]]).
- For ATOMIC FACTS the channel is already beaten: direct editing
  writes 10k facts at ~2.7s/fact with better specificity than
  distillation achieves ([[meng2022-memit]], [[meng2022-rome]] vs
  Snell's own concession). B2 is hereby NARROWED: the bet lives on
  behaviors/skills/procedures, not facts.
- Interference control is cheap at task scale: 1% replay retains
  98-99.8% over 8 sequential tasks ([[scialom2022-continual-t0]]).
- The value being thrown away is large: per-task test-time training
  buys +29 points on ARC, then the adapters are discarded
  ([[akyurek2024-ttt-arc]]). The TTT/Titans line builds the fast
  organ but RESETS it every sequence — the fast→slow channel is
  exactly what that line omits.

## The open territory (what 004 measures)

Nobody measures ACCUMULATION: repeated consolidation into ONE
persistent weight set. The only accumulation data anywhere is from
editing, and it is grim (ROME collapses by ~32 sequential edits;
MEMIT's specificity decays first). B2's "without catastrophic
interference" clause is currently UNTESTED, not supported.

## Rival mechanisms

**K1 — Compounding organ.** Sequential consolidation of skills into
one weight set retains most of each skill (with cheap replay), and
freed context makes the model strictly more capable over its
"lifetime" — capability-vs-accumulated-experience becomes a plottable
curve (the candidate scaling axis).
- PREDICTS: after consolidating skills 1..k, skill-i retention stays
  high (>90% of just-consolidated performance) with small fixed
  replay; re-prompting on top of consolidated weights ADDS capability
  (the Prompt Baking side of the tension below).
- KILLED-BY: retention collapsing with k like sequential editing
  does, at any replay budget we can afford; or consolidation cost
  growing with k.

**K2 — Cache, not organ.** Consolidation only caches what the prompt
did; re-prompting adds nothing (the Askell Fig-20 side); accumulation
works but capability never compounds — you save context tokens, no
more.
- PREDICTS: baked+re-prompted ≈ baked alone; total capability after
  k consolidations ≈ capability with the best single prompt.

**K3 — Interference wall.** The fast→slow write is fundamentally
edit-like at skill granularity too: one persistent weight set decays
with k regardless of routing (the reason every shipped system uses
swappable adapters instead).
- PREDICTS: the retention curve vs k matches the editing-collapse
  shape (fine at small k, cliff), replay only delays it.

Pre-registered secondary question (cross-card tension, decides B2's
payoff mechanism): does consolidation FREE context (gains stack when
you re-prompt on consolidated weights) or CACHE it? Askell Fig 20
and [[bhargava2024-prompt-baking]] currently disagree.

## Design sketch (to be finalized in its experiment DESIGN.md)

Small open model (1-8B, LoRA writes), a pool of N distinct skills
each teachable by a prompt (format transforms, tool conventions,
reasoning styles — NOT facts, per the narrowing). Loop k = 1..N:
distill skill k's prompted behavior into the SAME persistent adapter
(KL to own prompted distribution, per the Askell/Snell recipe),
with a fixed small replay budget; after each step measure every
prior skill (retention curve), a held-out generality suite
(collateral damage), and baked+re-prompted (the K1/K2 discriminator).
Controls: fine-tuning on the same examples (Snell's edge must
replicate or the premise falls), swappable per-skill adapters (the
ceiling accumulation must approach), no-replay ablation.
Cost: single GPU, LoRA-sized — waits on the quota like the m=8
column. Metrics, seeds, and kill thresholds get their own DESIGN.md
with a plain walkthrough BEFORE launch (standing rule).

## Why it matters for the goal

R2 is a row where the transformer recipe scores zero, and
accumulated experience is our leading candidate for a NEW SCALING
AXIS (capability vs lifetime, the curve nobody has plotted). K1
confirmed at small scale = the strongest single piece of evidence
the lab could produce for B2. K3 confirmed = B2 drops toward the
editing-collapse prior and the axis candidate dies honestly.

## Citation base

[[askell2021-context-distillation]], [[snell2022-context-distillation]],
[[ba2016-fast-weights]], [[schlag2021-fast-weight-programmers]],
[[meng2022-rome]], [[meng2022-memit]],
[[ovadia2023-finetuning-vs-retrieval]], [[scialom2022-continual-t0]],
[[bhargava2024-prompt-baking]], [[charakorn2026-doc-to-lora]],
[[akyurek2024-ttt-arc]], plus existing TTT/Titans cards.

## Rehearsal note (2026-08-16, e004r closed after two gated attempts)

The toy-scale rehearsal STOPPED at its own gates, twice, cleanly:
(1) on the deterministic corpus the pretrained base has no in-context
learning at all (nothing to consolidate); (2) on a corpus explicitly
built to reward in-context inference (per-document random mappings),
a 4-layer/128-dim model at 6M tokens still learns only the
unconditional distribution — first-occurrence and repeat-occurrence
accuracy both exactly chance. Transferable design fact: IN-CONTEXT
LEARNING IS ITSELF EMERGENT and absent at this scale/budget, so the
consolidation experiment PRESUPPOSES a genuinely pretrained ICL-capable
base — there is no rehearsal of the fast->slow channel below that
bar. The harness (skill definitions, gap gate, distillation loop,
retention metrics) is built, committed, and transfers as-is to the
GPU version. Status: 004 moves whole to the GPU queue.
