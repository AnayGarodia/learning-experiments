# Dossier 015 — Continual learning without replay: is rehearsal a crutch or the mechanism?

Status: PRE-REGISTERED 2026-08-16. Targets decision 0011 bar (b),
continual learning, named explicitly by Anay. Portfolio axis:
**learning**. Independence: independent of 004 (which measures whether
the write ACCUMULATES) — 015 asks whether accumulation survives when
the crutch is removed. Also gives sparsity a SECOND, independent reason
to matter, on the learning axis rather than the efficiency axis.

PROVENANCE NOTE (added 2026-08-16 after the e003 verifier flagged this
dossier for possibly leaning on unusable numbers): the ONLY e003 result
this dossier relies on is the ARITHMETIC ceiling, 12/(4+8k) = 2.31x at
k=15%. That is a property of the MLP being 8d^2 of a 12d^2 block and
holds independently of any run. e003's LOSS results are explicitly
NON-CITABLE (untuned LR), and the matched-token half of its win still
has regularisation as a live rival that its starvation arm exists to
test. Nothing here depends on those. If e003 later shows enforced
sparsity pays on efficiency after all, 015's motivation is unaffected —
the learning-axis question is independent either way.

## The question, and why replay makes it dishonest

The continual-learning literature's standard result is that a small
replay budget prevents catastrophic forgetting: 1% replay retains
98-99.8% over 8 sequential tasks ([[scialom2022-continual-t0]]). Our own
e004g pilot uses 25% replay, which its builder flagged as generous to
the favourable answer by construction.

But replay requires KEEPING THE OLD DATA and re-training on it. That is
not continual learning in any sense that scales:
- storage grows with everything ever learned;
- the old data may be unavailable (privacy, licensing, one-shot
  experience);
- biology does not rehearse raw episodes indefinitely;
- and a system that re-trains on everything is just retraining with
  extra steps, which is the cost we are trying to eliminate.

So the honest question is not "can forgetting be prevented" (yes, with
replay) but **"can it be prevented STRUCTURALLY, without storing the
past?"** That is the bottleneck. Nobody's replay-based result answers it.

## Rival mechanisms

**C1 — Structural allocation suffices.** Interference is a CAPACITY
COLLISION, not a law: if each skill is written into a different part of
the weight space (sparse allocation, orthogonal subspaces, gradient
projection away from directions prior skills use), retention at zero
replay approaches the replay-based ceiling.
- PREDICTS: retention at 0% replay with allocation >= retention at 25%
  replay without it; the gap to the swappable-adapter ceiling stays
  small as k grows.
- IMPLICATION IF TRUE: this is the continual-learning mechanism, it is
  cheap, and it makes sparsity independently valuable on the learning
  axis.
- KILLED-BY: allocation buying < half the retention that replay buys,
  at every allocation budget tried.

**C2 — Replay is doing the real work.** Structural methods delay
interference but do not prevent it; without rehearsal, retention decays
with k regardless of how writes are allocated.
- PREDICTS: allocation arms track the no-replay baseline, not the
  replay baseline; the decay shape is unchanged, only shifted.

**C3 — Capacity wall.** Both replay and allocation work until the
weight space runs out; retention is a function of total skills per
parameter, and the method barely matters.
- PREDICTS: curves for all methods collapse onto one when plotted
  against skills-per-parameter rather than k; the wall moves with model
  size and with nothing else.
- This is the deflationary reading and its arm (a second model size) is
  MANDATORY.

## Design commitments

Base: Qwen2.5-0.5B, the e004g-validated checkpoint. Harness: e004g's
skill suite, consolidation loop, retention metric and controls, reused
not forked, restricted to NON-DEGENERATE skills (copy_oracle <= 0.20).

Arms, all at matched write budget:
- replay 25% (the e004g setting, as reference);
- replay 1% (the literature's setting);
- replay 0% (the honest baseline);
- replay 0% + SPARSE ALLOCATION (each skill written into a masked
  subset of adapter parameters);
- replay 0% + ORTHOGONAL allocation (each write projected off the
  subspace prior writes used);
- swappable per-skill adapters (the ceiling any method must approach);
- k must reach at least 8, and ideally past the k~32 where sequential
  editing collapses, because "no collapse at k=3" is uninformative.

Measured per step k: retention of every prior skill, a held-out
generality suite (collateral damage), and the conditionality metric
e004g's K0 tripwire introduced (a write that leaks into unclaimed cues
is not a clean write).

## Decision-0012 compliance (all mandatory before launch)

- **Identifiability dry-run**: push C1's, C2's and C3's predicted
  retention curves through analyze.py and confirm each yields its
  intended verdict. In particular the design must show it can separate
  "allocation works" from "k is too small to have hurt yet" — that is
  precisely the ambiguity e004g's pilot could not resolve at k=3.
- **Known-positive**: the replay-25% arm must reproduce e004g's
  retention. If it does not, the harness has drifted.
- **Degeneracy check** (0010 am.3): copy-oracle and mismatched-shot on
  every skill before any curve is read.
- **Weakest-form reporting**: per-skill, per-seed retention; no mean
  that could hide a split (the e011 lesson).

## Why it matters for the goal

Continual learning is one of the two bottlenecks named as qualifying on
its own. If C1 holds, the mechanism is cheap, structural, needs no
stored data, and hands the sparsity line a second independent
justification after its efficiency ceiling came in at 2.31x. If C2, then
every continual-learning claim in the literature is conditional on
keeping the past, which is a real and publishable narrowing. If C3, the
problem is capacity and the answer is scale, which is the least
interesting outcome and the one we should know about before building
anything on top of 004.

Cost: CPU. e004g measured 567 forward tok/s and ~2.0 s/optimizer step
for this base on 16 cores; a k=8 run is within a few hours.
