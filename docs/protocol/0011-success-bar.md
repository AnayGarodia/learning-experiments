# Decision 0011 — The success bar: 1000x, or a fundamental bottleneck

Status: DECIDED 2026-08-16 by Anay (AXIOM — a goal choice, owned not
derived). Resolves assumption A2 in FOUNDATION.md, which had been on the
standing target list since the lab's founding.

## The decision

A result qualifies as a FINDING for this lab only if it does one of two
things:

**(a) It contributes to a compounding path to ~1000x** versus current
frontier models. No single mechanism delivers 1000x; the only route is
several independent multipliers multiplying together (illustratively:
~10x from data efficiency, a few times from sparsity, a few from
hardware fit, and something larger from a change in how learning works).
A qualifying result under (a) must therefore be a multiplier LARGE
enough and INDEPENDENT enough to be one of those factors.

**(b) It solves, or substantially cracks, a fundamental bottleneck.**
Named explicitly by Anay: sample efficiency and continual learning, "or
some fundamental bottleneck". A result qualifies under (b) even if it
produces no immediate multiplier, because removing a bottleneck changes
what is possible rather than what is cheap.

## AMENDMENT 1 (2026-08-16, same day, Anay) — bounded multipliers are
## good, but only as PORTFOLIO members

The first draft of this decision said bounded multipliers do not qualify
at all. Anay corrected it:

"bounded multiplier hunts are also good but only if they lead to a
combined 1000x efficiency and context and learning and energy and all
leap over existing models"

So the test on a bounded multiplier is NOT its size on its own. It is
whether it takes a named place in a portfolio that compounds to a leap
over existing models across FOUR axes, all of which count:

- **efficiency** (compute per unit capability)
- **context** (how much a model can hold and use)
- **learning** (sample efficiency and continual learning)
- **energy** (joules per unit capability, which is where hardware fit
  and sparsity cash out)

A 1.4x that is INDEPENDENT of the other factors and multiplies with them
is valuable. A 1.4x that overlaps an existing factor, or that stands
alone with nothing to multiply against, is not. This is a stricter test
than "is the number big" and a more permissive one than "reject anything
under 2x".

Operationally, every bounded-multiplier dossier must answer three
questions before it gets compute:
1. Which axis does this factor sit on?
2. Is it INDEPENDENT of the factors already claimed, or does it overlap
   one (in which case the portfolio gains the difference, not the
   product)?
3. What does the running product become if it lands at its predicted
   magnitude?

PORTFOLIO.md holds that accounting and is updated whenever a factor is
measured. A multiplier with no entry there is not a finding.

## What this still rules OUT

An efficiency trick that leaves the learning process unchanged AND has
no independent place in the portfolio. The 1.05x from objective
reweighting is the type case: it sits on the efficiency axis, overlaps
what data curation already does, and multiplies to nothing. It remains
an ELIMINATION, which has real value for a research program, but it is
not progress toward the goal and must not be presented as such.

## Why this is a correction, not just a clarification

Applied retrospectively, this bar indicts the lab's own compute
allocation to date. Of the six levers tested through 2026-08-16 —
objective reweighting (005), data selection (006), multi-horizon
prediction (011), adaptive compute (009/009b), growth (010), and bee
priors (002) — every one was a bounded-multiplier hunt. Even a clean WIN
on any of them would have failed bar (a) on magnitude and bar (b)
entirely, because none of them changes the learning process
structurally. They were cheap, CPU-viable, and fast to kill, which is
why they were run first; that was a resource-driven ordering, and it
produced a queue that could not have met the bar it was working toward.

The elimination value is real and is kept: three independent loss- and
data-side efficiency tricks measured ~1.0x against ground truth where
the literature reports 2-18x, and that pattern is itself informative.
But it is elimination, not progress.

## Consequence for the queue

Re-scored against this bar, in priority order:

**Qualifies under (b), the bottleneck bar:**
- **004 consolidation** — the context-to-weights channel. Attacks
  continual learning AND sample efficiency at once. The lab's headline
  bet (B2, 80%). UNTESTED. Blocked on GPU access.
- **012 ICL emergence** — where fast in-context learning switches on.
  A prerequisite for 004 and a sample-efficiency question in its own
  right. Instrument check under verification; the science is blocked on
  GPU access.
- **001 retrieval cap** — attacks a PROVEN fundamental limit (fixed-size
  state cannot do random-access retrieval). Running.
- **007 slop / diversity collapse** — attacks the verifiable-versus-taste
  asymmetry, a capability bottleneck rather than an efficiency one.
  Queued for GPU.

**Qualifies under (a) if the multiplier is large:**
- **003 sparsity** — the biological operating point implies ~85% of
  per-token activation is waste. Large enough to be a factor if it
  holds. Building.
- **008 local learning** — qualifies only through the hardware path: a
  constant-tax local rule that unlocks memory-compute colocation is a
  multiplier; a widening gap closes the door. Running.

**Does NOT qualify (elimination value only, do not extend):**
- 005, 006, 011 — closed, bounded, correctly dead.
- 009/009b — bounded by construction; the running grid finishes for the
  record and the T2 answer, and is not extended past it.
- 010 growth — a one-time discount; the redesign its open questions
  would need is NOT bought under this bar.

## Standing rule

Every new dossier states, in its first section, which bar it targets and
by what magnitude. A dossier that cannot name its bar does not get
compute.
