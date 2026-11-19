# Retrospective — the lab's first two days, audited against its own goal

Written 2026-08-16 at Anay's request. Scope: everything the lab has
done, judged against decision 0011's bar. Written to be uncomfortable
where the record is uncomfortable.

## 1. Progress toward the goal: none, and the portfolio says so

PORTFOLIO.md's measured product is **1.0x**. Nine experiments have
closed with verdicts. Not one produced a factor. The largest efficiency
number anywhere in the record is 1.05x.

That is the honest headline and no amount of process quality changes
it. What follows is about why the search has not produced its first
multiplier, and what to change.

## 2. The structural error: a resource-ordered queue

Recorded in decision 0011 and repeated here because it is the single
biggest finding of this retrospective. The first six levers tested —
objective reweighting, data selection, multi-horizon prediction,
adaptive compute, growth, bee priors — were ALL bounded-multiplier
hunts on the efficiency axis. Every one of them would have failed the
bar even had it won cleanly. They were run first because they were
cheap and CPU-viable.

Two days of compute went onto ONE of the four axes, and it is the axis
where the ceiling was always lowest. Three axes (context, learning,
energy) still carry no measurement.

## 3. The systematic bias: every correction went the same direction

Count of first-pass claims that were later corrected, voided or
withdrawn: **ten**.

- Four runs voided outright (e002 v1, e004r v1, e005 r1, e006 r1).
- e009: spliced seed endpoints; T1 downgraded to PARTIAL.
- e009: "80-88% at parity" downgraded to an upper bound once the union
  was actually executed.
- e011: verdict rewritten after per-seed data showed the means had
  hidden a 2-1 split.
- e010: an apparent exponent change was a fit-range artifact.
- e008: slopes marked non-citable (every rule's optimum at the sweep edge).
- e012b: a rotary result raised as the explanation, then withdrawn when
  its own discriminator showed it was constant-offset copying.

**In all ten cases the correction made the result less exciting.** Not
one correction went the other way. That is not random error; it is a
bias in first-pass analysis toward the interesting reading. The
verification layer catches it, which is why the record is trustworthy —
but the base rate of overclaiming is the thing to fix, not just the
catch rate.

## 4. The instrument sits partly below its own validated floor

e-retro FAILED: at ~1M non-embedding params the harness ranks an LSTM
above a matched transformer, both tuned to interior optima, both
confounds closed. It recovers the correct ranking by ~3M.

Most of this lab's work ran at or below that crossover. Decision 0007
(mechanism-first at small scale) is the lab's core method, and the
first honest test of it returned a measured floor that much of our own
output sits under. The correct reading is narrow — the failure is about
comparing architecture FAMILIES, not about within-family intervention
signs — but it was discovered, not designed for, and it should have
been the FIRST experiment the lab ran rather than the ninth.

## 5. The shared-corpus confound nobody priced

**Twelve of fifteen experiments reference the same e005 synthetic
corpus.** Reuse was deliberate (shared harness, faster builds) and its
cost was never stated: results across a dozen experiments are not
independent. If that corpus has a quirk — a too-easy structure, an
unrepresentative entropy floor, a saturation ceiling — it is in
everything at once.

Two specific places it already bit:
- e009b: "convergence" and "corpus exhaustion" are the same axis on
  this corpus, which is what made the regime claim uncallable.
- e005: "the deterministic body may compress differences" was named as
  a caveat and never tested against a second corpus.

The one experiment that used real data (e-retro, enwik8) produced the
biggest surprise. That is a signal.

## 6. The recurring design failure: resolution was checked, identifiability was not

Decision 0010 requires an MDE statement — can we resolve the effect?
Two grids nonetheless produced verdicts they could not call:
- **e010**: 89% of savings cells censored, and the two families shared
  1.2x of compute range. Two of three pre-registered questions were
  unanswerable, discovered only when analysis was written.
- **e009b pilot**: every oracle point fell in ONE frontier interval, so
  the bracket could not separate SURVIVES from INCONCLUSIVE no matter
  how many seeds ran.

Both passed their MDE check. Both failed on a different question:
**if the effect is exactly as predicted, does this design produce a
callable verdict?** Nobody asked it, because nothing required it.

## 7. What is genuinely good, and should not be lost

- Pre-registration with kill conditions, applied as written even when
  it hurt (e-retro FAIL, e011 bet lost, e012b's builder refusing to
  declare a verdict its own caveat forbade).
- The builder→verifier pipeline. Every serious error above was caught
  internally, usually within hours, several by agents attacking their
  own work unprompted.
- Killing our own good news fast: the e009 headline was attacked by the
  next experiment we built, not defended.
- Instrument guards written into analyze.py that fire mechanically
  (e008's edge-of-sweep warning, e010's censoring census).

## 8. Changes proposed (decision 0012)

1. **Identifiability dry-run before launch.** Simulate the predicted
   result and confirm the verdict rule fires. Would have caught e010
   and the e009b pilot pre-launch.
2. **Known-positive before any null is believed.** e012b was the first
   one the lab ever ran, reactively, after five negatives. It should be
   a precondition, not a rescue.
3. **Corpus independence.** Any claim intended to transfer needs a
   second, structurally different corpus — real text by default.
4. **Scale floor enforced.** No architecture-ranking claim below the
   measured 1M-3M floor without re-running the gate.
5. **Weakest-form reporting.** The builder states the result in the
   weakest form the data supports; the verifier may strengthen it.
   Directly targets the ten-for-ten bias in section 3.
6. **Portfolio gate.** Already decision 0011: name the axis, the
   independence, and the running product, or get no compute.

---

# Second synthesis (2026-08-17, after 12 experiments)

## The pattern nobody had named

Sort every experiment by what it varied:

**Group 1 — hold the learning process fixed, allocate better.**
Objective reweighting (1.05x), data selection (~1.0x), multi-horizon
(<1x), adaptive compute (dead, <2x ceiling), growth (~0.004 nats),
sparsity (2.31x arithmetic ceiling, 0.81x deployable). SIX experiments,
zero factors. Every one of them takes the learning machinery as given
and tries to be smarter about where loss, data, compute or parameters
go.

**Group 2 — change the learning process itself.** Exchange rate
(max 2.01x), consolidation (collapses at k=4), continual learning
(allocation buys 0.09% of rehearsal). THREE experiments, no factor
either — but they produced MECHANISM rather than bounds, and that is
the difference that matters.

**Group 3 — the instrument.** The retrodiction gate FAILED at ~1M
params; ten of ten first-pass claims were corrected downward.

## The two findings that reframe the programme

**(a) The exchange rate may be an information bound, not a machinery
gap.** e013's verifier found that with repetition free, the gradient
count and the context count measure the SAME quantity — how many
distinct examples identify the rule — so the ratio is pushed toward 1
by construction. If identification is the binding constraint, then no
architecture, objective or consolidation change can buy a large
sample-efficiency multiplier, because they are all downstream of an
information requirement. The only remaining lever would be PRIORS:
reducing how many rules are plausible before any example is seen.
This is testable directly on corpora where we know the rule count.
Dossier 016.

**(b) The continual-learning failure is ARBITRATION, not capacity.**
e015 + e004g together: the newest write is unconditional and answers
under every prior cue; prior skills survive in the weights and are
merely inaccessible; rehearsal's real function is making the newest
write conditional, not protecting representations. Both allocation arms
constrained WHERE an update lands and never what the model outputs on
old cues, so the hypothesis was never tested against the failure that
occurred. This is the first time the lab has LOCALISED a bottleneck to
a mechanism instead of bounding a multiplier, and a retrieval problem
is a far easier target than a destroyed one. Dossier 017.

## What this implies for what to run

- Group 1 is closed. Do not fund another allocation trick.
- The learning axis is not closed, but it is now bounded from below by
  016's question. Run 016 BEFORE any further consolidation work: if the
  identification floor binds, several queued experiments are already
  answered and should not be built.
- 017 is the only experiment in the lab's history aimed at fixing a
  mechanism the evidence located, rather than testing one chosen in
  advance. That makes it the highest-value build even though its prize
  is not yet sized.
- The instrument work is not overhead. Three grids were saved from
  producing uncallable verdicts by a check that costs minutes, and two
  experiments turned out to be measuring their own instruments.
