# Decision 0012 — Identifiability, known-positives, corpus independence, weakest-form reporting

Status: DECIDED 2026-08-16, from RETROSPECTIVE.md. Extends decision
0010 (pre-launch protocol). Every clause exists because something in
the record failed for want of it.

## 1. Identifiability dry-run (before launch, mandatory)

Decision 0010 requires an MDE: can the design RESOLVE the effect?
That is necessary and not sufficient. Two grids passed their MDE and
still could not call a verdict (e010: 89% of savings cells censored and
1.2x shared compute range; e009b pilot: every oracle point in a single
frontier interval).

New requirement, in DESIGN.md before launch: **simulate the predicted
result and show the verdict rule fires.** Concretely — write down the
outcome each rival mechanism predicts, push those synthetic numbers
through analyze.py, and confirm each produces its intended verdict
category. If a predicted outcome yields INCONCLUSIVE, the design is
broken and must change before compute is spent.

This is cheap (minutes, no training) and would have caught both
failures above at design time.

## 2. Known-positive before any null is believed

A null result may not be reported as a fact about the world until the
instrument has been shown to detect a POSITIVE it should detect. The
lab ran five ICL negatives before ever validating that measurement
path, and only did so reactively. The validation then changed what the
negatives meant.

Rule: any experiment whose interesting outcome is a null must include a
positive control in its pre-registration. If the positive control is
itself expensive, that cost is part of the experiment's price, not an
optional extra.

## 3. Corpus independence

Twelve of fifteen experiments share the e005 synthetic corpus. Results
across them are therefore not independent, and that was never priced.

Rule: a claim intended to TRANSFER requires a second, structurally
different corpus — real text by default. A single-corpus result is
labelled as a claim about that corpus. This does not forbid shared
harnesses (they are genuinely valuable); it forbids the silent
inference from one corpus to the world.

Standing note: the lab's one real-data experiment (e-retro on enwik8)
produced its biggest surprise. Treat synthetic corpora as the special
case requiring justification, not the default.

## 4. Scale floor enforced

e-retro measured the floor: at ~1M non-embedding params this harness
inverts a settled architecture ranking; it is correct by ~3M.

Rule: no cross-architecture-family RANKING claim from below ~3M
non-embedding params without re-running the retrodiction gate on that
configuration. Within-family intervention signs, mechanism claims and
cross-rung trends are not covered by this restriction, but any card
using a sub-floor rung must say so.

## 5. Weakest-form reporting

Ten first-pass claims have been corrected, voided or withdrawn. **All
ten corrections made the result less exciting.** That is a systematic
bias in first-pass analysis, not noise.

Rule: the BUILDER states every result in the weakest form the data
supports, and names the strongest alternative reading as a hypothesis
for the verifier. The VERIFIER may strengthen a claim; the builder may
not. Inverting who is allowed to escalate puts the incentive on the
side that is currently biased.

Corollary already learned the hard way: report per-seed values, not
means, wherever a mean could hide a split (e011), and never quote a
multiplier whose magnitude equals one grid step (e011 again).

## 6. Portfolio gate (from decision 0011)

Restated for completeness: every dossier names its axis, its
independence from existing factors, and what the running product
becomes if it lands. No entry in PORTFOLIO.md, no compute.

## 7. Two instrument traps, added 2026-08-16 from e014

Both were found by TESTING a confound rather than arguing about it, and
both had already produced a committed conclusion that was then withdrawn.

**(a) A ratio of gains is dominated by capability, not by the variable
you changed.** e014's transfer ratio R = held-out gain / training gain
moved from 0.018 to 0.920 and was reported as a breadth effect. A
matched-capability control settled it: a deliberately WEAK model on the
narrow 128-task corpus scores R = 0.849 and passes the gate outright.
Most of the movement was capability, not breadth. R was withdrawn as a
gate.
Rule: any metric built as a ratio (or fraction) of IMPROVEMENTS must be
checked against a matched-CAPABILITY control, not merely a
matched-budget or headroom-normalised one. Headroom normalisation is
NOT capability control — e014 tried it first and it pointed the wrong
way.

**(b) Loss flattening at the end of training measures the LR SCHEDULE,
not convergence.** e014 added a `tail_drop` guard (loss fall over the
final 20% of training) to detect under-trained points. Under OneCycle
or any annealing schedule the LR goes to ~0 at the end of EVERY run, so
loss always flattens: a run at 2.5M tokens scored "converged enough".
The guard was withdrawn one commit after it was added.
Rule: convergence guards must not read the tail of an annealed curve.
Use a schedule-independent signal (held-out loss still falling at
matched LR, a budget-doubling check, or distance to a measured
irreducible floor).

Both traps generalise beyond e014 and should be checked for in any
experiment reporting improvement ratios or claiming convergence.
