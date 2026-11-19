# 0004 — Exhaustive literature review before any experiment

Date: 2026-08-15
Status: accepted

## Context

The program's purpose is to question assumptions the field has stopped
questioning. A program that does not know the literature cannot tell its
new results from old news, and inherits hidden assumptions instead of
testing them.

## Options considered

1. Review as-needed, per experiment.
2. Build the complete knowledge base first (structured cards, graded
   claims, audited extraction, verified coverage), wire it into the
   workflow, and maintain it permanently.

## Decision

Option 2. Format and process in `library/FORMAT.md`. Standing rule: search
the library before proposing or running any experiment. The folklore
entries the review surfaces become the program's target list (MAP
Section 4).

## Sequencing

Hands-on first: card 10–20 foundational papers manually to prove the card
format against real papers, then design the fleet sweep and scale
extraction with mandatory second-reader audits.

## Evidence / reasoning

Extraction without audit fills the base with confident misreadings, which
is worse than nothing. Formats designed in the abstract break on contact
with real papers, hence the manual first batch.

## AMENDMENT 1 (2026-08-17) — enforcement, after the rule was broken twice

Anay restated this rule after e017. The rule did not need restating; it
needed teeth. It was accepted 2026-08-15 and broken on 2026-08-16 and
2026-08-17.

**What went wrong, specifically.** For e017 the sequence was: write the
dossier → dispatch the builder → run the prior-art sweep. The sweep then
found the experiment's central diagnosis published at LLM scale in 2023
([[kotha2023-implicit-inference]]), its classifier-scale ancestor in
2019 ([[wu2019-bic]], where the field already calls it TASK-RECENCY
BIAS), the framing paper in 2019 ([[vandeven2019-three-scenarios]]), and
the leading intervention published in 2016 ([[li2016-lwf]]). Five
planned arms were settled work. The dossier had to be amended and the
builder redirected mid-build.

The cost was not only wasted design. The result was reported to Anay as
the lab's best finding before anyone checked whether it was known.

**Binding sequence, from now on. No exceptions, including for
experiments that seem obviously novel — that is exactly when this fails.**

1. PRIOR-ART SWEEP FIRST. Before a dossier is written, not after. The
   sweep must answer, in writing: (a) is the MECHANISM already
   diagnosed? (b) is each proposed INTERVENTION already published, and
   what did it achieve? (c) which arms are settled and must not be run?
2. The dossier's first section cites the sweep and states what is
   UNCLAIMED, in the narrowest honest form. "Nobody has done X" is a
   claim requiring evidence, not a default.
3. Only then: DESIGN.md, identifiability dry-run (0012), build, verify.
4. A dossier with no prior-art sweep gets no builder and no compute.

**Reporting rule.** Any result whose mechanism turns out to be
published is reported to Anay as a REPRODUCTION, immediately and
unprompted. Reproduction is a legitimate contribution and is to be
stated as one — the failure is not reproducing, it is presenting a
reproduction as a discovery.

**Standing note.** The sweep pays for itself even when it kills the
novelty claim. The e017 sweep removed five arms as settled, supplied a
free predicted-negative control from LwF's own documented failure mode,
and surfaced three cheap external tests. It cost one agent-hour and
saved several days of compute.

## AMENDMENT 2 (2026-08-17) — sweep the PREMISE, not only the mechanism

Amendment 1 required a prior-art sweep before any dossier and specified
what it must answer: is the mechanism already diagnosed, is each
intervention already published, which arms are settled. Dossier 020
complied with all of that and still failed, because **every question in
amendment 1 is about the proposed EXPLANATION and none is about the FACT
being explained.**

Dossier 020 opened with "direct model editing collapses after roughly 32
SEQUENTIAL edits" and "the same method handles ~10,000 edits when they
are applied in ONE BATCH", called both established, and built its entire
rival structure on the asymmetry between them. Its verifier found the
sequential onset placed in the HUNDREDS in the literature, the batched
claim contested in the opposite direction (arXiv 2405.00664), and four
methods reaching 10-20k sequential edits. The dossier also dismissed the
field's own explanation as unable to account for an asymmetry that may
not exist at the asserted size.

An experiment designed to explain a fact that is not established is
unfalsifiable in the worst way: it will produce a mechanism for
something, and nobody will notice the something was wrong.

**Added to the mandatory sweep questions, before a dossier may be
written:**

1. **Source the anomaly.** Every quantitative claim in the "why this
   matters" or "the anomaly" section gets a specific paper, a specific
   table or figure, and the specific number. Not "it is known that", not
   a remembered magnitude. If the number cannot be sourced, it is written
   as UNSOURCED in the dossier and the design may not depend on it.
2. **Check the anomaly is not contested.** Search specifically for papers
   REPORTING A DIFFERENT MAGNITUDE, not only for papers explaining it. A
   contested premise is a finding and often a better experiment than the
   one planned.
3. **Steelman the field's own explanation before dismissing it.** If the
   dossier says "the standard account cannot explain X", find the
   strongest published version of the standard account and state why it
   fails against ITS OWN numbers. Dossier 020 dismissed parameter drift
   in one paragraph while four drift-side methods were reaching 10-20k
   sequential edits.

Rule of thumb, since this has now cost us twice: **the sweep must be able
to kill the QUESTION, not only the ANSWER.** A sweep that can only
discover "someone already explained this" is half a sweep.

Tonight's tally, recorded because it is the argument for this amendment:
of four things that looked like findings, three were reproductions of
published work (context distillation as our write mechanism; the CE-vs-KL
retention contrast; the missing-prior-edit-constraint reading of editing
collapse) and the fourth rested on an unsourced premise. All four were
caught internally.
