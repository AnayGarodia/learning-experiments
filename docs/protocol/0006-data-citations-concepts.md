# 0006 — Cards carry raw data and citation trees; a concepts layer teaches from first principles

Date: 2026-08-15
Status: accepted

## Context

Anay reviewed the first nine cards and set three requirements: (1) store
actual experimental data — exact values, tables, figure data; (2) store
citation trees, related work, further work; (3) explain core concepts from
first principles, because he wants to learn the field deeply while doing
the work and does not want expertise assumed.

## Decision

FORMAT.md (still a draft under D0) gains Data, Citation tree, Further
work, and Concepts sections. A new `library/concepts/` directory holds
plain-language, from-scratch explainers; every card links the concepts it
uses; naming a missing explainer marks it as owed. Existing cards get
retrofitted rather than grandfathered.

## Reasoning

(1) A card without the numbers forces a re-read, defeating the library's
purpose; downstream work (the S3 predictor, instrument design) needs the
actual constants, not summaries. (2) Citation trees are how coverage
verification (sweep 2 in FORMAT.md) actually executes — you cannot chase
what you did not record — and contested claims cannot be graded without
recording the contestation. (3) The lab has one permanent member whose
understanding compounds across everything: teaching artifacts are not
overhead, they are how the lab's judgment improves.
