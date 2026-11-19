# 0009 — Publishing is half the work; Anay learns by doing, not by delegation

Date: 2026-08-16
Status: accepted

## Context

Anay: talking about experiments, publishing them, and writing plain
explanations "is half of the work" — and he wants to learn from the
experiments himself, not have models do all the work while he learns
nothing. Also noted: the orders-of-magnitude goal will come from MANY
stacked optimizations, and "weird things can end up becoming scaling
laws" (inference-time compute is the precedent).

## Decision

1. **Anay writes all public prose; the lab produces the artifact
   pack.** (Clarified 2026-08-16.) For every experiment the lab
   delivers a complete artifact pack: publication-quality graphs,
   result tables, the verified facts and exact numbers with their
   provenance, method details sufficient for reproduction, and the
   honest caveats list (what could break at scale, what wasn't
   controlled). Anay writes the long-form technical blog posts and the
   X posts from that pack — no ghost-written prose. The
   pre-registration content is also packaged this way BEFORE results
   (calling shots in public is the honest version of prediction).
   Publishing is not marketing: per B7, the public track record of
   predictions-that-came-true IS the asset that buys compute and
   hardware conversations.
2. **Anay is in the loop as an experimenter, not an audience.** Every
   experiment produces: a plain-language walkthrough BEFORE the run
   (what we expect and why), the run itself operable by him (single
   command, readable output), and an after-lesson connecting the
   result to the map. The test: he can explain any arche result to a
   third person without notes. Fleet does mechanical volume; the
   science stays legible to and shared with him.
3. **Watch for new scaling AXES, not just better curves.** The biggest
   wins historically were new axes (params → data → inference-time
   compute). Candidate next axis on our map: accumulated experience —
   if the consolidation channel works, capability-vs-lifetime becomes
   a scaling law nobody has plotted. Every dossier closes with a
   "could this become an axis?" line.

## Reasoning

The lab's credibility strategy (predictability as the product) only
works in public. And a one-person-plus-agents lab whose person cannot
explain its results has failed its own knowledge-compounding premise —
the founder's understanding is the lab's most durable asset.
