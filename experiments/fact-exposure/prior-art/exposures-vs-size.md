# Prior art: exposures-to-store a fact vs model size (checked 2026-09-10)

Nobody has drawn exposures-needed vs parameters and fit it. Closest:
- Tirumala 2022 (2205.10770): T(N, tau) = passes needed for an N-param model to memorise a fraction tau of arbitrary text; decreasing in N; no fit; verbatim memorisation, not fact recall.
- Positional Fragility 2025 (2505.13171, App C.1): verbatim Rouge-L vs repetitions 1..256 at 1B/3B/8B; "1B needs ~64, 3B ~32, 8B ~16"; no fit; appendix. Also: memorisation depends on offset inside the context window (confound for any placement axis).
- Kandpal 2023 (2211.08411): QA accuracy vs number of relevant pretraining docs per size (observational); fit params-needed at fixed doc count, not the inverse.
- Physics of LMs 3.3 (2404.05405): bits/param at 1000 and 100 exposures across 1M-500M; nothing below 100; no exposures-vs-size curve.
- Chang 2024 (2406.11813): exposure count fixed at 10 (or 1), varies interval; 1B vs 7B qualitative; forgetting linear in log(steps).
- Lu 2024 (2406.15720): fact capacity linear in size; ~35-epoch floor; templates (up to 32) don't change memorisation rate on a completion metric.
- Allen-Zhu 3.1 (2309.14316): diversity of the fact's own text is everything on an OOD-QA metric (9.7% -> 96.6%).
- Auxiliary Views 2026 (2609.04180), Facts in Stats (2510.16096), Cram Less (2604.08519), Zucchet 2025 (2503.21676): diversity/frequency effects, none draw exposures vs size.

What is new here: (1) exposures-to-store vs N with a fitted form; (2) controlled 1-100 exposures for facts (every paper floors at 10, 35, or 100); (3) the 10M-100M range with per-exposure recall; (4) the "same fact string, fresh neighbours" axis: every diversity paper varies the fact's own wording, none varies only the surroundings.
Controls required: fix the offset inside the 512-token window for the identical-placement arm and randomise it for the fresh arm, else diversity is confounded with position (2505.13171). Report both a completion metric (Lu) and a rephrased-QA metric (3.1), since the two literatures disagree on whether diversity matters and the metric decides it.
