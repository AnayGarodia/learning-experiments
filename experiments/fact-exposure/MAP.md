# Map: where the sample inefficiency comes from, and what could close it

Written 2026-09-09 from three literature sweeps (prior-art/). Purpose: choose a paper topic from evidence. Nothing here is decided.

## 1. The gap, measured honestly

- Humans hear ~3-6M words/year (LENA daylong recordings): ~15-30M by age 5, ~55-110M by 18. Frank's upper bound is ~2e8 by 20.
- Frontier LMs train on 1e13 tokens. Text-only token gap: ~1e4-1e5. Not 1e6.
- Two confounds cut the other way. Multimodal input adds maybe 15x more bytes for a human, not orders of magnitude. And compute per token: a human brain at 1e14-1e15 FLOP/s spends ~1e14-1e15 FLOP per word heard versus ~1e12 for a 400B model per training token. Humans spend 100-1000x MORE compute per token, and a lifetime is within ~2 orders of magnitude of a frontier training run.
- What 100M words already buys a model: syntax and most lexical semantics (MiniBERTas, BabyLM), brain-alignment on par with big models. What it does not buy: world knowledge, commonsense, reasoning, and anything needing many exposures.

So the honest problem statement is: at ~1e8 tokens, models get the grammar and miss the knowledge and the reasoning; humans get both. And humans do it with far more compute per token, mostly offline (rehearsal, sleep, inner speech).

## 2. Where the tokens go (mechanisms of inefficiency, with the evidence)

| # | Mechanism | Evidence | Size |
|---|---|---|---|
| A | Gradient spent on already-learned tokens | Rho-1 trajectory audit: 51% of tokens are learned early and keep receiving gradient; 11% are never learnable | ~60% of gradient |
| B | Facts need 100-1000 exposures | Physics of LMs 3.3: 2 bits/param at 1000 exposures, 1 bit at 100; Chang et al.: per-exposure gain does not improve with more data; single exposure gone by end of run | 100-1000x vs a human's ~1 |
| C | Forgetting between exposures | power-law decay after each hit; learnability threshold shrinks tens of times at small batch | large, batch-dependent |
| D | Repetition stops paying | AR: ~4 free epochs, returns gone by 15-40; masked diffusion: ~500 useful epochs; masked-input AR closes part of the gap (regularization, not objective) | 4x to 16x epochs |
| E | Surface-form lock-in | facts seen in one form are stored but not extractable (Physics 3.1); paraphrase/auxiliary views keep improving to ~40 exposures | 2x exposures, more at scale |
| F | Optimizer noise | tuned benchmark: matrix preconditioners 1.4x at 0.1B, 1.1x at 1.2B | small |
| G | Junk dilution | 7:1 junk cuts capacity 20x at 100 exposures; a domain tag reduces it to 2x | 10x in dirty data |
| H | Objective | multi-token prediction helps only above ~1B; hybrid masked+causal wins BabyLM three years running | small at 100M scale |

The big terms are B, C, E (knowledge storage) and A, D (wasted gradient / repetition). F, H are small at this scale. G is a data-cleaning issue.

## 3. Candidate mechanisms (the answer space), and what is known about each

1. **Spend compute per token instead of tokens (offline rehearsal, self-generated views).** Humans do this. Model side: repetition is free to 4 epochs; diffusion tolerates 500; every synthetic-rephrasing multiplier (3-8x) uses an external model trained on 1e12 tokens. NOBODY has shown a model bootstrapping its own rephrases at 1e8 tokens. Open and directly testable.
2. **Fix the exposure count itself (one-shot knowledge storage).** Every method in the literature adds useful epochs; none reduces exposures-to-store below ~100. No measurement of bits/param below 100 exposures exists. Humans store a fact in ~1 exposure. This is the largest single term in the table and the least explored.
3. **Stop wasting gradient (token selection / selective LM).** Rho-1: 5-10x fewer tokens on math with a reference model; general-domain +6.8%. Needs a reference model; at 1e8 tokens the reference is the problem.
4. **Meta-learning / learning to learn.** Minnow: 5.8M tokens matches Llama-3-8B on novel-word learning. MAML-interleaved pretraining reaches target loss 1.6x sooner but hurts perplexity. Real but narrow.
5. **Priors / architecture.** Distilled Bayesian priors need 10x fewer examples on toy tasks; formal-language pre-pretraining saves 33% of NL tokens at 1B. Modest, and reviewers have seen it.
6. **Grounding, interaction, curriculum.** Three years of BabyLM: multimodal never beats text-only on text tasks; curriculum inconsistent; learner-chosen data has never been tried in an LM (all "interaction" work is teacher-selected). The one untried item there is active selection by the learner.
7. **Test-time compute for data.** Substitutes for extracting low-probability capabilities, not for acquiring new ones. Not the lever here.

## 4. What is untried and would matter (from the sweeps' own gap lists)

- Exposures-to-store below 100, as a function of objective, batch, paraphrase, and model size. Nobody has measured it.
- A model that generates its own auxiliary views from a 1e8-token budget, so the 3-8x rephrasing multipliers are tested without an external 1e12-token teacher.
- Why diffusion tolerates 16x more epochs: ordering, masking diversity, or ensembling. No mechanistic model; AR/diffusion hybrids unexplored.
- Selection combined with a repetition schedule (which examples to repeat, when): spaced repetition for pretraining. Selection beats ordering everywhere, but nobody has scheduled repeats.
- Learner-chosen data at pretraining scale.

## 5. Candidate paper topics (for discussion, not decided)

T1. **The exposure problem.** Measure exposures-to-store from 1 to 1000 across objective, batch, paraphrase, size; find what moves the curve; then a mechanism that stores facts in few exposures without forgetting (the human number). Biggest term, least explored, clean measurement, risk: the mechanism may not exist in 16 days; the measurement paper is still new.
T2. **Compute-for-data by self-generated views.** At fixed 1e8 tokens, can a model bootstrap its own paraphrases/views to earn the multi-epoch gains the literature gets only with an external teacher? Directly tests the human strategy (rehearsal with variation). Risk: small models paraphrase badly; the 2025 BabyLM result says synthetic text hurts as replacement, helps as ~50% augmentation.
T3. **Spaced repetition for pretraining.** Selection + repetition schedule per example driven by the model's own per-token loss trajectory (the Rho-1 audit as a control signal). Cheap, mechanistic, bitter-lesson-compatible (the schedule is learned from the model's own signal). Risk: incremental-looking unless the multiplier is large.
T4. **What diffusion's 500 epochs are made of.** Decompose the multi-epoch tolerance; if it is masking diversity, port it to AR and get the 16x without the diffusion cost. Risk: 2510.04071 already partly did this.

## 6. Diagnostics that would locate the inefficiency in a 10M-100M-token run on one T4 (each one experiment)

- D0. At fixed unique tokens, performance vs FLOPs at 1/4/16/64 epochs with the best known regularization (strong weight decay, input masking, paraphrase augmentation). If the curve keeps rising, the problem is compute allocation (mechanisms 1, 2); if it saturates, the exposure count is the wall. Comes before any mechanism.
- D1. Synthetic-fact injection (>=1000 facts, >=100 per exposure bucket, fixed schema so the template is learned from the aggregate; paraphrase arm varies only the context around the fact) at 1/3/10/30/100/300 exposures, duplicate vs paraphrase, probes after each exposure: gives the local exposures-per-bit curve and separates storage / extraction / forgetting. (Feeds T1.)
- D2. Rho-1 token-loss trajectories with a small reference model: fraction of gradient on already-learned tokens. (Feeds T3.)
- D3. AR vs input-masked AR vs masked diffusion at 1/10/50 epochs on 25M unique tokens: regularization vs objective. (Feeds T4/T2.)
- D4. Batch 32 vs 512 at fixed tokens scoring held-out loss AND injected-fact retention. (Feeds T1/T3.)
- D5. Self-paraphrase loop: model rephrases its own corpus at epoch k, trains on the mix, vs plain repetition and vs external-teacher paraphrase. (Feeds T2.)
- D6. pass@1 vs pass@64 on a probe: absent vs low-probability capability. (Rules T7 in/out.)
