# Sample efficiency of LM pretraining: literature sweep (2026-09-09)

Framing number: humans reach adult competence on ~1e8–1e9 words; LMs train on 1e12–1e13 tokens. Everything below is what is measured about squeezing more out of a fixed small budget.

## 1. Data-constrained scaling laws

**Muennighoff et al., 2305.16264 (May 2023; JMLR 2025).** 400 runs, up to 9B params / 900B tokens, C4. Up to ~4 epochs of repeated data costs almost nothing vs unique data; fitted repeat half-life R_D* ≈ 15.4 (repeated tokens keep ~63% value at that count), and value goes to ~zero past ~16–40 epochs. Excess parameters decay faster than repeated data (R_N* ≈ 5.3), so under a data constraint spend compute on epochs before width. Code mixing: up to 50% Python causes no downstream loss on English tasks (~2x effective tokens). Perplexity filtering helps only on noisy data; dedup did not help on C4. Limitation: one dataset, one architecture, hyperparameters fixed.

**Prabhudesai et al., "Diffusion beats AR in data-constrained settings", 2507.15857 (Jul 2025).** 200 models, 7M–2.5B params, 25M–500M unique tokens, up to 800 epochs. Fitted AR R_D* ≈ 31–32 (different fit range from Muennighoff's 15.4; report both). Masked diffusion R_D* ≈ 493–513: it keeps improving past 500 epochs with no overfitting, while AR degrades past ~50. Critical compute where diffusion wins: C_crit ≈ 2.12·10^1.956·U^2.174 (U = unique tokens). Diffusion also wins downstream (ARC-E, BoolQ, COPA, HellaSwag). Limitation: narrow fit range; AR/diffusion hybrids untested.

**2606.06888 (Jun 2026), "Data-constrained pretraining: improved regularization and scaling laws".** 72M–1.4B params, 100M–400M unique tokens, 4–32 epochs. Best weight decay is 4–32x the usual 0.1 (0.4 at 72M up to 3.2 at 1.4B on 100M tokens). Masked-input regularization (auxiliary NTP loss on randomly masked inputs) beats the strongly regularized baseline by an amount equivalent to ~1.3x more unique data; +10.2 BoolQ at 1.4B. Proposes "SoftQ" law (multiplicative N×D coupling). Limitation: ≤1.4B params, ≤400M tokens, architecture fixed.

**2605.01640 (May 2026), "Prescriptive scaling laws for data-constrained training".** One-parameter additive overfitting penalty; weight decay λ=1.0 removes ~70% of overfitting; beyond a repetition threshold spend on capacity, not epochs.

**2511.13421 (Nov 2025), theory.** Effective reuse rate E(K,N) ≈ K for small K, then plateaus at a level that grows with dataset size (Θ(log N) strongly convex). "4 free epochs" is not universal; bigger datasets tolerate more repeats. Linear regression only.

## 2. BabyLM 2023–2025

Note: scoring changed every year (2023 0–1 aggregate; 2024 per-benchmark; 2025 NLP-score + human-likeness composite), and each year's baseline is the prior winner. Do not compare across years.

**2023 (2504.08165, Warstadt et al.).** 31 papers, 162 models. Strict winner ELC-BERT (LTG-BERT variant): aggregate 0.74 vs OPT-125M baseline 0.60; Strict-Small 0.66 vs 0.50; BLiMP 0.85/0.80. Loose winner Contextualizer (recombining chunks across contexts) 0.73. Both beat Llama-2 (0.71 aggregate). What helped: LTG-BERT architecture, short sequences, 200+ epochs, ensemble distillation (BabyLlama 2308.02019: 58M student beats both teachers). What did not: curriculum — 13 of 31 teams (42%) tried it, no consistent gains; multimodal/Loose underperformed.

**2024 (2412.05149, Hu et al.).** 31 submissions. GPT-BERT (2410.24159), a 1:15 causal:masked hybrid, won both text tracks: 10M — BLiMP 81.2, GLUE 76.5, EWoK 54.6 (vs LTG-BERT 80.6/74.5); 100M — BLiMP 86.1, GLUE 81.5, EWoK 58.4 (vs 85.3/77.9/56.0). BabyLlama-2 (2409.17312): 345M distilled on 10M words beats 100M-word baselines. Organizers: gains from hybrid objectives, data modification, objectives; curriculum "modest or inconsistent"; strong FLOPs–performance correlation; no multimodal submission beat baseline.

**2025 (aclanthology 2025.babylm-main.28, Charpentier et al.).** 32 models, 15 participants. Strict: no submission beat the GPT-BERT baselines (best NLP score Simple-Diffusion 58.4 vs baselines 60.1–63.0). Strict-Small: AMLM-Hard-Decay (adaptive masking of hard tokens) 58.3 vs best baseline 57.3 (+1.0). Interaction: BLM (RL from teacher scores) 54.4 vs SimPO 54.1 (noise-level). Multimodal: below baseline, third year. Organizers' takeaways: objective/architecture changes win again (diffusion MLM, MoE, teacher RL); curriculum popular but not best; no FLOPs–performance correlation except in Interaction (contradicts 2024). Appendix D: model-driven selection (influence, surprisal) beats human-ordered curricula; synthetic/simplified text helps only as ~50% augmentation, hurts as replacement; contrastive good/bad pairs beat plain synthetic; tokenization (morphology-aware) gives ~20% EWoK and ~40% entity-tracking gains, "rivaling objective or architectural modifications"; MTP with reverse curriculum beats NTP on all tasks but BLiMP at 130M/10M (avg 38.5 vs 37.1, 2025.babylm-main.41).

## 3. Objective-level levers

- **Multi-token prediction, Gloeckle 2404.19737 (Apr 2024).** Gains grow with size; at 13B +12% HumanEval; helps under multi-epoch training. Small models (<1B) often hurt; curriculum (2505.22757) recovers some at 1.3B, and BabyLM 2025 shows +1.5 avg at 130M. No token multiplier stated.
- **Masked diffusion / random-order objectives.** 2507.15857 above: ~16x more useful epochs than AR. Consistent with BabyLM: masked/hybrid objectives win all three years; MIR (2606.06888) 1.3x. Mechanism claimed: random masking = implicit data augmentation over orderings.
- **Hybrid MLM+CLM (GPT-BERT).** Beats either alone at 10M/100M; no multiplier reported.
- **JEPA-style latent prediction for text.** LLM-JEPA 2509.14252 (Sep 2025): +14pp on NL-RX-SYNTH, but fine-tuning of an instruct model. Semantic Tube Prediction 2602.22617 (Feb 2026): "matches baseline with 16x less data" on the same single synthetic regex dataset. DLLM-JEPA 2606.00091 for diffusion LMs. Weak evidence: no general-pretraining result.
- **Contrastive/auxiliary.** Latent bootstrapping (2310.19420) failed at BabyLM 2023. Contrastive synthetic pairs helped in 2025 (data-level, not loss-level).
- **Pre-pretraining / meta-learning-flavoured.** 2502.19249 (Feb 2025): pre-pretraining a 1B model on a hierarchical formal language reaches the same loss with a 33% smaller natural-language budget (~1.6B tokens); attention heads from the formal stage carry syntactic evals. No explicit meta-learning pretraining paper with a token multiplier found.

## 4. Data levers

- **Curriculum, 2506.11300 (Jun 2025).** 200+ models to 100B tokens. Reaches baseline 18–45% fewer steps early/mid training; as warmup then random, +3.5% sustained. Gains fade toward convergence. Best difficulty signals: compression ratio, MTLD, Flesch. Matches BabyLM: ordering weak, selection stronger.
- **Selection/pruning.** Perplexed by Perplexity 2405.20541: 125M pruner, 3B target, +2.04 avg downstream, 1.45x fewer steps, holds in data-constrained regime. Perplexity correlations 2409.05816 beats DSIR at 160M–1.4B. OPUS 2602.05400: GPT-2 L/XL on 30B selected tokens beats 200B full; 6x on Qwen3-8B CPT; 4.7% overhead. Rho-1 2404.07965: token-level selection, matches DeepSeekMath with 3% of tokens (math CPT + FT, not general).
- **Metadata.** 2505.16570: URL prepending speeds training (secondary sources say ~33%; abstract gives no number); quality/topic tags don't; benefit needs longer inference prompts. Physics 3.3: domain tag cuts the junk-data penalty 20x → 2x.
- **Synthetic rephrasing.** WRAP 2401.16380 (Jan 2024): ~3x speedup on C4, >10% Pile ppl, 1.3B. BeyondWeb 2508.10975 (Aug 2025): 7.7x vs web, 2.7x vs Nemotron-Synth; 3B model on 180B tokens beats 8B on Cosmopedia; 3B rephraser gets +3.3pp of the 8B's +3.7pp; rephraser family barely matters; diversity needed at scale. Megadocs 2603.18534: 1.48x from 32 rephrases/doc, 1.80x stitched into long docs. EntiGraph 2409.07431: 1.3M source tokens → 455M synthetic, QuALITY 39.5 → 56.2 (raw repetition drops below base); log-linear in synthetic tokens; needs GPT-4-class generator, "no bootstrapping". All rely on an external model trained on ≥1e12 tokens and don't count generator compute.
- **Self-play / RL at small scale.** BabyLM Interaction track: +0.3 over baseline; 2510.20358 negative on dialogue+RL. No self-generated-data pretraining win found below ~1B.
- **Active selection.** Surprisal/influence-based selection helped at BabyLM 2025 (no multiplier); OPUS is the closest general result.

## 5. Knowledge-storage efficiency

**Physics of LMs 3.3, 2404.05405 (Apr 2024).** Synthetic biographies. 1000 exposures per fact → 2 bits/param, all architectures; 100 exposures → ~1 bit/param, and gated-MLP (LLaMA/Mistral) 1.3x worse than GPT-2 there. Junk 7:1 at 100 exposures: 20x capacity loss (3x/1.5x/1.3x at 300/600/1000); domain tag → 2x. int8 free, int4 halves capacity; MoE-32 only 1.3x loss. Limitation: synthetic facts, exposure counts not below 100.

**Physics 3.1, 2309.14316.** Without paraphrase/shuffle augmentation, extraction ≈0% even after instruction tuning; augmentation makes knowledge linearly encoded on entity names.

**Chang et al., 2406.11813 (NeurIPS 2024).** Facts are learned by micro-acquisitions that decay by a power law; more data doesn't improve per-fact acquisition; duplication accelerates forgetting; larger batch reduces forgetting.

**Kandpal et al., 2211.08411.** BLOOM-176B TriviaQA accuracy 25% → 55% as relevant docs go 1e1 → 1e4; extrapolation says 1e18 params to answer 1–10-document facts (R²=0.98).

**2609.04180 (Sep 2026), auxiliary views.** OLMo-2 1B–32B; source-only saturates ~20 exposures then degrades; paraphrase ~40; textbook/QA/blog "views" keep improving. Little advantage at 1B, widening at 7B–32B.

Sharpest framing: per-fact acquisition doesn't improve with more data (Chang), yet 2 bits/param needs ~1000 exposures (3.3). Humans get one.

## Measured multipliers

| Lever | Paper | Multiplier | Scale (params / unique tokens) | Held fixed | Caveat |
|---|---|---|---|---|---|
| Repeat data ≤4 epochs | 2305.16264 | ~4x tokens free; R_D*≈15 | ≤9B / ≤900B | compute | AR R_D*≈31 in 2507.15857 |
| Code mixing ≤50% | 2305.16264 | ~2x effective tokens | same | downstream tasks | English NL tasks only |
| Masked diffusion objective | 2507.15857 | R_D* ~500 vs ~31 (16x epochs) | 7M–2.5B / 25–500M | unique tokens | wins only past C_crit |
| MIR + strong WD | 2606.06888 | 1.3x unique-data equiv. | ≤1.4B / 100–400M | compute | ≤1.4B |
| Hybrid MLM+CLM | 2410.24159 | +1–4 pts, no multiplier | ~100M / 10M–100M | data | BabyLM only |
| Formal-language pre-pretraining | 2502.19249 | 1.5x (33% fewer NL tokens) | 1B / 1.6B | loss | one model size |
| MTP (small, reverse curriculum) | babylm-main.41 | +1.5 avg, no multiplier | 130M / 10M | data | hurts without curriculum |
| Curriculum ordering | 2506.11300 | 18–45% fewer steps early; +3.5% final | ≤100B tokens | steps | fades at convergence |
| Perplexity pruning | 2405.20541 | 1.45x fewer steps | 3B (125M pruner) | performance | holds data-constrained |
| Optimizer-space selection | 2602.05400 | 6.7x tokens (30B ≥ 200B) | GPT-2 XL | performance | needs proxy set |
| Rephrasing (WRAP) | 2401.16380 | ~3x | 1.3B / C4 | compute | external rephraser* |
| Rephrasing (BeyondWeb) | 2508.10975 | 7.7x vs web | 3B / 180B | tokens | external rephraser* |
| Megadocs | 2603.18534 | 1.48–1.80x | n/s | tokens | external rephraser* |
| EntiGraph | 2409.07431 | +16.7pp QA from 1.3M tokens | 8B / 455M synth | source docs | GPT-4 generator* |
| Domain tag on junk mix | 2404.05405 | 20x → 2x capacity loss | ≤1B synthetic | exposures | synthetic facts |
| JEPA/STP | 2602.22617 | "16x less data" | n/s | accuracy | one synthetic regex set; not pretraining |

*Generator compute and its ≥1e12-token pretraining not counted.

## What nobody has explained or tried

- Why per-fact acquisition needs ~100–1000 exposures and whether any objective (diffusion, MIR, auxiliary views) reduces the exposure count itself rather than just adding useful epochs; nobody has measured bits/param vs exposures below 100.
- Several levers are scale-gated (MTP hurts <1B, auxiliary views null at 1B, multimodal never wins at 100M). No account of what capacity threshold turns them on, or whether it is params, tokens, or exposures.
- Every synthetic-data multiplier uses a rephraser trained on ~1e12+ tokens; no result shows a model bootstrapping its own rephrases at 1e8–1e9 words. The human-vs-LM comparison is unanswered until this is tried.
- Diffusion's 16x epoch tolerance has no mechanistic model; whether it comes from order randomization, masking-rate diversity, or implicit ensembling is untested, and AR/diffusion hybrids are unexplored.
- Muennighoff (R_D* 15) vs Prabhudesai (R_D* 31) vs 2511.13421 (half-life grows with dataset size) disagree; nobody has fit the half-life as a function of unique-token count and regularization together.
- BabyLM finds tokenization and hyperparameters rival objectives at 10M–100M words, and 2025 found no FLOPs–performance correlation where 2024 did; no controlled study separates these confounds.
- Selection beats ordering everywhere, but no one has combined a selection policy with a repetition schedule (which examples to repeat, how often) — the natural analogue of spaced repetition.