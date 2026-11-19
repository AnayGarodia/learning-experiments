# Prior-art sweep: test-time-updated memory and fast weights (as of 2026-09-09)

Per system: mechanism / evaluated on / key number / stated limitation / scale. "n/s" = not stated in what I could verify; nothing backfilled.

## Named systems

**Fast Weight Programmers** (Schmidhuber/Irie; 2102.11174, Feb 2021, ICML'21). Slow net emits key/value outer products that program a fast weight matrix; linear attention shown to be exactly this; adds delta-rule write to correct stale key->value mappings. Eval: synthetic retrieval + language modeling. Key number: delta-rule FWP beats additive linear attention when pairs exceed d_k capacity. Limitation: capacity bounded by d_k; memory is sequence-local. Scale: n/s. Lineage: self-referential weight matrix (2202.05780, 2022); **Automated Continual Learning** (2312.00276, TMLR Feb 2025): self-referential net meta-learns its own in-context CL algorithm; Split-MNIST replay-free + image-classification sequences; beats hand-crafted meta-CL; limitation: lags pretrained-backbone CL methods. **Blending KV and fast-weight memory** (2506.00744, NeurIPS'25): three routings of inputs to softmax-KV vs fast-weight memory in a hybrid; LM, retrieval, algorithmic tasks; scale n/s.

**TTT layers** (Stanford; 2407.04620, Jul 2024). Hidden state is a small model (linear or 2-layer MLP) updated by one self-supervised gradient step per token; mini-batch TTT for parallelism. Eval: 125M-1.3B on the Pile (Chinchilla recipe), Books3 to 32K, vs Llama-style Transformer and Mamba. Key number: at 32K matched FLOPs, TTT-Linear/MLP keep improving with context while Mamba plateaus after 16K. Limitation: substantial wall-clock cost; TTT-MLP hits memory-I/O limits. Scale: 1.3B.
**LaCT** (2505.23884, May 2025): 2K-1M-token chunk updates, ~70% A100 utilisation, state up to 40% of params; NVS/LM/video.
**TTT-E2E** (2512.23675, Dec 2025). Sliding-window Transformer keeps doing next-token-prediction gradient steps on the context at test time; meta-learned init. Eval: 3B / 164B tokens, contexts to 128K vs full attention, Mamba2, GDN. Key number: matches full-attention loss-vs-context scaling; 2.7x faster at 128K. Limitation: weights reset per sequence; no cross-document retention measured. Scale: 3B, 164B tokens.

**Gated DeltaNet** (2412.06464, Dec 2024). Delta rule (targeted overwrite) plus scalar decay gate on a matrix fast-weight state; chunkwise parallel kernel. Eval: 400M/1.3B on 100B FineWeb-Edu tokens; LM-harness, S-NIAH, six real retrieval tasks. Key number: 1.3B wiki ppl 16.42 vs Mamba2 16.56; S-NIAH-PK at 8K 91.8% vs Mamba2 30.4%. Limitation: pure recurrent still far behind attention on real retrieval (30.6 vs 40.1 for the H2 hybrid); mixed length extrapolation. Scale: 1.3B/100B.
**DeltaProduct** (2502.10297, Feb 2025, NeurIPS'25). n_h delta steps per token = diagonal-plus-rank-n_h Householder transitions. Eval: state-tracking and LM. Key number: state tracking and extrapolation improve monotonically with n_h; numbers n/s. Limitation: n_h-fold cost. Scale: n/s.

**Titans** (Google; 2501.00663, Dec 31 2024). Deep MLP long-term memory updated at test time by the gradient of an L2 key-value loss with momentum ("surprise") and a data-dependent weight-decay forget gate; three placements (MAC/MAG/MAL) around sliding-window attention. Eval: 170M-760M on 15-30B FineWeb-Edu tokens; LM ppl, seven commonsense tasks, S-NIAH, BABILong, time series, DNA. Key number: 760M MAG wiki ppl 18.61 vs GDN-H2 19.88; BABILong above GPT-4 and Llama-3.1-70B; NIAH beyond 2M tokens. Limitation: authors say larger-model results unfinished; deep memory costs compute. Scale: 760M, 30B tokens.

**MIRAS** (Google; 2504.13173, Apr 2025, ICLR'26). Any sequence model = {memory architecture, attentional-bias loss, retention gate, optimizer}; instantiates Moneta (l_p loss, l_q retention), Yaad (Huber), Memora (softmax/KL retention). Eval: 120M-1.3B on 15B/30B/100B FineWeb-Edu; LM, commonsense, S-NIAH, recall. Key number: all three beat Transformer++, GDN and Titans at 340M-1.3B; best S-NIAH-PK. Limitation: delta-rule family has theoretical capacity limits; no CL eval. Scale: 1.3B/100B.

**ATLAS** (Google; 2505.23735, May 2025, ICML'26). Omega rule: memory optimised over a sliding window of the past c tokens rather than the current one; polynomial key features raise capacity O(d_k) -> O(d_k^p); Muon-style memory update. Eval: 340M-1.3B on 15-100B FineWeb; LM, reasoning, RULER, BABILong. Key number: ~80% BABILong at 10M context. Limitation: Transformers still win in-context recall. Scale: 1.3B/100B. Same group: TNT (2511.07343, ICLR'26) makes Titans/TTT chunkwise training 17x faster, still below Transformers; Lattice (2504.05646) orthogonal-update compression; Memory Caching (2602.24281, ICML'26) caches recurrent-state checkpoints, no CL claim.

**Nested Learning / HOPE** (Google; 2512.24695; NeurIPS 2025 paper, Nov 2025 blog, arXiv Dec 31 2025). Model = nested optimisation levels with different update frequencies; optimizers reread as associative memories; HOPE = self-modifying Titans (memory writes its own update rule) + Continuum Memory System (chain of MLPs, level l updated every C^(l) tokens). Eval: LM/commonsense at 760M/30B and 1.3B/100B; CL: class-incremental CLINC/Banking/DBpedia, CTNL new-language translation, LongHealth QA, using Llama-3B/8B with MLPs converted to CMS plus 15B tokens continual pretraining, baselines ICL/EWC/InCA; NIAH, BABILong. Key number: CL results are figure-only (Fig. 6): HOPE best of ICL/EWC/InCA; no table. Limitation: online consolidation only; CL eval is on a converted Llama backbone, not the from-scratch LM setup. Scale: 1.3B/100B (LM); Llama-8B + 15B tokens (CL).
**Language Models Need Sleep** (Google; 2606.03979, Jun 2026). Adds offline "sleep" to HOPE/CMS: upward distillation from a low-rank fast expert into slow weights, plus RL "dreaming" rehearsal on self-generated data. Eval: CLINC/Banking/DBpedia, CTNL, BABILong, SEAL-style knowledge incorporation, few-shot ARC, AIME. Key number: knowledge incorporation 48.9 vs SEAL 46.7; ARC 80.0 vs 72.5; AIME-24 79.2 vs 76.6 (Qwen3-8B). Limitation: SEAL-style SFT inner loop is expensive; forgetting can accumulate across sleep cycles; dreaming only samples existing knowledge. Scale: 1B-8B.

**Memory Layers at Scale** (Meta; 2412.09764, Dec 2024). Product-key sparse KV lookup (top-k of up to 64M slots) replacing some FFNs; custom kernels at 3 TB/s; memory sharded across GPUs. Eval: 134M-8B base, up to 128B memory params, 1T tokens; NQ, TQA, HellaSwag. Key number: 1.3B Memory+ NQ 13.68 vs dense 7.76, MoE 8.14; 8B/1T approaches Llama-3.1-8B (15T). Limitation: hardware efficiency still behind dense; factuality gain is empirical only. Scale: 8B + 128B memory, 1T tokens.
**Sparse Memory Finetuning** (Meta; 2510.15103, Oct 2025). Finetune only slots ranked by TF-IDF (activation on the new batch vs pretraining usage). Eval: 1.3B model, 1M slots, k=32; learn 1,000 TriviaQA facts or 100 SimpleQA docs; retention on NQ F1, HellaSwag, GSM8K NLL. Key number: NQ F1 drop 11% vs 71% LoRA vs 89% full FT at equal acquisition. Limitation: facts only; 1.3B only; optimizer interaction unexplored. Scale: 1.3B; hardware n/s.
**Improving Sparse Memory Finetuning** (2604.05248, Apr 2026). Retrofits Qwen2.5-0.5B with memory layers; KL-surprise slot selection; consumer hardware. Numbers n/s in abstract.

## 2026 work (Jan-Sep 9) building on these

| arXiv | Date | Mechanism | Scale / key number | What "CL" means |
|---|---|---|---|---|
| 2602.16704 REFINE | Feb 18 | GRPO trains fast-weight models on next-sequence prediction | LaCT-760M, DeltaNet-1.3B; beats NTP SFT on NIAH/LongBench | long context |
| 2603.06642 SR-TTT | Feb 26, v2 Jul 22 | surprisal-gated residual cache; v2 is a self-retraction (eval bug, 0% exact match) | small | long context |
| 2604.06169 In-Place TTT (ICLR'26 oral) | Apr 7 | MLP down-projection = fast weights, NTP-aligned chunk update, drop-in | 4B, 128K ctx | long context |
| 2604.07350 Elastic TTT | Apr 8 | Fisher-weighted elastic prior around an EMA anchor inside LaCT fast weights | 4D reconstruction only | forgetting inside fast weights |
| 2604.05248 Improving SMF | Apr 6 | KL slot selection, 0.5B retrofit | 0.5B consumer GPU | knowledge CL |
| 2605.12484 Fast-and-Slow | May 12 | optimized context as "fast weights" + RL slow weights (prompt method, boundary case) | n/s; 3x sample efficiency vs RL | task-sequence CL |
| 2606.03979 Sleep | Jun 2 | see above | 1B-8B | task/knowledge CL |
| 2606.21803 TTT-NTP (EMNLP'26 Findings) | Jun 19 | MLP fast weight supervised by next hidden state, drop-in, no pretraining change | 0.6B-8B; RULER +2.9..4.1, LongBench-v2 +3.7..5.6 | long context |
| 2606.25342 Lifelong ICL needs parametric attention | Jun 24 | position: KV cache cannot do lifelong ICL; fast weights currently lack capacity or cheap updates | none | lifelong |
| 2607.07847 When does CL require learning | Jul 8 | protocol recasting LLM benchmarks as sequences; prompts vs SFT vs RL vs Cartridges vs In-Place TTT | frontier + open | task/knowledge CL; context compression adds efficiency, not learning |
| 2608.01672 TTCD | Aug 3 | long-window teacher supervises short-window student fast weights; IP-TTCD retrofits MLPs | 340M/760M scratch 20-40B tok; CPT 10B tok on SmolLM2/Llama-8B/Qwen3; H100; RULER-32K 21.96 vs IP-TTT 9.29 | long context |
| 2608.02870 Maglev, 2608.12435 MARCH | Aug | recurrent memory via prefiller targets / cached state anchors | n/s | long context |
| 2608.21308 E2-TTT | Aug | closed-form chunk transition reproducing per-token momentum TTT | 1.3B scratch | long context |
| 2608.27763 Falcon (Zhang, Gu, Yao) | Aug 27 | NLMS-style normalized fast-weight updates; Falcon-1/2/3, Falcon-A | 124-130M, 50B FineWeb-Edu, 4xH100; ppl 17.10 best vs GDN; digit-addition extrapolation | long context framed as CL |
| 2603.07392 OAKS, 2606.05661 CL-Bench | Mar, Jun | benchmarks; both find memory systems fail or lose to naive ICL | frontier models | knowledge/agent CL |

Nothing in-slice dated Sep 1-9 2026. Adjacent but out of slice: FOREVER 2601.03938 (replay), 2601.18699 (forgetting analysis), 2511.14823 DNH (theory).

Pattern: every 2026 fast-weight/TTT paper except Sleep and Improving SMF defines continual learning as compressing one long sequence; fast weights reset at sequence end and none reports task-sequence metrics. Only HOPE/Sleep (dense gradient memory) and Meta's sparse memory finetuning (slot lookup) run classic CL protocols, and they have never been compared.

## Whitespace

1. No fast-weight/TTT layer (TTT, GDN, Titans, ATLAS, In-Place TTT, Falcon) has been evaluated with task-sequence CL metrics (average accuracy, backward transfer, forgetting) at from-scratch LM scale; HOPE is figure-only on a converted Llama-8B.
2. No paper persists fast weights across documents or episodes and measures retention and interference; CMS is the only multi-timescale design and reports no interference numbers.
3. Regularising forgetting inside the fast weights themselves (Fisher/EWC on the memory state) exists only in Elastic TTT, for 4D vision, never for language.
4. Sparse-slot memory (Meta, TF-IDF/KL slot selection) and dense gradient memory (Google/Stanford) have never been compared on the same CL benchmark at the same scale.
5. Fast-to-slow consolidation exists only as Sleep's offline RL+distillation pipeline; no learned in-network consolidation rule with a measured compute-vs-forgetting curve.

## Standard small benchmarks (CPU or one small GPU, hours)

- Split-MNIST, Split-CIFAR-10, image-class sequences (ACL; replay-free, CPU-minutes).
- CLINC-150, Banking-77, DBpedia-14 class-incremental text classification (HOPE, Sleep).
- TriviaQA 1,000-fact injection with NQ-F1 / HellaSwag / GSM8K-NLL retention (sparse memory finetuning; 0.5B on consumer GPU per Improving SMF).
- Variable-digit addition length extrapolation (Falcon, 124M).
- S-NIAH-PK/N/W from RULER at 1K-8K (GDN, MIRAS, Titans; fine on 125M-350M).
- MAD synthetic suite: recall, compression, copy (TTCD).
- OAKS-BABI synthetic streaming facts with revisions.
- Parity / group state tracking (DeltaProduct).
Too big for the budget: BABILong-10M, RULER-128K, CTNL, LongHealth, TiC-LM, CL-Bench.
