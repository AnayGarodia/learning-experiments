# Prior-art sweep 2: LLM-scale CL, plasticity, "solved" claims, benchmarks (2026-09-09)

Key papers: TFGN 2605.15053 (unverified "closes forgetting at 8B", no reception); RL's Razor 2509.04259 (forgetting ~ KL to base; on-policy forgets less); Retaining by Doing 2510.18874; SDFT 2601.19897 (self-distillation CL); Denser != Better 2607.01763 (dense SD forgets more than GRPO; on-policy alone insufficient); Merge before Forget 2512.23017; GCWM 2605.09608; Sparse Memory Finetuning 2510.15103 (NQ F1 drop 11% vs 71% LoRA vs 89% full FT) + 2604.05248; Self-generated replay 2605.26097 (fails near capacity saturation); Nested Learning/HOPE 2512.24695; Plasticity scaling 2606.24752 (scale delays plasticity loss sublinearly); Spectral Collapse 2509.22335; Weight decay plasticity 2602.11137; Plasticity survey 2411.04832; Face to Forgetting 2601.22012; Modular memory position 2603.01761; When does CL require learning 2607.07847 (weights vs scaffolding boundary); CL-Bench 2606.05661 (naive ICL beat memory systems); Forager 2605.01131; MEAL 2506.14990; CLEAR 2201.06289.

Open problems the field names:
- Why KL shift destroys prior knowledge; does the law hold at scale (Shenfeld/Agrawal).
- When must adaptation be in weights vs external memory (Harrington/Malik/Bai; Lambert vs Dwarkesh).
- Plasticity loss has no mechanism; scale only delays it (Klein; Millidge).
- Forgetting persists near capacity; replay-free tricks stop working there (Marek/Wilson).
- Evaluation broken: lifetime HP tuning erases differences; incremental classification is the wrong problem (Mundt; van de Ven).

Cheap benchmarks: Permuted/Split-MNIST, Split-CIFAR (Avalanche), CLEAR (frozen features), Forager (continual RL), MEAL (JAX), fact-injection on 0.5-3B LM with NQ/IF-eval retention, CLINC/Banking/DBpedia class-incremental text (HOPE/Sleep).
