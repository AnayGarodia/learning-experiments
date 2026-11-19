# Prior-art sweep 3: what accepted CL / TTT papers look like (ICLR 2026, NeurIPS 2025, ICML 2026)

ICLR 2026 orals: In-Place TTT 2604.06169 (MLP down-proj as fast weights; 500M scratch -> 14B pretrained; RULER, ppl, HellaSwag/ARC/MMLU; objection: Qwen-only, 4B small); FIRE 2602.08040 (isometric reinit; CIFAR/TinyImageNet/GPT-0.1B/Atari; objection: few competitors); CompSLOT (slot-attention plug-in). Posters: OSFT 2504.07097 (SVD subspace freezing; 13 baselines; survived a 2), LaCT 2505.23884 (rejected at NeurIPS first for thin eval).
NeurIPS 2025: AnaCP 2511.13880 spotlight, Nested Learning 2512.24695 (4/4/4/4, undefined jargon, theory intuitive), Titans 2501.00663 (4/5/5/5), SEAL 2506.10943 (penalised for non-standard evals), Compact Memory 2511.09167.
ICML 2026: SDFT 2601.19897 spotlight, VLAs resist forgetting 2603.03818 spotlight. Sparse Memory Finetuning 2510.15103 was DESK-REJECTED at ICLR 2026 (no signal).

What gets accepted: mechanism-level idea with a scaling story (repurpose something existing, show it across sizes/domains); retention on standard capability suites (HellaSwag/MMLU/IFEval or RULER/NIAH) or standard CIL benchmarks; broad baseline tables incl. strong non-CL controls (SFT/LoRA/replay/EWC/MTL upper bound); concrete rebuttal additions; framing "X already inside the model can do CL if used right"; simplicity + practicality over novelty.
What gets rejected: small custom setups instead of established benchmarks; thin eval for an architecture claim; recombination without differentiating experiments; benchmarks that concatenate datasets; vague theory.
Most-used baselines: ER, LoRA family (SeqLoRA/O-LoRA/InfLoRA), full SFT, EWC, Titans/DeltaNet/Mamba2 for TTT papers, RanPAC/L2P for PTM-CIL.
Most-used benchmarks: HellaSwag/PIQA/ARC/MMLU retention; CIFAR-100/ImageNet-R/CUB CIL; RULER/NIAH/BABILong; standard LLM-CL 5-task + TRACE; Wiki/Pile ppl at 170M-1.3B.
Kill-shots: Qwen-only, 3-4B-only, appendix-buried results, custom benchmarks.
Accepted shape for a scalable-CL ICLR 2027 paper: mechanism inside an existing pretrained LLM; >=2 model families, size sweep; forgetting on standard suite + a knowledge-incorporation task; SFT/LoRA/replay/memory-system baselines in one table; budget-matched cost.
