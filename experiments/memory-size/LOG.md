# Experiment log

Rules: every entry has a hypothesis, a prediction written BEFORE the run, the run, the measured result, and a conclusion. Numbers are real only if run. Held-out means never touched during development.

Compute (verified 2026-09-09): Mac M5 34GB with MPS (tiny runs); Kaggle free GPUs, account works (scale-ups); crunch = 8 CPU / 64GB, no GPU; Azure has zero GPU quota.

Directive (Anay, 2026-09-08 night): start tiny, scientific method, aim at solving continual learning, scalable methods, bitter lesson.

Deadline: abstract Sep 18, paper Sep 25 (ICLR 2027). One submission.

---

## E0. Harness and baselines (2026-09-09)

Purpose: a mechanism-independent testbed so every later hypothesis is measured the same way.

Metrics (standard, Lopez-Paz & Ranzato 2017): average accuracy at the end (ACC), backward transfer / forgetting (BWT), forward transfer (FWT). Task stream: Permuted-MNIST (domain-incremental, 10 tasks) and Split-MNIST (class-incremental, 5 tasks). Both run on MPS in minutes.

Baselines: fine-tune (lower bound), experience replay (the thing that actually works at scale), multitask joint training (upper bound).

Prediction: fine-tune forgets almost everything on Split-MNIST class-incremental (ACC near 20%), replay recovers most of it, joint is ~98%. On Permuted, fine-tune ACC ~70-80%, replay ~90%+. If the harness does not reproduce these known numbers, the harness is wrong.

Result (lr 0.1, 1 epoch, 3 seeds, Split-MNIST class-incremental): finetune ACC 0.185 BWT -0.99; replay(500) ACC 0.867 BWT -0.12; joint ACC 0.930 BWT -0.03. Matches the prediction. Harness is sound. (At lr 0.01 joint was 0.82: under-trained, so lr 0.1 is the default now.)

Conclusion: E0 passed. Use this harness for everything downstream.

---

## E1. Does forgetting scale away with sparse memory capacity? (2026-09-09, night)

Why this question: the sweep says (a) forgetting persists whenever a model is near capacity and replay-free tricks stop working there (Marek/Wilson 2605.26097); (b) Meta's sparse memory finetuning shows slot-local updates cut forgetting 89%->11% at one size, but nobody has drawn forgetting as a function of memory size; (c) the bitter lesson says the fix should be one that gets better with more compute/params, not a cleverer regularizer. If forgetting in a sparse-slot memory falls as a power law in the number of slots at constant active compute, that is a scalable route and a new measurement. If it does not, the shared parts (query net, readout) are the bottleneck, which tells us what the mechanism must protect.

Hypothesis H1: in a model with a product-key sparse memory layer (top-k of N slots, k fixed), backward transfer (BWT) on Split-MNIST and Permuted-MNIST improves monotonically with N, with no replay and no regularizer. A dense MLP of matched active width does NOT improve BWT with width.

Prediction (before running): memory model BWT goes from about -0.9 at N=256 to better than -0.4 at N=65536 on Split-MNIST; dense MLP stays around -0.95 across widths. Permuted-MNIST: memory BWT improves from about -0.3 to better than -0.1; dense flat.

Falsifier: memory BWT flat or non-monotone in N. Then interference is in the shared query/readout path and the next experiment isolates it (freeze query net, per-slot readout).

Setup: 784 -> d=128 trunk -> [memory(N, k=32) | dense MLP(width=32..)] -> 10, SGD lr 0.1, 1 epoch per task, 3 seeds. N in {256, 1024, 4096, 16384, 65536}.

Result: (pending)

Amendments written before E1 finished (2026-09-09, ~03:00):
- Read Permuted-MNIST as the primary test of H1. Split-MNIST single-head collapses old-class logits through the shared readout regardless of what sits before it (the dense R matrix shows every old task at exactly 0.0), so Split BWT says nothing about slot-locality. If Split is to say anything, it needs a task-incremental (multi-head) variant.
- Confound: the memory query uses BatchNorm (needed: without it the value table never trains; LayerNorm and key rescaling did not fix it in a 1-run check). BN running stats track the newest task, which shows up as forgetting by itself. Control after E1: re-estimate BN stats on a mix of all tasks with no weight update, re-evaluate; if BWT is unchanged, BN is not the cause.
- Axis note: the PKM sweep holds active compute fixed (k=32) while total params grow with N; the dense sweep grows both. That is the bitter-lesson axis on purpose, and the comparison is not param-matched.
- Optimizer: Adam 3e-3 for all params, SparseAdam for the value table so untouched slots do not drift through stale momentum. SGD could not train the memory at all.

Decision: ARC synthetic streams are OUT as a testbed (sweep 3: custom benchmarks are a kill-shot). Appendix at most.

## Paper claim and protocol (draft, written before results; Anay to approve)

One-sentence claim (draft): forgetting in a sparse-slot memory falls with memory size at constant active compute, and a learned write rule (which slots get updated, trained to minimise interference) turns this into a continual-learning method that improves with scale instead of with cleverer regularisers.

Eval protocol (from what gets accepted, sweep 3): knowledge incorporation (TriviaQA 1,000 facts or SQuAD passages) then retention on HellaSwag / ARC-easy / PIQA (+IFEval if instruct); two model families (SmolLM2 or Llama-3.2, plus Qwen2.5); a size sweep within what a T4 fits (135M-1.7B); baselines full SFT / LoRA / replay / EWC / sparse-memory-TF-IDF in one table with a cost column. MMLU dropped or subsampled on T4 (too slow), stated in the paper.

Compute budget: Kaggle 30 GPU-hours/week, ~60 hours total before Sep 25. Smoke test pushed as kernel anaygarodia/cl-smoke (SmolLM2-360M, 20 full-FT steps, lm-eval on 3 tasks x 500).

Mechanism to name (pending E1): learned slot-write gate on the memory, replacing the TF-IDF / KL slot-selection heuristics of Meta's sparse memory finetuning. E1's falsifier branch (forgetting flat in N) points at the shared query path, in which case the gate goes on the query side.

Kaggle smoke result (2026-09-09 03:40): P100 fails (image torch has no sm_60 kernels: pass --accelerator NvidiaTeslaT4). On T4: SmolLM2-360M loads in 46s, 20 full-FT AdamW steps fine, total 66s. lm-eval 0.4.8 is incompatible with the image's transformers (AutoModelForVision2Seq gone); install unpinned lm-eval or write the retention eval by hand. The scale-up rail exists.

## E2. LM-scale baselines: fact injection then retention (Kaggle T4, SmolLM2-360M)

Protocol: 1,000 TriviaQA facts as Q/A pairs, fine-tune, measure acquisition (exact match, greedy) and retention (HellaSwag / ARC-easy / PIQA loglik accuracy, 400-500 examples each). Baselines: full FT, replay (1:1 FineWeb-edu text), LoRA r=16.

Prediction: full FT at a lr that actually memorises the facts (acq > 80%) drops HellaSwag/ARC by several points; replay cuts the drop; LoRA in between. (Meta 2510.15103 at 1.3B: NQ F1 drops 89% full FT, 71% LoRA.)

Run 1 (lr 1e-5, 4 epochs): full FT acquisition 23% -> 26%, retention unchanged (HS .382->.380, ARC .68->.71, PIQA .724->.726). Under-powered: nothing was learned so nothing was forgotten. Not evidence about forgetting. LoRA crashed on a torchao/peft version clash. Eval cost ~13 min per pass on T4 at n=500.
Run 2 queued: lr 1e-4, 8 epochs, acquisition logged every 2 epochs, peft/torchao upgraded, n_eval 400.

Result E1 (Permuted-MNIST, 10 tasks, 3 seeds, PKM k=32, Adam 3e-3, 1 epoch/task):
| N | ACC | BWT |
| 256 | 0.345 | -0.676 |
| 1024 | 0.334 | -0.689 |
| 4096 | 0.318 | -0.707 |
| 16384 | 0.351 | -0.669 |
| 65536 | 0.353 | -0.668 |
Every task is learned (diag 0.955) and then destroyed (task 0 ends at 0.08-0.15, below chance). BWT is flat in N: the prediction (-0.3 -> better than -0.1) is wrong by a wide margin. H1 FALSIFIED as stated. Split-MNIST: BWT -0.99 at every N (shared-head collapse, uninformative).

Conclusion: at constant active compute, more slots do nothing while the addressing path (trunk, query, keys) and the head are trained on every task. Interference lives in the shared path, not in the slots. This is the falsifier branch, and it says what the mechanism must protect.

## E1b. Isolate the shared path (2026-09-09, 05:40)

Hypothesis H1b: if the addressing path (trunk + query + keys, BN frozen) is frozen after task 0 and only the value table and head train, forgetting falls with N. Variant (b): also freeze the head, so only values train. Then any remaining forgetting is slot collisions and should fall roughly as 1/N.

Prediction: variant (a) BWT improves from about -0.6 at N=256 to about -0.3 at N=65536 (head still interferes). Variant (b) BWT improves from about -0.5 to better than -0.15, and ACC at large N is close to the diag (learning intact) if the frozen random-ish features from task 0 are good enough to solve permuted tasks through the memory alone. Falsifier: (b) flat in N means the product-key routing itself collides regardless of N.

Result: (pending)

Run 2 (lr 1e-4, 8 epochs): full FT acquisition 23% -> 67% (still rising), retention UNCHANGED: HS .3825->.380, ARC .6825->.7075, PIQA .705->.7075. Replay run then OOM'd (previous model/optimizer not freed).
Conclusion: at 360M, injecting 1,000 facts by full FT does not move commonsense multiple-choice at all, even once the facts land. HellaSwag/ARC/PIQA are insensitive to this kind of damage; Meta measured forgetting as loss of previously-known facts (NQ F1). Protocol change for run 3: add "old-fact retention" = accuracy on a disjoint set of 1,000 TriviaQA questions restricted to the ones the base model already answers correctly. Also lr 2e-4, 12 epochs to drive acquisition up, free GPU memory between methods, n_eval 300 (the MC suite is kept only as the standard reviewer-facing column).

E1 addenda (06:30):
- Dense MLP width sweep on Permuted (same trunk/head, Adam 3e-3): width 32 BWT -0.607, 128 -0.594, 512 -0.644, 2048 -0.601. Flat, as predicted for dense. PKM (-0.67 to -0.71) is slightly WORSE than dense at every size.
- BN control (N=65536, 2 seeds): re-estimating BN stats on all tasks with weights frozen moves BWT -0.730 -> -0.707 / -0.675 and ACC +0.02 / +0.05. BN accounts for a small slice; the forgetting is real.

Result E1b (Permuted, 10 tasks, 3 seeds):
| freeze | N=256 | 1024 | 4096 | 16384 | 65536 |
| addr (trunk+query+keys frozen after task 0) BWT | -0.383 | -0.410 | -0.432 | -0.432 | -0.416 |  ACC 0.17-0.19
| addr+head BWT | -0.324 | -0.322 | -0.354 | -0.331 | -0.341 |  ACC 0.21-0.23
Forgetting halves when the addressing path stops moving, but it is still FLAT in N, and plasticity collapses (ACC ~0.2: later tasks are barely learned through task-0 features). H1b FALSIFIED: with fixed addressing, product-key routing on task-0 features sends different permutations to overlapping slots, so more slots do not separate them.

Conclusion of the E1 pair: slot-locality gives no free forgetting-resistance. What matters is whether the addressing features separate the tasks. Meta's result at 1.3B works because a pretrained trunk already gives separating, stable features; on Permuted-MNIST with a task-0 trunk they neither separate nor stay stable. This is a real finding about WHERE the capacity story can hold: on top of a stable, semantically meaningful feature space, not in a from-scratch model.

## E1c. Random-hash routing control + slot diagnostics (2026-09-09, 07:20)

Unmeasured claim in E1b's conclusion ("routing sends permutations to overlapping slots"): measure it. Diagnostic: slots touched per task and Jaccard overlap between tasks at N=256 and N=65536 with frozen addressing.

Hypothesis H1c: if addressing is a FIXED random projection of the raw pixels (no trunk, no learned query/keys; only the value table and head train), different permutations hash to different slot sets, and forgetting falls with N.
Prediction: rhash BWT improves from about -0.3 at N=1024 to better than -0.1 at N=65536; ACC rises with N. Falsifier: flat in N under random routing too, which kills the capacity story at this scale outright.

E2 contingency (written before run 3 lands): if old-fact retention also shows no drop for full FT at 360M / 1,000 facts, there is no forgetting to fix in that regime. Then move to a regime where damage is measurable: 10k facts, or a sequence of 5 fact batches, or SmolLM2-135M. Prediction for run 3: old-fact retention drops by 10-30 points under full FT, less under replay, LoRA in between.

Result: (pending)

Result E1c + diagnostics (08:10):
Slot diagnostic (frozen addressing, 5,000 test examples per task):
- N=65536: each task touches 480-680 slots (~1%), task 0 touches 2,148; union over all 10 tasks = 2,357 of 65,536 (3.6%); mean Jaccard overlap between tasks 0.53.
- N=256: each task touches ~50% of slots; union 245 of 256; Jaccard 0.79.
So the learned product-key addressing collapses onto a few thousand slots however large N is. E1 and E1b never tested capacity; "flat in N" was a routing-collapse artifact. This is measured, not inferred.

Random-hash routing (rhash: fixed random projection of raw pixels, values + head train), Permuted, 3 seeds:
| N | 1024 | 4096 | 16384 | 65536 | 262144 |
| ACC | 0.164 | 0.189 | 0.222 | 0.250 | 0.286 |
| BWT | -0.368 | -0.402 | -0.419 | -0.428 | -0.457 |
| diag (learned) | 0.51 | 0.55 | 0.63 | 0.63 | 0.71 |
Plasticity rises with N (diag 0.51 -> 0.71, ACC 0.16 -> 0.29): capacity does buy learning. Forgetting does not fall; old tasks end near chance at every N. H1c FALSIFIED on the forgetting half, confirmed on the plasticity half.
Why (partly unmeasured): 60k examples x 32 slots per task is ~2M slot-touches, enough to overwrite most of even 262k slots, and the shared head is trainable in this variant. Whether the residual forgetting is slot saturation or the head is NOT yet separated. Next diagnostic if this line continues: rhash with frozen head (values only) and utilization count.

Conclusion of E1 (a/b/c): (1) learned sparse addressing collapses utilization to a few percent, which by itself explains why "more memory" did nothing; (2) with the collapse removed, capacity buys plasticity but not retention in this regime. The lever that the bitter lesson points at is therefore ALLOCATION: a write rule whose utilization grows with N. That is the mechanism candidate for the paper. Not yet chosen; needs Anay.

## E1d. Random-hash addressing + frozen head: only the value table trains (2026-09-09, 09:10)

Hypothesis: E1c's residual forgetting was the shared trainable head. With the head frozen (random), writes are slot-local end to end, and forgetting falls with N.
Prediction: BWT improves monotonically with N; ACC rises with N.
(First run was an artifact: the freeze code path dropped the value optimizer, BWT exactly 0 and nothing learned after task 0. Guard fixed, rerun.)

Result (Permuted, 10 tasks, 2 seeds):
| N | ACC | BWT | diag |
| 4096 | 0.318 | -0.204 | ~0.51 |
| 65536 | 0.457 | -0.134 | ~0.57 |
| 262144 | 0.575 | -0.070 | ~0.64 |
CONFIRMED. Forgetting roughly halves for each 4-16x more slots, plasticity rises with N at the same time, and task 0 ends at 0.54 of its 0.68 at N=262k. This is the capacity story, and it only appears once (a) addressing spreads writes across the memory (random hashing; learned product-key addressing collapsed to 3.6% of slots) and (b) nothing shared is trained. Compared with E1c (same routing, head trainable): the head alone turned -0.07 into -0.46 at N=262k.

Conclusion: in this from-scratch regime, forgetting is (shared-parameter interference) + (slot collisions), and only the second term falls with memory size. The mechanism the paper needs is therefore: learned features that stay fixed, addressing whose utilization grows with N (measured, not assumed), and writes confined to the slots. On a pretrained trunk, (a) is what the pretrained features give for free, which is the regime Meta's result lives in.
Next: extend to N=1M (does the trend continue?), then the same module on frozen pretrained features (stable, separating) where accuracy is not capped by random pixel hashing.

Run 3 (lr 2e-4, 12 epochs, old-fact retention added): full FT acquisition 23% -> 99.8%. Old-fact retention: of the 229 held-out facts the base model answered correctly, it now answers 51.5%, i.e. it lost ~48% of previously-known facts. Commonsense suites again unchanged (HS .397 -> .407, ARC .68 -> .66, PIQA .70 -> .70). Prediction (10-30 point drop) was too small: the drop is ~48 points. THE FORGETTING REGIME EXISTS and the old-fact metric is the sensitive one. Replay OOM'd again (12.6 GB still allocated after freeing); each method now runs in its own subprocess. Run 4 queued: replay, lora.

E1e (1M slots, rhash + frozen head, 2 seeds): ACC 0.584, BWT -0.057. Trend continues but flattens (-0.070 -> -0.057 for 4x slots); ACC saturates near 0.58 because random pixel hashing caps what is learnable (diag ~0.68). Full curve: 4k -0.204, 65k -0.134, 262k -0.070, 1M -0.057. The next factor of improvement has to come from better (fixed, separating) features, not more slots.

Run 4: replay OOM'd again even in its own process (the 1:1 replay batch doubles the sequences; 13.8 GB). LoRA r=16 at lr 4e-3 (20x base), 12 epochs: acquisition 91%, old-fact retention 13.5%, HellaSwag .397 -> .313, ARC .68 -> .42. That is a mis-tuned baseline (lr too high), not a finding; it does show LoRA can forget MORE than full FT when pushed. Run 5 queued: replay texts 300 chars as a separate half-size micro-batch; LoRA lr 1e-3.
Reference row so far (SmolLM2-360M, 1,000 facts): full FT lr 2e-4 x 12 ep: acq 99.8%, old-fact retention 51.5%, MC unchanged.

Run 5: replay (FineWeb-edu text, half-size micro-batch, lr 2e-4, 12 ep): acq 99.5%, old-fact retention 52.0%, MC unchanged (HS .43, ARC .65, PIQA .70). Generic-text replay gives ZERO protection of known facts versus full FT (51.5%). Consistent with Marek/Wilson: replay only helps if it contains the knowledge being protected. LoRA row repeated run 4 (the lr edit had not applied; deterministic seeds gave identical numbers). Run 6 queued: LoRA lr 1e-3 only.

Baseline table so far (SmolLM2-360M, 1,000 TriviaQA facts, 12 epochs):
| method | acq | old-fact retention | HS/ARC/PIQA delta |
| base | 23% | 100% (229 known) | - |
| full FT 2e-4 | 99.8% | 51.5% | ~0 |
| replay generic text | 99.5% | 52.0% | ~0 |
| LoRA r16 4e-3 (mis-tuned) | 90.8% | 13.5% | -8/-26/-12 |

---
# Phase 2 (2026-09-09, day): the study

Anay's directive: full research-paper-worth of study, understand deeply, aim at a breakthrough. Direction = allocation mechanism on fixed features, then inside a pretrained LM.

## Claim (hypothesis, not a fact) and falsifier

Claim: a continual learner inside a pretrained LM whose forgetting falls as a power law in memory size at constant active compute, with plasticity intact, beats full FT / LoRA / replay on old-fact retention at matched acquisition.

Decomposition (measured on MNIST, E1): forgetting = shared-parameter interference + slot collisions; only the collision term falls with N; LEARNED addressing kills the law by collapsing utilization (3.6% of slots).

Mechanism hypothesis (chosen for the Sep 10 gate): addressing pretrained WITH a load-balance objective then FROZEN; writes confined to slots; utilization a first-class reported metric. Prior art to beat/cite: Kanerva SDM (fixed random addressing), Hash Layers (fixed hash routing), Meta sparse memory finetuning (learned addressing, TF-IDF slot selection, one N).

Falsifier: in the LM, old-fact retention does not improve with N under condition (a) below, or improves under (a) but the load-balanced learned keys (c) do no better than random keys (then the mechanism is just a hash table and the paper is a measurement paper).

## LM N-sweep predictions (written before running)
Conditions, memory layer retrofit into SmolLM2-360M (then Qwen2.5-0.5B), only value slots train during fact injection:
(a) random frozen keys: old-fact retention rises with N (prediction: from ~60% at 16k slots to >85% at 1M), acquisition lower than full FT at small N, rising with N.
(b) keys warmed by LM training WITHOUT load balance, then frozen: utilization collapses (<10%), retention flat in N.
(c) keys warmed WITH load balance, then frozen: retention rises with N like (a), acquisition higher than (a) at every N.
Baselines: full FT (51.5% retention, 99.8% acq), LoRA (run 6), replay of KNOWN facts (disjoint from the test 229), sparse memory finetuning with TF-IDF selection (Meta).

## Gates
Sep 11: law measured on frozen vision features with exponent; retrofit recipe understood. Sep 13: first LM N-sweep (a/b/c) on SmolLM2. Sep 16: second family. Sep 18 abstract. Sep 22 freeze.

Compute: ~55 Kaggle GPU-hours left. Eval cut to old-fact retention + n=100 MC for iteration runs. No more baseline-tuning kernels after run 6; every kernel is a sweep.

## E1d law fit
E1d 3-parameter fit is ill-conditioned on 4 points (negative asymptote, meaningless); use the pure power law below and note the flattening at 1M.
pure power law on 4k-262k: alpha = 0.242

## E4. Memory-size sweep inside a pretrained LM (2026-09-09, afternoon)

Setup: SmolLM2-360M frozen; additive product-key memory h + mem(h) at layers 8, 16, 24 (of 32); 4 heads, k=16, key dim 256; values N x 960 fp32, zero-init; scale fixed at 1 (a trainable shared scale blew up to 14-20 under SGD and is itself an interference path, so it was removed). Injection: 1,000 TriviaQA facts, 12 epochs, ONLY value rows train, SparseAdam lr 0.1 (updates only rows that received gradient, so no stale-momentum drift; SGD at lr 5-50 could not land the facts on a cold memory, unlike Meta's pretrained-with-memory setting). Local check on 135M: 32 facts reach 100% acquisition in 8 epochs.
Metrics per run: acquisition, old-fact retention (base-known disjoint facts), slot utilization per layer, MC n=100.
Kernel cl-e4-random: N in {4096, 16384, 65536, 262144}, random frozen keys. Predictions as in the phase-2 section: retention rises with N; acquisition rises with N.

## E4 random-key retrofit, N sweep (Kaggle cl-e4-random, logged 2026-09-09 for the record; line scratched)
SmolLM2-360M, memory at layers 8/16/24, random-hash addressing, only values train (SparseAdam lr 0.1), 12 epochs, 1000 facts.
Prediction on record: old-fact retention rises with N.
Measured: acquisition 0.998 at every N. Old-fact retention 0.457 (4k) / 0.509 (16k) / 0.552 (65k) / 0.574 (262k). Full FT baseline 0.515.
Utilization ~1.0 at 4k-65k, 0.83-0.86 at 262k. HellaSwag/ARC-easy unmoved.
Reading: retention rises slowly with N (~+4 points per 4x) and only clears the full-FT baseline above 16k slots. Weak effect; consistent with E1d but far from the human number. Not pursued.

## E2 run 6, LoRA lr 1e-3 (Kaggle cl-e2-baselines, logged 2026-09-09 for the record; line scratched)
Acquisition 0.998, old-fact retention 0.472 (full FT 0.515, replay 0.520, random-key memory 0.46-0.57). LoRA at a sane lr forgets as much as full fine-tuning. Nothing in the baseline set protects old facts.
