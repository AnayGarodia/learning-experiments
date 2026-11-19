# Sample-efficiency log. Predictions written before each run.

## Setup
Tiny GPT (d384 L6, ~10M non-embedding params), GPT-2 tokenizer, 25M unique FineWeb-edu tokens, block 512, batch 32, AdamW lr 1e-3 cosine.
Invented town: 600 residents, each with job/shop/street/pet/spouse; residents split evenly across exposure buckets 1,3,10,30,100 (per epoch).
Probes: direct recall (job, street, pet, spouse) and composition (spouse's street / job, never stated in one sentence). Greedy 4 tokens, exact-prefix match.

## R1 (Kaggle se-r1): rereading baseline x fact curve. Epochs {1,4,16} x paraphrase {off,on}. 6 runs.
Predictions (2026-09-09, before launch):
- Held-out loss: 4 epochs clearly better than 1; 16 better than 4 by less than half the 1->4 gain (Muennighoff: ~4 free epochs). Paraphrase changes held-out loss by <0.02 at any epoch count.
- Facts at 1 epoch: exposure 1-3 near 0 on all probes; 10 begins to rise (0.1-0.3 on job); 100 above 0.8 on job/pet/street; spouse lower than job because two names must be tied.
- Total exposures (k x epochs) is what matters: 16 epochs x 1 exposure ~ 1 epoch x 10-30 exposures. If duplicates at 16 epochs land well below 16 exposures' worth, that is the "repetition stops paying" wall showing up for facts.
- Paraphrase (context varied, fact sentence fixed) helps little for direct recall at low exposures, maybe +0.1 at 10-30.
- Composition: near 0 everywhere at this scale. If it rises above 0.3 at 100 exposures that is a surprise worth chasing.
Kill: if job accuracy at 100 exposures x 16 epochs is under 0.5, the model is too small or the lr wrong; fix the harness before reading anything.

## R2 (Kaggle se-r2, queued behind R1): the yardstick. 1 pass over 6.25/12.5/25/50/100M unique tokens, same model, no facts.
Purpose: held-out loss vs unique words. Every trick's gain is converted to "equivalent extra words" against this curve.
Prediction: loss falls roughly linearly in log(unique tokens) over this range, ~0.15-0.25 nats per doubling at 10M params, flattening toward 100M as the model saturates (10M params is under-sized for 100M tokens by Chinchilla, so the last doubling is worth the least).

### R1 result: ep1_dup (2026-09-10, Kaggle se-one)
Held-out loss 5.16 after 25M tokens (10.8M non-embedding params). T4 throughput 8.7k tok/s: same as the Mac, so the GPU is barely used (fp32 head over a 50k vocab, bf16 autocast does nothing on a T4). 48 min per pass; the six-run kernel can never finish inside Kaggle's 12h.
Facts (1 pass): chance is 1/20 for job, 1/16 for street, 1/20 for pet name. Measured at 100 exposures: job 0.075, street 0.183, pet 0.125, spouse 0.033. At 1-30 exposures: at chance. Composition: chance.
Reading: at one pass, nothing sticks even at 100 exposures, on a model this small and this undertrained (loss 5.16 is far above a converged small GPT). Consistent with the prediction that low exposures give zero, but 100 exposures was predicted >0.8 and came in near chance. Before reading more into it: fix throughput (fp16 + grad scaling), then get the 4- and 16-pass points; the kill rule (job <0.5 at 100 x 16) decides whether the harness is sized right.
Replicate of ep1_dup (fp16 code, 43k tok/s): held 5.14; 100-exposure job 0.125, street 0.275, pet 0.133; 1-30 exposures at chance. Same picture, so seed noise is ~±0.05 on the 100 bucket.

### R1 result: ep4_dup (2026-09-10)
Held-out 4.41 (vs 5.14 at 1 pass). 4 passes is worth a lot here: the model is far from converged at 25M tokens.
Facts by exposures-per-pass (x4 total): 1: chance. 3: chance. 10 (40 total): job 0.16 street 0.27 pet 0.24 spouse 0. 30 (120 total): 0.42 / 0.45 / 0.53 / 0.06. 100 (400 total): 0.72 / 0.82 / 0.70 / 0.41. Composition: chance everywhere.
Readings (tentative, one seed): (1) 100 exposures inside a single pass gave ~0.1-0.2, but 30 per pass x 4 = 120 total gives ~0.45. Exposures count for more once the model has learned the language; early exposures are largely wasted. (2) Threshold for a fact to stick sits around 30-100 exposures per pass at this scale, in line with Physics of LMs. (3) Spouse (two names to tie) lags single-attribute facts by ~2x. (4) Composition is zero: the model memorizes sentences, it does not build the town.
Prediction check: "16 passes x 1 exposure ~ 1 pass x 10-30 exposures" is now testable with ep16.
Launched ep4_para (same, facts wrapped in varied context). Prediction: held-out unchanged (±0.02); direct recall at 30 and 100 within ±0.1 of dup (context variation should not matter much when the fact sentence itself is fixed); composition still chance.

### R1 result: ep16_dup (2026-09-10)
Held-out by pass: 4.98, 4.56, 4.45, 4.36, 4.36, 4.28, 4.27, 4.26 (min, pass 8), 4.28, 4.36, 4.36, 4.46, 4.43, 4.53, 4.54, 4.58 (final). Train 3.44 at the end: overfitting from pass ~9.
So at 25M unique tokens and 10.8M params, rereading pays to ~8 passes (4.41 -> 4.26, i.e. 0.15 nats) and then costs. Muennighoff's "~4 free, dead by 15-40" reproduced in shape at tiny scale.
Facts by exposures-per-pass (x16 total): 1 (16): job 0.09 pet 0.14 street 0.11 (just above chance). 3 (48): 0.21/0.23/0.13. 10 (160): 0.32/0.42/0.46. 30 (480): 0.58/0.60/0.61, spouse 0.29. 100 (1600): 0.71/0.75/0.80, spouse 0.48. Composition chance.
The important comparison, equal total exposures, different split:
  100/pass x 4 passes (400 total): job 0.72  vs  30/pass x 16 passes (480 total): job 0.58
  30/pass x 4 (120):              0.42       vs  10/pass x 16 (160):              0.32
  10/pass x 4 (40):               0.16       vs  3/pass x 16 (48):                0.21
Distinct placements beat repeats of the same placement at equal total count, at 120+ total. And the 100/pass bucket saturates at ~0.7-0.8 whether it gets 400 or 1600 total: past a point, more repeats of the same sentences do nothing, while the held-out loss says the model is by then overfitting. Ceiling ~0.75 on single-attribute facts even at 1600 exposures is itself notable (Physics of LMs 3.3 needs 1000 exposures for 2 bits/param; we're at 10.8M params, 600 people x ~5 attributes, far under capacity).
Tentative reading: what a fact needs is not N exposures but N *different contexts*. That is the cheapest thing to test next (paraphrase arm running on the Mac; then a "same sentence, new random neighbours each pass" arm which is free: re-splice facts at fresh positions every pass).

## Next arms (all at 4 passes, same everything, one flag each), queued for GPU:
- reshuffle: same fact sentences, fresh random neighbours every pass. Prediction: if "different contexts" is what counts, 30/pass x 4 lands near 100/pass x 4 dup (0.72 job) instead of 0.42, and held-out is unchanged. If it stays ~0.42, the effect was about distinct sentence *positions in the schedule*, not neighbours.
- paraphrase (running on Mac): fact wrapped in varied lead-ins. Prediction as logged above (±0.1 of dup).
- hard (flashcards): from pass 2, reread only the worst half twice. Prediction: held-out better than dup by 0.03-0.08 at 4 passes; facts at 10-30/pass up, since fact chunks are high-loss and get reread.
- mask 0.15 (hidden text): prediction: held-out slightly worse at 4 passes (regularizer costs early), better than dup by pass 8-16 (2510.04071); facts unchanged or slightly down.

## Reframe (2026-09-10, after Anay: "not interesting enough"): the deliverable is a LAW, not a sweep.
Exposures-to-store (per pass, at 4 passes) as a function of model size N, unique tokens D, and context diversity. Define E50(N, D, arm) = exposures/pass at which job recall crosses 0.5 (interpolated on log-exposure).
Points so far: N=10.8M, D=25M, dup: E50 ~ 40 (job 0.42 at 30, 0.72 at 100).
Predictions before the size runs:
- E50 falls with N as a power law, E50 ∝ N^-a with a ≈ 0.4-0.6 (Tirumala 2022: larger models memorise faster; Kandpal 2023: long-tail accuracy rises with size at fixed document count). So 30M -> E50 ~ 25, 100M -> E50 ~ 13. If a < 0.2 the wall is flat and scaling does not close it.
- Reshuffle (fresh neighbours each pass) lowers E50 by ~2x at every N (contexts, not repeats).
- D axis (10.8M at 12.5M vs 50M unique tokens, 4 passes): E50 is lower with more D (the language is learned earlier so exposures stop being wasted), ~1.5x from 12.5M to 50M.
Composition stays a figure, not a target. R2 yardstick dropped.
Compute: Kaggle out until Sat; Azure has no GPU quota on either sub (T4/A10/A100 requests filed on the pay-as-you-go sub 2026-09-10); Azure CPU is 1k tok/s (useless). Mac excluded by Anay.
