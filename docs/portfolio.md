# Portfolio — the factors that must multiply to a leap

Created 2026-08-16 under decision 0011 amendment 1. This file turns
"1000x" from a slogan into an accounting. Every claimed multiplier gets
a named place here, an independence judgement, and a measured or
predicted magnitude. A multiplier with no entry here is not a finding.

Four axes, all of which count (Anay): **efficiency**, **context**,
**learning**, **energy**.

Rule for the running product: factors multiply ONLY if independent.
Where two factors act on the same underlying waste, the portfolio gains
the difference, not the product. Overlap is stated explicitly per row.

## Status: MEASURED product = 1.0x

Nothing measured so far contributes a factor. That is the honest
headline and it should stay at the top of this file until it changes.

## The ledger

| axis | candidate factor | target | measured | status |
|---|---|---|---|---|
| efficiency | objective reweighting (005) | >=5x | **1.05x** | DEAD. Overlaps data curation; multiplies to nothing. |
| efficiency | learner-chosen data (006) | >=2-3x | **~1.0x** | DEAD. Even a free perfect scorer buys ~nothing in that world. |
| efficiency | multi-horizon prediction (011) | >=1.5x | **<1x (a tax)** | DEAD at this scale. |
| efficiency | adaptive compute (009/009b) | >=2x vs model selection | **ceiling never reaches 2x (max 1.68 over the whole grid); implementable arm <1x above the smallest rung** | DEAD as a portfolio factor. Full grid landed (76/76 cells, 5 training levels x 2 seeds). The POST-HOC ceiling — which no router can exceed, because it reads the full model's own logits — has a fit-free worst-seed lower bound of at most 1.68 against a 2.0 kill line, at every level, seed and axis, under both FLOPs accountings. The arm a real system could build (depth-only, CALM state copy) has multipliers 0.83/0.75 down to 0.075/0.076 across rungs xs0..l at convergence: a smaller dense model is strictly cheaper at matched capability. Formal verdict is INCONCLUSIVE (the two capability axes disagree; consequence accuracy misses the 1.6 survive line by 0.01), but "does not reach 2x" is callable and is the portfolio-relevant question. Conditional on: e005 synthetic corpus, block granularity, verdict-bearing rungs below the ~3M scale floor. See [[arche-e009b-t2-overlay]]. |
| efficiency | growth / stacking (010) | >=30%/hop | **~0.004 nats at matched compute** | TINY. Real but not portfolio-sized. Redesign not bought. |
| efficiency+energy | enforced sparsity (003) | **ceiling x2.3, not 3-6x** | **deployable-today x0.81 (a LOSS, measured)**; co-design ceiling x1.8-2.2 | PRE-LAUNCH, verifier-priced. Target column was arithmetically unreachable and is corrected; wall clock is measured uncontended, not inferred. See the row detail below. |
| energy | local learning -> memory-compute colocation (008) | 3x+ energy | not yet | RUNNING. Qualifies only through the hardware path: a CONSTANT tax unlocks colocated silicon; a widening gap closes the door. |
| context | retrieval cap / update rules (001) | qualitative | not yet | RUNNING. Attacks a proven fundamental limit (fixed-size state cannot do random-access retrieval). Contributes to the context axis rather than a clean multiplier. |
| learning | consolidation: context->weights (004) | unbounded / paradigm | **retention 0.922 at k=8, TEACHER-FORCED (resolved: the freegen gap is NON-TERMINATION, not wrong content)** | ALIVE. The "collapses at k=4" reading came from a DIFFERENT skill sequence than any grid seed and is not directly comparable (it also contained the colliding skill), so it is set aside rather than refuted. That was ONE SEED (mean 0.779). The completed 3-seed grid, read from committed records, retains 0.922 at k=8 on the four CLEAN skills (0.946 as first reported, before removing a prefix-colliding pair: B_double's output is a prefix of B_triple's, and writing either made the other read 1.0 even when never written -- 8 instances across all arms; the harness's own max_unprompted_before_write tripwire read 1.0 and nobody looked) against a no-replay arm at 0.000 (0.000 / 0.000 / 0.000; the previously reported 0.059 was entirely seed 2's colliding pair -- corrected 2026-08-17, see e004g DESIGN 'POOL GATE'). So the consolidation write accumulates across a sequence twice as long as the one thought to break it. Two caveats keep this from being a paradigm claim. (a) The retaining arm CARRIES 25% REPLAY, which is continued training on the very skills retention measures (+37% extra batches on skill 1 by k=3, +65% by k=8); the no-replay collapse to 0.000 is the honest statement of the unsolved problem. (b) analyze.py's own caution: "a no-cliff reading bounds nothing beyond the k actually run" and the editing literature's collapse lives at k~32, which is where e020 is looking. Mechanism, replicated: conditionality is ACQUIRED not conferred, and the replay-vs-no-replay conditionality contrast (umar_age0 0.527 vs 0.957) REPRODUCES e015's 0.537 vs 0.954. Also from this grid: the WRITE OBJECTIVE matters (see the new "write objective" row). The separate_adapters arm is BROKEN, not null (18 rows, retention nan) and may not be cited. |
| learning + efficiency | write objective: KL-to-self vs CE-on-truth (004g) | unclear | **NOT RESOLVED (3 arms x 3 seeds, Task A)** | REPRODUCTION, not a finding: the CE-vs-KL retention contrast is published (SDFT arXiv 2601.19897 Jan 2026, 64.5% vs 53.4% prior-task average; DiSC arXiv 2602.16093 confirms across four models), and the "3.3x capability damage" figure is WITHDRAWN (it was a max-minus-min pooled across seeds; per-seed CE drift is 0.048/0.278/0.056 vs KL 0.021/0.020/0.034, so the effect rests on seed 1 alone). Also, our write objective is itself published as CONTEXT DISTILLATION (Snell/Klein/Zhong arXiv 2209.15189, 2022) which must be cited not claimed. UNRESOLVED PROBLEM: SDFT reports a scale crossover where distillation UNDERperforms SFT at 3B (+4 at 7B, +7 at 14B), and we ran 0.5B, below it, yet measured KL clearly ahead. Either we measure something else (most likely: our CE arm swaps the objective on replayed priors too) or one result is wrong. Do not cite until resolved. Our numbers, READABLE (3 seeds, k=8, matched 25% replay). Distilling the base's own few-shot behaviour retains 0.938 and drifts general capability 0.096 nats; supervised CE on ground truth retains 0.817 and drifts 0.318 nats. So HOW you write a skill, not just where you put it, changes both retention and collateral damage. Not yet CALLABLE as a single-factor result: the CE arm swaps the objective on the replayed prior examples too, so it is not a clean contrast on the write alone. This is also the cost side of the harness scope limit (library/sweeps/harness-scope-limit.md): the gentle KL write can only install skills the base can already demonstrate. |
| learning | where ICL switches on (012) | prerequisite | **not reached** | The five negatives were UNDERPOWERED, not informative: every run sat 10-200x below the 1e9-5e9 tokens at which the literature reports induction forming. A rotary probe that looked like a fix was withdrawn on its own discriminator (it was fixed-offset copying, not induction). Nothing CPU-trainable has shown ICL. Consequence: 004 needs a DOWNLOADED base pretrained past 1e9 tokens; the GPU-block challenge fails for home-trained bases. |
| learning | rehearsal DOSE and its mechanism (015/022) | a cheap-rehearsal regime would be a real factor | **UNKNOWN; the earlier "threshold" is withdrawn** | e015's 1%-vs-25% contrast confounded FOUR things at once: total budget (24x), aux step count (15x), recency (last aux at step 30 vs 56), and COVERAGE (the 1% arm rehearsed exactly ONE of ten prior skills, because the harness rehearses one prior per aux step). Published work reports 0.25-1% rehearsal working well when spread through every batch (Scialom 2022, Ibrahim ICLR 2024, Chaudhry 2019), so our failure at 1% is most likely our schedule. e022 separates all four knobs and runs the unclaimed fixed-budget/varied-spread ablation; Hickok 2025 is its predicted negative. |
| learning | ICL-vs-gradient exchange rate (013) | R >= 20 to be a factor | **max measured R = 2.01** | X1 KILLED (kill line 5.0). Verifier measured B_first 1.70, B_upper 2.01, and the three NOVEL skills never reach even C=0.25 within 64 shots while gradients need <=128 — so no world consistent with the data yields R>=5. Two deeper results: (a) clause 7(a) BITES — R is monotone in the base model's 1-shot competence across all four skills, so R is NOT a property of the two learning routes; (b) with repetition free, n_grad measures the same quantity as n_ctx (distinct examples needed to identify the rule), pushing R toward 1 BY CONSTRUCTION. The literature's folk 'ICL is vastly more sample-efficient' figure is not this quantity. Novel arm: the memorization objection FAILS (a rule-shaped novel skill also fails in-context, flat 0.035-0.105 to 64 shots) but gradients learned it as a TABLE too (unseen-letter 0.000 until coverage completes). Net: gradients beat context on novel skills by STORAGE, not by learning. |
| capability | diversity collapse / taste (007) | qualitative | not yet | QUEUED. A capability bottleneck rather than an efficiency one. |

## Row detail: enforced activation sparsity (003)

Written 2026-08-16 by the e003 verifier, from the pilot plus the
mechanism's arithmetic, BEFORE the grid. It replaces a placeholder.

**Which axis.** Efficiency and energy, counted once (per the independence
note below). Not context, not learning: nothing about sparsity changes what
the model retains or how much it can hold.

**Magnitude — three different numbers, and only the first is deployable.**

| accounting | what it means | k=15% |
|---|---|---|
| **measured wall clock** | today's silicon, our kernel, uncontended | **x0.81 — a 19% LOSS** |
| conservative FLOPs | today's GPUs, a perfect sparse down-projection kernel | x1.18-1.35 |
| optimistic FLOPs | ideal predictor / event-driven silicon | x1.83-2.21 |

The wall-clock row is measured, not assumed: on arche-sparse, one run at a
time, one thread, eval time excluded, `text`/s/swiglu ran 2066.1 ms/step
dense against 2564.4 ms/step at k=15% — the top-k costs **x1.24 per step**,
so as a speed multiplier sparsity delivers **x0.81**. (A Mac reproduced
x1.51 at k=50% against the box's x1.47, so the figure holds across two
architectures.) On top of that, matched-FLOP budgeting adds x1.83 more
steps because the FLOP saving is unrealisable, giving **x2.27 dense wall
clock for a matched-FLOP sparse run**. The builder's x1.86-2.05 came from
15 workers on a 16-core box (3.3x contention inflation on the dense cell)
with eval time inside the ratio.

The ceiling is arithmetic, not empirical: activation sparsity touches the
MLP, the MLP is 8d² of a 12d² block, so per-token FLOPs fall by at most
12/(4+8k) = **x2.31 at k=15%, x3.00 as k→0**, and attention is untouched.
The 3-6x this row used to target and the "10-100x" in dossier 003's "Why it
matters" are both above that ceiling and were never reachable by this
mechanism. Corrected here rather than left to be discovered by a win.

**Independence.** Still overlapping adaptive compute (009), which did not
pay, so 003 has the per-token-waste axis to itself. But the overlap that
now matters more is with the *token* axis: 40-49% of the pilot's
matched-FLOP win is the extra tokens a cheaper model buys, which is
budget allocation and is already counted under data efficiency. Only the
matched-token half is a sparsity factor.

**Running product.** Unchanged at **1.0x**. On the axis the portfolio
actually spends — wall clock on real silicon — the measured factor is
**x0.81, a net loss**, not a multiplier and not even a free no-op. Per decision 0008 and dossier 003 Revision 1,
an S1 win at FLOP parity with a wall-clock parity/loss is a **demand story
for co-designed silicon (0008 path b), not a deployable multiplier**. The
honest entry is therefore conditional: 003 contributes **x1.8-2.2 on the
efficiency/energy axis IF event-driven or gather-native silicon exists to
price unstructured per-token sparsity, and x1.0 until then.** A
conditional factor does not enter the running product; it enters the
demand case a vendor would need.

**What would move it.** Not a better kernel — the conservative column caps
that at x1.2-1.4, and at h=256-683 a gather/scatter down-projection is
memory-bound with nothing to recover. What would move it is structured
sparsity at a granularity the hardware can already execute (which is MoE,
already carded and already in production, i.e. not a new factor), or the
silicon in 0008 path (b).

## What the ledger makes obvious

1. **The efficiency axis is close to exhausted at the trick level.** Five
   candidates, five deflations. Every one left the learning process
   unchanged, and the pattern across them is that loss- and data-side
   reweighting measures ~1.0x against ground truth where the literature
   reports 2-18x.
2. **Three of the four axes have nothing measured on them at all.**
   Context, learning and energy are each carried by exactly one or two
   live experiments, and the two that matter most are GPU-blocked.
3. **The portfolio cannot reach a leap on the efficiency axis alone**,
   which is where all spent compute went. This is the concrete form of
   decision 0011's retrospective critique.
4. **The learning axis is gated by pretraining budget, not by a fixable
   detail** (2026-08-16, after one intermediate claim was raised and
   withdrawn the same day). In-context learning did not appear in any
   configuration we could train, and the literature puts its onset
   10-200x beyond our largest run. The axis holding the only
   independent factor in this table is therefore reachable only
   through a downloaded pretrained base, or through GPUs.

## Note on e015 (added 2026-08-16 after its verifier ruling)

The continual-learning result is ARBITRATION, not allocation. Without
rehearsal the newest write is unconditional and takes every cue; prior
skills survive in the weights and are inaccessible, not destroyed.
Parameter-space allocation (sparse or orthogonal) cannot fix that
because it never constrains what the model outputs on old cues. So the
allocation arms were never tested against the failure mode that
occurred. Any future continual-learning factor must attack ARBITRATION
first; a retrieval problem is a far easier engineering target than a
destroyed representation, and that is the one hopeful part of this
result.

## Independence notes (the part that is easy to get wrong)

- Sparsity (003) and adaptive compute (009) both attack per-token
  compute waste. They are NOT independent; claiming both products would
  double-count.
- Sparsity contributes to BOTH efficiency and energy, but not
  independently: the energy gain is the same physical saving priced
  differently. Count it once, on the axis where it is demonstrated.
- Consolidation (004) is independent of everything else here: it changes
  what the model retains between episodes rather than what a forward
  pass costs. This is why it is the highest-value untested item.
- Local learning (008) contributes energy only if it also survives on
  capability. A local rule that is 3x cheaper per joule and 2x worse on
  loss is a net loss, not a factor.
