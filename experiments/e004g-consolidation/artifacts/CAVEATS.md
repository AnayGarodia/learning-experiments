# e004g — CAVEATS

Everything a reader needs in order to distrust these numbers correctly.
Written by the builder; the verifier should add to it, not replace it.

## Provenance

- Base: `Qwen/Qwen2.5-0.5B`, fp32, CPU. 494,032,768 params, 24 layers,
  d_model 896, **rotary** positions, no absolute position embedding
  (recorded by `pos_scheme()` into every gate file — the e012b lesson).
- Box: `arche-consol`, eastus2, 16 cores / 32GB, `~/venv/bin/python`,
  torch 2.13.0+cpu, `torch.set_num_threads(16)`.
- Generality corpus: `generality.txt`, 41,323 bytes, Project Gutenberg
  *Alice's Adventures in Wonderland* from "CHAPTER I",
  sha256 `f18f78a9138665aa0881af14baaec05bbbe20c125ccbd63807ad56c6522da41b`.
- `--selfcheck` (7 asserts, tokenizer only, ~2s) passes on the committed
  code. Run it before trusting anything here.

## 1. Measured CPU throughput

Measured, not assumed, by `--throughput` on the box:

| quantity | measured |
|---|---|
| forward tokens/sec (batch 32, seq_len 36, fp32, 16 threads) | **567** |
| forward sequences/sec at that length | 16.2 |
| Stage 1 gate, 8 skills x 3 conditions x n=200 + free-gen | 262 s |
| Stage 1 gate, 14 skills, same conditions | ~460 s |
| consolidation optimizer step, no replay | ~2.0 s |
| consolidation optimizer step, with replay 0.25 amortized | ~2.7 s |
| 60-step write, no replay (k=1) | ~119 s |
| 60-step write, with replay (k>=2) | ~160 s |
| eval of ONE skill at n=200 (prompted + unprompted) | ~20 s |
| generality probe (16 x 256 tokens) | ~7 s |

This is close to the machine's fp32 peak (~0.6 TFLOP/s observed against
~0.64 TFLOP/s theoretical for 16 AVX cores), so it is a hardware wall,
not a code inefficiency. bf16 would not help much: torch CPU bf16 matmul
has no throughput advantage on this SKU.

## 2. Projected full-grid wall-clock — **the grid IS CPU-feasible**

Cost model fitted to the pilot's own per-k timings. For a run of N
skills, one (arm, seed):

```
cost(N) ~= 119 + 160(N-1)          # writes
         + N * (20N + 7)           # evals: all N skills every step + generality
         + 4.5 * N(N-1)/2          # intra-skill retention probes
```

| N skills | per (arm, seed) | 4 arms x 3 seeds |
|---|---|---|
| 3 (pilot) | ~12 min | — |
| 6 | ~27 min | **~5.4 h** |
| 10 (all passing, incl. the 4 copyable type-A) | ~62 min | **~12.5 h** |

**Recommended grid: the 6 non-degenerate skills, 4 arms, 3 seeds ≈ 5.4
hours on this one box.** Adding a replay-0.10 arm makes it ~6.8 h.
Nothing here needs a GPU. That is a genuine and slightly surprising
result: the highest-priority item in PORTFOLIO.md is CPU-affordable
once the base is downloaded rather than trained, and the honest GPU
statement is "not required at this scale", not "here is the wall".

The wall that DOES exist is scale, not cost: everything here is at
0.5B. Snell's quality edge was measured at 7B+, and nothing in this
experiment can tell you whether accumulation behaves the same two
orders of magnitude up. That is the GPU ask, and it is a different ask
from the one 004 has been blocked on.

## 3. The pilot is a PILOT

1 seed, 3 skills, 2 of the 4 arms. A retention curve over k=1..3 bounds
nothing about k=8. `analyze.py` prints this caution into
`verdict.json` and it should stay attached to every number quoted from
the pilot. `direct_ft` and `no_replay` have not been run at all, so the
Snell-edge control and the replay ablation are **unmeasured**.

## 4. Known confounds and open attack surface

1. **K0 (the write may not be marker-conditional) is the live threat.**
   Pilot 1 showed `unknown_marker_agree = 0.000` at k=1 in both arms.
   If `unknown_marker_applies_rule > 0.20`, every K1/K2/K3 reading is
   blocked and the result is about conditionality, not accumulation.
   Registered in DESIGN amendment 2 before pilot output was read
   (commit `5a5276e`), and the amendment landed before the run that
   measures it.
2. **Pilot 1 and 2 records are DISCARDED and must not be cited.**
   Pilot 1 used a per-eval drifting RNG, so retention differences
   carried +-0.035 of sampling noise that paired evaluation removes.
   Pilot 2 ran with `UNKNOWN_MARKER = "<<K9>>"`, which the 14-skill
   extension had turned into B_first's own marker. Only `pilot4.log`
   and the `runs/` it produced are valid, and they were produced by the
   committed code.
3. **Four of six non-degenerate skills are one mechanism** — copy the
   input and decorate it (B_double, B_title, B_plural, B_triple). Only
   B_upper and B_first transform the input's characters. Read the
   retention curve per skill, never only as a mean.
4. **Type A skills are fixed-offset copyable at ~0.95** (`copy_oracle`
   in the gate file). They are in the suite as the fact-like arm and
   must not carry a conclusion about skills.
5. **Teacher-forced scoring is an upper bound on free generation.**
   Validated: across all 10 passing skills the two agree within 0.016.
   It FAILS on B_bang (0.000 vs 1.000), which fails the gate anyway;
   see DESIGN amendment 3 for the owned deviation from the
   pre-registered response to that guard.
6. **The teacher is the frozen base, not the current model.** e004r
   re-cloned the current weights as teacher before each skill, so its
   target drifted with k. Here the target is fixed for all k. This is a
   deliberate deviation (DESIGN, "The write") and it makes retention
   measurable against a fixed reference — but it also means this
   experiment does not test the drifting-teacher variant at all.
7. **Replay is 25%, not Scialom's 1%.** At 60 steps per skill, 1% is
   not expressible. The design is therefore GENEROUS to K1: an
   interference wall at replay 0.25 is a strong negative, while a flat
   curve at 0.25 says nothing about how low replay could go.
8. **LoRA r=8 on all seven projections in all 24 layers** is one point
   in a large space. Rank, target modules, learning rate (1e-4) and the
   60-step budget are all unswept. The e012b verifier's standing note
   that learning rate was never exonerated in the ICL series applies
   here too, and applies to `direct_ft` as much as to `consolidate` —
   the Snell-edge comparison is only fair if neither arm is mistuned,
   and neither has been tuned at all.
9. **Answers are short (1-3 tokens).** Whether consolidation of longer
   procedural behaviour behaves the same is untested and is closer to
   what dossier 004 actually cares about.
10. **`age` in results.csv is computed from `order.index`**, so it is
    only meaningful within a run; do not aggregate it across seeds
    with different orders without re-deriving it.

## 4b. Corrections made after the pilot ran (read before the numbers)

- **`unknown_marker_applies_rule` collapsed to a `max` is not a
  conditionality test.** Read PER SKILL AND PER AGE — which is how
  DESIGN amendment 2 defines it — it is. See the VERIFIER RULING in
  DESIGN.md section 1. By age since its own write the probe reads
  age 0 → 0.70, age 1 → 0.445, age 2 → **0.000**. Conditionality is
  ACQUIRED: a rule is format-attached when fresh and marker-attached
  once a later write takes the format default.
- **`cross_marker_leak` is a diagnostic, NOT the K0 reading rule
  (verifier ruling).** It is near-arithmetically implied by retention —
  if skill j's own rule fires under marker j at 0.98, a different rule
  cannot also fire there — so it cannot independently confirm what
  retention already says.
- **CORRECTION to the builder's mechanism claim.** "Markers that own
  nothing fall back to the newest rule" is false on this data. At k=2
  the unclaimed marker returns the OLDEST rule (B_title 0.890), not the
  newest (B_upper 0.125). The format default holds one rule at a time
  and changes hands late.
- **k=1 leak is 1.000 and that is not a defect**, it is what "one rule
  in the adapter" looks like. It does mean the k=1 point is not
  comparable to later k on any curve.
- **CORRECTION: pilot 4 vs pilot 5 is a DETERMINISM check, not a
  reproduction.** Same seed, same order, same code path, so agreement is
  expected; and they agree at all THREE k, not two. What it does show,
  narrowly, is that adding the `cross_marker_leak` evaluation did not
  perturb the training RNG stream. Independent replication is the
  verifier's seed-1 run.
- **CORRECTION: `runs/` was produced by pilots 5 and 5b, not pilot 4.**
  `runs/consolidate_s0.json` came from pilot 5 (which added
  `cross_marker_leak`) and `runs/separate_adapters_s0.json` from pilot
  5b. Pilot 4's records are quarantined in `runs_pilot4_noXmarker/` on
  the box and are not in the repo.
- **`git_sha` is the EMPTY STRING in every pilot record.** The box is an
  rsync of the experiment directory, not a git checkout, so
  `git rev-parse` returned nothing and the code recorded it silently.
  Nothing in the pilot record proves which code produced which number;
  the code-identity check is the md5 of `runs/*.json` matching between
  box and repo, done by the verifier. Fixed going forward: `git_sha()`
  falls back to a sha256 of `consolidate.py`.
- **`separate_adapters` has no `cross_marker_leak_by_k`** — by
  construction each skill owns its own adapter, so there are no
  already-trained markers to leak into. The empty dict is correct, not
  a missing measurement.

## 4b-bis. THE PILOT HEADLINE IS SUPERSEDED (verifier replication, k=4)

**Do not quote "retention 0.979, min 0.957, no collapse at k<=3" as a
result.** The verifier ran one more skill at a different seed
(`runs_verifier_s1/`, order B_upper -> B_double -> B_title -> B_first)
and retention collapses at k=4: **B_upper 0.812, B_double 0.570,
B_title 0.954**, mean at max k 0.779, against a 0.90 threshold and an
MDE of 0.060. Both conditionality-clean rows (age >= 2) are below 0.90,
so it is not a K0 artefact. The pilot stopped exactly one step before
the effect.

The intra-skill probe shows the shape: within the k=4 write, prior
skills crash by step 20 (B_upper 0.375, B_double 0.000, B_title 0.000)
and replay claws them partway back by step 60 (0.797 / 0.500 / 0.984).
End-of-skill evaluation sees only the recovered state.

## 4c. Verifier additions to the confound list

11. **Only ONE of the pilot's six retention rows is conditionality-clean
    (age >= 2).** "Retention mean 0.979, min 0.957" is mostly a mean over
    age-1 rows, where the measured skill may still hold the format
    default and part of its apparent retention is the unconditional
    fallback. The clean pilot number is a single point of 1.000.
12. **Replay 0.25 is not a retention control, it is CONTINUED
    TRAINING.** Round-robin replay gives the first-written skill
    `15 * sum(1/i, i=1..k-1)` extra batches on top of its 60-batch
    write: **+37% by k=3, +57% by k=6, +65% by k=8**. Scialom's 1%
    would be about +2.5%. The `consolidate` arm therefore measures
    "retention under 25% rehearsal"; the uncontaminated measurement is
    the `no_replay` arm, which had not been run when the pilot was
    reported.
13. **The K1-vs-K2 discriminator is CEILING-CENSORED, not merely
    tight.** Four of six non-degenerate skills consolidate to unprompted
    1.000, where a +0.06 re-prompting gain is arithmetically impossible.
    Before the verifier's fix the verdict rule would have returned "K2,
    cache not organ" regardless of the world. Expect
    `INCONCLUSIVE_CEILING_CENSORED` on this suite.
14. **k=8 cannot confirm the interference wall.** K3 predicts "flat then
    cliff" and the sequential-editing literature puts the cliff at
    k ~ 32. A flat curve to k=8 lies inside K3's own flat region. The
    grid can KILL K1 (a cliff by k=8); it cannot confirm K3.

## 5. What the verifier should attack first

In priority order:
1. Re-run `--selfcheck`, then re-derive K0 from `runs/*.json` by hand.
   If `unknown_marker_applies_rule` is high, stop and rewrite the
   reading — do not let a K3 headline through.
2. Check that `k0_frozen_base.json` really is the frozen base (no
   adapter attached) and that its `unprompted` values are ~0. If they
   are not, the null level is wrong and every threshold moves.
3. Attack the interference finding as an artifact of B_upper and
   B_double competing for the same answer slot (both are input-derived
   at the same position; A_zeta, a constant, survived pilot 1 while
   B_upper did not). A skill-pair-level analysis would separate
   "interference between any two skills" from "interference between
   two skills of the same shape".
4. Re-run `analyze.py` on the pilot output and confirm the printed
   spread separates the arms beyond the stated MDE of 0.060.
5. Question the 60-step budget: if the write is still moving at step
   60, retention differences may be budget artifacts. The intra-skill
   probes at steps 20/40/60 are in the records for exactly this.
