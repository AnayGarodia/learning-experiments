# e017 pilot — partial results at k = 4 of 10, all 7 arms

Box `arche-arb` [address removed] (westus2, 16 cores / 62 GB). Launched
2026-08-17 ~10:20 UTC, 7 arms x seed 0 x k = 10, one process per arm at 2
threads. `arb.py` sha256 `32c225718396866e...`, computed BEFORE the first
optimizer step, identical on the box and in the commit — verified, not
assumed. Read `CAVEATS.md` before any number below.

Screen-read from the run logs. The run RECORDS (`runs/*.json`) are written at
the end of each arm, so `analyze.py` has not yet run on real output and no
verdict category has been computed. **Nothing here is a verdict.**

## Both quantities, per k, as measured

`umar0` = `unknown_marker_applies_rule` at age 0 = does the NEWEST write fire
under a cue no skill owns. Lower is more conditional. `prior` = absolute
unprompted accuracy of each already-written skill, oldest first.

| arm | k=1 | k=2 | k=3 | k=4 | prior accuracies at k=4 |
|---|---|---|---|---|---|
| `no_replay` | 0.805 | 1.000 | 1.000 | 1.000 | 0.0, 0.0, 0.0 |
| `replay25` | 0.805 | 0.000 | 0.835 | 0.945 | 0.805, 1.0, 1.0 |
| `negcue` | 0.735 | 1.000 | 0.445 | 1.000 | 0.0, 0.0, 0.085 |
| `reserve` | **0.000** | **0.000** | **0.035** | 0.745 | 0.0, 0.0, 0.0 |
| `selfdistil` | 0.805 | 0.640 | 0.765 | 0.910 | 0.855, 0.96, 1.0 |
| `selfdistil_offcue` | 0.805 | 1.000 | 0.980 | 1.000 | 0.0, 0.0, 0.0 |
| `cueanchor` | 0.805 | 1.000 | 1.000 | (running) | 0.0, 0.0 |

Writes land in every arm at every k (0.805-1.000 against a 0.60 tripwire), so
no arm is disqualified and no arm is write-taxed so far. The k = 1 value of
0.805 is shared by six arms because it is the same skill (`B_plural`) written
by the same seed before any arm's aux loss can differ — a useful internal
consistency check that the arms are otherwise identical.

## What this supports, in the weakest form

1. **`reserve` is the clearest arbitration effect in the pilot, and it does
   NOT buy retention.** Format-default reservation drives the newest write's
   conditionality to 0.000/0.000/0.035 at k = 1-3 against 1.000 in the
   baseline, at identical write strength — while prior-skill accuracy stays
   0.000 throughout. That is the **A3 signature**: arbitration restored,
   retention not. It is exactly the deflation the dossier said must be
   measurable, and it is only visible because both quantities are reported.
   Weakest form: three clean-ish k values, one seed, and `reserve` trains on
   the same cue family the probe is drawn from (CAVEATS §1), which is the
   single most likely way this number is measuring the intervention rather
   than the model.
2. **`selfdistil` (= LwF) retains, and its newest write stays largely
   unconditional.** Prior accuracies 0.855/0.96/1.0 at k = 4, comparable to
   `replay25`'s 0.805/1.0/1.0, with `umar0` 0.640-0.910. If this holds, the
   pattern is `RETENTION_WITHOUT_ARBITRATION`, which the pre-registered rules
   treat as a signal that the mechanism acting is NOT the one this experiment
   named. Reported as predicted-to-work per the dossier, not as a surprise.
3. **The LwF predicted-negative control BEHAVED.** `selfdistil_offcue` —
   same method, proxy cues that do not exercise the old behaviour — tracks the
   no-replay baseline exactly (umar0 1.000, priors 0.000). Per LwF's own
   authors that is the expected failure, so the on-cue arm's retention is
   coming from preserving the old cue's function rather than from generic
   regularisation. This is the strongest single piece of construct validity in
   the pilot.
4. **`negcue` is not yet separable from the baseline.** One non-zero prior row
   (0.085) and one partially conditional k (0.445) against a noisy 1.000
   elsewhere.

## The number the verifier should attack first

**`replay25`'s `umar_age0` is running at 0.805-0.945 here against e015's
0.537.** That is the headline quantity of this whole experiment — rehearsal's
function as conditionality restoration — and the pilot is not reproducing the
e015 level so far. Three readings, none yet excluded:
- e015's 0.537 is a mean over k = 1-12; these are k = 1-4 individually, and
  0.000 at k = 2 shows the low values do occur. The comparison is not yet
  like-for-like and the arm mean at k = 10 is the number to compare.
- the suite differs (N = 10 prefix-free vs e015's N = 12).
- or the effect is weaker than e015's single seed suggested.

The retention known-positive (`replay25` clean retention in [0.55, 0.83]) is
unaffected by this and is checked mechanically at completion. **If the
retention known-positive passes while the conditionality level does not
reproduce, the headline claim weakens and the retention comparisons stay
readable.** That split is worth stating explicitly rather than discovering in
the results card.

## Throughput and projected grid cost, measured

Per-arm wall clock to reach each k, from the logs:

```
no_replay          1082  2267  3331  4525
replay25           1139  2593  3878  5425
negcue             1195  2542  3771  5208
reserve            1091  2397  3665  5091
selfdistil         1143  2231  3542  4939
selfdistil_offcue  1109  2291  3500  4954
cueanchor          1583  3403  5107   —
```

- ~1100-1300 s per k for six arms; `cueanchor` is ~1.6x slower (it pays an
  extra teacher forward per step instead of an aux batch).
- 7 concurrent arms at 2 threads on 16 cores, load average 13.8-14.0, total
  RSS **24.3 GB of 62** (3.47 GB per process — matching e015's measured
  3.55 GB at this base, so worker count was sized from measured RSS, not from
  cores).
- **Pilot completes ~14:00-15:00 UTC**, i.e. ~4-5 h for k = 10, consistent
  with e015's 12,445 s at k = 12.
- **Projected grid**: 7 arms x 3 seeds = 21 runs. At 7 concurrent (the memory
  ceiling is ~17 processes but the CPU ceiling is ~7-8 at 2 threads), that is
  3 waves x ~4.5 h = **~13.5 h wall on this box**, `cueanchor` on the critical
  path. Dropping to 6 arms x 3 seeds by removing an intervention the pilot
  shows redundant saves ~4.5 h. A k ~ 20 extension would need a fresh gated
  skill batch and roughly doubles per-run cost, so ~27 h for 3 seeds; that
  decision is the verifier's.

## Everything else the verifier should attack

0. **THE VERDICT RULE INHERITS ITS THRESHOLD FROM THE NUMBER THAT IS NOT
   REPRODUCING, AND THIS BLOCKS THE RESULTS CARD.** `analyze.py`'s
   `arb_restored_vs_replay` is `umar_age0 <= replay25's umar_age0 + MDE`, and
   the A1/A3 branch keys off it. If `replay25`'s arm mean lands near 0.85
   rather than e015's 0.537, that criterion is nearly vacuous: `selfdistil` at
   umar0 0.64-0.91 with strong retention would satisfy BOTH tests and return
   `A1_ARBITRATION_FIXES_RETENTION` for a write that is plainly still
   unconditional — the exact wrong headline — and
   `RETENTION_WITHOUT_ARBITRATION`, the category built for that case, could
   never fire. The identifiability dry-run did not catch this because no
   synthetic world has `replay25` conditionality HIGH while its retention is
   in band; every world inherited e015's 0.537. **Not fixed here on purpose:**
   the builder may not edit a verdict rule after seeing data, and the file is
   hashed and must not change while runs are live. The absolute criterion
   `arb_restored_full` (`umar_age0 <= 0.20`, e004g's pre-registered threshold,
   independent of `replay25`) still discriminates in that case, and the
   verifier must rule on which of the two carries the verdict — and add the
   missing dry-run world — before any results card is written.
1. **The `reserve` probe-contamination question above is the load-bearing
   one.** If `reserve`'s low `umar0` is generalisation from its trained cue
   pool rather than a property of the write, the arm's arbitration claim
   collapses. The discriminator already in the design is `negcue`, which uses
   identical contexts with a different target and is NOT showing the same
   effect — which cuts toward the effect being real and target-specific
   rather than contamination. Not settled at one seed.
2. **`cueanchor` is one k behind and on the critical path.** Its weight spread
   is well defined (sd 0.96, min 0.20, max 4.33 in smoke) so a null from it
   would be a real null, but it has produced no k = 4 row yet.
3. **λ = 1, untuned, for every aux loss.** A null is a null for this λ at this
   write budget, nothing more.
4. **The k = 2 `replay25` umar0 of 0.000 next to k = 3's 0.835** is a large
   swing on n = 200. Per-k values are reported precisely so this is visible;
   it argues for reading the arm mean over k, not any single k.
5. **`analyze.py` has not run on real output yet.** Its 7-world identifiability
   dry-run passes, but the reading code has not touched a real record. The
   verifier must execute it on pilot output per decision 0010 amendment 2.
6. **One seed.** No per-seed split is detectable at n = 1, so every category
   above is READABLE and none is CALLABLE.
