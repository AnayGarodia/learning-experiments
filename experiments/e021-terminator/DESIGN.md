# e021 — The terminator fix: is the no-stored-data method capped by a corrupted teacher?

Pre-registration: `dossiers/021-terminator-fix.md` (committed BEFORE any code
here was written). This file records the FIX, the guard, the arms and the
reading rules. Rivals T1-T4 are the dossier's and are not restated.

Status: BUILT + IDENTIFIABILITY DRY-RUN PASSED 2026-08-17. Grid launched on
`arche-arb`.

## 1. What the fix is, exactly

One idea, three call sites. `consolidate.py` gains

```python
TERM = "\n"
def term(ans): return ans + TERM
```

and the answer STRING handed to the write objectives becomes `term(ans)`:

- `C.kl_loss` — the write, shared by e004g, e015, e017, e021 (both the
  teacher's prompted items and the student's unprompted items; their answer
  spans still align, asserted as before);
- `C.ce_loss` — the `direct_ft` control's write, so the two writes stay
  comparable;
- `arb.selfdistil_loss` — the self-distillation aux, so the terminator is
  preserved on OLD cues as well as taught on the new one. Without this the
  arms would differ in aux CONTENT and not only in aux TARGET.

That is the whole mechanism change. The scored span of the write is now
answer-plus-newline, so **stopping is a trained target**. Before this, nothing
in the objective ever mentioned the terminator, which is the root cause the
e004g verifier ruled on ("NON-TERMINATION": the model emits the right answer
and keeps going on the same line, `" wagon???????"`).

### Why newline and not EOS

The write is KL to the FROZEN BASE under a newline-joined few-shot context, so
the base already puts mass on `"\n"` at the position after the answer and
essentially none on EOS. A write toward EOS would be fighting its own teacher,
and the student could only reach it by drifting off the teacher's
distribution — which is the thing this write exists to avoid. Newline is also
the terminator the eval's string comparison already assumes. **Choice:
NEWLINE.**

### What was deliberately NOT changed

- **Eval metrics.** `C.accuracy` and `C.freegen_accuracy` still score the
  ANSWER ONLY. This is load-bearing for the T4 guard: if `accuracy` also
  required the terminator, `mean_just_after` would drop for a
  metric-definition reason and could not be compared to the pre-fix
  0.927-0.937. A change in the guard is now a change in write strength.
- **No post-processing of generated text.** The dossier forbids it and it
  would mask the defect: the point is that the TRAINED model must stop,
  because the self-distillation teacher is recorded by free generation and is
  then trained on.
- **`lambda_aux = 1.0`**, unchanged from e017. A null here is a null for that
  lambda only.
- **`NULL_ANSWER = "?"`** for `reserve` (single-token, first-position only, no
  terminator involved).

### The terminator gate (new, BLOCKING)

`C.term_clashes` / `C.assert_term_clean`. If BPE merges the terminator
backwards into the last answer token, the scored span silently shifts and both
the answer and the terminator are trained at the wrong positions. Four suite
skills really do this — `B_bang` (`apple!`), `B_dot`, `B_angle`, `B_upbang` —
and `"?" + "\n"` is a single Qwen token. **None of the four is in
`GRID_SKILLS`**, so the grid is clean, and the gate is asserted for the grid
suite in `arb.selfcheck` alongside the pool gate. The gate is self-tested in
`C.selfcheck`: it must FIRE on `B_bang` and stay silent on `[5, 8, 9, 11]`.

Consequence: `self_targets`' fallback for an EMPTY greedy generation could no
longer be `"?"`. It is now `SELF_NULL = "none"` (terminator-clean, recorded per
run). This is a rare arbitrary null, not a measurement.

## 2. Arms, k=10, 3 seeds

| arm | stores old data | aux content | aux target | role |
|---|---|---|---|---|
| `selfdistil` | NO | prior skills' own cues | the model's own recorded outputs | the method under test |
| `replay25` | YES | prior skills' own cues | true answers | stored-data CEILING |
| `no_replay` | NO | none | none | FLOOR / dynamic range |
| `selfdistil_offcue` | NO | unclaimed cues | own outputs there | PREDICTED NEGATIVE; if it retains, VOID |
| `selfdistil_front` | NO | as `selfdistil` | as `selfdistil` | T3, equal-cost front-loaded schedule |

**T3's equal-cost construction.** Every arm gets 25% replay's schedule: 15 aux
batches of 16 per write. `selfdistil` spends them round-robin over ALL priors
(`order[n_aux % i]`); `selfdistil_front` spends the SAME 15 batches on the TWO
MOST RECENT priors (`order[i-1-(n_aux % min(2,i))]`), i.e. each skill receives
concentrated aux in the two writes immediately after it is learned and none
later. Asserted in `arb.selfcheck` item 9: equal batch count at every i, and
front's prior set is exactly the last two. Nothing else differs, so a
difference cannot be explained by aux budget.

`selfdistil_front` also RECORDS snapshots and fidelity for all priors, exactly
as `selfdistil` does — only the loss CONSUMPTION is front-loaded — so the
mediator is reported for the T3 arm too.

Routing bug avoided: the snapshot cue selector tested `arm == "selfdistil"`,
which would have made any new `selfdistil_*` arm silently an OFF-CUE arm, i.e.
the predicted negative. It now tests `arm != "selfdistil_offcue"`, asserted.

## 3. Primary outcomes

1. `selfdistil_target_fidelity` **per prior per k** — the MEDIATOR, primary,
   not a diagnostic. Pre-fix it read 0.000 for every prior from k=6 onward.
   `analyze21.fid_late` is exactly that population (age>0, k>=6).
   `selfdistil_target_samples` commits the first three teacher STRINGS per
   prior per k, because a scalar cannot show a reader `wagon` vs
   `wagon???????`.
2. Retention, **teacher-forced AND free-generation, side by side**, per skill
   per k per seed. `cont.eval_step` now scores `C.freegen_accuracy` on the SAME
   items as the teacher-forced row (`freegen_n=32`, default 0 so e015's numbers
   are untouched).
3. `mean_just_after` — the T4 guard, read FIRST.

## 4. Reading rules (`analyze21.reading`), fixed before any e021 number

0. **T4 WRITE GUARD.** `mean_just_after` below `0.927 - 0.05 = 0.877` in any
   arm -> `T4_WRITE_BROKEN_BY_TERMINATOR`, STOP. Nothing below is read.
1. **DYNAMIC RANGE.** `no_replay` clean retention > 0.85 -> suppressed.
2. **LEAK / VOID.** `selfdistil_offcue` retention > 0.15, or within the 3-seed
   MDE of `selfdistil` -> `VOID_OFFCUE_CONTROL_RETAINS`.
3. **TERMINATION ACTUALLY FIXED.** teacher-forced vs free-generation prior
   accuracy must agree within 0.05 in every arm, or
   `SUPPRESSED_TERMINATION_NOT_FIXED`.
4. **THE MEDIATION.** `fid_late < 0.30` -> `T0_FIX_DID_NOT_RAISE_FIDELITY`.
   `fid_late >= 0.60` AND `replay25 - selfdistil <= 0.15` ->
   `T1_TEACHER_CORRUPTION_WAS_BINDING`. Otherwise
   `T2_FIDELITY_UP_RETENTION_CAPPED`.

T3 gets its OWN verdict beside the main one (`front - uniform > MDE_3SEED` ->
`T3_CRITICAL_WINDOW_SUPPORTED`), because it is a different manipulation and can
succeed or fail independently of T1. **T3's interpretation is GATED on the
prior-art check the dossier requires** — front-loading rehearsal just after
acquisition is close to consolidation-window findings in the memory literature
and may be known. The arm may be RUN before that check; it may not be CLAIMED
before it.

Retention is a RATIO to each skill's own just-after accuracy (decision 0012
clause 7a) with the absolute printed beside it, and rows younger than age 2 are
excluded (the e004g verifier's ruling). Both inherited from e017's `analyze.py`,
whose `rows()` this reuses unchanged apart from carrying the new `freegen`
field.

## 5. Identifiability dry-run (decision 0012 clause 1) — PASSED

`python analyze21.py --dryrun`. Eight predicted worlds, each pushed through the
real `per_arm` + `reading` code as synthetic ROWS indistinguishable from
`A17.rows` output. Every one yields its intended verdict:

```
OK   T1_TEACHER_CORRUPTION_WAS_BINDING   -> T1_... / T3_NULL_SCHEDULE_DOES_NOT_MATTER
OK   T2_FIDELITY_UP_RETENTION_CAPPED     -> T2_... / T3_NULL_SCHEDULE_DOES_NOT_MATTER
OK   T2_FIDELITY_UP_RETENTION_CAPPED+T3  -> T2_... / T3_CRITICAL_WINDOW_SUPPORTED
OK   T4_WRITE_BROKEN_BY_TERMINATOR       -> T4_... / None
OK   T0_FIX_DID_NOT_RAISE_FIDELITY       -> T0_... / T3_NULL_SCHEDULE_DOES_NOT_MATTER
OK   VOID_OFFCUE_CONTROL_RETAINS         -> VOID_... / None
OK   SUPPRESSED_TERMINATION_NOT_FIXED    -> SUPPRESSED_... / None
OK   SUPPRESSED_NO_DYNAMIC_RANGE         -> SUPPRESSED_... / None
DRYRUN OK
```

The dry-run caught one design breaker before launch: with the leak check
placed before the dynamic-range check, a world in which NOTHING degrades (every
arm retains, including the control) was read as
`VOID_OFFCUE_CONTROL_RETAINS` — a leak verdict on an unstressed harness. The
order is now T4 -> dynamic range -> leak -> termination -> mediation, and the
world reads correctly. This is the sixth consecutive experiment in which the
clause-1 dry-run has caught something.

## 5b. GO/NO-GO VERIFICATION — the fix does what it claims (READABLE, one seed)

Record: `runs_verify/selfdistil_s0.json` (committed). `selfdistil`, seed 0,
skills `B_first,B_upper,B_double` (8,5,7), k=3, `n_eval=64`, `freegen_n=32`,
`arb_sha256 = 4f10d01fea13c2d5`, `consolidate = code:2d25eb1ae46dbfd9`. This is
a SMOKE, not the grid: 3 skills, one seed, a quarter of the eval resolution.

**1. The teacher strings are clean.** `selfdistil_target_samples`, verbatim:

| k | prior | prefix | recorded teacher target |
|---|---|---|---|
| 2 | 8 | `<<K8>> river =` | `River` |
| 3 | 8 | `<<K8>> glass =` | `Glass` |
| 3 | 5 | `<<K5>> glass =` | `GLASS` |

No `wagon???????`. The free generation now terminates and the newline cut
fires on an actual newline.

**2. The mediator moved.** `selfdistil_target_fidelity`: k=2 `{8: 1.000}`,
k=3 `{8: 0.938, 5: 0.875}`. Pre-fix e017 read 0.88/0.94/0.75 in the first step
after a write and **0.000 for every prior from k=6 onward**; this run does not
reach k=6, so it shows the early cells are intact, not that the late cells are.

**3. Free generation agrees with teacher-forced.** Same items, same k:

| k | skill | teacher-forced (n=64) | free generation (n=32) | gap |
|---|---|---|---|---|
| 2 | 8 | 1.000 | 1.000 | 0.000 |
| 2 | 5 | 0.922 | 0.969 | 0.047 |
| 3 | 8 | 1.000 | 1.000 | 0.000 |
| 3 | 5 | 0.922 | 0.969 | 0.047 |
| 3 | 7 | 1.000 | 1.000 | 0.000 |

Max gap **0.047**, inside the 0.05 requirement that had been failing
grid-wide. Free generation reading slightly ABOVE teacher-forced is a sample
artifact, not an impossibility: the free-generation row is the first 32 of the
same fixed 64-item draw.

**4. T4 pre-grid smoke, same skills.** just-after accuracy, e017 (pre-fix) vs
this run: skill 8 `1.000 -> 1.000`, skill 5 `0.890 -> 0.922`, skill 7
`1.000 -> 1.000`. Writes are not weaker; if anything skill 5 is stronger. The
FORMAL T4 guard is `mean_just_after` over 10 skills at `n_eval=200` from the
grid records and is not this number.

## 5c. GRID LAUNCH

`grid21.sh` on `arche-arb` [address removed], launched 2026-08-17T20:27:31Z
detached (`setsid`), `arb_sha256 = 1f51ebfbf765095e` (the verify run predates
the `.inprogress` addition, hence the different sha; no scoring or training
line differs). 15 runs = 5 arms x 3 seeds, `xargs -P 5`, `--threads 3`,
`--n-eval 200`, `--freegen-n 32`. Sizing is from MEASURED e017 RSS (5.5 GB
peak/run on a 62 GB box), not core count. e017's wall clock was ~4.2 h/run, so
~13 h for three waves. `~/e021/STATUS` is rewritten on every exit path;
`~/e021/runs/*.json.inprogress` is a complete record through the last finished
k, so the T4 guard is readable from wave 1 without waiting for the grid.

Read with `python analyze21.py --runs ~/e021/runs` after rsyncing records into
`runs/` and committing them. `runs_verify/` is deliberately a separate
directory: it is a 3-skill `n_eval=64` run and must never be ingested into the
grid read.

## 5d. FIRST GRID PARTIALS (k=1, committed) — per-skill write strength is IDENTICAL

`runs_partial/` (committed), 3 of the 5 wave-1 runs at k=1, `n_eval=200`, all
carrying `arb_sha256 = 1f51ebfbf765095e` and `git_sha = code:2d25eb1ae46dbfd9`
(no mixed sha inside the grid).

| skill | pre-fix e017 just-after | e021 grid partial |
|---|---|---|
| 11 `B_plural` | 0.805 | 0.805, 0.805 (two runs) |
| 8 `B_first` | 1.000 | 1.000 |

The terminator changed the write objective and changed these numbers by
**0.000**. That is the strongest available form of the T4 guard so far, and it
is per-skill rather than a mean.

The 3-row MEAN of those same values is 0.870, which is below the 0.877 floor —
and it is a COMPOSITION ARTIFACT, not a tax: `mean_just_after` is a mean over
SKILLS and the pre-fix band is a 10-skill mean that itself contained skill 11's
0.805 (e017 `selfdistil_s0`'s own 10-skill mean was 0.9405). `analyze21`
therefore now REFUSES the guard on any arm below k=10 and returns
`t4 = NOT-COMPARABLE-PARTIAL` / `verdict = INCOMPLETE_GRID`, so this misread
cannot be made by running the reading code. `load_runs` reads `.inprogress`
partials and skips truncated ones.

## 6. What a verifier should attack

- **The fix is on the write, so it changes the WHOLE family.** e004g's, e015's
  and e017's committed numbers were produced by the pre-fix write and are NOT
  comparable to e021's except through the T4 guard. e021 runs go to a separate
  directory (`ARB_RUNS`), never mixed with e017's.
- **The T4 guard is a mean.** A terminator tax concentrated on a few skills
  could hide inside a mean that stays in band. Per-seed `just_after` is
  recorded; per-skill is in the records.
- **`fid_late >= 0.60` and `gap <= 0.15` are thresholds chosen from the
  dossier's prose**, not from a power calculation. An outcome just either side
  of them is not a strong result.
- **The free-generation agreement test uses `freegen_n = 32`** against
  teacher-forced `n_eval = 200`. The two rows are the same items (the first 32
  of the same fixed draw) but not the same sample size, so the 0.05 tolerance
  is doing work at a resolution of 1/32.
- **`selfdistil_front` concentrates aux on recent priors, which are also the
  rows `CLEAN_AGE = 2` partly excludes.** If T3 reads positive, check whether
  the gain is in old skills or is an age-composition artifact.
- **One write path was deliberately NOT updated.** `arb.cueanchor_kl` builds
  its own teacher/student/alt-cue items with raw answers, so it is the one
  write in the family that does not score the terminator. `cueanchor` is not in
  the e021 grid (it failed at lambda=1 in e017), so nothing here depends on it,
  but any cueanchor re-run must apply `C.term()` to all three item lists first
  or its write is not comparable to the others'.
- **The terminator gate certifies TRUE answers and `SELF_NULL`, not recorded
  teacher strings.** `selfdistil_loss` trains on `term(recorded_string)`, and a
  recorded string is arbitrary model output; if one ended in a character that
  BPE-merges with `"\n"` (the `"?"` case), the aux would train a merged
  terminator token. Benign for termination — the merged token still contains
  the newline, so both the freegen cut and the `self_targets` cut fire — but it
  is not certified.
- **Single corpus, single base** (`Qwen2.5-0.5B`, the e005-descended synthetic
  skill suite). Decision 0012 clause 3: this is a claim about this corpus.
- **The method is published** — LwF (Li & Hoiem 2016) / SDFT, write = context
  distillation (Snell et al. 2022). What is ours is the MEASUREMENT that a
  termination defect capped it, the fidelity mediator, and the critical-window
  hypothesis.

## 7. Provenance

- Local repo commit and `arb_sha256` / consolidate hash are recorded inside
  every run record at LAUNCH (not at write).
- Box: `arche-arb` [address removed], `~/experiments` (rsync, not a git
  checkout — `git_sha()` therefore falls back to a content hash, which is the
  thing that has to match).
- Records: `~/e021/runs/*.json` on the box, rsynced to
  `experiments/e021-terminator/runs/` and committed. A number is citable only
  from a committed record.
