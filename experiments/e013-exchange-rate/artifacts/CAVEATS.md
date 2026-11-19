# e013 caveats

## Per-seed exchange rates (decision 0012: never a single headline number)

| skill | family | seed | C | n_ctx | n_grad | R | status |
|---|---|---|---|---|---|---|---|
| B_first | natural | 0 | 0.25 | 1.0 (at_first) | 1.317 (ok) | 1.32 | R_lower_bound |
| B_first | natural | 0 | 0.5 | 1.018 (ok) | 1.733 (ok) | 1.7 | ok |
| B_first | natural | 0 | 0.75 | 1.587 (ok) | 4.899 (ok) | 3.09 | ok |
| B_upper | natural | 0 | 0.25 | 1.0 (at_first) | 1.419 (ok) | 1.42 | R_lower_bound |
| B_upper | natural | 0 | 0.5 | 1.0 (at_first) | 2.012 (ok) | 2.01 | R_lower_bound |
| B_upper | natural | 0 | 0.75 | 1.019 (ok) | 2.687 (ok) | 2.64 | ok |
| N_map | novel | 0 | 0.25 | 64.0 (censored) | 19.398 (ok) | 0.3 | R_upper_bound |
| N_map | novel | 0 | 0.5 | 64.0 (censored) | 52.831 (ok) | 0.83 | R_upper_bound |
| N_map | novel | 0 | 0.75 | 64.0 (censored) | 59.742 (ok) | 0.93 | R_upper_bound |
| N_shift2 | novel | 0 | 0.25 | 64.0 (censored) | 21.791 (ok) | 0.34 | R_upper_bound |
| N_shift2 | novel | 0 | 0.5 | 64.0 (censored) | 32.334 (ok) | 0.51 | R_upper_bound |
| N_shift2 | novel | 0 | 0.75 | 64.0 (censored) | 71.382 (ok) | 1.12 | R_upper_bound |

Verdict object: `artifacts/verdict.json`. Family medians:
{
 "natural": {
  "0.25": 1.37,
  "0.5": 1.85,
  "0.75": 2.87
 },
 "novel": {
  "0.25": 0.32,
  "0.5": 0.67,
  "0.75": 1.03
 }
}

## Resolution

- Accuracy: teacher-forced exact match over 200 queries drawn from 64
  DISTINCT held-out words. The distinct-input count, not the query count,
  bounds the effective sample: treat SE as ~0.06 worst case, ~0.04 at
  p=0.9, NOT the 0.035 a naive binomial on n=200 would give.
- Example axes are sqrt(2)-ish grids, finer than doubling; crossings are
  interpolated in log2(n)
  between measured points. **No R at or below 2.0x is reported as
  a magnitude** — that is one grid step and e011 voided a claim on it.
- `at_first` means the curve was already above C at n=1: n_ctx is then an
  UPPER bound and R a LOWER bound. `censored` means the curve never
  reached C inside the grid.

## Known weaknesses, stated by the builder

1. N_map is coverage-limited by construction: an unseen first letter is
   unanswerable in EITHER arm. `coverage_oracle` in results.csv is the
   ceiling. A low R on N_map is therefore consistent both with X2 and
   with "this novel skill is a memorisation task rather than a rule".
   N_shift2 and N_perm are the systematic novel skills that answer that,
   and N_shift2 is only WEAKLY novel ("next letter" plausibly occurs in
   pretraining).
2. The gradient arm is LoRA-r8 with a swept LR and dev-based early
   stopping. It is not full fine-tuning; an untuned or under-capacity
   gradient arm inflates R in ICL's favour, which is the direction of
   the exciting answer. The LR optimum must be INTERIOR — see
   artifacts/lr_sweep.md.
3. Sample efficiency, not compute: the gradient arm takes many passes
   over its N examples. R counts EXAMPLES on both sides and says nothing
   about FLOPs.
4. One base (Qwen2.5-0.5B), one corpus family (marker-tagged one-line
   skills inherited from e004g). R is a property of this base and this
   skill format until a second base says otherwise.

## 5. The natural skills are in the wrong regime for the measurement

B_first scores 0.490 at ONE shot and 0.880 at two; B_upper 0.745 at one.
Read every natural-skill R as "small on a skill this base learns in one or
two shots", not as "R is small". NOTE (verifier, 2026-08-17): this is NOT
a floor artifact. With n_ctx = 1, R = n_grad and the gradient grid runs to
128, so a denominator of 1 MAXIMISES R rather than capping it. R came out
small because LoRA also reached C = 0.50 from 1.73 distinct examples and
C = 0.75 from 4.90. Both routes are fast on this skill. X2's R_natural >= 5
clause was empirically unfired, not structurally unfireable: it needed
n_grad >= 5.00 / 5.09 / 7.93 at C = 0.25 / 0.50 / 0.75, all inside the
grid, and the measured value at C = 0.75 missed by a factor of 1.6.

## 6. X3 is unreadable in this pilot, and the guard said so

All three capability crossings fall inside ONE measured grid interval on
both skills. Not fixable by a finer grid -- the in-context axis is integer
and 1 is its floor. It IS fixable by a skill whose in-context arm crosses
C = 0.75 between 8 and 30 shots; see the DESIGN's VERIFIER RULING section 3.

## 7. One seed

Every number is seed 0 unless labelled otherwise. The gradient curves are
visibly noisy at one seed (B_first 0.630 -> 0.330 -> 0.590 at N = 2, 3, 4).
The crossing rule takes the first point above C, so noise shortens n_grad
and SHRINKS R: the one-seed bias runs toward the kill, not away from it.
With every measured n_grad doubled, B_first's median R is 3.4, still under
the 5.0 kill line.

## 8. Novel skills without a gradient arm

N_perm has an in-context curve (0.000 at every shot count) but no gradient
arm, so it carries no R and no swept LR. Its 0.000 may be a tokenization
effect on a 5-character permuted answer (line length 9.67 +/- 0.53 tokens
against 8.06 for the single-character skills) rather than a novelty
effect -- e004g's B_reverse failed its gate the same way. Do not read
N_perm as evidence.

## 9. Bounds are not the same as censoring (verifier, 2026-08-17)

`censored_frac` counts cells with no R at all; `bounded_frac` counts cells
whose R is a bound. DESIGN's INCONCLUSIVE rule named both, the code counted
only the first. Rather than blanket-refusing a verdict when bounds are
present -- a bound on the decisive side of a threshold is informative --
`bound_stress()` pushes every bounded cell to its worst case for the flag
under test and the flag must survive. See `verdict.json/kill_bound_stress`.

## 10. Rule-vs-table split on the gradient arm (verifier, 2026-08-17)

`acc_seen_letter` / `acc_unseen_letter` in results.csv split eval accuracy
by whether the query word's FIRST LETTER appeared in the gradient arm's
training set. A rule learner answers unseen letters; a table learner
cannot. Read it together with the B_first control in `runs_ctl/`, which
runs the same split on a rule the base already has, so that a low
unseen-letter score cannot be blamed on letter rarity alone.
