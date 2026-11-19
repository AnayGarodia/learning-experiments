# e021 — rehearsal that stores no old data recovers most of what stored replay buys

Status: CALLABLE (3 seeds, 10 skills, k=10, one base). Verdict from the
pre-registered analyzer: `A1_ARBITRATION_FIXES_RETENTION`. Known positive in
band. Pool gate clean (max unprompted-before-write 0.094, threshold 0.20).

## The numbers

Qwen2.5-0.5B, LoRA r=8, 10 gate-clean prefix-free skills written sequentially
with no task identity given, measured on held-out words. Retention is each
skill's accuracy now over its accuracy just after it was written, on rows at
age >= 2 with conditionality-blocking rows dropped (the e015 verifier's
pre-stated row-level procedure, 93-108 clean rows per arm).

| arm | retention (mean) | per seed | umar_age0 |
|---|---|---|---|
| no_replay | 0.000 | 0.000 / 0.000 / 0.000 | 0.947 |
| replay25 (stores old data) | 0.778 | 0.782 / 0.684 / 0.867 | 0.379 |
| **selfdistil (stores nothing)** | **0.695** | 0.733 / 0.740 / 0.613 | 0.294 |
| selfdistil_front | 0.300 | 0.255 / 0.327 / 0.319 | 0.453 |
| selfdistil_offcue (control) | 0.016 | 0.008 / 0.011 / 0.030 | 0.725 |

Write strength is matched across arms (mean_just_after 0.940-0.942), so no arm
is write-taxed and the ratios are not inflated by a lowered denominator.

## What is ours and what is not

**The mechanism is NOT ours.** Constraining a new write to preserve the
model's own outputs on old cues is Learning without Forgetting
(Li & Hoiem 2016), and the apparent-forgetting-is-inference diagnosis is
Kotha et al. 2023. This is a reproduction at LLM scale in the sequential,
no-task-identity setting.

**The mandatory predicted-negative control fired.** LwF's own documented
failure mode is proxy inputs that do not exercise the old behaviour. The
off-cue arm uses exactly such proxies and retains 0.016 against the on-cue
arm's 0.695 (difference 0.679, `control_behaved: true`). Without this the
retention could not be attributed to preserving the old cue's function.

**What the sweep found unclaimed, and what this measures:** rehearsal's
FUNCTION as conditionality restoration. `umar_age0` -- how often the newest
write answers under a cue that is not its own -- is 0.947 with no rehearsal
and 0.294-0.379 with either kind. The field treats replay as protecting
representations; here both kinds of rehearsal restore conditionality, and the
one that stores nothing restores it as well as the one that stores data.
No source found reports this quantity.

## Caveats

- One base, one model size, k=10. Sequential editing is published to collapse
  around k~32 and this bounds nothing past k=10.
- selfdistil is BELOW replay25 in mean and in two of three seeds. The claim is
  "recovers most of", not "matches".
- 0.294 vs 0.379 umar_age0 is inside seed spread; do not claim selfdistil is
  MORE conditional than stored replay.
- Retention is a ratio. `prior_abs_clean` (absolute accuracy) is 0.643 for
  selfdistil against 0.719 for replay25 and 0.000 for no_replay.
- Not verifier-reviewed yet.
