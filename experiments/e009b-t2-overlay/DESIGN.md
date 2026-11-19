# e009b — T2 discriminator: oracle-skipped compute vs the dense inference frontier

Status: DESIGN 2026-08-16, written before code (decision 0010 am.1).
Gating follow-up named by dossier 009's VERDICT and by the e009 results
card ("Further work (1)"). Nothing in dossier 009 may be read as
"adaptive compute is a lever" until this runs.

## The question, stated so it can lose

e009 measured that a post-hoc oracle can skip 80.5/84.2/88.1% of
(token, block) compute at s/m/l at argmax parity, growing with training
and with scale. T2 (small-model-equivalence) says this is nothing new:
the savings are just "a smaller dense model would have sufficed for
most tokens". Dossier 009's pre-registered line:

> **T2 — Small-model-equivalence.** PREDICTS: oracle-skipped compute
> curve overlays the dense compute-optimal frontier; KILLED-BY: joint
> oracle beating the dense frontier by >=2x at matched capability.

This experiment places both curves on one plot and measures the
horizontal gap.

## THE ORACLE IS POST-HOC AND NON-REALIZABLE. THIS IS A CEILING.

Say it twice because the whole reading depends on it. The oracle needs
the full model's answer in order to decide what to skip: the depth
criterion is "the intermediate readout already argmax-matches the
final logits", which cannot be evaluated without the final logits.
Nothing here is a mechanism, and no measured number here is achievable
by any router, gate, or predictor. Every number is an upper bound on
what any adaptive-compute mechanism at this granularity could deliver.

That is exactly what makes this a cheap kill. **If the ceiling does not
beat the dense frontier by 2x, adaptive compute dies as a lever** —
not "is hard", dies, because no realizable mechanism can exceed the
oracle that sees the answer. A kill here saves the mechanism-building
program named in dossier 009 ("BLT entropy-patching plus MoD are the
mechanism candidates"). A survival licenses it. Both outcomes are
worth the compute; the null is the cheaper and more likely one.

Two further respects in which the ceiling is inflated, both retained
deliberately (they can only help T2 die, and T2 is what we are trying
to kill):
- **Temporal reuse is FLOPs-free but KV-full.** When the oracle reuses
  block b's output from t-1 at position t, we charge zero FLOPs, but
  the measurement still lets later positions attend to position t's
  true keys/values at block b. A real implementation would have to
  produce that KV. Depth-skip does NOT have this problem: the
  passthrough state is what later positions attend to, exactly as in
  CALM, so the depth-only arm is implementation-faithful.
- **Iso-token, not iso-training-FLOPs.** All rungs see 6M tokens, so
  the large rungs received more training compute than a
  compute-optimal allocation would give them. That raises the
  high-FLOPs end of the frontier, which raises the capability the
  oracle rung is credited with, which *increases* the measured
  multiplier. Generous to the oracle by construction.

## Rival readings (pre-registered)

**T2-SURVIVES (multiplier ~1x).** The oracle's effective FLOPs land on
or above the dense frontier: some smaller dense model reaches the same
capability for the same or less inference compute. Adaptive compute
buys nothing that model selection does not already buy. R5 closes
deflationary like the rest of the series.

**T2-KILLED (multiplier >=2x).** At matched capability the oracle rung
costs less than half of the cheapest dense model reaching that
capability. There is a per-input adaptivity premium that no static
size choice captures, and the mechanism program is licensed.

**DEGENERATE-FRONTIER (the artifact reading, and the one we expect at
the end of training).** If every rung reaches the same capability, the
"frontier" is a horizontal line, the smallest rung trivially suffices,
and the multiplier is a statement about the corpus rather than about
adaptivity. A pre-flight probe on the existing e009 checkpoints already
shows this happens at 6M tokens:

| rung | tokens | next-tok acc | consequence acc | struct acc | loss |
|------|--------|--------------|-----------------|-----------|------|
| s (0.89M) | 6.0M | 0.8339 | 1.000 | 1.000 | 1.027 |
| m (3.07M) | 6.0M | 0.8332 | 1.000 | 1.000 | 1.021 |
| l (9.44M) | 6.0M | 0.8336 | 1.000 | 1.000 | 1.020 |
| s | 229k | 0.6190 | 0.186 | 0.790 | 6.077 |
| m | 229k | 0.7433 | 0.431 | 0.932 | 3.331 |
| l | 229k | 0.7549 | 0.461 | 0.940 | 1.566 |

The e005 corpus is mostly irreducible entropy by construction (fork
tokens carry log2(8) bits, noise spans are uniform), so accuracy has a
hard ceiling near 0.834 that all three rungs already sit on. The
capability axis therefore has dynamic range only at earlier training
levels. This is stated up front so that the primary-level selection
rule below is a rule and not a post-hoc rescue.

## Design

### Ladder (extending e009, same recipe family, same pinned corpus)

Reuses e009's `corpus.py`, `model.py`, `train.py` by import — the
corpus is not forked and the recipe is not re-specified. Three new
rungs are added to `model.SIZES` at runtime:

| rung | d | L | nonemb params | status |
|------|---|---|---------------|--------|
| xxs | 32 | 4 | 0.049M | new |
| xs | 64 | 6 | 0.294M | new |
| s | 96 | 8 | 0.885M | e009, reused |
| m | 160 | 10 | 3.07M | e009, reused |
| l | 256 | 12 | 9.44M | e009, reused |
| xl | 384 | 14 | 24.8M | new |

d/L follows e009's family (d x ~1.5, L +2 per rung). Adjacent rungs are
2.19x-2.95x apart in inference FLOPs/token (483.8k / 1.426M / 3.124M /
8.560M / 23.00M / 56.15M, from `ladder.py`) — see the MDE section, this
spacing is load-bearing for a 2x verdict line. Two rungs are added
BELOW e009's smallest deliberately: the frontier bends at the small
end, and the small end is where a dense model can undercut the oracle.
Same optimizer, batch, sequence length, checkpoint schedule
(4/10/25/50/100% of 6M tokens) and corpus seed as e009, so the reused
s/m/l checkpoints are valid frontier points.

### Inference FLOPs per token (honest accounting)

Matmul FLOPs only, 2 FLOPs per multiply-accumulate, forward pass,
average over positions of one seq_len=256 sequence with causal masking
(mean context = (T+1)/2 = 128.5). Per block, per token:

    qkv projection      2 * d * 3d          = 6 d^2
    QK^T                2 * d * ctx
    attn @ V            2 * d * ctx
    output projection   2 * d * d           = 2 d^2
    SwiGLU gate + up    2 * (2 * d * h)     = 4 d h
    SwiGLU down         2 * h * d           = 2 d h
    ------------------------------------------------
    per block           8 d^2 + 6 d h + 4 d ctx      (h = int(8d/3))

Plus once per token, outside the blocks and NOT skippable by any
oracle: the unembedding head, 2 * d * V with V=5000. Embedding and
positional lookups are gathers (0 matmul FLOPs); RMSNorm, SiLU,
softmax, and residual adds are not counted (elementwise, o(d) or
o(d*ctx) with tiny constants, and counting them would only shrink the
skippable share since the head-side elementwise cost is unskippable).

Hand-check that the unit test asserts, rung s (d=96, L=8, h=256,
ctx=128.5): 8*96^2 = 73,728; 6*96*256 = 147,456; 4*96*128.5 = 49,344;
per block 270,528; x8 blocks = 2,164,224; head 2*96*5000 = 960,000;
total 3,124,224 FLOPs/token. `ladder.py` asserts this exactly.

Note what the head does to the story: it is 66.1% of inference FLOPs at
xxs, 44.9% at xs, 30.7% at s, and 6.8% at xl. **The unembedding is a
floor no adaptive-compute scheme can skip.** An oracle that removes 88%
of block compute removes only 78% of total compute at l, 48% at xs and
30% at xxs; the whole-token speedup from an 88% block-skip runs
1.42x (xxs) / 2.56x (s) / 4.59x (l) / 5.55x (xl). Reporting
skip fractions against block compute alone (as e009 did, correctly for
its question) overstates the end-to-end saving; this experiment prices
the whole token.

    effective FLOPs/token = head + (1 - skip_fraction) * block FLOPs

Skipped (token, block) passes are charged zero. Whether a real kernel
can realize that is the T3 question, answered in e009 (the redundancy
is coarse-structured, hence priceable); it is not re-litigated here.

### Capability axis — and the subtlety that decides what may be plotted

The oracle's criterion is argmax-agreement with the full model. This
means **accuracy-type metrics are preserved by construction and loss is
not**. A metric that reads only the argmax cannot distinguish the
oracle-executed model from the full model wherever the criterion holds;
a metric that reads the whole distribution can, and will look worse,
because an early-exit readout is under-confident and mis-calibrated
even when its argmax is right. Plotting loss against a criterion that
never promised to preserve loss would manufacture a null.

So, pre-registered:
- **Capability axis 1 (primary): next-token accuracy**, all positions,
  the metric the argmax criterion actually preserves.
- **Capability axis 2 (primary): consequence accuracy**, the e005
  downstream metric — accuracy on positions whose target token carries
  oracle label 3 (the token determined by a fork that is body+noise
  tokens back). This is the metric with dynamic range across the
  ladder, and it is the one that tests whether skipping compute breaks
  long-range dependency use rather than just local prediction.
- **Loss: reported, plotted in its own panel, and labelled NOT
  PRESERVED BY THE ORACLE CRITERION.** It is a diagnostic of what the
  argmax criterion discards, not a capability axis for the verdict.
  Any T2 verdict read off the loss panel is invalid and CAVEATS.md
  says so.

Capability of the oracle configurations is **measured, not assumed**.
This is a change from e009's accounting and it matters. e009's depth
and temporal fractions are per-(token, block) counterfactuals evaluated
independently against the full model's hidden states; their union is
an upper bound on a jointly valid configuration, not a verified one.
Here every oracle point is produced by a single masked forward pass
that applies ALL of that configuration's skips simultaneously:

    for each block b:
        out = block_b(h)
        h[t] <- h[t]        if (t, b) depth-skipped   (CALM state copy)
        h[t] <- out[t-1]    if (t, b) temporally reused
        h[t] <- out[t]      otherwise

and capability is then measured on that model's actual outputs. If the
union is jointly valid, measured capability equals the full model's; if
the skips interact, the drift shows up as lost capability and the point
moves DOWN the plot instead of silently keeping a parity it does not
have. Reporting the drift is a deliverable
(`joint_agreement_measured`).

### The four points per (rung, checkpoint)

1. **dense full** — (full FLOPs/token, measured capability)
2. **depth-only oracle, argmax** — implementation-faithful arm
3. **joint oracle (depth ∪ temporal), argmax** — the dossier's headline
   oracle, the ceiling arm
4. **the same two oracles under KL<=0.1** — the criterion gap is a
   pre-registered reporting requirement of dossier 009 (constraint d),
   so the whole overlay is emitted a second time under KL. e009 found
   depth skippability falls 0.72-0.81 -> 0.26-0.45 between criteria;
   the overlay must be read knowing which criterion bought it.

### Frontier fit

At a fixed training-token level, the dense frontier is the Pareto
envelope over the ladder of (inference FLOPs/token, capability): the
upper-left staircase, then log-linear interpolation in
(log10 FLOPs, capability) between adjacent frontier rungs.

Why iso-token rather than a single pooled frontier: inference FLOPs do
not depend on the checkpoint, so pooling training levels would let a
6M-token xs point undercut a 229k-token l point and call it a frontier.
The comparison T2 makes is "a smaller dense model, trained the same
way, would have sufficed", so the frontier is drawn among models
trained the same way.

**Interpolation is load-bearing and the fit is stated, not hidden.**
Alternative fits (log-linear in capability vs log-FLOPs; monotone
piecewise-linear) are both computed and both reported; if they
disagree about which side of the 2x line the multiplier falls on, the
result is INCONCLUSIVE by rule, not by taste.

### The multiplier (the verdict statistic)

For an oracle point at effective FLOPs F_o and measured capability C_o:

    M = F_frontier(C_o) / F_o

F_frontier(C_o) is the interpolated dense FLOPs at capability C_o.
M > 1 means the oracle undercuts the frontier; M < 1 means a dense
model of that capability is already cheaper than the oracle ceiling.
Headline M is the maximum over rungs (the arrangement most favourable
to killing T2), for the joint argmax oracle, at the primary training
level. Reported alongside: M for every rung, for depth-only, and for
both criteria.

Out-of-support handling, pre-registered: if C_o exceeds the capability
of every dense rung, F_frontier is undefined above support and M is
reported as ">= F(largest rung)/F_o" (a lower bound, which is the
direction that favours killing T2). If C_o falls below the smallest
rung's capability, M is reported as "< F(smallest rung)/F_o", an upper
bound. No extrapolation of the fit beyond the rungs.

## Verdict rule (pre-registered, against the 2x line)

Primary training level: **the largest training-token level at which the
dense ladder still separates** — formally, the largest level at which
at least one adjacent-rung capability gap on the primary axis exceeds
5x the bootstrap CI half-width. If no level qualifies, the experiment
is UNINTERPRETABLE (see below). The final (6M-token) level is ALWAYS
reported too, whether or not it qualifies, with its degeneracy stated.

Let M be the headline multiplier with its 95% bootstrap CI:

- **M_lower >= 2.0** -> **T2 KILLED.** Per-input adaptivity beats model
  selection at matched capability. Dossier 009's R5 lever survives; the
  mechanism program (MoD, entropy-patching) is licensed.
- **M_upper <= 1.6** -> **T2 SURVIVES, and adaptive compute dies as a
  lever at this setting.** The ceiling itself does not beat picking a
  smaller model, so no mechanism can. R5 closes deflationary.
- otherwise -> **INCONCLUSIVE**, and the reason (CI width, frontier
  spacing, fit disagreement) is named in the results.

The verdict is evaluated independently on each of the two primary
capability axes and on each of the two frontier fits. **All must
agree**; any disagreement across the 2x line is INCONCLUSIVE. One axis
killing T2 while the other does not is a statement about which
capability was preserved, not a kill.

The asymmetric band (2.0 / 1.6) is not hedging: it is the MDE below.
The dossier's line is 2x, and 1.6 is where the measurement can
distinguish "clearly under 2x" from "could be 2x".

Secondary, reported but not verdict-bearing: the multiplier under
KL<=0.1, the depth-only multiplier, the per-rung multipliers at every
training level, and the trend of M with training tokens.

## MDE and resolution (decision 0010 items 3 and 4)

**Eval size.** 256 sequences x 256 positions = 65,280 next-token
predictions, of which ~4,040 carry consequence labels. Held-out corpus
seed 9999, identical to e009's eval stream.

**Capability resolution.** Binomial 95% half-widths: next-token
accuracy +/-0.0029 at p=0.83; consequence accuracy +/-0.0154 at p=0.5.
Reported CIs are bootstrap over sequences (1000 resamples), which will
exceed the binomial figure because positions within a sequence are
correlated; the bootstrap number is the one that gates.

**Grid-snapping guard.** The artifact that killed e005 run 1 and e006
run 1 was a capability axis quantised coarser than the effect. Here the
capability axis is continuous at resolution ~0.003 (accuracy) and
~0.015 (consequence accuracy), against inter-rung gaps that must exceed
5x that to qualify as the primary level — the gate is defined in units
of the instrument's own resolution, so a snapped axis cannot pass it.
The FLOPs axis carries the mirror-image risk: it is quantised by having
only six rungs. Adjacent rungs are 2.19-2.95x apart in FLOPs, which is
the same order as the 2x verdict line, so **the frontier grid is the
binding resolution limit, not the capability grid.** Two mitigations,
both pre-registered: log-linear interpolation between rungs (which
makes the FLOPs axis continuous within a rung interval), and reporting
both fits so that a verdict that depends on the interpolation choice
is declared inconclusive rather than believed.

**MDE on the multiplier.** With capability CI half-width e and adjacent
rung capability gap g, interpolation error in log10-FLOPs is at most
(e/g) * log10(rung spacing). At the qualifying gate (g >= 5e) and the
worst rung spacing 2.95x, that is <= 0.2 * 0.47 = 0.094 dex, a factor
of 1.24. The skip-fraction contribution is much smaller (e009's
fractions carry +/-0.004 at 65k x L samples, propagating to <5% on
effective FLOPs). So the multiplier is resolved to roughly +/-25%
multiplicatively:
M >= 2.0 with CI is distinguishable from M <= 1.6, and the 1.6-2.0
band is where the instrument cannot call it. Effects worth detecting
(2x) are ~11x the resolution floor at the gate.

**Spread check (0010 item 4).** The comparison mechanism is the
frontier, so the pilot prints the variance of what it compares: the
capability spread across rungs at each training level, and the spread
of skip fractions across rungs. Near-zero spread at every level blocks
launch. Near-zero spread at the final level only is expected and does
not block (it is the DEGENERATE-FRONTIER reading, reported as a
result about the corpus).

**Seeds.** Seeds 0 and 1 at every rung (12 training runs; s seed 1 and
s/m/l seed 0 already exist from e009 and are reused). Bootstrap
resamples eval sequences and, where two seeds exist, seeds as well.
Seed-to-seed spread on skip fractions in e009 was 0.01-0.05, largest at
the least-trained checkpoint — that magnitude propagates to <10% on M.

## What would make this uninterpretable (pre-registered)

1. **No training level passes the dynamic-range gate.** Then the ladder
   never separates on this corpus and no frontier exists to overlay
   against. Verdict: UNINTERPRETABLE, T2 stays undiscriminated, and the
   named next step is a corpus with a non-saturating capability axis —
   NOT a rescue of this run by switching metrics after seeing them.
2. **Joint masked-forward capability collapses** (measured agreement
   with the full model far below the union's implied parity). Then the
   joint point is not "the same capability more cheaply", it is a
   different, worse model, and only the depth-only arm carries a
   verdict. Reported either way; the joint arm is demoted, not dropped.
3. **The two frontier fits disagree across the verdict line.** Rung
   spacing is then too coarse for the question at this ladder density.
   Fix is more rungs, not a chosen fit.
4. **Non-monotone frontier beyond CI** (a larger rung worse than a
   smaller one at matched tokens by more than the bootstrap CI). Then
   the training recipe is not size-transferable at these scales and the
   Pareto envelope is measuring recipe noise.
5. **FLOPs counter fails its hand-computed unit test**, or the
   recomputed oracle fractions disagree with e009's published values on
   the shared s/m/l checkpoints beyond eval noise. Either means the
   instrument is not the one described here.

## Expectations (pre-registered, so the result is not read backwards)

Stated before running, from the pre-flight probe above and from e009's
numbers:
- The final-checkpoint overlay is expected to be DEGENERATE and to
  produce M < 1 (rung s at 3.12M FLOPs/token already reaches the
  saturated capability, while the oracle at rung l needs ~5.0M
  effective FLOPs/token). Predicted, not discovered.
- At 229k tokens the ladder separates and M is expected to be
  ABOVE 1 and possibly above 2 (rung l reaches capability no smaller
  rung reaches, at an oracle-effective cost between rungs s and m).
- Therefore the headline verdict is expected to hinge on the primary
  level rule, and the honest summary is likely to be "the multiplier
  is large only where the dense ladder has not yet converged, and goes
  below 1 once it has" — which is a T2-survives reading with a
  training-dependence caveat. If the numbers land otherwise, the
  pre-registration above is what decides, not this paragraph.

## Deliverables

- `ladder.py` — extra rungs, FLOPs counter, unit test vs the hand
  computation, self-check on `__main__`.
- `overlay.py` — per (rung, checkpoint): skip masks under both
  criteria, masked forward, measured capability, effective FLOPs;
  one JSON record per measurement in `runs/`.
- `analyze.py` — frontier fit, overlay table, multipliers with
  bootstrap CIs, verdict against the 2x rule; writes
  `artifacts/results.csv`, `artifacts/results.png`,
  `artifacts/CAVEATS.md` (decision 0010 am.2).
- `pilot/` — the pilot record.

## Pilot (decision 0010 item 1-2)

Two maximally-different cells at one seed, full analysis path
exercised: rung xxs (0.049M) and rung l (9.44M), at the 229k-token
checkpoint where the ladder is known to separate. Because the new
small rungs train in minutes and the s/m/l checkpoints already exist on
lab-2, the pilot in practice runs the complete five-rung frontier at
two training levels, which is the smallest configuration that
exercises frontier fitting, interpolation, out-of-support handling, and
the verdict rule at once. xl and seed 1 are NOT run before verifier
sign-off.

Pilot outcome and any design amendments are appended to this file
below the line before the pilot is committed.

## Pilot outcome (2026-08-16, decision 0010 items 1-4)

Run on lab-2. New rungs xxs (d=32, L=4, 0.049M nonemb) and xs (d=64,
L=6, 0.295M nonemb) trained on the pinned e005 corpus by e009's
train.py at the same 6M-token schedule (204s and 320s wall). The s/m/l
seed-0 checkpoints were reused from e009 unmodified. Overlay measured
at five rungs x two training levels (229,376 and 5,996,544 tokens),
seed 0, 256 eval sequences; analyze.py run end to end. Outputs in
`pilot/`, artifacts in `artifacts/`.

### Instrument checks

**FLOPs counter** passes its hand-computed unit test exactly (rung s =
3,124,224 FLOPs/token; `ladder.py` asserts the itemised terms too).

**Same instrument as e009.** Recomputed fractions on the five shared
checkpoints, at 256 eval sequences against e009's 64:

| ckpt | depth_arg here / e009 | joint here / e009 |
|------|----------------------|-------------------|
| s 229k | 0.3799 / 0.3804 | 0.3880 / 0.3885 |
| m 229k | 0.6885 / 0.6870 | 0.6920 / 0.6903 |
| l 229k | 0.7961 / 0.7924 | 0.8036 / 0.8013 |
| s 6.0M | 0.7167 / 0.7150 | 0.8034 / 0.8052 |
| m 6.0M | 0.7650 / 0.7629 | 0.8413 / 0.8422 |

All within eval noise. Uninterpretable condition 5 does not fire.

**Dynamic range.** Both levels pass the gate on both axes:

| level | axis | max adjacent gap | CI half-width | gap/CI | gate |
|-------|------|------------------|---------------|--------|------|
| 229k | acc | 0.5525 | 0.0044 | 126x | PASS |
| 229k | cons | 0.2509 | 0.0161 | 15.6x | PASS |
| 6.0M | acc | 0.0455 | 0.0041 | 11.0x | PASS |
| 6.0M | cons | 0.4171 | 0.0151 | 27.6x | PASS |

The 6M level passes only because the ladder was extended DOWNWARD:
xxs does not saturate (acc 0.7916, consequence 0.5770) while xs/s/m/l
all sit at 0.837/1.000. Had the ladder started at e009's s, the final
level would have been degenerate and the primary verdict would have
had to be taken at an undertrained checkpoint. Adding rungs below the
existing smallest was the single most load-bearing design choice.

**Capability is measured, not assumed — and the union is NOT jointly
valid.** Executing all skips at once costs real capability:

| rung @6M | config | skip | eff FLOPs | cons acc | argmax agreement |
|----------|--------|------|-----------|----------|------------------|
| l | full | 0 | 23.00M | 1.0000 | 1.0000 |
| l | depth_arg | 0.808 | 6.49M | 0.5837 | 0.8993 |
| l | joint_arg | 0.880 | 5.01M | 0.9625 | 0.9815 |
| s | depth_arg | 0.717 | 1.57M | 0.5977 | 0.8785 |
| s | joint_arg | 0.803 | 1.39M | 0.9248 | 0.9452 |

e009's headline "80-88% of (token, block) compute is skippable at
argmax parity" does not survive execution: at l the executed joint
configuration agrees with the full model on 98.2% of tokens, not 100%,
and the executed depth-only configuration on 89.9%. The union of
independently-verified counterfactuals is an upper bound, as suspected;
now it is measured. Every overlay point is placed at its measured
capability, so this costs the oracle rather than hiding.

**Controls order correctly (item 2), and they had to.** The executed
joint arm scores HIGHER than executed depth-only despite skipping
strictly more — counterintuitive enough to be a suspected masking bug.
Controls at s@6M (`pilot/controls.txt`): random mask of matched density
0.7537 < reuse from the WRONG neighbour t+1 0.8466 < depth-only 0.8745
< true t-1 reuse 0.9467. An arbitrary perturbation of the same density
makes things worse, as it must; the gain is specific to genuine
temporal reuse. A separate no-drift control (final state taken from the
full model's own hidden state at the exit block) returns exactly 1.0000
agreement, proving the exit criterion is implemented correctly and that
everything the depth arm loses is execution drift — frozen early-exit
states feeding later blocks' attention for OTHER positions, the
CALM state-copy pathology. Both controls are now computed inside
overlay.py at every cell, so the ordering is a per-cell tripwire on the
grid, not a one-off.

**Spread (item 4).** Skip fractions across rungs at 6M: 0.493 / 0.730 /
0.803 / 0.842 / 0.880; capability across rungs at 6M (consequence):
0.577 / 0.994 / 1.000 / 0.9995 / 1.000. Neither is near-zero spread.

### Pilot verdict statistic (NOT the verdict — one seed, two levels)

Primary level by the pre-registered rule (largest level passing the
gate) = 6.0M tokens. Headline joint-argmax multiplier, max over
in-support rungs:

- next-token accuracy: M = 1.08, CI [1.05, 1.11], logit fit 1.05
- consequence accuracy: M = 1.45, CI [1.42, 1.48], logit fit 0.98

Both axes and both fits land below the 1.6 line: **T2 SURVIVES —
the oracle ceiling does not beat picking a smaller dense model.** At
rung l the joint oracle needs 5.01M FLOPs/token to reach consequence
accuracy 0.9625, while the dense frontier reaches that capability at
about 1.3M: M = 0.26. The oracle is nearly 4x WORSE than model
selection at matched capability.

Secondary, and the one genuinely interesting pattern: at the 229k level
the multipliers are much larger and rise with rung size — m 2.26 (acc)
/ 2.21 (cons), l 2.73 [2.47, 3.00] (acc) / 1.76 [1.56, 1.97] (cons).
On accuracy that clears the 2x line; on consequence accuracy it does
not, so the two primary axes disagree and 229k would read INCONCLUSIVE
under the rule even if it were primary. The shape of the result is
that **the adaptivity premium exists while the dense ladder is still
separating and disappears once capability saturates.** Whether that
survives more levels, a second seed, and rung xl is exactly what the
grid is for. No verdict is claimed from the pilot.

### MDE, as measured rather than as predicted

Bootstrap CI half-widths at 256 eval sequences: next-token accuracy
+/-0.004, consequence accuracy +/-0.015-0.016 (4,040 labelled
positions), against a 2x verdict line and rung spacing of 2.19-2.95x.
The resulting multiplier CIs are +/-3% at well-separated rungs and
+/-10% where the frontier is flat (l@229k on accuracy: [2.47, 3.00]).
Both are inside the +/-25% budgeted in the MDE section, so the
instrument resolves the 2x line wherever the frontier is not flat.
Where the frontier IS flat — adjacent rungs within noise of each other
— the bootstrap goes bimodal (rungs swap places on the Pareto envelope
between resamples; see xs@229k on consequence accuracy, CI [0.39,
1.18]). That is the instrument correctly reporting that it cannot
place a frontier there, and it is why the verdict is gated on the
dynamic-range test rather than on the CI alone.

### Amendments made after the pilot

All are instrument fixes; none moves the 2.0 / 1.6 thresholds, and the
originally pre-registered quantity is retained wherever one is replaced.

1. **Second frontier fit changed from global log-linear to piecewise
   log-linear in LOGIT capability.** The pre-registered second fit was
   a least-squares straight line through (log10 FLOPs, capability).
   That model cannot represent a saturating frontier, and it disagreed
   with local interpolation almost everywhere (e.g. 3.41 vs 1.46 at
   s@229k), which would have forced a permanent INCONCLUSIVE for a
   reason having nothing to do with interpolation error. The
   replacement is also a local, monotone interpolation and differs only
   in shape assumption within a rung interval, which is the
   uncertainty the second fit is meant to probe. The global log-linear
   value is still computed and still written to results.csv as
   `M_acc_loglin_diag` so the original pre-registration remains
   auditable. This amendment makes verdicts MORE callable, so it is
   flagged loudly rather than buried.
2. **The headline max over rungs now considers only in-support
   points.** A below-support point overstates M (the true frontier lies
   left of the smallest rung), so allowing one to set the headline
   could manufacture a kill. In the pilot this changed the accuracy
   headline from 1.20 (xxs, below-support) to 1.08 (xs, in support);
   the verdict was unchanged. Out-of-support points are still reported
   as one-sided bounds.
3. **Both fits must now agree on the VERDICT CATEGORY**, not merely on
   which side of the 2.0 line they fall. The original check would have
   allowed a SURVIVES call at pw 1.5 while the other fit said 1.9.
4. **Two controls added to every cell** (wrong-neighbour reuse, random
   matched-density mask), per the finding above.
5. **Eval chunked at 64 sequences.** At 256 sequences a logits tensor
   is 1.3GB at the large rungs; sequences never interact, so chunking
   is exact and keeps peak memory near 3GB while three e010 jobs hold
   ~6GB on lab-2.

### Cost note for the grid

Oracle measurement is dominated by readouts over a 5,000-token vocab:
the depth probe alone needs L of them. At 256 eval sequences, rung l
took ~25 minutes per checkpoint on a contended 16-core box. Rung xl
(L=14, d=384) will be ~2.5x that per checkpoint, so the full grid
(6 rungs x 5 levels x 2 seeds) is dominated by xl and is the item to
size or trim first. Training is comparatively free (xxs 204s, xs 320s,
l 2569s for 6M tokens).

---

## VERIFIER RULING (2026-08-16, decision 0010 am.1 — independent context)

Attacked: the post-pilot analysis amendment, the execution finding and its
upstream consequences, the joint-beats-depth anomaly, the regime claim, and
the standard instrument checks. Everything below was recomputed from the
committed `runs/*.json` or re-measured from the checkpoints on lab-2 with
code that does not import `overlay.py`.

### 1. The post-pilot fit amendment — NOT COSMETIC, but the replacement is right and the fix is to need neither fit

**(a) The verdict category DOES differ between the two fits.** At the
primary level (6.0M tokens), headline joint-argmax multiplier, max over
in-support rungs:

| fit | acc | verdict | cons | verdict |
|-----|-----|---------|------|---------|
| piecewise-linear (primary, pre-registered) | 1.08 [1.05,1.11] | SURVIVES | 1.45 [1.42,1.48] | SURVIVES |
| piecewise-logit (the amendment) | 1.05 [1.02,1.08] | SURVIVES | 0.98 [0.93,1.02] | SURVIVES |
| **global log-linear (originally pre-registered second fit)** | **1.46 [1.27,1.64]** | **INCONCLUSIVE** | **2.14 [2.07,2.21]** | **T2 KILLED** |

So the builder's statement that the amendment is an instrument fix that
does not move the verdict is wrong: under the fit that was actually
pre-registered, the pilot reads T2 KILLED on the consequence axis and
INCONCLUSIVE on accuracy — the two primary axes disagree, and the primary
fit disagrees with the second fit, both of which force INCONCLUSIVE under
the rule as written before the pilot. Replacing that fit is what makes
SURVIVES callable. The amendment note in the section above understates
this and has been left in place with this ruling as its correction; the
worked example it cites (3.41 vs 1.46 at s@229k) is at a non-primary level
and a non-headline rung, and is not the disagreement that matters.

**(b) The original fit is genuinely misspecified, not merely inconvenient,
and this is provable without appeal to taste.** F_frontier(C) is bounded
below by the last frontier rung with capability under C and above by the
first rung at or over it — those are measured points, not modelling
choices. The global least-squares log-linear fit lands OUTSIDE that
interval in 20 of the 32 in-support pilot cases. It reports a dense-FLOPs
requirement that a directly measured rung contradicts. It is not a fit of
this frontier: it does not pass through the frontier points, and with the
saturating shape here (three Pareto points at 6M on consequence accuracy,
capability 0.577 -> 0.994 -> 1.000) a two-parameter straight line cannot.
The stated justification is correct.

**(c) The replacement is defensible but carries one tunable, and the whole
argument is unnecessary.** Piecewise-logit is a monotone interpolant
through the measured points, so it is always inside the bracket; its only
free parameter is the logit clip (`1e-4`), which is load-bearing exactly
where a frontier rung sits at capability 1.0000 — which is the regime this
experiment is in. That is a small knob pointing at the answer, and pw vs
logit already disagree by 48% on the headline (1.45 vs 0.98).

**FIX APPLIED.** The verdict no longer runs on any fit. `analyze.py` now
computes the FIT-FREE BRACKET on M and the pre-registered rule is applied
to it: KILLED needs some rung's bracket lower bound >= 2.0, SURVIVES needs
every rung's bracket upper bound <= 1.6. Thresholds, statistic and the
max-over-rungs headline are unchanged. This can only make a verdict harder
to call, never easier, and it removes the fit choice entirely, so it is
the opposite of the failure mode being audited. Out-of-support handling
falls out for free and correctly (above support gives a valid lower bound
and no upper; below support the mirror image), which also retires
amendment 2. All three point fits are still computed and written to
results.csv, the original global log-linear among them, now with a printed
`LOGLIN-OUT-OF-BRACKET` flag wherever it violates the measured rungs.

**On the pilot numbers the fit-free rule reads INCONCLUSIVE**, by a hair:
KILL-side max lower bound 0.57, SURVIVE-side max upper bound 1.68 against
the 1.6 line. Under the KV-floor accounting (below) the same bound is 1.55
and it reads SURVIVES. The statement every fit and every accounting
agrees on, and the only one the pilot supports: **T2 is not killed at the
primary level, at any rung** — the best case for a kill is 0.57 against a
line at 2.0.

**The three smaller amendments.** #2 (in-support only) was correct in
motivation but wrong in symmetry: it also dropped ABOVE-support points,
whose M is a lower bound and therefore safe to use for a kill, so as
written it could suppress a kill. Subsumed and fixed by the bracket. #3
(fits must agree on verdict CATEGORY) is strictly tighter than what it
replaced — legitimate, and now moot. #4 (two controls per cell) is strictly
more scrutiny — legitimate, and it is what caught the anomaly. #5
(chunk at 64) is exact: nothing in `measure_chunk` crosses sequences. One
real defect: the random-mask control was drawn unseeded, so it differed
per chunk and per re-run. Seeded.

### 2. The execution finding — CONFIRMED INDEPENDENTLY, and upstream corrected

Re-measured on lab-2 from `l_s0_t5996544.pt` and `s_s0_t5996544.pt` with an
independent reimplementation of the exit criterion, the temporal probe and
the masked forward. At l@6M, 256 eval sequences: depth-only argmax
agreement **0.8993**, joint **0.9815**, skip fractions 0.8076 / 0.8798.
Identical to the builder's 0.8993 / 0.9815 / 0.808 / 0.880. At s@6M, 64
sequences: depth 0.8745, joint 0.9470, matching `pilot/controls.txt`
exactly. The finding is right.

Consequence: e009's headline is a bound, not a parity result. Corrected in
`dossiers/009-temporal-redundancy.md`, `library/cards/arche-e009-redundancy-oracle.md`
and the B3 repricing paragraph in `BELIEFS.md`. The correction is bounded:
80-88% of (token, block) compute passes the per-pair test, and executing
the whole union retains 98.2% argmax agreement — a real measurement worth
having, just not "at parity". B3 is left at 65%; the direction of the
e009 evidence is unchanged and only its precision is.

### 3. The anomaly — EXPLAINED, and the brief's hypothesis is REFUTED

Joint executes better than depth-only (0.9815 vs 0.8993 at l). Localised:

- The gain is entirely on MID-EXIT positions (exit block in (L/2, L)):
  agreement 0.3485 -> 0.8060 at s, 0.3365 -> 0.9667 at l. Early-exit
  positions (86% of tokens) are unaffected, never-exit positions get
  slightly worse. The temporal mask covers 0.67-0.80 of mid-exit positions
  and ~0.00 of the others, because depth wins where both hold, so temporal
  reuse only has room where depth-skipping is late.
- It is OWN-state, not context. Reuse switched off for mid-exit positions
  but left on everywhere else: mid 0.3489 (no gain at all). Reuse left on
  ONLY for mid-exit positions: mid 0.8086 (the entire gain).
- **The proposed hypothesis is refuted on both halves.** Off-distribution
  norm: rescaling the frozen depth state to the live state's RMS changes
  agreement by 0.0000. In-distribution substitute: the copied value is
  MORE drifted in L2 than the value it replaces at every block (0.916 vs
  0.904 at block 7), and substituting the CLEAN t-1 output instead of the
  executed one changes agreement by -0.0024 at s. Reuse is not a
  regularizer and it is not importing clean state.
- What it is: a valid-substitute certificate, not a low-sensitivity slot.
  Injecting other values at exactly the certified (t,b) slots, s@6M,
  mid-exit agreement: t-1 **0.806**, t-2 0.452, do-nothing 0.349,
  passthrough 0.176, matched-norm noise 0.173, zero 0.188. Everything
  except the genuine neighbour is worse than doing nothing.

Mechanism: the depth certificate is a claim about the CLEAN state at the
exit block, and execution does not deliver that state — mid-exit positions
run 5-11 blocks through a context already polluted by early-exit freezes,
so they arrive at their exit block badly drifted and read out the wrong
token. The temporal certificate is a claim about a SUBSTITUTION, and
execution performs exactly that substitution. Applying it discards the
position's own drifted computation and replaces it with a value the clean
model verified would produce the right final argmax. The joint arm is
therefore repairing damage the depth arm caused.

Bound on the reading: that repair requires the clean model both to select
the slots and to supply the values, so no realizable mechanism has it. The
joint arm's 98.2% is inflated relative to any implementable scheme and
must NOT be read as "the union of skips is nearly jointly valid". The
inflation is generous to the oracle and therefore conservative for a
T2-survives conclusion, so it does not threaten the direction of the
result. Written into CAVEATS.md.

### 4. The regime claim — NOT SUPPORTED BY THE PILOT

"The adaptivity premium exists in the undertrained regime and vanishes at
convergence" is not callable from two training levels at one seed, and
three separate things block it:

1. **Two points are not a trend.** The design specifies five checkpoints;
   the pilot ran two.
2. **At 229k the claim's own half fails the rule.** The axes disagree
   (acc 2.73 -> KILLED, cons 1.76 -> not), which is INCONCLUSIVE, and
   fit-free the brackets straddle 2.0 (acc l joint [1.30, 3.50], cons m
   joint [0.84, 2.29]). The 229k premium is fit-dependent.
3. **The confound is fatal as stated.** On this corpus "convergence" means
   the dense ladder has hit the corpus's irreducible-entropy ceiling, so
   training tokens and distance-from-ceiling are the same axis. The pilot
   cannot separate "adaptivity is an undertraining artifact" from "the
   frontier degenerates once every rung saturates" — and DESIGN.md already
   names the second as the artifact reading.

For the claim to be made at all the grid must show: all five levels and
both seeds, with M falling monotonically across at least four levels
beyond seed spread; the fall present on BOTH primary axes; the fit-free
bracket, not just the point fit, moving across the levels; and the
dynamic-range gate passing at the levels being compared. Even then the
strongest honest wording is "the multiplier falls as the dense ladder
converges onto this corpus's ceiling". Promoting that to "adaptive compute
is an artifact of undertraining" needs a corpus whose capability axis does
not saturate, and that is a different experiment, not a reading of this
one.

### 5. Standard checks

- **FLOPs counter: correct as specified, but the specification is generous
  to the oracle in one place.** The hand-check passes (rung s = 3,124,224)
  and the counter's h matches the model's h at every rung. However a
  skipped (token, block) is charged ZERO while later positions still
  attend to that position at that block, so its keys and values must be
  produced: 2*d*(2d) = 4 d^2 per skipped pass. DESIGN flags this for
  temporal reuse and explicitly exempts depth-skip, but CALM's state copy
  pays it too — the copy exists precisely so later layers can compute K/V
  from it. At l@6M the honest figure is 7.78M effective FLOPs/token
  against the 5.01M charged. The pre-registered zero-charge accounting is
  RETAINED as the verdict statistic (changing the verdict statistic after
  seeing data is the error this review exists to prevent); the KV floor is
  reported alongside as `flops_eff_kvfloor`. It moves M down, away from a
  kill, so it can never rescue a verdict. **Pre-registered now, before
  launch: if the two accountings ever disagree on the verdict category,
  the result is INCONCLUSIVE.**
- **MDE arithmetic: the algebra is right, the gate is applied in the wrong
  place.** The dynamic-range gate tests the MAXIMUM adjacent capability
  gap, but the multiplier's resolution depends on the gap in the interval
  that actually contains C_oracle. At 6M on accuracy the max gap is 0.0455
  (xxs->xs) and passes at 11x the CI, while the s->m->l intervals have
  gaps of 0.0009-0.0001, four times SMALLER than the CI half-width. That
  is the source of the 15x-wide CIs on the KL arms (s depth_kl acc
  [0.60, 8.97]), which the pilot writeup attributes to bimodality without
  naming the cause. It does not bite the verdict here because every
  argmax-arm oracle point at 6M lands in the one wide interval. The
  fit-free bracket makes it self-correcting: an unresolvable interval
  produces a wide bracket and an INCONCLUSIVE, automatically.
- **Capability axes: correct choice, and the stated reason is now void.**
  "Accuracy is preserved by construction wherever the criterion holds"
  no longer applies, because the criterion does not hold under execution.
  Capability being MEASURED rather than assumed is what saves this, and it
  is the single best decision in the design. Both axes are argmax-type, as
  required. Consequence accuracy is much the more sensitive of the two
  (depth-only at l@6M: acc 0.838 -> 0.802, cons 1.000 -> 0.584), which is
  the right direction — the harsher axis is the one with dynamic range.
- **The frontier does not have enough points where it matters — this is
  the launch blocker.** At the primary level, on consequence accuracy, the
  Pareto envelope is THREE points (0.484M@0.577, 1.426M@0.994,
  3.124M@1.000): m and l are dropped as ties at 1.000, and xl will be
  dropped for the same reason. Every argmax oracle point at 6M, on both
  axes, falls in the SINGLE interval [xxs, xs], which spans 2.95x in
  FLOPs. The verdict therefore rests entirely on one 2.95x-wide gap, the
  bracket there is [0.57, 1.68] against a survive line of 1.6, and **no
  number of seeds or training levels narrows it.** Only rungs inside that
  interval do.
- **Cost sizing.** Measurement cost scales as L^2 d^2 + L d V; on the
  builder's l = 25 min/checkpoint anchor the full 6 x 5 x 2 grid is
  ~16 h single-stream, ~4-6 h wall at realistic parallelism, plus ~6 h
  single-stream of training — call it $4-8 on a D16als at list price,
  against $79 spent of $20,000. **Cost is not a reason to trim anything.**
  xl is ~57% of measurement time and contributes ZERO frontier points at
  the primary level, so trimming it is right on scientific grounds and the
  saving pays for the rungs that are actually needed.

### RULING: NEEDS-WORK

Nothing found is fatal and nothing needs redesigning. The instrument is
sound, the execution finding is real and correctly reported, the anomaly
is a post-hoc-oracle artifact that biases conservatively, and the amended
analysis — once the fit choice is removed — supports "T2 is not killed"
robustly. Three things before launch:

1. **Add two rungs inside (xxs, xs)** so the primary-level frontier has
   more than one usable interval. `d=48, L=5` (0.880M FLOPs/token) and
   `d=56, L=6` (1.184M) both sit under the 1.363M a frontier point needs
   to be at for the headline bracket to clear 1.6. They train in ~4
   minutes each and measure in ~2. Without this the grid cannot separate
   SURVIVES from INCONCLUSIVE at the primary level no matter how long it
   runs, and the pilot's 1.68 is the proof.
2. **Run xl only at the training levels where the dense ladder still
   separates** (229k / 590k / 1.49M). At 3.0M and 6.0M it is a tie at the
   corpus ceiling and adds no frontier point, at ~62 min/checkpoint.
3. **Re-run the pilot cells through the amended `analyze.py`** and confirm
   the fit-free bracket, the KV-floor column and the seeded control
   reproduce. No re-measurement is needed; the records are unchanged.

Then launch. The verdict rule is now fit-free and pre-registered against
both FLOPs accountings, which is what makes the grid's answer callable in
either direction.

---

## FINAL RESULT (2026-08-16, full grid, 76/76 cells)

8 rungs x 5 training levels x 2 seeds, xl restricted to the three levels
where the dense ladder still separates. All records in `runs/`, pulled
from lab-2 before analysis. Verdict by the pre-registered rule applied to
the fit-free bracket, per seed, both FLOPs accountings.

### Instrument checks — all pass

- **e009 cross-check**: all 20 shared checkpoints reproduce e009's
  fractions within eval noise (largest gap 0.0037 on depth_arg).
- **Control ordering**: 0 violations in 76 cells (random <
  wrong-neighbour < depth-only everywhere).
- **Dynamic-range gate**: 20/20 (5 levels x 2 seeds x 2 axes) PASS. The
  final level passes only because the ladder was extended downward —
  xxs still does not saturate (acc 0.792/0.796, consequence 0.577/0.608)
  while xs1/xs/s/m/l all sit at 0.837/~1.000.
- **Seed handling defect found and fixed during analysis.** `analyze.py`
  keyed cells by training level alone, so with two seeds both seeds'
  points entered ONE Pareto envelope and the multiplier dict collided on
  rung name (seed 1 silently overwrote seed 0). Every frontier, gate and
  multiplier is now computed WITHIN a seed, and seeds combine
  conservatively: a kill must hold on the worst seed, a survive must hold
  on the worst seed. No number below was computed with the pooled code.

### The verdict

**COMBINED: INCONCLUSIVE (the two primary capability axes disagree).**

Primary level by the pre-registered rule = 5,996,544 tokens (the largest
level passing the gate; here every level passes).

| axis | worst-seed KILL-side lower bound | worst-seed SURVIVE-side upper bound | KV-floor | call |
|------|------|------|------|------|
| next-token accuracy | 0.78 | 1.43 | 1.36 | **T2 SURVIVES** |
| consequence accuracy | 1.03 | 1.61 | 1.50 | **INCONCLUSIVE** |

The consequence axis misses the survive line by **0.01** on the
pre-registered zero-charge accounting (1.61 vs 1.60) while the KV-floor
accounting clears it (1.50). That triggers the disagreement clause
pre-registered before launch, and the clause is honoured: INCONCLUSIVE.
The point fits at that rung are pw 1.17/1.18 and logit 1.15/1.17, so the
multiplier is very probably near 1.2 — but 1.61 is the bound the rule
uses and the rule is not being relaxed after seeing it.

### What IS callable, at every level and on both axes

**T2 is never killed.** The fit-free, worst-seed, bootstrapped KILL-side
lower bound across all five training levels:

| level | acc | cons |
|-------|-----|------|
| 229,376 | 1.23 | 1.68 |
| 589,824 | 1.42 | 1.44 |
| 1,490,944 | 1.51 | 1.59 |
| 2,998,272 | 1.23 | 1.21 |
| 5,996,544 | 0.78 | 1.03 |

Maximum over the whole grid is 1.68, against a kill line of 2.0. **No
training level, no seed, no axis, no fit and neither FLOPs accounting
puts the post-hoc oracle ceiling at 2x the dense frontier.** That is the
weakest-form statement the data supports and it is also the one the
experiment was built to make.

### The implementation-faithful arm is worse than model selection

Depth-only (CALM-style state copy, the arm a real system could build) at
the final level, multiplier on consequence accuracy by rung, both seeds:
xxs 1.19/1.19, xs0 0.83/0.75, xs1 0.68/0.64, xs 0.62/0.59, s 0.32/0.32,
m 0.16/0.16, l 0.075/0.076. **Above the smallest rung it is below 1 at
every rung on both seeds** — a smaller dense model is strictly cheaper at
matched capability. Its executed argmax agreement is 0.873-0.912 and its
consequence accuracy falls from ~1.000 to 0.54-0.67.

### Regime claim: 2 of 4 conditions met -> NOT SUPPORTED

The four conditions set before launch:

1. **M falling monotonically across at least four levels beyond seed
   spread — FAILS.** Point estimates are non-monotone on both axes (the
   consequence-axis point multiplier peaks at 1.49M tokens, 4.15/2.97 by
   seed, not at the least-trained level). On the bootstrapped fit-free
   bracket the first four levels are statistically flat (KILL-side 1.23,
   1.42, 1.51, 1.23 on accuracy) with a single drop at the last level.
   One transition is not a monotone fall across four.
2. **Fall present on both axes — MET**, for that single final transition
   (acc 1.23 -> 0.78, cons 1.21 -> 1.03).
3. **Fit-free bracket, not just the point fit, moving across levels —
   FAILS.** The brackets at levels 1-4 overlap heavily; only the last
   transition separates.
4. **Gate passing at the compared levels — MET**, 20/20.

**The claim "the adaptivity premium exists in the undertrained regime and
vanishes at convergence" is NOT supported by this grid.** What the data
show is flat-then-drop, with the single drop landing exactly at the level
where the dense frontier saturates on this corpus — so the confound named
before launch is untouched: one transition cannot separate "model
convergence" from "corpus exhaustion". Any regime claim needs a corpus
whose capability axis does not saturate.

### Decision 0012 compliance

- **Clause 5 (weakest-form).** The headline is stated in its weakest
  supported form: "never reaches 2x", not "survives". The SURVIVES call
  on the accuracy axis is reported as the axis-level result it is, and
  the combined verdict stays INCONCLUSIVE.
- **Clause 3 (corpus independence).** Single corpus (e005 synthetic).
  Everything here is a claim about that corpus; the transferable content
  is the structural argument (the unembedding floor) and the direction.
- **Clause 4 (scale floor).** Six of eight rungs sit below the ~3M
  non-embedding floor e-retro measured, and the verdict-bearing rungs
  (xs0/xs1/xs, 0.13-0.30M nonemb) are far below it. This is a
  within-family comparison, not a cross-architecture ranking, so the
  clause-4 prohibition does not bite — but the frontier's SHAPE at the
  small end is exactly what the verdict rests on, and e-retro says this
  harness is untrustworthy about small-rung orderings. Named as the
  primary threat to the result.
- **Clause 1 (identifiability dry-run).** Not run before launch; the
  fit-free bracket plus the added rungs were the ad-hoc equivalent. The
  bracket at the primary level (1.43-1.61 against a 1.6 line) shows the
  design is still only marginally able to separate SURVIVES from
  INCONCLUSIVE — decision 0012 clause 1 would have predicted that, and
  this run is a datum in its favour.
