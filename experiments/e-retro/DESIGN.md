# e-retro — Retrodiction pilot (FOUNDATION S3): design

Status: DESIGN 2026-08-16, written before code. Instrument validation,
not a discovery experiment: given only tiny models, does our
methodology recover the KNOWN verdict — transformer beats LSTM on
language modeling, with the gap stable or growing in scale?

## Data (decision)
enwik8 (first 20MB Wikipedia XML, mattmahoney.net/dc/enwik8.zip),
byte-level. Reasons: one dependency-free HTTP fetch; byte vocab (256)
keeps embeddings tiny so matched NON-embedding params is honest; it is
the canonical char-LM benchmark where the historical verdict is
documented (Al-Rfou 2018 transformer ~1.06 bpc vs LSTM-line ~1.23+),
so "history's answer" is on the record for this exact data. TinyStories
rejected: needs HF datasets and its simplicity may flatten the gap.
Split: bytes 0-20M train, 20-22M val.

## Models (decision)
Matched non-embedding params (blocks/recurrent stack only; embedding +
head excluded on BOTH sides), 3 rungs ~1M / ~3M / ~9M:
- transformer: e005 Block (copied, pinned), d/L = 128/5, 192/7, 256/11.
- LSTM: nn.LSTM 2 layers, hidden 256, 444, 752 (16h^2 params).
Both: embedding + LayerNorm + linear head, same everything else.
DEVIATION noted: both archs train on independent 256-byte windows (no
LSTM state carry across windows). This handicaps the LSTM's long-memory
story equally at train and eval; acceptable for a SIGN-and-TREND
instrument test, named as a caveat on any absolute bpc.

## Protocol (decision)
AdamW wd 0.1, LR in {3e-4, 1e-3, 3e-3} (coarse fairness sweep per
arch x rung, e001 protocol), 2 seeds at best LR implied by running all
(cheap enough to run all 3 LRs x 2 seeds). Token budgets by rung:
10M / 20M / 30M bytes. batch 32 x seq 256. Held-out loss (nats/byte,
reported also as bits/byte) on 512 fixed val windows, every ~1/10 of
budget. Runs: 2 x 3 x 3 x 2 = 36.

## Verdict rule (pre-registered)
Per (arch, rung): best final val loss over LRs, mean of 2 seeds.
- PASS: transformer wins all 3 rungs AND gap (nats) non-shrinking with
  rung (within seed noise) — history recovered.
- PARTIAL: transformer wins all rungs but trend flat/noisy.
- FAIL: LSTM wins any rung at tuned LR — instrument failure; gates all
  upstream small-scale claims with a named asterisk until diagnosed.
Prediction: PASS ~0.6, PARTIAL ~0.3, FAIL ~0.1.

## Budget
36 runs; heaviest (9M params x 30M tokens) ~4-5 h at 1 CPU thread;
whole grid ~6-9 h wall on a 16-core box, 12 workers. fp32, JSON
records with git SHA per run.

## VERDICT (2026-08-16, 56/68 records; rung 1 settled and decisive)

**FAIL by the pre-registered rule.** The rule reads: "FAIL: LSTM wins
any rung at tuned LR — instrument failure; gates all upstream
small-scale claims with a named asterisk until diagnosed." At rung 1
the LSTM wins, and both architectures are now properly tuned, so the
rule fires. The verdict is recorded as written rather than
renegotiated after seeing the data.

Best final val loss (nats/byte), mean of 2 seeds, at each arch's own
best LR:

| rung | LSTM | transformer | winner | gap |
|---|---|---|---|---|
| r1 (~1M) | **1.5118** (lr 3e-3) | 1.6415 (lr 6e-3) | LSTM | 0.130 |
| r2 (~3M) | 1.3988 (lr 3e-3) | **1.3593** (lr 3e-3) | transformer | 0.040 |
| r3 (~9M) | 1.3369 (lr 1e-3) | **1.2871** (lr 1e-3) | transformer | 0.050 |

**The tuning objection is closed.** The original sweep {3e-4, 1e-3,
3e-3} put the transformer's rung-1 optimum at the hot EDGE (1.6848 at
3e-3), so a hotter LR was the live rescue hypothesis and was the
pre-registered conditioned follow-up. Extending to {6e-3, 1e-2}
resolves it: the transformer improves to 1.6415 at 6e-3 then degrades
to 1.7410 at 1e-2, so its optimum is INTERIOR. The LSTM's rung-1
optimum is likewise interior (1.5118 at 3e-3, bracketed by 1.5860 at
1e-3 and 1.5170 at 6e-3). Hotter learning rates narrow the rung-1 gap
from 0.173 to 0.130 and do not close it. Both architectures are tuned;
the LSTM's small-scale win is real.

**The LSTM state-carry objection is also closed.** The design's named
deviation (independent 256-byte windows, no state carried across them)
was the other candidate confound, in the direction of handicapping the
LSTM. Carrying state adds nothing: r1 1.5131 vs 1.5118 plain, r2
1.4051 vs 1.3988 plain — both marginally WORSE. The deviation was not
load-bearing.

**Diagnosis, stated separately from the verdict.** The failure is not
noise: it is a CROSSOVER. The LSTM wins at ~1M, the transformer wins
at ~3M and ~9M, and its margin grows (0.040 -> 0.050). That is the
historically correct shape — the transformer's documented advantage on
enwik8 was established at far larger scale, and RNNs were competitive
below it. So the instrument recovers the right ORDERING TREND while
getting the ranking backwards at its smallest rung.

**Consequence, applied.** The asterisk the rule demands is now
attached and it is specific rather than vague: at ~1M non-embedding
parameters this harness ranks two architecture families in the wrong
order, so no architecture-ranking claim from the ~1M rung is
trustworthy on its own. Claims about MECHANISM, SIGN of an
intervention within one architecture, and TRENDS across rungs are not
directly implicated by this failure, but they inherit the general
caution. Directly affected: e009's ladder (0.89M/3.07M/9.44M) straddles
this crossover, so its smallest rung sits in the untrusted zone.

**Still outstanding (12 runs):** transformer r2/r3 and lstm r3 at
{6e-3, 1e-2}. Since the transformer preferred a HOTTER LR than the
LSTM at r1, those cells could improve the transformer at r2/r3 and
widen its wins there. They can only strengthen the crossover reading;
they cannot undo the rung-1 FAIL, which is bracketed on both sides.

## VERDICT CONFIRMED FINAL (2026-08-16, 64/64 records)

The 12 outstanding hot-LR cells landed. Every optimum, both
architectures, all three rungs, is now INTERIOR — the tuning objection
is closed everywhere, not just at rung 1:

| rung | LSTM best | transformer best | winner | gap |
|---|---|---|---|---|
| r1 | 1.5118 @ 3e-3 | 1.6415 @ 6e-3 | **LSTM** | 0.1297 |
| r2 | 1.3988 @ 3e-3 | 1.3593 @ 3e-3 | transformer | 0.0396 |
| r3 | 1.3369 @ 1e-3 | 1.2871 @ 1e-3 | transformer | 0.0498 |

The extension did what extensions do: it moved the transformer's rung-1
number (1.6848 -> 1.6415) without moving the verdict. FAIL stands, and
the crossover diagnosis is now the fully-tuned reading rather than a
provisional one. The transformer's margin still grows r2 -> r3
(0.0396 -> 0.0498), so the trend half of the gate is recovered while
the ranking half is inverted at r1.

No further follow-ups. Dossier-level status: e-retro CLOSED.
