# Small experiments on how language models learn

Small, pre-registered experiments on sample efficiency and continual learning in small language models. Most results are negative.

## Summary

Two questions run through this work:

1. Why does a language model need so many examples to learn something?
2. Why does learning something new erase what it knew before?

Fifteen experiments on models from under 1M to 500M parameters gave three results:

- Ideas that spend a fixed training budget more cleverly are worth at most about 2x, even when given perfect information.
- When a model was taught skills one after another, the old skills stayed in the weights but could no longer be reached. Rehearsal fixed this, including rehearsal that stores no old data.
- A larger memory reduced forgetting only once its addressing was fixed so that the extra slots were actually used.

Nothing here is a claim about large models.

## Findings

| # | Question | Result | Experiments |
|---|---|---|---|
| 1 | Can the test setup reproduce a known ranking? | No below about 1M parameters. An LSTM beat a transformer there (1.51 against 1.64 nats per byte, both tuned). The expected ranking returned by about 3M | [`e-retro`](experiments/e-retro) |
| 2 | Does spending the same budget more cleverly help? | Best cases between 1.0x and about 2x across six ideas | see table below |
| 3 | Is learning from the prompt cheaper than learning through weight updates? | Largest ratio 2.01 against a bar of 5. Below 1 for skills the model had never seen | [`e013-exchange-rate`](experiments/e013-exchange-rate) |
| 4 | Why are old skills lost when new ones are written? | The newest skill answers every cue. Old skills survive in the weights. Rehearsal from the model's own outputs kept 0.70, stored rehearsal 0.78, none 0.00 | [`e015-no-replay`](experiments/e015-no-replay), [`e004g-consolidation`](experiments/e004g-consolidation), [`e017-arbitration`](experiments/e017-arbitration), [`e021-terminator`](experiments/e021-terminator), [`e021-arbitration-grid`](experiments/e021-arbitration-grid) |
| 5 | Does a bigger sparse memory forget less? | Flat at first, because learned addressing used 3.6% of slots. With fixed random addressing, forgetting fell from -0.20 at 4k slots to -0.06 at 1M | [`memory-size`](experiments/memory-size) |
| 6 | How many times must a fact be seen to stick? | About 30 or more per pass. Spreading exposures over distinct placements beat repeating the same ones (0.72 against 0.58 recall). Combining two facts stayed at chance | [`fact-exposure`](experiments/fact-exposure) |

### Finding 2 in detail

Each experiment kept the learning process fixed and tried to spend loss, data or compute better. Where possible the best case was measured with an oracle that knows the right answer, so a weak implementation could not be blamed.

| Experiment | Idea | Result |
|---|---|---|
| [`e005-oracle`](experiments/e005-oracle) | Weight the loss toward tokens that matter, using a perfect labeller | 0.96x to 1.06x. A later audit found the setup itself capped the possible gain near 1.1x to 1.2x, so this is a weak bound |
| [`e006-selection`](experiments/e006-selection) | Let the learner pick its own training data | The first run was void by construction (best and worst pickers tied). The redesign found selection loses to random once its own cost is counted |
| [`e009-redundancy`](experiments/e009-redundancy) | How much computation could be skipped in hindsight? | 80% to 88% of block computations, as an upper bound |
| [`e009b-t2-overlay`](experiments/e009b-t2-overlay) | Does skipping it beat using a smaller model? | No. The best case peaked at 1.68x against a 2x bar, and the buildable version was worse than a smaller model |
| [`e010-growth`](experiments/e010-growth) | Grow a small trained model into a bigger one | Ahead by 0.003 to 0.005 nats at equal compute |
| [`e011-horizons`](experiments/e011-horizons) | Predict several tokens ahead to learn long-range structure sooner | Slower. Plain next-token training got there first in every seed |

My reading is that these ideas all rearrange a fixed amount of information, and that rearranging it is not where the waste is. [`docs/allocation-ceiling.md`](docs/allocation-ceiling.md) gives the argument.

### Finding 4 in detail

Ten skills were written into a 500M-parameter model in sequence. With no rehearsal, retention of old skills was 0.000 to 0.002. Forcing each write into separate or sparse weights did not change that. Rehearsing old examples in 25% of steps kept 0.79 to 0.90.

Probing showed the old skills were still present and the newest one fired on every cue. That suggested rehearsing on the model's own earlier outputs for old cues, with no stored data. Over 3 seeds this kept 0.70 of old-skill accuracy. The mechanism is published under other names, and the write-ups say where.

### Finding 5 in detail

Fine-tuning a 360M model on 1,000 new facts erased about 48% of the facts it already knew, while three standard benchmarks did not move. LoRA and replay of unrelated text gave no protection. A random-addressed memory added to the same model gave a weak improvement that grew slowly with size.

## Method

Each experiment has:

- a written case and a prediction, made before the run (`docs/proposals/`, or the `DESIGN.md` and `LOG.md` in the experiment folder)
- a rule, fixed in advance, for what counts as failure
- the raw run records next to the code
- a list of caveats

The rules the experiments ran under are in [`docs/protocol/`](docs/protocol). [`docs/retrospective.md`](docs/retrospective.md) is an early audit of the experiments against their own goal.

I chose the questions, set the rules and made the decisions. AI coding agents wrote most of the code and the first-pass analysis. For the thirteen `e…` experiments, a second agent with no access to the first one's reasoning re-derived each result from the raw records before it counted. That pass corrected ten first-pass claims, each toward a less exciting result. The corrections are left in the files.

## Limitations

- Small models, mostly synthetic data, two or three seeds per cell.
- Most of the `e…` experiments share one synthetic corpus.
- The test setup is unreliable below about 1M parameters (finding 1), and some experiments sit near that floor.
- Finding 4 uses one base model and one family of string-transformation skills. Its headline numbers are scored with the answer prefix supplied. `e021` reports free generation beside them.
- Finding 5's positive curve is two seeds on Permuted-MNIST with low final accuracy (0.58). Slot use under random addressing was not measured. The language-model sweep is one seed. Fixed random addressing is close to older work on sparse distributed memory.
- Finding 6 is one seed per run, and the planned sweep over model size was not run.

Experiments whose results were withdrawn, unfinished, or not independently checked are left out. [`docs/portfolio.md`](docs/portfolio.md) mentions them by number.

## Repository layout

```
experiments/
  e-retro … e021-*     one folder each: code, DESIGN.md with the prediction, runs/, artifacts/
  memory-size/         code, LOG.md, results/, logs/, kaggle/
  fact-exposure/       code, LOG.md, MAP.md (background reading), results/, logs/, kaggle/
docs/
  proposals/           the written case for each experiment, before it ran
  results/             one-page result summaries
  protocol/            the rules the experiments ran under
  retrospective.md     early audit of the experiments
  portfolio.md         running ledger of every measured effect
  allocation-ceiling.md  the argument behind finding 2
```

Some files refer to experiments by an older path or mention experiments that are not included here.

## Running the code

The code is plain Python with PyTorch. Dependencies are not pinned. Each experiment folder has its own entry script and the shell scripts that launched its runs. Several experiments import code from a sibling folder, so keep `experiments/` intact. Model checkpoints and files over 2 MB are not included.
