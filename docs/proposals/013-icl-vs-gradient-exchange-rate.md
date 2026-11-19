# Dossier 013 — The exchange rate: how many gradient examples buy one in-context example?

Status: PRE-REGISTERED 2026-08-16. Targets decision 0011 bar (b),
sample efficiency, and prices bet B2 directly. Portfolio axis:
**learning**. Independence: this measures the SIZE of the prize
consolidation (004) is trying to bank, so it is not an additional
factor — it is the ceiling on 004's factor, and therefore must be
measured before 004's result can be interpreted.

## The question

A model shown 5 examples in context often performs a task it would need
hundreds of gradient examples to learn by fine-tuning. That gap is the
most striking sample-efficiency phenomenon in current models and, as far
as the lab's reading has found, **nobody has measured it as a ratio on
a matched task with a matched capability target.**

Define the EXCHANGE RATE R = (gradient examples to reach capability C) /
(in-context examples to reach capability C), same base model, same task,
same C.

R is the ceiling on what any context-to-weights channel can be worth.
- If R ~ 2-5, consolidation is a convenience and B2 is badly overpriced
  at 79%.
- If R ~ 100+, then fast learning is enormously more sample-efficient
  than slow learning, the prize is real and portfolio-sized, and the
  only question left is whether the channel can bank it.

Either way the number is load-bearing and cheap. It should have been
measured before 004 was ever queued.

## Rival mechanisms

**X1 — Large exchange rate, and it is real learning.** R >= 20 on
non-degenerate skills, stable across skills and across capability
targets; the in-context route reaches capability the gradient route
needs an order of magnitude more data for.
- KILLED-BY: R < 5 on every non-degenerate skill.

**X2 — Small or illusory exchange rate.** The apparent ICL advantage
comes from the base having seen the skill in pretraining, so context
merely SELECTS an existing capability while gradients must build one.
Then R measures pretraining coverage, not a learning-speed difference.
- PREDICTS: R collapses on skills constructed to be absent from
  pretraining (novel symbol systems, arbitrary invented mappings), and
  survives only on skills with natural analogues.
- This is the deflationary reading and it is the one most likely true.
  It must be tested, not assumed away: the novel-skill arm is
  mandatory, not optional.

**X3 — Regime-dependent.** R is large at low capability targets and
collapses near ceiling (context gets you started fast, gradients get you
all the way).
- PREDICTS: R falls monotonically as C rises; the two curves cross.

## Design commitments

Base: Qwen2.5-0.5B, the checkpoint already validated by e004g's gate
(rotary, 18T pretraining tokens, CPU-runnable, 262s for a full gate
sweep on 16 cores). Second base for robustness if throughput allows.

Skills: e004g's suite, restricted to the NON-DEGENERATE skills (copy
oracle <= 0.20 — the copyable ones cannot carry this reading either),
PLUS a mandatory arm of newly constructed skills designed to have no
plausible pretraining analogue. X2 lives or dies on that arm.

Two learning curves per skill, same capability metric, same eval:
- IN-CONTEXT: capability vs number of in-context examples (1, 2, 4, 8,
  16, ... to context limit). No weight updates.
- GRADIENT: capability vs number of training examples (LoRA, tuned LR,
  early stopping on a held-out split). No context examples.
R is read at several capability targets C, not one, so X3 is visible.

Controls, all mandatory:
- mismatched-shot control on the in-context arm (e004g's, which
  scores 0.000) — the model must be reading the examples;
- copy-oracle on every skill;
- unprompted baseline, so C is measured above the right floor;
- LR sweep on the gradient arm with the optimum required to be
  INTERIOR (the e-retro and e008 lesson: an edge optimum means the
  gradient arm is untuned and R is inflated in ICL's favour).

## Decision-0012 compliance

- **Identifiability dry-run**: before launch, push each rival's
  predicted curves through analyze.py and confirm each yields its
  intended verdict category.
- **Known-positive**: the in-context arm on a natural skill IS the
  positive control (e004g measured gaps well above floor on 6/8
  skills). A null there means the instrument is broken, not that ICL
  is absent.
- **Corpus independence**: the novel-skill arm is structurally
  different from the natural-skill arm by construction; that is the
  second corpus.
- **Weakest-form reporting**: R is reported per skill with per-seed
  values, never as a single headline number, and never as a ratio
  whose magnitude equals one grid step.

## Why it matters for the goal

Sample efficiency is one of the two bottlenecks Anay named. R is the
cleanest number in that whole area: it says how much more efficient the
fast route is than the slow one, on the same model, on the same task.
It sets the ceiling on 004's value, it prices B2, and if X2 wins it
deflates the entire consolidation programme cheaply — which is worth
knowing in days rather than after a GPU-scale build.

Cost: CPU. Forward-only for the in-context arm; LoRA-sized writes for
the gradient arm. Same box class as e004g.
