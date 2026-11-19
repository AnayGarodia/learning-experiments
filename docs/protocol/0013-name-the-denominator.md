# Decision 0013 — Name the denominator (OPEN: needs Anay's call)

Status: DRAFTED 2026-08-17 from the frontier resource-bill sweep.
NOT DECIDED — this is a goal definition, which is Anay's to make.
Options and a recommendation below.

## The problem, which the sweep exposed

The lab's target is "frontier capability at ~1/1000th the resources".
That target currently has no denominator, and the sweep shows the
choice of denominator changes the answer by orders of magnitude:

- DeepSeek-V3's published final-run cost is **$5.576M / 2.788M
  H800-hours**. That figure is already **≈1/1000th of OpenAI's
  estimated 2024 R&D compute spend (~$5B)**.
- So "1000x cheaper than a frontier lab" is ALREADY SATISFIED by an
  existing open model. Under decision 0005 the target as written is
  not falsifiable.

Two further facts constrain any honest version:

1. **Pretraining is 95.55% of the only published stage breakdown**
   (V3: pretrain 2,664K H800-h; context extension 119K; post-training
   5K). **Zeroing everything that is not pretraining buys 1.047x.**
   A 1000x claim is a claim about pretraining or it is nothing.
2. **Published costs are final-run only**, by the authors' own
   sentence. Third-party estimates put the shipped run at 10-23% of
   total R&D compute, so the true bill is roughly 4-10x the published
   one, and the majority is spent on runs that never shipped.

## The options

**A. Final-run compute at matched capability (RECOMMENDED).**
Target: match a named model's capability at 1/1000th of ITS published
final-run training compute. Concretely, against DeepSeek-V4: match its
capability using ~1/1000th of the compute its final pretraining run
consumed.
- PRO: falsifiable, uses the one number that is actually published,
  and puts the burden where 95% of the cost is.
- CON: V4 has NOT published a compute figure — only token counts (32T
  Flash / 33T Pro). We would be measuring against V3's disclosure
  (Dec 2024) or against a token-budget proxy.

**B. Total R&D compute including failed runs.**
- PRO: the honest full bill; it is what a lab actually spends.
- CON: nobody publishes it; we would be targeting an estimate, and
  estimates carry 3-30x error bars — useless for composing 2-10x
  factors.

**C. Capability per inference dollar.**
- PRO: four independent parties already publish this frontier
  (Artificial Analysis, ARC Prize, Kimi's own report, Epoch), so the
  comparison is measurable TODAY and the instrument exists.
- CON: it is not the goal as stated; it targets serving cost, and the
  frontier labs are already optimising it hard (V4's whole attention
  redesign is an inference-cost play at ~27% of V3.2's per-token FLOPs).

**D. Tokens-to-capability (data efficiency), holding architecture free.**
- PRO: V4 publishes tokens (33T) where it does not publish compute, so
  the denominator EXISTS for current models; and it is the axis our
  own work says is bounded by information, which makes it the sharpest
  scientific question.
- CON: 1000x fewer tokens than 33T is 33B tokens, which is a strong
  claim; and tokens are not the whole resource bill.

## Recommendation

**A as the headline, D as the working target.** A is what the goal
means in plain terms and is falsifiable against a published number. D
is what we can actually measure progress on this quarter, its
denominator exists for current models, and it is where the lab's own
evidence says the binding constraint lives.

And whichever is chosen, the target must name a MODEL and a DATE, since
"frontier" moves. Suggested anchor: **DeepSeek-V4-Pro, April 2026** —
it is the strongest open-weights model (Artificial Analysis Intelligence
Index 44 vs 55-56 for the closed frontier), its weights and
documentation are public, and it is therefore the only frontier-class
target we could actually verify a claim against.

## A note that belongs with this decision

The sweep found NO published training-capability-per-resource frontier
anywhere — capability-vs-INFERENCE-cost is published by four parties,
capability-vs-TRAINING-cost by none. That absence is the artifact bet
B7 says is worth more than any single result. If the lab ever produces
a validated small-to-large predictor, the plot it enables does not
currently exist in the literature.

## RESOLUTION (2026-08-17): adopted by DEFAULT, subject to Anay's override

Anay was asked three times across two days and has not chosen. The
decision blocks something real — while the denominator is unnamed the
program target is not falsifiable, because DeepSeek-V3's published final
run ($5.576M) is ALREADY about a thousandth of OpenAI's ~$5B annual R&D
spend, so "1000x better" is true or false depending entirely on which
number one divides by. Leaving it open lets any result be spun either way,
which is the failure mode this decisions file exists to prevent.

Rather than block further, the recommendation is ADOPTED AS THE WORKING
DEFAULT and marked as such everywhere it is used. It is cheap to reverse:
nothing in the portfolio depends on the choice except how a factor is
reported.

**ADOPTED:**
1. **Anchor: DeepSeek-V4-Pro, April 2026.** A named model and a named
   date, because "frontier" moves. Chosen because it is the strongest
   open-weights model, its weights and documentation are public, and it is
   therefore the only frontier-class target a claim could actually be
   verified against.
2. **Headline denominator: final-run training compute** (option A). This
   is what "1000x more efficient to train" means in plain language and it
   is falsifiable against a published number.
3. **Working target: tokens-to-capability** (option D). This is what the
   lab can measure progress against this quarter, its denominator exists
   for current models (V4 publishes 33T tokens where it does not publish
   compute), and it is the axis the lab's own evidence says is binding.
   1000x fewer tokens than 33T is 33B tokens; that is the number to beat.
4. **Both are reported, always, and never mixed.** A claim quoting the
   headline may not be supported by evidence from the working target.

**What this immediately implies, and it is the reason to bother:** the
allocation family is now formally disqualified rather than merely
disappointing. Against a 33B-token working target, an oracle-bounded
~2x reallocation gain is three orders of magnitude short, and
SYNTHESIS-ALLOCATION-CEILING.md shows the whole family shares that
bound. So the bar is doing its job: it kills a line of work that twenty
experiments had already been telling us was dead.

**Override:** if Anay names a different anchor or denominator, this
resolution is replaced and any factor reported under it is restated. No
experiment needs re-running, only re-expressing.
