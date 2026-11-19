# e013 LR sweep on the gradient arm (N=32, 1 seed, 200 steps)

Decision-0010/e-retro/e008 requirement: the gradient arm's LR optimum must
be INTERIOR. An edge optimum means the arm is untuned and R is inflated in
ICL's favour, which is the direction of the exciting answer. Every skill
that carries an R has an interior optimum.

| skill | 1e-05 | 3e-05 | 0.0001 | 0.0003 | 0.001 | 0.003 | best | interior? |
|---|---|---|---|---|---|---|---|---|
| B_first | 0.945 | 0.975 | 0.955 | **0.990** | 0.930 | 0.580 | 0.0003 | YES |
| B_upper | 0.875 | 0.900 | 0.890 | **0.905** | 0.765 | 0.075 | 0.0003 | YES |
| N_map | 0.060 | 0.095 | 0.100 | 0.210 | **0.395** | 0.110 | 0.001 | YES |
| N_shift2 | 0.155 | 0.155 | 0.115 | **0.495** | 0.475 | 0.115 | 0.0003 | YES |

Values are eval accuracy at the best-dev checkpoint. Bracketing:
B_first 1e-4 0.955 -> **3e-4 0.990** -> 1e-3 0.930; B_upper 1e-4 0.890 ->
**3e-4 0.905** -> 1e-3 0.765; N_map 3e-4 0.210 -> **1e-3 0.395** -> 3e-3
0.110; N_shift2 1e-4 0.115 -> **3e-4 0.495** -> 1e-3 0.475.

B_upper and N_shift2 were swept by the VERIFIER (2026-08-17); B_first and
N_map by the builder. No R is reported for N_perm: it has no gradient arm
and its in-context curve is tokenization-confounded (line length 9.67 +/-
0.53 tokens against 8.06 for the single-character skills).

## Rule-vs-table split at the swept point (verifier addition)

`acc_unseen_letter` is accuracy on eval words whose FIRST LETTER never
appeared in the gradient arm's training set. It separates a learner that
has the RULE from one that has a TABLE.

| skill | N | seen-letter | unseen-letter | n_unseen |
|---|---|---|---|---|
| B_first (rule the base HAS) | 8 | 0.958 | **0.857** | 105 |
| B_first (rule the base HAS) | 32 | 0.988 | **1.000** | 32 |
| B_upper (not first-letter based) | 32 | 0.915 | **0.833** | 24 |
| N_shift2 (novel rule) | 32 | 0.553 | **0.000** | 21 |

The B_first and B_upper rows are the rarity control: words with rare first
letters are answered at 0.83-1.00, so a 0.000 on N_shift2 is not a
frequency artifact. See the VERIFIER RULING in DESIGN.md section 1.1.
