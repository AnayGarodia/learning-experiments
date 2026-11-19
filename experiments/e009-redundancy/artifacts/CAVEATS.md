# e009 artifact caveats

- Tiny models (0.89M/3.07M/9.44M nonemb) on the e005 synthetic corpus,
  6M tokens. Transferable claims per dossier 009: trend direction,
  criterion gap, granularity gap — NOT the absolute fractions.
- Joint fraction is block-granularity, argmax criterion, positions 1:
  (position 0 excluded — no t-1 to reuse).
- Temporal oracle substitutes ALL positions at once (conservative;
  fractions are lower bounds on per-token reuse).
- width_fine_local uses a per-unit activation-magnitude criterion, not
  end-to-end zeroing (computationally out of reach); the coarse-vs-fine
  comparison is criterion-mismatched by necessity and labeled as such.
- 1 seed for m/l; seed pair on s only (spread there: ~0.01-0.03 on all
  fractions, small vs the effects).
- T2 (small-model-equivalence) is NOT discriminated by this grid: the
  oracle-vs-dense-frontier overlay arm was not run. Named follow-up.
- pred_frac criterion lives in hidden-space (ridge probe, rel-err<0.1),
  not end-to-end; reported separately from the joint by design.
