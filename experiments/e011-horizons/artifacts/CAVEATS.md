# e011 artifact caveats

- The effect equals the instrument's resolution. The eval grid is 10
  points spaced 655,360 tokens apart (11% of a run). next_token and
  multi_horizon/annealed land on ADJACENT grid points, so the ordering
  is solid (all seeds agree on which side) but the magnitude is not:
  the true ratio is bounded only as >1. Do NOT quote 2.0x as a
  measured multiplier.
- ld=0.25 is UNDER-RESOLVED: all four arms cross on the same grid
  point. No ordering claim is made for that column.
- mtp4 vs next_token is a 2-1 seed split at matched data — unresolved.
  It separates only under FLOP-equivalent accounting, where the extra
  heads cost a deterministic 1.90x per token.
- 3 seeds per cell. Tiny models, 6M tokens, synthetic corpus with a
  planted long-range dependency; the transferable claim is the SIGN at
  this scale, not the size of the tax.
- The known scale caveat runs the other way: gloeckle2024 reports MTP
  paying only above ~3B params. A tiny-scale loss is consistent with
  that and does NOT license a claim about large models.
- Auxiliary-head FLOPs are counted in flop_eq_tokens; the raw-token
  column deliberately ignores them (matched-data view). Both are in
  results.csv.
