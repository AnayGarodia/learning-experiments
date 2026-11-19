# 0008 — Hardware realism: every candidate needs a hardware story

Date: 2026-08-15
Status: accepted

## Context

Anay: a successor architecture "must scale predictably and be built for
the hardware. The hardware takes years to build and the NVIDIAs and AMDs
need to know demand will be there... it's billions+ of investment."
Library support: the hardware lottery ([[hooker2020-hardware-lottery]])
names the selection mechanism; Switch/MoE scaled BECAUSE its sparsity
stays dense-matmul ([[fedus2021-switch-transformer]]); capsules died
partly because per-example routing was accelerator-hostile
([[sabour2017-capsules]]); Hyena stalled on kernel efficiency, not
mechanism ([[poli2023-hyena]]).

## Decision

Every arche candidate architecture must state, at proposal time, one of:

(a) **Runs on today's stack**: its inner loop reduces to dense matmul /
    operations current accelerators execute near peak, with an argument
    (arithmetic-intensity level, memory-traffic estimate), or
(b) **Co-design path**: it needs different hardware, and the proposal
    names what hardware, why the capability win would justify the
    multi-year/billions investment, and what demand-certainty story a
    vendor could believe — plus an interim way to demonstrate the
    capability win on existing hardware at small scale.

Candidates with neither are not pursued, regardless of elegance
(capsules lesson). FLOP-matched and wall-clock comparisons are BOTH
reported and labeled: FLOP-matched measures the idea, wall-clock
measures the idea-on-this-hardware; conflating them bakes the lottery
into our own results.

## Reasoning (DERIVED)

The lottery is not a lament, it is a boundary condition. An architecture
1000x better that cannot run is 0x better. But the constraint cuts
carefully: hardware evolves BEHIND demonstrated demand (GPUs got tensor
cores and FP8 transformer engines after transformers proved demand), so
(b) is a real path — it just must be walked explicitly, with the
capability demonstration leading the hardware ask, never assumed.
