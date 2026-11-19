# 0001 — Run this as a lab, not a project

Date: 2026-08-15
Status: accepted

## Context

The goal is long-horizon: question the foundations of the current paradigm
(why transformers scale, whether text is the right substrate, continual
learning, sample efficiency) and find architectures that scale past
transformers. Multiple people and agents will work on it over months to
years.

## Options considered

1. Single research project: one question, one codebase, one push.
2. A lab: standing infrastructure (shared instrument, experiment records,
   decision log, library) that many questions and people run on.

## Decision

Option 2. The questions are coupled, the horizon is long, and contributors
will change. Infrastructure that makes results comparable and knowledge
durable is worth its upfront cost.

## Evidence / reasoning

Prior experience: benchmarks built without methodological rigor were later
voided wholesale (all pre-Aug-2026 AgentWeb benchmarks). Long-horizon work
without decision records and reproducible experiment records loses its own
history and repeats itself.
