#!/bin/bash
# TASK A: the single-factor write-objective triplet.
#   consolidate        = KL write / KL replay   (context distillation, Snell 2022)
#   ce_write_kl_replay = CE write / KL replay   (the new arm: write only)
#   direct_ft          = CE write / CE replay   (the grid's confounded arm)
#
# POOL: 0,1,2,3,5,8,9,11 -- every e004g skill that passes the gate EXCEPT the
# prefix-colliding pair 7/13. The published grid TRAINED on 7 and 13 and only
# excluded them at READ time, so its exact 8-skill list is now blocked by
# consolidate.assert_pool_clean and its per-seed orders are unreproducible.
# All THREE arms are therefore re-run on this clean pool: the decomposition is
# read WITHIN the triplet, and the old grid is context only.
#
# One worker, 16 threads: the builder measured 567 fwd tok/s at 16 threads,
# ~94% of this SKU's fp32 peak, and peak RSS 3.8 GB of 31 GB. Sizing by
# MEASURED memory, not core count.
#
# Seed 0 runs first for all three arms and analyze.py must pass on it before
# the remaining six cells are spent (decision 0010 items 1-2).
cd ~/e004g-taskA || exit 1
SK=0,1,2,3,5,8,9,11
ARMS="consolidate ce_write_kl_replay direct_ft"
mkdir -p logs
run() {   # arm seed freegen
  ~/venv/bin/python consolidate.py --arm "$1" --seed "$2" --skills "$SK" \
    --n-eval 200 --threads 16 --freegen-final "$3" > logs/"$1"_s"$2".log 2>&1
}
for arm in $ARMS; do
  run "$arm" 0 64 || { echo "CELL FAILED $arm s0" > ~/STATUS_TASKA; exit 1; }
  echo "done $arm s0 $(date -u)"
done
if ! ~/venv/bin/python analyze.py > logs/analyze_pilot.log 2>&1; then
  echo "PILOT ANALYSIS BLOCKED, seeds 1-2 NOT spent $(date -u)" > ~/STATUS_TASKA
  exit 1
fi
for seed in 1 2; do
  for arm in $ARMS; do
    run "$arm" "$seed" 0 || { echo "CELL FAILED $arm s$seed" > ~/STATUS_TASKA; exit 1; }
    echo "done $arm s$seed $(date -u)"
  done
done
~/venv/bin/python analyze.py > logs/analyze_final.log 2>&1
echo "TASKA DONE $(date -u)" > ~/STATUS_TASKA
