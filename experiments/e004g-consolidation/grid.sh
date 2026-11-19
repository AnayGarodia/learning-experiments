#!/bin/bash
# e004g full grid, launched by the VERIFIER after the pre-launch review.
# 8 skills (6 non-degenerate + 2 type-A copyables that extend k to the
# pre-registered 8) x 4 arms x 3 seeds, replay 0.25, n_eval 200.
#
# ONE worker at 16 threads, deliberately: the builder measured 567 fwd tok/s
# at 16 threads, ~94% of this SKU's fp32 peak, so a single process already
# saturates the box and concurrency would buy ~nothing while multiplying
# memory. Peak RSS is 3.8 GB (measured) of 31 GB. Sizing by MEASURED memory
# rather than by core count is the rule after a cpu_count-1 runner wedged
# another 32 GB box until sshd stopped answering.
cd ~/e004g || exit 1
SK=5,7,8,9,11,13,0,1
mkdir -p logs
for seed in 0 1 2; do
  for arm in consolidate no_replay direct_ft separate_adapters; do
    fg=0; [ "$seed" = 0 ] && fg=64
    ~/venv/bin/python ~/e004g/consolidate.py --arm "$arm" --seed "$seed" \
      --skills "$SK" --n-eval 200 --threads 16 --freegen-final "$fg" \
      > ~/e004g/logs/"$arm"_s"$seed".log 2>&1
    echo "done $arm s$seed $(date)"
  done
done
echo "GRID DONE $(date)"
~/venv/bin/python ~/e004g/analyze.py > ~/e004g/logs/analyze.log 2>&1
