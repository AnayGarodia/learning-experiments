#!/bin/bash
# e013 pilot runner. Workers are sized by MEMORY, not cores: measured
# per-job max RSS is 3.5GB, box has ~28GB free -> 6 workers (21GB), 2
# threads each. A runner defaulting to cpu_count-1 wedged another box today.
cd "$(dirname "$0")"
PY=~/venv/bin/python
W=6
T=2
stage=$1

case "$stage" in
icl)
  printf '%s\n' B_first B_upper N_map N_shift2 N_perm | xargs -P $W -I{} \
    sh -c "$PY exchange.py --icl --skills {} --threads $T --tag icl_{}_s0 \
           > logs/icl_{}.log 2>&1"
  ;;
lrsweep)
  for s in B_first N_map; do for lr in 1e-5 3e-5 1e-4 3e-4 1e-3 3e-3; do
    echo "$s $lr"; done; done | xargs -P $W -n2 sh -c \
    "$PY exchange.py --grad --skills \$0 --ngrid 32 --lr \$1 --threads $T \
     --tag lr_\$0_\$1_s0 > logs/lr_\$0_\$1.log 2>&1"
  ;;
grad)
  lrf=$2; lrm=$3
  for n in 1 2 3 4 6 8 12 16 24 32 48 64 96 128; do
    echo "B_first $lrf $n"; echo "N_map $lrm $n"; done | xargs -P $W -n3 sh -c \
    "$PY exchange.py --grad --skills \$0 --lr \$1 --ngrid \$2 --threads $T \
     --tag grad_\$0_n\$2_s0 > logs/grad_\$0_\$2.log 2>&1"
  ;;
*) echo "usage: run_pilot.sh {icl|lrsweep|grad <lr_B_first> <lr_N_map>}"; exit 1;;
esac
echo "stage $stage done"
