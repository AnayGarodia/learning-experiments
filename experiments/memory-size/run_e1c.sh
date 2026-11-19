#!/bin/zsh
cd ~/Desktop/Projects/continual-learning
P=.venv/bin/python
{ $P memory.py --stream permuted --tasks 10 --kind rhash --Ns 1024,4096,16384,65536,262144 --seeds 3; } 2>&1 | grep -v -i warn > logs/e1c_rhash.log &
{ $P slot_diag.py 65536; $P slot_diag.py 256; } 2>&1 | grep -v -i warn > logs/e1_slotdiag.log &
wait; echo DONE >> logs/e1c_rhash.log
