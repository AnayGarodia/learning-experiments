#!/bin/zsh
cd ~/Desktop/Projects/continual-learning
P=.venv/bin/python
{ $P memory.py --stream permuted --tasks 10 --kind pkm --seeds 3 --freeze addr+head; } 2>&1 | grep -v -i warn > logs/e1b_addrhead.log &
{ $P memory.py --stream permuted --tasks 10 --kind pkm --seeds 3 --freeze addr; } 2>&1 | grep -v -i warn > logs/e1b_addr.log &
{ $P bn_control.py permuted 65536; } 2>&1 | grep -v -i warn > logs/e1_bncontrol.log &
wait; echo DONE >> logs/e1b_addr.log
