#!/bin/zsh
cd ~/Desktop/Projects/continual-learning
P=.venv/bin/python
{ $P memory.py --stream split --kind pkm --seeds 3; $P memory.py --stream split --kind dense --Ns 32,128,512,2048 --seeds 3; } 2>&1 | grep -v -i warn > logs/e1_split.log &
{ $P memory.py --stream permuted --tasks 10 --kind pkm --seeds 3; $P memory.py --stream permuted --tasks 10 --kind dense --Ns 32,128,512,2048 --seeds 3; } 2>&1 | grep -v -i warn > logs/e1_permuted.log &
wait
echo DONE >> logs/e1_split.log
