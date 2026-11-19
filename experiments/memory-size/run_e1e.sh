#!/bin/zsh
cd ~/Desktop/Projects/continual-learning
{ .venv/bin/python memory.py --stream permuted --tasks 10 --kind rhash --Ns 1048576 --seeds 2 --freeze head; } 2>&1 | grep -v -i warn > e1e_rhash_1M.log; echo DONE >> e1e_rhash_1M.log
