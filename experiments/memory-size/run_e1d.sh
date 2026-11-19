#!/bin/zsh
cd ~/Desktop/Projects/continual-learning
{ .venv/bin/python memory.py --stream permuted --tasks 10 --kind rhash --Ns 4096,65536,262144 --seeds 2 --freeze head; } 2>&1 | grep -v -i warn > logs/e1d_rhash_headfrozen.log; echo DONE >> logs/e1d_rhash_headfrozen.log
