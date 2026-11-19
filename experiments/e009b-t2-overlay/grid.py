"""Full-grid driver for e009b: train what is missing, then measure every cell.

Grid: 8 rungs x 5 checkpoints x 2 seeds, MINUS rung xl at the two largest
training levels. That trim is the one exclusion the verifier allowed, and it
is on scientific grounds, not budget: at 3.0M and 6.0M tokens the dense
ladder has saturated on this corpus, xl ties the rungs below it and is
dropped from the Pareto envelope, so it contributes no frontier point there
— at ~62 min/checkpoint, the most expensive cell in the grid.

Idempotent: skips training whose checkpoints exist and measurement whose
runs/*.json exists, so it can be re-run after an interruption.

    python grid.py plan          # what would run, and why
    python grid.py train [W]     # train missing rungs, W workers
    python grid.py measure [W]   # measure missing cells, W workers
"""
import os
import subprocess
import time
import sys
from pathlib import Path

import ladder  # registers the rungs

HERE = Path(__file__).resolve().parent
CKPTS = [HERE / "ckpts", HERE.parent / "e009-redundancy" / "ckpts"]
RUNS = HERE / "runs"
SEEDS = [0, 1]
# e009's schedule: 4/10/25/50/100% of 6M tokens, as written to disk
LEVELS = [229376, 589824, 1490944, 2998272, 5996544]
# rung -> levels to measure. xl only where the dense ladder still separates.
XL_LEVELS = LEVELS[:3]


def cells():
    for size in ladder.LADDER:
        for seed in SEEDS:
            for t in (XL_LEVELS if size == "xl" else LEVELS):
                yield size, seed, t


def ckpt_path(size, seed, t):
    for d in CKPTS:
        p = d / f"{size}_s{seed}_t{t}.pt"
        if p.exists():
            return p
    return None


def missing_training():
    """A (size, seed) needs training if any of its levels lacks a checkpoint."""
    need = []
    for size in ladder.LADDER:
        for seed in SEEDS:
            lv = XL_LEVELS if size == "xl" else LEVELS
            if any(ckpt_path(size, seed, t) is None for t in lv):
                need.append((size, seed))
    # longest pole first: xl trains for hours and would otherwise start last
    return sorted(need, key=lambda c: -ladder.LADDER.index(c[0]))


def missing_measurement():
    todo = [c for c in cells() if not (RUNS / f"{c[0]}_s{c[1]}_t{c[2]}.json").exists()]
    # longest pole first: measurement cost goes as L^2 d^2 + L d V, so xl is
    # ~2.3x rung l and would otherwise be the tail of the run
    return sorted(todo, key=lambda c: -ladder.LADDER.index(c[0]))


def run_parallel(cmds, workers, log_dir):
    log_dir.mkdir(exist_ok=True)
    env = dict(os.environ, OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "2"))
    live, queue = [], list(cmds)
    while queue or live:
        while queue and len(live) < workers:
            name, cmd = queue.pop(0)
            fh = open(log_dir / f"{name}.log", "w")
            live.append((name, subprocess.Popen(cmd, cwd=HERE, env=env,
                                                stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"start {name}  ({len(queue)} queued)", flush=True)
        done = [i for i, (_, p, _) in enumerate(live) if p.poll() is not None]
        for i in reversed(done):
            name, p, fh = live.pop(i)
            fh.close()
            print(f"done  {name}  rc={p.returncode}", flush=True)
        if not done and live:
            time.sleep(5)  # poll, never block on live[0] — it may be the longest


def main():
    what = sys.argv[1] if len(sys.argv) > 1 else "plan"
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    tr, ms = missing_training(), missing_measurement()
    if what == "plan":
        print(f"grid: {len(list(cells()))} cells "
              f"({len(ladder.LADDER)} rungs x {len(LEVELS)} levels x "
              f"{len(SEEDS)} seeds, xl restricted to {XL_LEVELS})")
        print(f"training missing: {len(tr)} -> {tr}")
        print(f"measurement missing: {len(ms)}")
        for c in ms:
            print(f"  {c[0]:4} s{c[1]} t={c[2]:>8}")
    elif what == "train":
        run_parallel([(f"train_{s}_{d}", [sys.executable, "ladder.py", "train", s, str(d)])
                      for s, d in tr], workers, HERE / "logs")
    elif what == "measure":
        todo = []
        for size, seed, t in ms:
            p = ckpt_path(size, seed, t)
            if p is None:
                print(f"SKIP {size} s{seed} t={t}: no checkpoint")
                continue
            todo.append((f"measure_{size}_s{seed}_t{t}",
                         [sys.executable, "overlay.py", str(p)]))
        run_parallel(todo, workers, HERE / "logs")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
