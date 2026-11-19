import itertools, pathlib
from train import run_arm, RUNS
ARMS = ["next_token", "multi_horizon", "annealed", "mtp4"]
for arm, ld, s in itertools.product(ARMS, (1.0, 0.25), range(3)):
    f = RUNS / f"{arm}_ld{ld}_s{s}.json"
    if not f.exists():
        print("running", arm, ld, s, flush=True)
        run_arm(arm, s, ld)
print("all filled")
