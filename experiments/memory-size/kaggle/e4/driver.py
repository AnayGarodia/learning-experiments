# Kaggle driver: sweep N for one addressing condition, one subprocess per run (fresh GPU each time).
import subprocess, sys, json, os
COND = os.environ.get("COND", "random"); NS = os.environ.get("NS", "4096,16384,65536,262144").split(",")
EXTRA = os.environ.get("EXTRA", "").split()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "e4_retrofit.py")).read() if os.path.exists("e4_retrofit.py") else None
results = {}
for N in NS:
    out = f"/kaggle/working/e4_{COND}_{N}.json"
    subprocess.run([sys.executable, "/kaggle/working/e4_retrofit.py", "--cond", COND, "--N", N, "--lr", "0.1", "--out", out] + EXTRA, check=False)
    if os.path.exists(out): results[N] = json.load(open(out))
    json.dump(results, open(f"/kaggle/working/e4_{COND}.json", "w"), indent=1)
    print("RESULT", COND, N, {k: v for k, v in results.get(N, {}).items() if k in ("acq_before", "acq_after", "n_known", "old_fact_retention", "util", "mc_after", "secs")}, flush=True)
