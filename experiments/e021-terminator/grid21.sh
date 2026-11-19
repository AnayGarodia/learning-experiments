#!/usr/bin/env bash
# e021 grid: 5 arms x 3 seeds, k=10, on arche-arb.
# Sized by MEASURED RSS (e017: 5.5 GB peak/run, 62 GB box) and by e017's
# measured threads=3 per run, NOT by core count: 5 workers x 3 threads on 16
# cores. Detached (setsid), STATUS written on EVERY exit path, one record per
# run written by arb.py itself (crash-safe: a killed run leaves the earlier
# runs' records intact).
set -u
E=$HOME/experiments/e017-arbitration
R=$HOME/e021/runs
L=$HOME/e021/logs
S=$HOME/e021/STATUS
mkdir -p "$R" "$L"
PY=$HOME/venv/bin/python

status() { echo "$1 $(date -u +%FT%TZ) done=$(ls "$R"/*.json 2>/dev/null | wc -l)/15" > "$S"; }
trap 'status ABORTED' INT TERM
status STARTING

cd "$E" || { status FAILED_NO_DIR; exit 1; }
ARB_RUNS=$R $PY arb.py --selfcheck > "$L/selfcheck.log" 2>&1 || { status FAILED_SELFCHECK; exit 1; }

for a in selfdistil replay25 no_replay selfdistil_offcue selfdistil_front; do
  for s in 0 1 2; do echo "$a $s"; done
done | xargs -P 5 -n 2 bash -c '
  a=$0; s=$1
  [ -f '"$R"'/${a}_s${s}.json ] && exit 0
  ARB_RUNS='"$R"' '"$PY"' '"$E"'/arb.py --arm "$a" --seed "$s" \
     --threads 3 --n-eval 200 --freegen-n 32 > '"$L"'/${a}_s${s}.log 2>&1
  echo "exit=$? $a s$s" >> '"$L"'/exits.txt
'
status DONE
