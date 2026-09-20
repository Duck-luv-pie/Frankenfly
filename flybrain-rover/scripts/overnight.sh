#!/bin/bash
# Overnight training queue (2026-09-17). Sequential so the MPS GPU is never shared.
# Each run: CSV in logs/<run>.csv, stdout in logs/overnight_<run>.log, checkpoints/<run>_latest.pt
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() {  # name minutes args...
  local name=$1 mins=$2; shift 2
  echo "[$(date '+%H:%M')] start $name ($mins min): $*" | tee -a logs/overnight_queue.log
  $PY train.py --run "$name" --minutes "$mins" "$@" > "logs/overnight_$name.log" 2>&1
  echo "[$(date '+%H:%M')] end $name (exit $?)" | tee -a logs/overnight_queue.log
}
mkdir -p logs checkpoints
run tfA_both   75 --stage A --learn three_factor --plastic both  --envs 64  --seconds 20
run esA_dnin   75 --stage A --learn es --plastic dn_in --envs 128 --seconds 10 --sigma 0.05
run esA_lcdn   75 --stage A --learn es --plastic lc_dn --envs 128 --seconds 10 --sigma 0.05
run tfA_kcmbon 75 --stage A --learn three_factor --plastic kc_mbon --envs 64 --seconds 20
echo "[$(date '+%H:%M')] queue done" | tee -a logs/overnight_queue.log
