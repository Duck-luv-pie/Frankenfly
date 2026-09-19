#!/bin/bash
# Training queue for the Vast box (CUDA sparse engine, 512 envs). Sequential. Logs in logs/, checkpoints in checkpoints/.
cd "$(dirname "$0")/.."
PY=.venv/bin/python
run() { local name=$1 mins=$2; shift 2
  echo "[$(date '+%H:%M')] start $name ($mins min): $*" | tee -a logs/queue.log
  $PY train.py --run "$name" --minutes "$mins" --device cuda --engine event "$@" > "logs/train_$name.log" 2>&1
  echo "[$(date '+%H:%M')] end $name (exit $?)" | tee -a logs/queue.log; }
mkdir -p logs checkpoints
# 1. the demo-brain candidate: three-factor on eye/DN + KC->MBON, population forward drive, retinotopy, balanced dopamine
run tfA_both_512   60 --stage A --learn three_factor --plastic both --envs 512 --seconds 20 --no-promote
# 2. same with a stronger punishment channel (makes the -0.02/step penalty visible to PPL1)
run tfA_both_pun10 60 --stage A --learn three_factor --plastic both --envs 512 --seconds 20 --pun_gain 10 --no-promote
# 3. ES on the DN inputs, now that forward drive exists so fitness has a gradient
run esA_dnin_512   60 --stage A --learn es --plastic dn_in --envs 512 --seconds 10 --sigma 0.05 --no-promote
echo "[$(date '+%H:%M')] queue done" | tee -a logs/queue.log
