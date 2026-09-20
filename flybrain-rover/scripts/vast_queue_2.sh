#!/bin/bash
# Afternoon queue for the A100. Waits for scripts/vast_queue.sh to finish, then keeps training until ~00:20 box
# time (20:20 Toronto). Run inside tmux:  tmux new -d -s fly 'nohup bash scripts/vast_queue_2.sh > logs/queue2.log 2>&1'
cd "$(dirname "$0")/.."
PY=.venv/bin/python
while pgrep -f "vast_queue\.sh" > /dev/null; do sleep 60; done
run() { local name=$1 mins=$2; shift 2
  echo "[$(date '+%H:%M')] start $name ($mins min): $*" | tee -a logs/queue.log
  $PY train.py --run "$name" --minutes "$mins" --device cuda --engine event "$@" > "logs/train_$name.log" 2>&1
  echo "[$(date '+%H:%M')] end $name (exit $?)" | tee -a logs/queue.log; }
mkdir -p logs checkpoints
# 4. stage A with the exploratory state on (search moments inside stage A episodes)
run tfA_both_explore 60  --stage A --learn three_factor --plastic both --envs 512 --seconds 20 --no-promote --explore
# 5. ES on all eye/DN synapses, now that fitness has a gradient
run esA_lcdn_512     60  --stage A --learn es --plastic lc_dn --envs 512 --seconds 10 --sigma 0.05 --no-promote
# 6. long continuation of the demo-brain run
run tfA_both_long    180 --stage A --learn three_factor --plastic both --envs 512 --seconds 20 --no-promote --checkpoint checkpoints/tfA_both_512_latest.pt
# 7. stage B (search) training from the stage-A brain, explorer on
run tfB_explore      90  --stage B --learn three_factor --plastic both --envs 512 --seconds 20 --no-promote --explore --checkpoint checkpoints/tfA_both_512_latest.pt
echo "[$(date '+%H:%M')] afternoon queue done" | tee -a logs/queue.log
