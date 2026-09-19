#!/bin/bash
# Runs ONE extra job after the main queue (scripts/overnight.sh) has exited. Uses the tonic-locomotion
# FALLBACK (amp_tonic 0.6, not biology, see STATUS.md Task 1). Ends before 07:00.
cd "$(dirname "$0")/.."
while pgrep -f "bash scripts/overnight.sh" > /dev/null; do sleep 60; done
echo "[$(date '+%H:%M')] start tfA_both_tonic (60 min, FALLBACK amp_tonic 0.6)" | tee -a logs/overnight_queue.log
.venv/bin/python train.py --run tfA_both_tonic --minutes 60 --stage A --learn three_factor --plastic both --envs 64 --seconds 20 --amp_tonic 0.6 > logs/overnight_tfA_both_tonic.log 2>&1
echo "[$(date '+%H:%M')] end tfA_both_tonic (exit $?)" | tee -a logs/overnight_queue.log
