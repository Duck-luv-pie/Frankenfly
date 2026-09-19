#!/bin/bash
# One-time (or repeat) copy of this repo to a Vast.ai box. Run this YOURSELF from the repo's parent folder:
#     bash flybrain-rover/scripts/sync_to_box.sh <port> <host>          e.g. ... 30714 150.136.39.147
# Excludes secrets (.env), the venv, checkpoints and logs. brain.npz/brain_v2.npz/lc_columns.json ARE copied.
# To pull results back:  rsync -az -e "ssh -p <port>" root@<host>:~/flybrain-rover/checkpoints/ flybrain-rover/checkpoints_box/
set -e
PORT=${1:?port}; HOST=${2:?host}
cd "$(dirname "$0")/../.."
rsync -az -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
  --exclude .venv --exclude .env --exclude __pycache__ --exclude .pytest_cache \
  --exclude 'checkpoints/' --exclude 'logs/overnight_*' --exclude 'logs/*.log' --exclude '*.zip' \
  flybrain-rover root@$HOST:~/
echo "synced. next, on the box:  bash ~/flybrain-rover/scripts/vast_setup.sh"
