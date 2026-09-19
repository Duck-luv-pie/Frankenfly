#!/usr/bin/env bash
# Copy the brain project (code + connectome data + checkpoints, no .venv) to a rented GPU box.
#   tools/cloud/sync.sh user@host [port]
# Re-run it any time: rsync only sends what changed. Pull results back with tools/cloud/fetch.sh.
set -euo pipefail
HOST="${1:?usage: sync.sh user@host [ssh-port]}"
PORT="${2:-22}"
SRC="$(cd "$(dirname "$0")/../../brain" && pwd)"
# minimal images (Vast, RunPod) often lack rsync and tmux
ssh -p "$PORT" "$HOST" 'command -v rsync >/dev/null && command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync tmux curl) >/dev/null'
ssh -p "$PORT" "$HOST" 'mkdir -p ~/companion/brain'
rsync -avz --progress -e "ssh -p $PORT" \
  --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude .ruff_cache --exclude '*.pyc' \
  "$SRC/" "$HOST:~/companion/brain/"
scp -P "$PORT" "$(dirname "$0")/setup.sh" "$(dirname "$0")/train.sh" "$HOST:~/companion/"
echo "synced. next:  ssh -p $PORT $HOST 'bash ~/companion/setup.sh'"
