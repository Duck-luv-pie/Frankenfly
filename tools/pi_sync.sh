#!/bin/sh
# Copy this repo (plus the cached, git-ignored brain files) from the Mac to the Raspberry Pi.
# Usage: tools/pi_sync.sh [user@host]     default: companion@companion-pi.local
# Re-run after every change; only differences are sent.
set -e
PI="${1:-companion@companion-pi.local}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
echo "syncing $HERE -> $PI:companion/"
rsync -az --delete \
  --exclude '.git' --exclude '.venv' --exclude '__pycache__' --exclude '.pio' --exclude '.pytest_cache' \
  --exclude 'brain/data/raw' --exclude 'brain/data/atlas' --exclude 'node_modules' \
  "$HERE/" "$PI:companion/"
echo "done. On the Pi:  ssh $PI  then  bash companion/tools/pi_setup.sh"
