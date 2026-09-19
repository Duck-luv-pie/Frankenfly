#!/usr/bin/env bash
# Run ON the GPU box, inside tmux so it survives the SSH session:  tmux new -s train 'bash ~/companion/train.sh'
# Default: the connectome fly by PPO against the frozen trained runners: 4096 rooms, backprop through 4 ticks, half the
# episodes starting at contact (tracking practice), the LC9 -> DNp09 brake switched on, resumed from the best brain so far.
# Override anything by passing your own arguments, e.g.  train.sh hunt-gpu --train --envs 8192
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/companion/brain
export UV_NO_SYNC=1          # keep the torch build setup.sh chose for this host's driver
OUT=data/cache/cloud_$(date +%Y%m%d_%H%M)
mkdir -p "$OUT"
if [ $# -gt 0 ]; then
  uv run companion "$@" --out "$OUT" 2>&1 | tee "$OUT/train.log"
else
  START=${START:-$(ls -t data/cache/cloud_*/hunter_brain.npz 2>/dev/null | head -1)}
  uv run companion --set hunt_gpu.brain.bptt_ticks=4 --set hunt_gpu.track.start_locked=0.5 --set hunt.near_gain=1.0 \
    --set hunt_gpu.brain.lr_head="${LR_HEAD:-0.01}" \
    hunt-brain --train --device cuda --envs "${ENVS:-4096}" --updates "${UPDATES:-200}" --humans "${HUMANS:-wander}" \
    ${START:+--load "$START" --fresh-brake} --out "$OUT" 2>&1 | tee "$OUT/train.log"
fi
echo "done -> $OUT"
