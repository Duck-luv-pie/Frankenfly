#!/usr/bin/env bash
# Run ON the GPU box after sync.sh: installs uv, the project with the gpu + fast extras, and checks CUDA.
set -euo pipefail
cd ~/companion/brain
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv sync --extra gpu --extra fast
# the default torch wheel targets the newest CUDA; a host with an older driver needs the matching build
CUDA_VER=$(nvidia-smi 2>/dev/null | grep -o "CUDA Version: [0-9.]*" | grep -o "[0-9.]*$")
if [ -n "$CUDA_VER" ] && ! uv run --no-sync python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  TAG="cu$(echo "$CUDA_VER" | tr -d .)"
  echo "driver supports CUDA $CUDA_VER: installing torch for $TAG"
  uv pip install --python .venv --reinstall torch --index-url "https://download.pytorch.org/whl/$TAG"
fi
export UV_NO_SYNC=1
uv run --no-sync python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"
uv run --no-sync pytest -q tests/test_hunt_gpu.py
echo "ready. start a run with:  bash ~/companion/train.sh"
