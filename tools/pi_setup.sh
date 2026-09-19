#!/bin/bash
# One-time setup on the Raspberry Pi (64-bit Raspberry Pi OS). Installs uv, the brain with numba and
# the S1 driver, runs the benchmark. Usage on the Pi:  bash companion/tools/pi_setup.sh
set -e
cd "$(dirname "$0")/../brain"
if [ "$(uname -m)" != "aarch64" ]; then
  echo "This is a $(uname -m) system. Use the 64-bit Raspberry Pi OS: numba/opencv have no 32-bit Arm wheels." >&2
  exit 1
fi
sudo apt-get update -qq && sudo apt-get install -y -qq git rsync libgl1 libglib2.0-0 >/dev/null
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  grep -q '.local/bin' "$HOME/.bashrc" || echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$HOME/.bashrc"
fi
sudo usermod -aG dialout "$USER" || true           # serial ports (/dev/ttyUSB0) without sudo; takes effect at next login
uv sync --extra fast --extra s1 --extra gpu
echo
echo "--- brain benchmark (the pruned circuit; 1.0x or more = real time) ---"
uv run companion bench
echo
echo "next: plug in the ESP32 bridge and run  uv run python ../tools/s1_sbus.py --port /dev/ttyUSB0 --neutral"
