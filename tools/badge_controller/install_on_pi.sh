#!/bin/sh
# One-time Raspberry Pi installer for the read-only badge radio probe.
set -eu

if [ "$(uname -s)" != "Linux" ]; then
  echo "Run this on the Raspberry Pi, not on the Mac."
  exit 1
fi

SOURCE_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEST="$HOME/badge-controller"

echo "Installing the badge radio probe into $DEST"
mkdir -p "$DEST"
cp "$SOURCE_DIR/pi_badge_receiver.py" "$SOURCE_DIR/requirements.txt" "$SOURCE_DIR/README.md" "$DEST/"
cp "$SOURCE_DIR/run_receiver.sh" "$DEST/"
chmod +x "$DEST/pi_badge_receiver.py" "$DEST/run_receiver.sh"

echo "Enabling the Raspberry Pi Bluetooth service"
sudo rfkill unblock bluetooth || true
sudo systemctl enable --now bluetooth

if ! command -v bluetoothctl >/dev/null 2>&1; then
  echo "BlueZ is missing from this OS image. Connect the Pi to the internet and run:"
  echo "  sudo apt-get update && sudo apt-get install -y bluez"
  exit 1
fi

OFFLINE="$SOURCE_DIR/offline"
if [ -f "$OFFLINE/virtualenv.pyz" ] && [ -d "$OFFLINE/wheels" ]; then
  echo "Creating a private Python environment from the offline card bundle"
  python3 "$OFFLINE/virtualenv.pyz" "$DEST/.venv"
  "$DEST/.venv/bin/pip" install --no-index --find-links "$OFFLINE/wheels" -r "$DEST/requirements.txt"
else
  echo "Creating a private Python environment (internet required)"
  if ! python3 -m venv "$DEST/.venv"; then
    echo "Install python3-venv first, then rerun this script."
    exit 1
  fi
  "$DEST/.venv/bin/pip" install -r "$DEST/requirements.txt"
fi

echo "Checking the receiver parser"
"$DEST/.venv/bin/python" "$DEST/pi_badge_receiver.py" --self-test

cat <<EOF

Installation complete.

Start the badge receiver with:
  $DEST/run_receiver.sh

Then open Fly Pi Probe on the badge and press A, then B.
This probe only prints messages; it cannot control the robot.
EOF
