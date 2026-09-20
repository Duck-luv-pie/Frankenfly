#!/bin/bash
# Offline bootstrap used by the headless boot-card test. Its stdout/stderr is
# redirected to web/receiver.log by systemd so failures are visible in a browser.
set -u

TRANSFER=/boot/firmware/badge-controller-transfer
[ -d "$TRANSFER" ] || TRANSFER=/boot/badge-controller-transfer
DEST=/home/companion/badge-controller

echo
echo "=== FlyBadge receiver bootstrap $(date -Is) ==="
echo "kernel: $(uname -a)"
echo "python: $(python3 --version 2>&1)"
echo "transfer: $TRANSFER"

systemctl start bluetooth 2>&1 || true
rfkill unblock bluetooth 2>&1 || true

mkdir -p "$DEST"
cp "$TRANSFER/pi_badge_receiver.py" "$TRANSFER/requirements.txt" "$TRANSFER/run_receiver.sh" "$DEST/"
chmod +x "$DEST/pi_badge_receiver.py" "$DEST/run_receiver.sh"

if [ ! -x "$DEST/.venv/bin/python" ]; then
  echo "Creating the offline Python environment..."
  python3 "$TRANSFER/offline/virtualenv.pyz" "$DEST/.venv"
  "$DEST/.venv/bin/pip" install --no-index --find-links "$TRANSFER/offline/wheels" -r "$DEST/requirements.txt"
fi

chown -R companion:companion "$DEST" 2>/dev/null || true
echo "parser check:"
"$DEST/.venv/bin/python" "$DEST/pi_badge_receiver.py" --self-test
echo "Starting BLE scan. Open Fly Pi Probe and press A, then B."
exec "$DEST/.venv/bin/python" "$DEST/pi_badge_receiver.py"
