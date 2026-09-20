#!/bin/sh
# Push a newer remote_button.py to the REMOTE Pi and restart its service. From the Mac on the robot's Wi-Fi "companion":
#   tools/remote_push.sh [companion@companion-remote.local]     (or companion@10.42.0.NN if .local does not resolve)
set -e
REMOTE="${1:-companion@companion-remote.local}"
HERE="$(cd "$(dirname "$0")" && pwd)"
scp -q "$HERE/pi/remote_button.py" "$REMOTE:remote_button.py"
scp -q "$HERE/pi/remote-button.service" "$REMOTE:remote-button.service"
ssh "$REMOTE" "sudo cp remote-button.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now remote-button >/dev/null 2>&1; sudo systemctl restart remote-button; sleep 2; journalctl -u remote-button -n 8 --no-pager"
