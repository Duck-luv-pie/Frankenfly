#!/bin/sh
# Push the code to the Pi on the robot and (re)start the fly there. Run from the Mac while it is on the
# Pi's Wi-Fi "companion". Usage: tools/pi_push.sh [CAM_URL] [on|off]   e.g. tools/pi_push.sh http://10.42.0.157:4747/video on
set -e
PI=companion@10.42.0.1
CAM="${1:-http://10.42.0.157:4747/video}"
ROVER="${2:-off}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
echo "== copying the code"
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pio --exclude .pytest_cache \
  --exclude brain/data/raw --exclude brain/data/atlas --exclude node_modules "$HERE/" "$PI:companion/"
echo "== camera $CAM, rover $ROVER"
ssh "$PI" "printf 'CAM_URL=$CAM\nROVER=$ROVER\n' > ~/companion-hunt.env; sudo cp ~/companion/tools/pi/companion-hunt.service /etc/systemd/system/; sudo systemctl daemon-reload; sudo systemctl restart companion-hunt; echo '== restarted; the brain takes ~80 s to load'; ls /dev/serial/by-id/ 2>/dev/null | grep -qi cp210 && echo 'bridge: present' || echo 'bridge: NOT on the Pi (the fly will have no legs)'; curl -s -m 3 -o /dev/null -w 'phone camera: HTTP %{http_code}\n' $CAM || true"
echo "== watch it at http://10.42.0.1:8601 (page shows the camera, the fly's retina and drive; the rover button toggles driving)"
