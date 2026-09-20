#!/bin/sh
# Push the code to the Pi on the robot and (re)start the fly there. Run from the Mac while it is on the
# Pi's Wi-Fi "companion". Usage: tools/pi_push.sh [CAM_URL] [on|off] [udp|usb|serial] [fly|scripted]
#   env SCALE=0.2 sets the forward speed for the run (default: the config's body.s1.scale)
#   e.g. tools/pi_push.sh http://10.42.0.157:4747/video on udp     (udp = the body ESP32 streams the S-Bus over Wi-Fi)
set -e
PI=companion@10.42.0.1
CAM="${1:-http://10.42.0.157:4747/video}"
ROVER="${2:-off}"
S1_MODE="${3:-udp}"
MODE="${4:-fly}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
echo "== copying the code"
rsync -az --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pio --exclude .pytest_cache \
  --exclude brain/data/raw --exclude brain/data/atlas --exclude node_modules "$HERE/" "$PI:companion/"
echo "== camera $CAM, rover $ROVER, S-Bus via $S1_MODE, mode $MODE, scale ${SCALE:-config}"
ssh "$PI" "printf 'CAM_URL=$CAM\nROVER=$ROVER\nS1_MODE=$S1_MODE\nMODE=$MODE\nSCALE=${SCALE:-}\n' > ~/companion-hunt.env; sudo cp ~/companion/tools/pi/companion-hunt.service /etc/systemd/system/; sudo systemctl daemon-reload; sudo systemctl restart companion-hunt; echo '== restarted; the brain takes ~80 s to load'; if [ '$S1_MODE' = udp ]; then getent hosts companion-body.local >/dev/null && echo 'body ESP32: on the network' || echo 'body ESP32: not seen on the network yet (it joins companion by itself; give it a minute)'; elif [ '$S1_MODE' = usb ]; then ls /dev/serial/by-id/ 2>/dev/null | grep -qi cp210 && echo 'body ESP32: on the Pi USB' || echo 'body ESP32: NOT on the Pi USB (plug its USB-C cable into the Pi)'; else ls /dev/serial/by-id/ 2>/dev/null | grep -qi cp210 && echo 'bridge: present' || echo 'bridge: NOT on the Pi (the fly will have no legs)'; fi; curl -s -m 3 -o /dev/null -w 'phone camera: HTTP %{http_code}\n' $CAM || true"
echo "== watch it at http://10.42.0.1:8601 (page shows the camera, the fly's retina and drive; the rover button toggles driving)"
