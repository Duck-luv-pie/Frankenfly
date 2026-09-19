#!/bin/bash
# The fly on the robot, started by systemd (tools/pi/companion-hunt.service): the spiking hunter with the
# ESP32-CAM as its eyes and the S1 as its legs. The rover starts OFF; enable it from the page at :8601.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/companion/brain" || exit 1
BRIDGE=$(ls /dev/serial/by-id/*CP210* 2>/dev/null | head -1)         # the S-Bus bridge ESP32 (CP2102), whatever ttyUSB number it got
CAM="${CAM_URL:-http://companion-cam.local:81/stream}"
echo "[companion-hunt] camera $CAM, bridge ${BRIDGE:-none}"
exec uv run companion hunt-brain --real --camera "$CAM" --load data/cache/hunter_brain.npz --rover-off ${BRIDGE:+--s1 "$BRIDGE"}
