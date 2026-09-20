#!/bin/bash
# The fly on the robot, started by systemd (tools/pi/companion-hunt.service): the spiking hunter with the
# ESP32-CAM as its eyes and the S1 as its legs. The rover starts OFF; enable it from the page at :8601.
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/companion/brain" || exit 1
BRIDGE=$(ls /dev/serial/by-id/*CP210* 2>/dev/null | head -1)         # the S-Bus bridge ESP32 (CP2102), whatever ttyUSB number it got
[ "${S1_MODE:-}" = "udp" ] && BRIDGE=udp                             # S1_MODE=udp in companion-hunt.env: the body ESP32 does the S-Bus, packets over Wi-Fi
[ "${S1_MODE:-}" = "usb" ] && BRIDGE=usb                             # S1_MODE=usb: the body ESP32 does the S-Bus, packets down its USB cable (no Wi-Fi needed)
CAM="${CAM_URL:-http://companion-cam.local:81/stream}"
echo "[companion-hunt] camera $CAM, bridge ${BRIDGE:-none}"
ROVER_FLAG="--rover-off"; [ "${ROVER:-off}" = "on" ] && ROVER_FLAG=""      # ROVER=on in companion-hunt.env: drive from the start
MODE_FLAG=""; [ "${MODE:-fly}" = "scripted" ] && MODE_FLAG="--scripted"      # MODE=scripted: the OpenCV chaser instead of the fly brain
SCALE_FLAG=""; [ -n "${SCALE:-}" ] && SCALE_FLAG="--set body.s1.scale=$SCALE"  # SCALE=0.2: forward speed for this run
echo "[companion-hunt] rover ${ROVER:-off} at start, mode ${MODE:-fly}, scale ${SCALE:-config}"
exec uv run companion $SCALE_FLAG hunt-brain --real --camera "$CAM" --load data/cache/hunter_brain.npz $ROVER_FLAG $MODE_FLAG ${BRIDGE:+--s1 "$BRIDGE"}
