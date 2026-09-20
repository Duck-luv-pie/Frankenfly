#!/bin/sh
# Build the body firmware on the Mac and flash the ESP32 that hangs off the robot Pi's USB, from the Pi.
# Run from the Mac while on the Pi's Wi-Fi "companion". The fly service is stopped for the flash and restarted.
set -e
PI=companion@10.42.0.1
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE/firmware/body" && pio run 2>&1 | grep -E "error|SUCCESS|FAILED" | tail -2
ssh "$PI" 'mkdir -p ~/body-fw'
scp -q .pio/build/body/firmware.bin "$PI:~/body-fw/"
ssh "$PI" 'export PATH=$HOME/.local/bin:$PATH; P=$(ls /dev/serial/by-id/*CP210* | head -1); echo "flashing $P"; sudo systemctl stop companion-hunt; for i in 1 2 3 4 5 6 7 8 9 10; do fuser "$P" >/dev/null 2>&1 || break; sleep 1; done; (uvx --offline --from esptool esptool.py --chip esp32 --port "$P" --baud 460800 write_flash -z 0x10000 ~/body-fw/firmware.bin || uvx --from esptool esptool.py --chip esp32 --port "$P" --baud 460800 write_flash -z 0x10000 ~/body-fw/firmware.bin) 2>&1 | grep -E "Hash of data verified|Hard resetting|error|Failed|fatal|cause" | tail -4; sudo systemctl start companion-hunt; echo "fly restarted (~2 min to load)"'
