#!/bin/sh
# Ground truth from the body ESP32 on the robot Pi's USB: stop the fly, reset the board, show its boot log,
# send it three test packets, show its reaction, restart the fly. Run from the Mac while on "companion".
ssh -o ConnectTimeout=8 companion@10.42.0.1 '
export PATH=$HOME/.local/bin:$PATH
P=$(ls /dev/serial/by-id/*CP210* 2>/dev/null | head -1); [ -z "$P" ] && { echo "no CP210x ESP32 on the Pi USB"; exit 1; }
echo "== stopping the fly to borrow $P"; sudo systemctl stop companion-hunt
for i in 1 2 3 4 5 6 7 8 9 10; do fuser "$P" >/dev/null 2>&1 || break; sleep 1; done
uvx --offline --from pyserial python - "$P" <<"EOF"
import serial, sys, time, json
p = sys.argv[1]
def grab(baud, secs):
    s = serial.Serial(p, baud, timeout=0.1)
    s.setDTR(False); s.setRTS(True); time.sleep(0.2); s.setRTS(False)      # reset the board
    t = time.time(); buf = b""
    while time.time() - t < secs:
        try: buf += s.read(4096)
        except serial.SerialException: time.sleep(0.05)
    return s, buf.decode(errors="replace")
s, txt = grab(460800, 8)
baud = 460800
if "companion body" not in txt:
    s.close(); s, txt = grab(115200, 8); baud = 115200
print(f"== boot log at {baud} baud (8 s):"); print("\n".join(l for l in txt.splitlines() if l.strip() and not l.startswith("load:"))[-1500:])
if baud == 115200: print("== NOTE: the board answers at 115200: it is still running the OLD firmware (the flash has not happened)")
eye = {"px": 0, "py": 0, "pr": 0.3, "ut": 0.4, "lt": 0.4, "tint": [255, 120, 40], "tilt": 0.3}
pkt = {"t": 1, "state": "angry", "eyes": {"l": eye, "r": eye}, "blink": False, "sound": {"track": 0, "vol": 0}, "s1": {"ch": [1024, 1024, 1024, 1024, 352, 352, 1696]}}
line = (json.dumps(pkt, separators=(",", ":")) + "\n").encode()
print(f"== sending 3 test packets of {len(line)} bytes")
for i in range(3): s.write(line); time.sleep(0.3)
t = time.time(); buf = b""
while time.time() - t < 3:
    try: buf += s.read(4096)
    except serial.SerialException: time.sleep(0.05)
print("== board says:"); print(buf.decode(errors="replace").strip() or "(nothing)")
s.close()
EOF
echo "== restarting the fly"; sudo systemctl start companion-hunt
'
