#!/bin/sh
# Ground truth from the body ESP32 on the robot Pi's USB: stop the fly, reset the board, show its boot log,
# send it three test packets, show its reaction, restart the fly. Run from the Mac while on "companion".
ssh -o ConnectTimeout=8 companion@10.42.0.1 '
export PATH=$HOME/.local/bin:$PATH
P=$(ls /dev/serial/by-id/*CP210* 2>/dev/null | head -1); [ -z "$P" ] && { echo "no CP210x ESP32 on the Pi USB"; exit 1; }
echo "== stopping the fly to borrow $P"; sudo systemctl stop companion-hunt; sleep 1
uvx --from pyserial python - "$P" <<"EOF"
import serial, sys, time, json
p = sys.argv[1]
s = serial.Serial(p, 460800, timeout=0.1)
s.setDTR(False); s.setRTS(True); time.sleep(0.2); s.setRTS(False)      # reset the board
t = time.time(); buf = b""
while time.time() - t < 8: buf += s.read(4096)
print("== boot log (8 s):"); print("\n".join(l for l in buf.decode(errors="replace").splitlines() if l.strip() and not l.startswith("load:"))[-1500:])
eye = {"px": 0, "py": 0, "pr": 0.3, "ut": 0.4, "lt": 0.4, "tint": [255, 120, 40], "tilt": 0.3}
pkt = {"t": 1, "state": "angry", "eyes": {"l": eye, "r": eye}, "blink": False, "sound": {"track": 0, "vol": 0}, "s1": {"ch": [1024, 1024, 1024, 1024, 352, 352, 1696]}}
line = (json.dumps(pkt, separators=(",", ":")) + "\n").encode()
print(f"== sending 3 test packets of {len(line)} bytes")
for i in range(3): s.write(line); time.sleep(0.3)
t = time.time(); buf = b""
while time.time() - t < 3: buf += s.read(4096)
print("== board says:"); print(buf.decode(errors="replace").strip() or "(nothing)")
s.close()
EOF
echo "== restarting the fly"; sudo systemctl start companion-hunt
'
