#!/usr/bin/env python3
"""Drive the body ESP32's eyes over its USB serial port with the brain's own expression poses (no Wi-Fi needed):
the same JSON packets the brain sends over UDP, one per line, 30 a second. Bench tool for the eye animations.

  uv run python ../tools/eyes_demo.py --port /dev/cu.usbserial-210 --cycle angry,searching --hold 4
"""
from __future__ import annotations

import argparse
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("/tools/", 1)[0] + "/brain")
from companion_brain.body.eyes import eyes_for, POSES   # noqa: E402
from companion_brain.body.decode import Decoded         # noqa: E402

try:
    import serial
except ImportError:
    sys.exit("pip install pyserial")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/cu.usbserial-210")
    p.add_argument("--cycle", default="angry,searching", help="expressions to cycle through: " + ", ".join(POSES))
    p.add_argument("--hold", type=float, default=4.0, help="seconds per expression")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--echo", action="store_true", help="print what the ESP32 says back")
    a = p.parse_args()
    names = [n.strip() for n in a.cycle.split(",") if n.strip()]
    unknown = [n for n in names if n not in POSES]
    if unknown:
        sys.exit(f"unknown expression(s) {unknown}; known: {', '.join(POSES)}")
    ser = serial.Serial(a.port, 460800, timeout=0)
    time.sleep(0.3)
    t0 = time.monotonic()
    i = -1
    print(f"[eyes] {a.port}: cycling {names} every {a.hold:g} s (Ctrl-C to stop)", flush=True)
    try:
        while True:
            t = time.monotonic() - t0
            k = int(t / a.hold) % len(names)
            if k != i:
                i = k
                print(f"[eyes] {names[i]}", flush=True)
            d = Decoded(state=names[i])
            pkt = {"t": int(t * 1000), "state": names[i], "eyes": eyes_for(d, 0.0, 0.0, expression=names[i], now=t),
                   "blink": False, "sound": {"track": 0, "vol": 0}}
            ser.write((json.dumps(pkt, separators=(",", ":")) + "\n").encode())
            if a.echo and ser.in_waiting:
                print("[esp32]", ser.read(ser.in_waiting).decode(errors="replace").strip(), flush=True)
            time.sleep(1.0 / a.fps)
    except KeyboardInterrupt:
        pass
    finally:
        ser.close()


if __name__ == "__main__":
    main()
