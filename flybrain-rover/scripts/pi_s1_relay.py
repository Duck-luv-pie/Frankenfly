#!/usr/bin/env python3
"""
pi_s1_relay.py -- runs ON THE PI: motor packets in over UDP, S-Bus out to the RoboMaster S1.

    python3 pi_s1_relay.py                          # /dev/ttyAMA0 (Pi UART through the inverter), UDP :4310
    python3 pi_s1_relay.py --port /dev/ttyUSB0      # FTDI / ESP32 adapter instead
    python3 pi_s1_relay.py --fake                   # no serial: count frames, for testing the network path

The brain runs on the laptop (15,000 neurons plus the person detector do not fit on a Pi). The Pi rides
the robot, holds the S-Bus wiring, and is the access point the ESP32-CAM streams through. This is the
one missing piece between them: the laptop's scripts/demo.py --s1-udp <pi>:4310 sends one small JSON
packet per frame, {"forward": f, "turn": t} in [-1, 1], and this hands it to Ducks's S1Body, which streams
S-Bus frames at 70 Hz and centres the sticks by itself if packets stop for half a second. Nothing here
decides anything; it is a wire.

Needs his companion_brain package on the Pi (tools/pi_sync.sh puts it in ~/companion) and pyserial.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time


def find_companion(explicit=None):
    for d in ([explicit] if explicit else []) + [os.path.expanduser("~/companion/brain"),
                                                 os.path.expanduser("~/hunting-fly/brain"),
                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hunting-fly-s1", "brain"),
                                                 os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hunting-fly", "brain")]:
        if d and os.path.isdir(os.path.join(d, "companion_brain", "body")):
            sys.path.insert(0, os.path.abspath(d))
            return os.path.abspath(d)
    return None


class FakeSerial:
    """Counts bytes instead of sending them, so the relay can be tested on a laptop."""
    def __init__(self):
        self.n = 0
        self.is_open = True

    def write(self, b):
        self.n += len(b)
        return len(b)

    def flush(self):
        pass

    def close(self):
        self.is_open = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=4310, help="UDP port for motor packets from the laptop")
    ap.add_argument("--port", default="/dev/ttyAMA0", help="serial port to the S1 S-Bus pins")
    ap.add_argument("--fake", action="store_true", help="no serial port; count frames (network test)")
    ap.add_argument("--companion-dir", default=None)
    ap.add_argument("--speed", default="slow", help="S1 speed preset the sticks were measured on")
    ap.add_argument("--sign-yaw", type=int, default=1)
    ap.add_argument("--sign-forward", type=int, default=1)
    ap.add_argument("--stick-forward", type=float, default=0.6, help="full forward = this fraction of stick throw")
    a = ap.parse_args()

    where = find_companion(a.companion_dir)
    if not where:
        sys.exit("cannot find companion_brain (run tools/pi_sync.sh first, or pass --companion-dir)")
    from companion_brain.body import s1
    cfg = dict(rotation_only=False, speed=a.speed, free_mode=True, timeout_s=0.5,
               stick_forward=a.stick_forward, stick_yaw=1.0, sign_yaw=a.sign_yaw, sign_forward=a.sign_forward)
    fake = FakeSerial() if a.fake else None
    body = s1.S1Body(cfg, ser=fake, port=None if a.fake else a.port)
    print(f"S1 over S-Bus on {'FAKE' if a.fake else a.port} (driver from {where}); "
          f"listening for motor packets on udp://0.0.0.0:{a.listen}", flush=True)

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", a.listen))
    sock.settimeout(0.5)
    n_pkt, last_from, last_line, t_first = 0, None, time.time(), None
    fwd = turn = 0.0
    try:
        while True:
            try:
                data, addr = sock.recvfrom(256)
            except socket.timeout:
                data = None
            if data:
                try:
                    m = json.loads(data.decode())
                except ValueError:
                    continue
                if m.get("stop"):
                    fwd = turn = 0.0
                else:
                    fwd = max(-1.0, min(1.0, float(m.get("forward", 0.0))))
                    turn = max(-1.0, min(1.0, float(m.get("turn", 0.0))))
                body.send({"motor": {"forward": fwd, "turn": turn}})
                n_pkt += 1
                if last_from != addr[0]:
                    last_from = addr[0]
                    print(f"  motor packets arriving from {addr[0]}", flush=True)
                t_first = t_first or time.time()
            if time.time() - last_line >= 2.0:
                last_line = time.time()
                st = body.status()
                idle = "" if data else "  (no packets, sticks centred by failsafe)"
                fr = f"{fake.n // 25} frames" if fake else f"{st.get('frames', 0)} frames"
                print(f"  {n_pkt} packets | sticks fwd {st['sticks']['forward']:+.2f} yaw {st['sticks']['yaw']:+.2f} "
                      f"| {fr} sent{idle}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        body.send({"motor": {"forward": 0.0, "turn": 0.0}})
        time.sleep(0.1)
        body.close()
        print("sticks centred, closed", flush=True)


if __name__ == "__main__":
    main()
