#!/usr/bin/env python3
"""Drive a DJI RoboMaster S1 from a Raspberry Pi over the motion controller's S-Bus port.

Wiring (Pi 5, 40-pin header):
  GPIO14 / TXD0 (pin 8) -> inverter -> S1 S-BUS "Signal"
  GND (pin 6)           -> S1 S-BUS "GND"
S-Bus is an inverted UART (idle low), so the Pi's TX needs one inverter: an NPN transistor
(base 1k from TXD, emitter GND, collector to Signal with 10k to 3.3 V) or a 74HC14 gate.
Enable the UART with `dtparam=uart0=on` in /boot/firmware/config.txt -> /dev/ttyAMA0.

Frame: 100000 baud, 8E2, 25 bytes every 14 ms: 0x0F, 16 x 11-bit channels LSB-first, flags, 0x00.
S1 channel map (user manual v1.8, "Using S-Bus Port"), sticks are 1024 +/- 672:
  1 strafe   2 forward/back   3 gimbal pitch   4 yaw (gimbal in chassis-lead, chassis in free mode)
  5 speed preset (3-pos)   6 chassis-lead(+)/free(-) mode   7 chassis set(+)/release(-)
Firmware >= 00.05.0046. Signs of channels 1/2/4 are unverified on hardware: flip SIGN below if a key
drives the wrong way.

Usage:  python3 tools/s1_sbus.py                  # keyboard: w/s forward, a/d strafe, q/e yaw, space stop, x quit
        python3 tools/s1_sbus.py --neutral        # just stream centered frames (does the S1 arm?)
        python3 tools/s1_sbus.py --port /dev/ttyUSB0   # e.g. an FTDI adapter with TXD inverted in its EEPROM
"""
from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty

try:
    import serial
except ImportError:
    sys.exit("pip install pyserial")

CENTER, SPAN = 1024, 672
LOW, HIGH = CENTER - SPAN, CENTER + SPAN
FRAME_S = 0.014
SIGN = {"strafe": +1, "forward": +1, "yaw": +1}   # flip after the first bench test if needed


def encode(ch: list[int], failsafe: bool = False, lost: bool = False) -> bytes:
    """16 channels of 11 bits -> 25-byte S-Bus frame."""
    buf = bytearray(25)
    buf[0] = 0x0F
    bits = n = 0
    i = 1
    for v in (ch + [CENTER] * 16)[:16]:
        bits |= (max(0, min(2047, int(v))) & 0x7FF) << n
        n += 11
        while n >= 8:
            buf[i] = bits & 0xFF
            i += 1
            bits >>= 8
            n -= 8
    buf[23] = (0x04 if lost else 0) | (0x08 if failsafe else 0)
    buf[24] = 0x00
    return bytes(buf)


def stick(x: float) -> int:
    """-1..1 -> S1 stick value."""
    return int(round(CENTER + SPAN * max(-1.0, min(1.0, x))))


class S1Link:
    def __init__(self, port: str, free_mode: bool = True, speed: str = "slow"):
        self.ser = serial.Serial(port, baudrate=100000, bytesize=8, parity=serial.PARITY_EVEN, stopbits=2, write_timeout=0.05)
        self.strafe = self.forward = self.yaw = self.pitch = 0.0
        self.mode_free = free_mode
        self.speed = {"slow": LOW, "medium": CENTER, "fast": HIGH}[speed]
        self.released = False
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()

    def frame(self) -> bytes:
        ch = [
            stick(SIGN["strafe"] * self.strafe),
            stick(SIGN["forward"] * self.forward),
            stick(self.pitch),
            stick(SIGN["yaw"] * self.yaw),
            self.speed,
            LOW if self.mode_free else HIGH,
            LOW if self.released else HIGH,
        ]
        return encode(ch)

    def _loop(self) -> None:
        nxt = time.perf_counter()
        rx = b""
        while not self._stop.is_set():
            try:
                self.ser.write(self.frame())
                if self.ser.in_waiting:                      # the ESP32 bridge reports once a second
                    rx += self.ser.read(self.ser.in_waiting)
                    while b"\n" in rx:
                        line, rx = rx.split(b"\n", 1)
                        print("[esp32]", line.decode(errors="replace").strip(), flush=True)
            except serial.SerialException:
                pass
            nxt += FRAME_S
            time.sleep(max(0.0, nxt - time.perf_counter()))

    def stop(self) -> None:
        self.strafe = self.forward = self.yaw = 0.0

    def close(self) -> None:
        self.stop()
        time.sleep(5 * FRAME_S)   # a few centered frames before the signal disappears
        self._stop.set()
        self._t.join(timeout=0.5)
        self.ser.close()


def keyboard(link: S1Link, gain: float) -> None:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    print("w/s forward, a/d strafe, q/e yaw, i/k gimbal pitch, space stop, r release/set chassis, x quit", flush=True)
    try:
        while True:
            if not select.select([sys.stdin], [], [], 0.2)[0]:
                continue
            k = sys.stdin.read(1)
            if k == "x":
                break
            elif k == " ":
                link.stop()
            elif k == "w": link.forward = gain
            elif k == "s": link.forward = -gain
            elif k == "a": link.strafe = -gain
            elif k == "d": link.strafe = gain
            elif k == "q": link.yaw = -gain
            elif k == "e": link.yaw = gain
            elif k == "i": link.pitch = min(1.0, link.pitch + 0.1)
            elif k == "k": link.pitch = max(-1.0, link.pitch - 0.1)
            elif k == "r":
                link.released = not link.released
                print("chassis", "released" if link.released else "set", flush=True)
            print(f"\rfwd {link.forward:+.2f} strafe {link.strafe:+.2f} yaw {link.yaw:+.2f} pitch {link.pitch:+.2f}   ", end="", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        print()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", default="/dev/ttyAMA0")
    p.add_argument("--neutral", action="store_true", help="stream centered frames only, no keyboard")
    p.add_argument("--pulse", default=None, metavar="AXIS:AMOUNT:SECONDS",
                   help="hands-off test: 3 s centred, then e.g. forward:0.3:2, then centred (axes: forward, strafe, yaw)")
    p.add_argument("--chassis-lead", action="store_true", help="channel 6 = chassis-lead mode (default free: ch4 turns the chassis)")
    p.add_argument("--speed", choices=["slow", "medium", "fast"], default="slow")
    p.add_argument("--gain", type=float, default=0.4, help="stick deflection for a key press, 0..1")
    a = p.parse_args()
    link = S1Link(a.port, free_mode=not a.chassis_lead, speed=a.speed)
    print(f"[s1] streaming S-Bus on {a.port} (free mode={not a.chassis_lead}, speed {a.speed})", flush=True)
    try:
        if a.pulse:
            axis, amount, secs = a.pulse.split(":")
            print(f"[s1] centred 3 s, then {axis} {float(amount):+.2f} for {float(secs)} s, then centred", flush=True)
            time.sleep(3)
            setattr(link, axis, float(amount))
            time.sleep(float(secs))
            link.stop()
            print("[s1] centred", flush=True)
            time.sleep(2)
        elif a.neutral:
            while True:
                time.sleep(1)
        else:
            keyboard(link, a.gain)
    except KeyboardInterrupt:
        pass
    finally:
        link.close()


if __name__ == "__main__":
    main()
