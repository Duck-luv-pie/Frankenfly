"""badge_link.py -- the badge's two robot controls, over the USB serial console.

    .venv/bin/python scripts/badge_link.py --dry-run          # print what it would press
    .venv/bin/python scripts/badge_link.py                    # for real, into scripts/demo.py

The badge game prints one line when you press its lesion or stop button:

    @FLY LOBO 1   /   @FLY LOBO 0        cut the visual pathway, and restore it
    @FLY STOP 1   /   @FLY STOP 0        wheels off, and back on

This reads those with pyserial and sends a single hotkey byte to 127.0.0.1:9600, the UDP port
`scripts/robot_bridge.py` already binds and `scripts/talk.py` already uses. So pressing B on the badge
lobotomises the rover in front of you, and the same press makes the mosquitoes on the badge stop
escaping. One button, both brains.

**Serial, not Bluetooth.** `badge.radio.enable()` panics this badge with the circuit loaded
(`BLE_INIT: Malloc failed`, then a Guru Meditation; see badge/SDK_NOTES.md), and `badge.radio` filters
receive to its own LUA1 frames with no documented way for a laptop to join. This is the channel
PROJECT_SPEC section 4c chose, for those two reasons.

Nothing depends on this. The badge game plays identically with no laptop listening, and the rover takes
the same hotkeys from the keyboard. If it does not work on the night, close it and press 8 yourself.

Two traps that have each cost real time:
  * the badge IDE holds the port exclusively. `[Errno 16] Resource busy` means click Disconnect in the
    IDE tab first.
  * it must be a data cable. A charge-only cable enumerates nothing at all.
"""
from __future__ import annotations

import argparse
import glob
import socket
import sys
import time

# LOBO/STOP -> the hotkey scripts/demo.py binds. 8 lobotomise, 9 wake, 0 stop/resume.
KEY_LOBO_ON, KEY_LOBO_OFF, KEY_STOP = b"8", b"9", b"0"
DEBOUNCE_S = 0.4
MARKER = "@FLY"


def find_port() -> str | None:
    """The badge shows up as a USB JTAG/serial debug unit. Prefer an explicit --port over this."""
    for pattern in ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/ttyACM*", "/dev/ttyUSB*"):
        hits = sorted(glob.glob(pattern))
        if hits:
            return hits[0]
    return None


class Link:
    """Parses badge console lines into hotkey bytes.

    Kept free of pyserial and of the socket so the whole decision can be tested over a pty, which is
    what tests/test_badge_link.py does. `send` is injected.
    """

    def __init__(self, send, now=time.monotonic, debounce_s: float = DEBOUNCE_S):
        self.send, self.now, self.debounce_s = send, now, debounce_s
        self.last_fire = {}
        # The badge sends absolute state ("STOP 1"); the rover's hotkey 0 is a toggle. So a datagram
        # goes out only on a CHANGE, and the two stay in step as long as they start in step. Pressing
        # 0 on the keyboard behind its back is what would desynchronise them.
        self.lobo = False
        self.stopped = False

    def _debounced(self, key: str) -> bool:
        t = self.now()
        if t - self.last_fire.get(key, -1e9) < self.debounce_s:
            return False
        self.last_fire[key] = t
        return True

    def feed(self, line: str) -> list[bytes]:
        """One console line in, the bytes actually sent out. Anything without the marker is ignored,
        because the badge logs plenty of other things and a stray word must never drive a robot."""
        if MARKER not in line:
            return []
        tail = line.split(MARKER, 1)[1].split()
        if len(tail) < 2:
            return []
        what, arg = tail[0].upper(), tail[1]
        if arg not in ("0", "1"):
            return []
        on = arg == "1"
        out: list[bytes] = []

        if what == "LOBO" and on != self.lobo and self._debounced("LOBO"):
            self.lobo = on
            out.append(KEY_LOBO_ON if on else KEY_LOBO_OFF)
        elif what == "STOP" and on != self.stopped and self._debounced("STOP"):
            self.stopped = on
            out.append(KEY_STOP)

        for b in out:
            self.send(b)
        return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", default=None, help="serial port (default: the first badge-looking one)")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--host", default="127.0.0.1", help="loopback only; this never leaves the laptop")
    ap.add_argument("--cmd-port", type=int, default=9600, help="the hotkey port robot_bridge binds")
    ap.add_argument("--dry-run", action="store_true", help="print the keypresses, send nothing")
    a = ap.parse_args(argv)

    if a.host not in ("127.0.0.1", "localhost", "::1"):
        print("refusing: the control loop is local (CLAUDE rule 5). --host must be loopback.",
              file=sys.stderr)
        return 2

    port = a.port or find_port()
    if not port:
        print("no serial port found. Plug the badge in with a DATA cable, and click Disconnect in the "
              "badge IDE tab if it is open.", file=sys.stderr)
        return 1

    try:
        import serial
    except ImportError:
        print("this needs pyserial:  .venv/bin/pip install pyserial", file=sys.stderr)
        return 1

    try:
        ser = serial.Serial(port, a.baud, timeout=0.2)
    except OSError as e:
        extra = ("  <- the badge IDE is holding the port. Click Disconnect in that tab."
                 if getattr(e, "errno", None) == 16 else "")
        print(f"cannot open {port}: {e}{extra}", file=sys.stderr)
        return 1

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    names = {KEY_LOBO_ON: "LOBOTOMISE", KEY_LOBO_OFF: "WAKE", KEY_STOP: "STOP/RESUME"}

    def send(b: bytes):
        print(f"  badge -> rover: {names.get(b, b.decode())}  (hotkey {b.decode()})", flush=True)
        if not a.dry_run:
            sock.sendto(b, (a.host, a.cmd_port))

    link = Link(send)
    print(f"listening on {port} at {a.baud}"
          f"{' (dry run)' if a.dry_run else f' -> udp {a.host}:{a.cmd_port}'}", flush=True)
    print("press B on the badge to lobotomise the rover, RIGHT to stop its wheels. Ctrl-C to stop.",
          flush=True)

    buf = b""
    try:
        while True:
            chunk = ser.read(256)
            if chunk:
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    link.feed(raw.decode("utf-8", "replace").strip())
            if len(buf) > 4096:        # a console that never sends a newline must not grow without end
                buf = buf[-512:]
    except KeyboardInterrupt:
        print("\nstopped. The rover keeps whatever state it was last put in.")
    finally:
        ser.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
