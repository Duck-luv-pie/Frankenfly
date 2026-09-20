"""UDP link to the body ESP32 (and a dry-run printer)."""
from __future__ import annotations

import json
import socket
import time


class BodyLink:
    def __init__(self, host: str, port: int, listen_port: int, verbose: bool = False):
        self.host, self.port = host, port
        self.addr: tuple[str, int] | None = None
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setblocking(False)
        self.sock.bind(("0.0.0.0", listen_port))
        self.verbose = verbose
        self.last_resolve = 0.0
        self.pir = 0
        self.busy = 0
        self.last_rx = 0.0
        self._resolve()

    def _resolve(self) -> None:
        self.last_resolve = time.time()
        try:
            self.addr = (socket.gethostbyname(self.host), self.port)
            print(f"[body] {self.host} -> {self.addr[0]}")
        except OSError:
            self.addr = None

    def send(self, packet: dict) -> None:
        if self.addr is None:
            if time.time() - self.last_resolve > 5:
                self._resolve()
            return
        try:
            self.sock.sendto(json.dumps(packet, separators=(",", ":")).encode(), self.addr)
        except OSError:
            pass
        if self.verbose:
            print(json.dumps(packet))

    def poll(self) -> dict | None:
        """Read the most recent packet from the body, update pir/busy, return it (or None)."""
        last = None
        while True:
            try:
                data, addr = self.sock.recvfrom(512)
            except (BlockingIOError, OSError):
                break
            try:
                last = json.loads(data.decode())
            except ValueError:
                continue
            if self.addr is None or addr[0] != self.addr[0]:
                self.addr = (addr[0], self.port)  # learn the body's address from its packets
        if last is not None:
            if not self.last_rx:
                print(f"[body] hearing from the body at {self.addr[0]} (rssi {last.get('rssi')})", flush=True)
            self.pir = int(last.get("pir", 0))
            self.busy = int(last.get("busy", 0))
            self.last_rx = time.time()
        return last

    def close(self) -> None:
        self.sock.close()


class DryBody:
    """Prints packets instead of sending them; PIR can be faked with `set_pir`."""

    def __init__(self, every: int = 4):
        self.every = every
        self.i = 0
        self.pir = 0
        self.busy = 0
        self.last_rx = time.time()
        self.addr = ("dry", 0)

    def send(self, packet: dict) -> None:
        self.i += 1
        if self.i % self.every == 0:
            e = packet["eyes"]["l"]
            print(f"[body] t={packet['t']:>8} state={packet['state']:<9} snd={packet['sound']['track']} "
                  f"pupil=({e['px']:+.2f},{e['py']:+.2f}) r={e['pr']:.2f} lids=({e['ut']:.2f},{e['lt']:.2f}) tint={e['tint']}")

    def poll(self) -> dict | None:
        return None

    def set_pir(self, v: int) -> None:
        self.pir = v

    def close(self) -> None:
        pass


class SerialBodyLink:
    """The same packets as BodyLink, as JSON lines down the body ESP32's USB cable (firmware/body reads them
    on its serial port). No Wi-Fi involved: on the robot the ESP32 hangs off the Pi's USB anyway.
    `port` = a serial device, or None to pick the first CP210x under /dev/serial/by-id (the ESP32 DevKit)."""

    def __init__(self, port: str | None = None, baud: int = 115200, ser=None, verbose: bool = False):
        self.pir = 0
        self.busy = 0
        self.last_rx = 0.0
        self.verbose = verbose
        self.port = port or self._find()
        self.addr = (self.port or "no-serial", 0)
        if ser is not None:
            self.ser = ser
        elif self.port:
            import serial
            self.ser = serial.Serial(self.port, baud, timeout=0, write_timeout=0.05)
            print(f"[body] ESP32 over USB serial at {self.port}", flush=True)
        else:
            self.ser = None
            print("[body] no ESP32 on USB serial (no CP210x under /dev/serial/by-id)", flush=True)
        self._rx = b""

    @staticmethod
    def _find() -> str | None:
        import glob
        hits = sorted(glob.glob("/dev/serial/by-id/*CP210*")) or sorted(glob.glob("/dev/cu.usbserial-*"))
        return hits[0] if hits else None

    def send(self, packet: dict) -> None:
        if self.ser is None:
            return
        try:
            self.ser.write((json.dumps(packet, separators=(",", ":")) + "\n").encode())
        except Exception:
            pass
        if self.verbose:
            print(json.dumps(packet))

    def poll(self) -> dict | None:
        """The ESP32's serial log lines (it reports PIR / status over UDP only); returned as {'log': line}."""
        if self.ser is None:
            return None
        try:
            n = self.ser.in_waiting
            if n:
                self._rx += self.ser.read(n)
        except Exception:
            return None
        last = None
        while b"\n" in self._rx:
            line, self._rx = self._rx.split(b"\n", 1)
            text = line.decode(errors="replace").strip()
            if text:
                last = {"log": text}
                self.last_rx = time.time()
                if text.startswith("PIR "):
                    self.pir = int(text[4:5] or 0)
        return last

    def close(self) -> None:
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
