#!/usr/bin/env python3
"""The remote: a second Pi with three buttons and three lights that runs the fly on the robot Pi.

This Pi is the master. It joins the robot's Wi-Fi "companion" (the robot Pi is 10.42.0.1 there), polls
the robot's `GET /status` a few times a second for the lights, and the buttons `POST /control`:

  START     the trained (GPU-evolved) fly drives the rover: {"rover": true, "lobotomy": false, "paused": false}
  STOP      no movement, the fly keeps watching with the sticks centred: {"rover": false}
  LOBOTOMY  the untrained, senseless spare brain takes over (moving or not): {"lobotomy": true}; START restores the trained fly
  READY     lit when the robot answers with the brain loaded, the camera streaming and the body attached;
            slow blink = the brain is up but the camera or the body is missing; off = the robot is not answering
  RUNNING   lit while the rover is driving (rover on)
  LOBOTOMY  lit while the spare brain drives

Wiring on this Pi's header (every button between its GPIO and GND, internal pull-up; every LED GPIO -> 330 ohm -> LED -> GND):

  START button     GPIO 5  (pin 29)     READY light     GPIO 22 (pin 15)
  STOP button      GPIO 6  (pin 31)     RUNNING light   GPIO 23 (pin 16)
  LOBOTOMY button  GPIO 17 (pin 11)     LOBOTOMY light  GPIO 27 (pin 13)
  GND pins nearby: 6, 9, 14, 20, 25, 30, 34, 39

  python3 remote_button.py [--robot http://10.42.0.1:8601] [--start 5 --stop 6 --lobotomy 17] [--led-ready 22 --led-running 23 --led-lobotomy 27]
  python3 remote_button.py --keys          # no GPIO: s / x / l + Enter press the buttons, the lights print (for a test from a laptop)
A pin of -1 leaves that button or light out.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.request

ROBOT = "http://10.42.0.1:8601"
PRESS = {                          # what each button sends to the robot
    "start": {"rover": True, "lobotomy": False, "paused": False},
    "stop": {"rover": False},
    "lobotomy": {"lobotomy": True},
}
KEYS = {"s": "start", "x": "stop", "l": "lobotomy"}
POLL_S = 0.25


def lights(status: dict | None) -> dict:
    """The three lights from one /status snapshot (None = the robot did not answer). Values: True, False or "blink"."""
    if status is None:
        return {"ready": False, "running": False, "lobotomy": False}
    ready = bool(status.get("ready"))
    up = "t" in status                                        # the brain has ticked at least once
    return {"ready": True if ready else ("blink" if up else False),
            "running": bool(status.get("rover_on")), "lobotomy": bool(status.get("lobotomized"))}


def describe(status: dict | None) -> str:
    if status is None:
        return "robot not answering"
    if "t" not in status:
        return "robot up, brain still loading"
    parts = ["READY" if status.get("ready") else ("camera down" if not status.get("camera_ok") else "no body attached")]
    parts.append("moving" if status.get("rover_on") else "stopped")
    parts.append("LOBOTOMIZED" if status.get("lobotomized") else "trained fly")
    return ", ".join(parts)


class Robot:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def status(self) -> dict | None:
        try:
            with urllib.request.urlopen(self.base + "/status", timeout=1.5) as r:
                return json.loads(r.read().decode() or "{}")
        except Exception:
            return None

    def control(self, payload: dict) -> bool:
        req = urllib.request.Request(self.base + "/control", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        try:
            urllib.request.urlopen(req, timeout=2).read()
            return True
        except Exception as e:
            print(f"[remote] robot unreachable: {e}", flush=True)
            return False


class GpioPanel:
    """The breadboard: gpiozero buttons and LEDs."""

    def __init__(self, a, on_press):
        try:
            from gpiozero import LED, Button
        except ImportError:
            sys.exit("sudo apt install python3-gpiozero python3-lgpio")
        self.buttons = {}
        for name, pin in (("start", a.start), ("stop", a.stop), ("lobotomy", a.lobotomy)):
            if pin >= 0:
                b = Button(pin, pull_up=True, bounce_time=0.05)
                b.when_pressed = (lambda n: (lambda: on_press(n)))(name)
                self.buttons[name] = b
        self.leds = {name: LED(pin) for name, pin in (("ready", a.led_ready), ("running", a.led_running), ("lobotomy", a.led_lobotomy)) if pin >= 0}
        self.shown: dict = {}

    def show(self, want: dict) -> None:
        for name, led in self.leds.items():
            v = want[name]
            if self.shown.get(name) == v:
                continue
            self.shown[name] = v
            if v == "blink":
                led.blink(0.5, 0.5, background=True)
            elif v:
                led.on()
            else:
                led.off()


class KeyPanel:
    """No GPIO: the keyboard presses the buttons, the lights print when they change."""

    def __init__(self, on_press):
        self.shown: dict = {}
        print("[remote] keys: s = START, x = STOP, l = LOBOTOMY (then Enter); q quits", flush=True)

        def reader():
            for line in sys.stdin:
                k = line.strip().lower()[:1]
                if k == "q":
                    import os
                    os._exit(0)
                if k in KEYS:
                    on_press(KEYS[k])
        threading.Thread(target=reader, daemon=True).start()

    def show(self, want: dict) -> None:
        if want != self.shown:
            self.shown = dict(want)
            print("[remote] lights: " + "  ".join(f"{n} {'~' if v == 'blink' else ('ON' if v else 'off')}" for n, v in want.items()), flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default=ROBOT)
    p.add_argument("--start", type=int, default=5)
    p.add_argument("--stop", type=int, default=6)
    p.add_argument("--lobotomy", type=int, default=17)
    p.add_argument("--led-ready", type=int, default=22)
    p.add_argument("--led-running", type=int, default=23)
    p.add_argument("--led-lobotomy", type=int, default=27)
    p.add_argument("--keys", action="store_true", help="no GPIO: press the buttons from the keyboard")
    a = p.parse_args()
    robot = Robot(a.robot)
    wake = threading.Event()

    def on_press(name: str) -> None:
        print(f"[remote] {name.upper()} pressed -> {json.dumps(PRESS[name])}", flush=True)
        robot.control(PRESS[name])
        wake.set()                                            # refresh the lights right away

    panel = KeyPanel(on_press) if a.keys else GpioPanel(a, on_press)
    print(f"[remote] robot {a.robot} (Ctrl-C to stop)", flush=True)
    last_desc = None
    while True:
        st = robot.status()
        panel.show(lights(st))
        desc = describe(st)
        if desc != last_desc:
            print(f"[remote] {desc}", flush=True)
            last_desc = desc
        wake.wait(POLL_S)
        wake.clear()


if __name__ == "__main__":
    main()
