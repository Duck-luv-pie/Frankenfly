#!/usr/bin/env python3
"""The remote: a button on this Pi's GPIO lobotomizes (and restores) the fly on the robot Pi.

Wiring on this Pi's header: button between GPIO 17 (pin 11) and GND (pin 9); optional LED from GPIO 27
(pin 13) through 330 ohm to GND, lit while the fly is lobotomized. This Pi joins the robot's Wi-Fi
"companion"; the robot Pi is 10.42.0.1 there. Each press toggles; the LED follows what the robot reports.

  python3 remote_button.py [--robot http://10.42.0.1:8601] [--button 17] [--led 27]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

try:
    from gpiozero import Button, LED
except ImportError:
    sys.exit("sudo apt install python3-gpiozero")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--robot", default="http://10.42.0.1:8601")
    p.add_argument("--button", type=int, default=17)
    p.add_argument("--led", type=int, default=27)
    a = p.parse_args()
    button = Button(a.button, pull_up=True, bounce_time=0.05)
    led = LED(a.led) if a.led >= 0 else None
    state = {"lobotomized": None}

    def status():
        with urllib.request.urlopen(a.robot + "/status", timeout=2) as r:
            return json.loads(r.read().decode())

    def post(payload: dict):
        req = urllib.request.Request(a.robot + "/control", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=2).read()

    def pressed():
        want = not bool(state["lobotomized"])
        try:
            post({"lobotomy": want})
            print(f"[remote] {'LOBOTOMIZE' if want else 'restore'} -> sent", flush=True)
        except Exception as e:
            print(f"[remote] robot unreachable: {e}", flush=True)

    button.when_pressed = pressed
    print(f"[remote] button on GPIO {a.button}, robot {a.robot} (Ctrl-C to stop)", flush=True)
    while True:
        try:
            st = status()
            lob = bool(st.get("lobotomized"))
            if lob != state["lobotomized"]:
                state["lobotomized"] = lob
                print(f"[remote] robot says: {'lobotomized' if lob else 'trained fly'}", flush=True)
            if led is not None:
                (led.on if lob else led.off)()
        except Exception:
            if led is not None:
                led.blink(0.1, 0.9, n=1, background=True)   # a short blink: the robot is not answering
        time.sleep(0.5)


if __name__ == "__main__":
    main()
