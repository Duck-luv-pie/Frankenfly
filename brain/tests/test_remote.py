"""The remote Pi's controller (tools/pi/remote_button.py): the lights from the robot's /status, and the three
buttons against a fake robot that answers /status and /control like hunt_gpu/real.py."""
from __future__ import annotations

import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "pi" / "remote_button.py"
spec = importlib.util.spec_from_file_location("remote_button", SCRIPT)
remote = importlib.util.module_from_spec(spec)
sys.modules["remote_button"] = remote
spec.loader.exec_module(remote)


def test_lights_follow_the_robot():
    assert remote.lights(None) == {"ready": False, "running": False, "lobotomy": False}           # not answering
    assert remote.lights({}) == {"ready": False, "running": False, "lobotomy": False}             # server up, brain still loading
    up = {"t": 1.0, "camera_ok": True, "rover": None, "rover_on": False, "lobotomized": False, "ready": False}
    assert remote.lights(up)["ready"] == "blink"                                                  # no body attached
    ready = {**up, "rover": {"sticks": {}}, "ready": True}
    assert remote.lights(ready) == {"ready": True, "running": False, "lobotomy": False}
    assert remote.lights({**ready, "rover_on": True}) == {"ready": True, "running": True, "lobotomy": False}
    assert remote.lights({**ready, "rover_on": True, "lobotomized": True}) == {"ready": True, "running": True, "lobotomy": True}
    assert "not answering" in remote.describe(None)
    assert remote.describe({**ready, "rover_on": True}).startswith("READY, moving, trained fly")


class FakeRobot:
    """Answers like RealHunt's server: /status is the state, /control merges the posted keys."""

    def __init__(self):
        self.state = {"t": 1.0, "camera_ok": True, "rover": {"sticks": {}}, "rover_on": False, "paused": False, "lobotomized": False, "ready": True}
        self.posts: list[dict] = []
        robot = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                body = json.dumps(robot.state).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

            def do_POST(self):
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))).decode())
                robot.posts.append(req)
                if "rover" in req:
                    robot.state["rover_on"] = bool(req["rover"])
                if "lobotomy" in req:
                    robot.state["lobotomized"] = bool(req["lobotomy"])
                if "paused" in req:
                    robot.state["paused"] = bool(req["paused"])
                self.send_response(204); self.send_header("Content-Length", "0"); self.end_headers()

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()


@pytest.fixture
def fake():
    r = FakeRobot()
    yield r
    r.close()


def test_buttons_drive_the_robot(fake):
    robot = remote.Robot(fake.url)
    assert remote.lights(robot.status()) == {"ready": True, "running": False, "lobotomy": False}
    assert robot.control(remote.PRESS["start"])
    assert fake.state["rover_on"] and not fake.state["lobotomized"] and not fake.state["paused"]
    assert remote.lights(robot.status())["running"] is True
    assert robot.control(remote.PRESS["lobotomy"])
    assert fake.state["lobotomized"] and fake.state["rover_on"]                                   # lobotomy leaves the movement as it was
    assert remote.lights(robot.status()) == {"ready": True, "running": True, "lobotomy": True}
    assert robot.control(remote.PRESS["stop"])
    assert not fake.state["rover_on"] and fake.state["lobotomized"]                               # stop does not restore the brain
    assert robot.control(remote.PRESS["start"])                                                   # start does
    assert fake.state["rover_on"] and not fake.state["lobotomized"]
    assert fake.posts == [remote.PRESS["start"], remote.PRESS["lobotomy"], remote.PRESS["stop"], remote.PRESS["start"]]


def test_unreachable_robot_is_dark():
    robot = remote.Robot("http://127.0.0.1:9")                                                    # nothing listens on the discard port
    assert robot.status() is None
    assert robot.control(remote.PRESS["start"]) is False
    assert remote.lights(robot.status()) == {"ready": False, "running": False, "lobotomy": False}
