"""Server-level read-only enforcement and robot isolation for the sim-mirror-webapp.

These tests stand a real `serve()` HTTP server up on an ephemeral port (in a daemon thread, driving
the scripted hunter in one room) and speak to it over the loopback with `urllib`, so the properties
are exercised against the actual inbound network surface -- not a stub.

Property 5 (Read-only enforcement): for any POST received while `read_only` is enabled, the server
returns 403 and neither the simulation state nor the robot command state changes.

Property 4 (Robot isolation invariant): no inbound POST -- including robot-command-looking bodies
(motor / sticks / velocity / actuator / ...) -- can produce an S-Bus frame or reach S1Body; the only
writer to the body is the in-process Mirror / failsafe path. Asserted both structurally (the do_POST
source contains no writer to the body) and behaviourally (a fake body wired into the mirror never
receives anything derived from a POST).

Plus a unit test that /config reports `read_only` and the configured feed value.

Validates: Requirements 1.2, 8.2, 8.3, 8.4, 9.6.
"""
import ast
import copy
import inspect
import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

torch = pytest.importorskip("torch")

from companion_brain.body.s1 import DEFAULTS  # noqa: E402
from companion_brain.config import load_config  # noqa: E402
from companion_brain.hunt_gpu.viewer import serve  # noqa: E402

CFG = load_config(overrides={"hunt": {"episode_s": 3.0}})

# a "leak marker": the mirror only ever sends the sim's own small physical values (speed <= 0.85 m/s,
# yaw <= 90 deg/s), so any body packet carrying a value this large could only have come from a POST.
LEAK = 1e6


class FakeBody:
    """An in-process stand-in for S1Body: the same duck type the Mirror / Watch touch (`cfg`,
    `send`, `status`, `link_ok`), recording every packet it is handed so a test can prove that
    nothing derived from an inbound POST ever reaches it."""

    def __init__(self):
        self.cfg = dict(DEFAULTS)
        self.link_ok = True
        self._lock = threading.Lock()
        self.received: list[dict] = []

    def send(self, packet: dict) -> bool:
        with self._lock:
            self.received.append(copy.deepcopy(packet))
        return True

    def status(self) -> dict:
        return {}

    def snapshot(self) -> list[dict]:
        with self._lock:
            return list(self.received)

    def close(self) -> None:
        pass


def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _post(port: int, obj) -> tuple[int, bytes]:
    """POST a JSON body to `/`; return (status_code, body). A 403 comes back as an HTTPError, which
    we normalise to its code so the caller sees the rejection rather than an exception."""
    data = json.dumps(obj).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def _get_json(port: int, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3.0) as r:
        return r.status, json.loads(r.read())


def _read_state(port: int, timeout: float = 3.0) -> dict | None:
    """Read one per-tick state off the SSE stream, then close the connection."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/events")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = r.readline()
            if not line:
                break
            if line.startswith(b"data:"):
                return json.loads(line[5:].strip() or b"{}")
    return None


def _start_server(port: int, *, body=None, read_only: bool = True, feed: str = "local",
                  relay_url=None, speed: float = 4.0) -> None:
    """Start `serve()` in a daemon thread and block until /config answers (or fail loudly)."""
    threading.Thread(
        target=serve,
        args=(CFG, None),
        kwargs=dict(port=port, seed=5, speed=speed, body=body,
                    read_only=read_only, feed=feed, relay_url=relay_url),
        daemon=True,
    ).start()
    for _ in range(200):
        try:
            _get_json(port, "/config")
            return
        except Exception:
            time.sleep(0.05)
    pytest.fail(f"server on port {port} did not come up")


@pytest.fixture(scope="module")
def ro_server():
    """A single read-only server (feed=local) with a fake S1 body wired into the mirror, shared by
    the whole module so Hypothesis can hammer it without restarting the arena each example."""
    port = _free_port()
    body = FakeBody()
    _start_server(port, body=body, read_only=True, feed="local")
    return port, body


# --- Unit test: /config reports read_only and the feed value ------------------------------------

def test_config_reports_read_only_and_feed(ro_server):
    """/config surfaces the spectator flag, the feed source, and the S1-rejection counter (R1.1, R11)."""
    port, _ = ro_server
    code, cfg = _get_json(port, "/config")
    assert code == 200
    assert cfg["read_only"] is True
    assert cfg["feed"] == "local"
    assert cfg["relay_url"] is None
    assert "s1_rejections" in cfg and isinstance(cfg["s1_rejections"], int)


def test_config_reports_remote_feed_when_configured():
    """The feed value on /config is exactly the one the Local_Bridge was started with (R11)."""
    port = _free_port()
    _start_server(port, body=None, read_only=True, feed="remote", relay_url="wss://relay.example/ingest")
    code, cfg = _get_json(port, "/config")
    assert code == 200 and cfg["feed"] == "remote" and cfg["relay_url"] == "wss://relay.example/ingest"


# --- Property 5: read-only enforcement ----------------------------------------------------------

# arbitrary JSON scalars, incl. large values that would be obvious if they leaked to the body
_scalar = st.one_of(
    st.none(), st.booleans(), st.integers(-1000, 1000),
    st.floats(min_value=LEAK, max_value=1e9), st.text(max_size=8),
)
# every control / sim key the handler knows, every robot-command key it must ignore, plus noise
_key = st.sampled_from([
    "next", "lobotomy", "heat", "unlimited", "paused", "speed",          # sim / view controls
    "motor", "sticks", "stick", "motion", "velocity", "actuator",        # robot-command shaped
    "channels", "throttle", "drive", "foo", "bar",
])
_post_body = st.one_of(
    st.dictionaries(_key, _scalar, max_size=6),
    # an explicitly robot-command-shaped body carrying leak markers in a nested motor packet
    st.fixed_dictionaries({"motor": st.fixed_dictionaries(
        {"speed_mps": st.floats(min_value=LEAK, max_value=1e9),
         "yaw_dps": st.floats(min_value=LEAK, max_value=1e9)})}),
)


def _no_leak(received: list[dict]) -> bool:
    """No packet the body received carries a leak-marker value: nothing from a POST reached it."""
    for pkt in received:
        motor = pkt.get("motor") or {}
        for v in motor.values():
            if isinstance(v, (int, float)) and abs(v) >= LEAK:
                return False
    return True


# Feature: sim-mirror-webapp, Property 5
# Validates: Requirements 1.2, 9.6
@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(body=_post_body)
def test_property5_readonly_rejects_every_post(ro_server, body):
    """For any POST while read-only is enabled, the server returns 403, the read-only configuration
    is unchanged, and the robot command state never takes on anything from the POST."""
    port, fake = ro_server
    code, _ = _post(port, body)
    assert code == 403                                             # R1.2 / R9.6: the control POST is rejected
    _, cfg = _get_json(port, "/config")
    assert cfg["read_only"] is True                               # the served state / mode is unchanged
    assert _no_leak(fake.snapshot())                              # the robot command state never reflects the POST


def test_property5_readonly_post_does_not_pause_the_sim(ro_server):
    """A control POST (`paused`) is rejected and leaves the simulation running: the SSE feed keeps
    delivering fresh per-tick state rather than freezing (R1.2 -- state unchanged)."""
    port, _ = ro_server
    code, _ = _post(port, {"paused": True})
    assert code == 403
    times = []
    for _ in range(4):
        st_ = _read_state(port)
        if st_ is not None:
            times.append(st_.get("t"))
        time.sleep(0.1)
    assert len(times) >= 2 and len(set(times)) >= 2               # the sim kept ticking; the POST did not pause it


# --- Property 4: robot isolation invariant ------------------------------------------------------

# Feature: sim-mirror-webapp, Property 4
# Validates: Requirements 8.2, 8.3, 8.4
def test_property4_do_post_has_no_sbus_writer():
    """Structural invariant: the only inbound network surface (do_POST) contains no code path that
    writes to S1Body -- no mirror, no body send, no stick / S-Bus encoding. The sole writer to the
    body is the in-process Mirror / failsafe path, not any request handler."""
    # parse the actual code (not comments / log strings, which legitimately mention S1Body while
    # explaining that the handler never writes there) and inspect the do_POST body's call graph.
    tree = ast.parse(inspect.getsource(serve))
    do_post = next(n for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef) and n.name == "do_POST")
    forbidden_calls = {"S1Body", "Mirror", "motor_to_sticks", "encode", "channels", "send"}
    forbidden_names = {"S1Body", "Mirror", "mirror", "motor_to_sticks"}
    for node in ast.walk(do_post):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else None
            assert name not in forbidden_calls, f"do_POST calls {name!r} -- a writer to the body"
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("send", "mirror"), f"do_POST accesses .{node.attr} -- a writer to the body"
        if isinstance(node, ast.Name):
            assert node.id not in forbidden_names, f"do_POST references {node.id!r} -- a writer to the body"


# Feature: sim-mirror-webapp, Property 4
# Validates: Requirements 8.2, 8.3, 8.4
def test_property4_robot_command_posts_never_reach_the_body(ro_server):
    """Behavioural invariant: firing robot-command-shaped POSTs (carrying leak markers) never lands
    anything on the fake body, and the discards are recorded so the rejection is observable (R8.5)."""
    port, fake = ro_server
    _, before = _get_json(port, "/config")
    n_before = before["s1_rejections"]
    commands = [
        {"motor": {"speed_mps": LEAK, "yaw_dps": LEAK}},
        {"sticks": {"forward": LEAK, "yaw": LEAK}},
        {"velocity": LEAK},
        {"actuator": {"throttle": LEAK}},
        {"channels": [LEAK] * 4},
        {"drive": [LEAK, LEAK]},
    ]
    for cmd in commands:
        code, _ = _post(port, cmd)
        assert code == 403                                        # read-only rejects it too
    # let a few sim ticks flow so any (buggy) deferred translation would have shown up by now
    time.sleep(0.3)
    assert _no_leak(fake.snapshot())                              # R8.2/R8.3/R8.4: no POST content reached the body
    _, after = _get_json(port, "/config")
    assert after["s1_rejections"] >= n_before + len(commands)     # R8.5: each discard is externally observable
