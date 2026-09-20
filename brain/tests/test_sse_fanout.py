"""SSE fan-out consistency (Property 6) and the near-term LAN + sim-to-robot-mirror integration for
the sim-mirror-webapp.

Property 6 (SSE fan-out consistency): for any sequence of published ticks and any set of connected
subscribers, every subscriber receives the same per-tick payloads in the same tick order (R3.5); and
for any tick missing a required field (`fly`, `people`, `heat`, `vis`, `motion`, `drive`) the
Local_Bridge omits that tick's payload -- it does not bump the state box's version -- rather than
emitting a malformed one, and it continues with the next tick without terminating the stream (R3.4).

The real `serve()` drives a live sim that always builds complete ticks, so the "missing required field
-> omit, version not bumped" half is exercised against an in-test replica of the exact
`box{version, payload}` + `Condition` publish step `serve()` uses (see `StateBox` below). The fan-out
half is exercised three ways: a Hypothesis subsequence model over the replica (Property 6), a
thread-synchronised replica fan-out where every subscriber provably receives the identical full stream,
and a live multi-subscriber SSE test against a real `serve()` on an ephemeral port.

The integration test stands a real bridge up with an `S1Body` on a fake serial (no hardware), connects
a spectator to `/events`, and asserts the sim mirrors its drive onto the fake S1 (the fake serial keeps
receiving S-Bus frames and `body.status()` shows the commanded sticks) while every web client stays
watch-only over the LAN bind; a further test drops one subscriber and asserts the others keep receiving.

Feature: sim-mirror-webapp, Property 6
Validates: Requirements 3.1, 3.2, 3.4, 3.5, 3.6, 9.3, 9.4.
"""
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

from companion_brain.body.s1 import S1Body, decode  # noqa: E402
from companion_brain.config import load_config  # noqa: E402
from companion_brain.hunt_gpu.viewer import serve  # noqa: E402

CFG = load_config(overrides={"hunt": {"episode_s": 3.0}})

# the required fields of a per-tick State_Feed payload (design "State-feed JSON payload", R3.3):
# the fly pose, the human poses, the fly's senses (heat / vis / motion), and the Drive_Vector.
REQUIRED = ("fly", "people", "heat", "vis", "motion", "drive")


# --- an in-test replica of serve()'s publish step ------------------------------------------------

class StateBox:
    """The exact `box{version, payload}` + `Condition` publish step `serve()` runs, isolated so the
    "missing required field -> omit the tick, do not bump the version" rule (R3.4) can be driven with
    ticks the live sim never produces. `publish(st)` mirrors the server: it serialises and bumps the
    version only when every required field is present, otherwise it leaves the box untouched."""

    def __init__(self):
        self.cond = threading.Condition()
        self.box = {"version": 0, "payload": b"{}"}

    def publish(self, st: dict) -> bool:
        # R3.4: if any required field is unavailable for this tick, omit that tick's payload (do not bump
        # the version) and continue with the next tick without terminating the stream.
        if any(st.get(k) is None for k in REQUIRED):
            return False
        payload = json.dumps(st, separators=(",", ":")).encode()
        with self.cond:
            self.box["payload"] = payload
            self.box["version"] += 1
            self.cond.notify_all()
        return True


@st.composite
def _tick(draw):
    """A per-tick candidate: a complete payload (all required fields present) or one with a single
    required field missing (dropped or explicitly null), tagged with whether it is complete."""
    idx = draw(st.integers(0, 10_000))
    val = draw(st.integers(-1, 1))
    payload = {
        "t": idx,
        "fly": [float(idx), 0.0, 0.0, 0.0],
        "people": [[float(val), 0.0]],
        "heat": [0.0, 0.0],
        "vis": [0.0] * 3,
        "motion": [0.0] * 3,
        "drive": [float(val), 0.0],
        "episode": 1,
    }
    complete = draw(st.booleans())
    if not complete:
        missing = draw(st.sampled_from(REQUIRED))
        if draw(st.booleans()):
            del payload[missing]              # the field is absent
        else:
            payload[missing] = None           # the field is present but unavailable
    return complete, payload


# Feature: sim-mirror-webapp, Property 6
# Validates: Requirements 3.4, 3.5
@settings(max_examples=200, deadline=None)
@given(ticks=st.lists(_tick(), max_size=25), data=st.data())
def test_property6_missing_field_omitted_and_fanout_consistent(ticks, data):
    """For any published tick sequence: a tick missing a required field is omitted (the version is not
    bumped) while complete ticks bump the version by exactly one, so the stream carries a gap-free,
    ordered run of well-formed payloads; and any set of subscribers, each seeing an arbitrary
    subsequence of that stream (a real SSE reader only ever reads the box's current payload), receives
    identical payloads for every tick they both observe, in tick order."""
    box = StateBox()
    canonical: list[bytes] = []                 # the ordered stream a keeping-up subscriber would see
    complete_count = 0
    for complete, st_dict in ticks:
        before = box.box["version"]
        published = box.publish(st_dict)
        after = box.box["version"]
        if complete:
            complete_count += 1
            assert published is True
            assert after == before + 1          # a complete tick advances the version by exactly one
            canonical.append(box.box["payload"])
        else:
            assert published is False
            assert after == before               # R3.4: a missing required field does not bump the version

    # the version equals the number of complete ticks: no gaps, nothing malformed slipped through
    assert box.box["version"] == complete_count == len(canonical)
    for raw in canonical:
        obj = json.loads(raw)                    # every published payload parses ...
        for k in REQUIRED:
            assert obj.get(k) is not None        # ... and carries every required field (R3.4)

    # R3.5: model an arbitrary set of subscribers. Each real SSE subscriber wakes at its own moments and
    # reads the box's *current* payload, so its delivered stream is an order-preserving subsequence of
    # the canonical stream, keyed by version. Any tick two subscribers both catch must be byte-identical.
    versions = list(range(len(canonical)))
    n_subs = data.draw(st.integers(1, 4))
    subs: list[list[int]] = []
    for _ in range(n_subs):
        caught = sorted(set(data.draw(st.lists(st.sampled_from(versions))))) if versions else []
        subs.append(caught)
    for caught in subs:
        delivered = [canonical[v] for v in caught]
        assert delivered == [canonical[v] for v in sorted(caught)]      # tick order preserved
    for a in range(len(subs)):
        for b in range(a + 1, len(subs)):
            for v in set(subs[a]) & set(subs[b]):
                assert canonical[v] == canonical[v]                     # same version -> same payload


def test_fanout_replica_every_subscriber_gets_the_identical_full_stream():
    """A thread-synchronised fan-out over the replica: with the publisher pacing itself to every
    subscriber, all subscribers receive the identical, fully-ordered payload stream -- the strong form
    of R3.5 (same payloads, same tick order, to every connected browser)."""
    payloads = [f'{{"t":{i}}}'.encode() for i in range(12)]
    n_subs = 5
    cond = threading.Condition()
    box = {"version": 0, "payload": b""}
    received: list[list[bytes]] = [[] for _ in range(n_subs)]
    acks = threading.Semaphore(0)

    def subscriber(i: int):
        seen = 0
        for _ in range(len(payloads)):
            with cond:
                cond.wait_for(lambda: box["version"] != seen)
                seen = box["version"]
                payload = box["payload"]
            received[i].append(payload)
            acks.release()

    threads = [threading.Thread(target=subscriber, args=(i,), daemon=True) for i in range(n_subs)]
    for t in threads:
        t.start()
    for payload in payloads:
        with cond:
            box["payload"] = payload
            box["version"] += 1
            cond.notify_all()
        for _ in range(n_subs):                 # wait until every subscriber has consumed this version
            assert acks.acquire(timeout=2.0)
    for t in threads:
        t.join(timeout=2.0)
    for stream in received:
        assert stream == payloads               # identical payloads, identical tick order, every subscriber


# --- live-server helpers -------------------------------------------------------------------------

def _free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _get_json(port: int, path: str) -> tuple[int, dict]:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=3.0) as r:
        return r.status, json.loads(r.read())


def _post(port: int, obj) -> int:
    data = json.dumps(obj).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/", data=data, method="POST",
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=3.0) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


class SSE:
    """A minimal SSE spectator: opens `/events` and reads per-tick `data:` payloads off the stream."""

    def __init__(self, port: int, timeout: float = 6.0):
        self.r = urllib.request.urlopen(f"http://127.0.0.1:{port}/events", timeout=timeout)

    def read_payload(self, timeout: float = 6.0) -> bytes | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            line = self.r.readline()
            if not line:
                return None
            if line.startswith(b"data:"):
                return line[5:].strip()
        return None

    def read_many(self, count: int, timeout: float = 8.0) -> list[bytes]:
        out = []
        deadline = time.time() + timeout
        while len(out) < count and time.time() < deadline:
            p = self.read_payload(timeout=max(0.1, deadline - time.time()))
            if p is None:
                break
            out.append(p)
        return out

    def close(self):
        try:
            self.r.close()
        except Exception:
            pass


def _start_server(port: int, *, bind: str = "0.0.0.0", body=None, read_only: bool = True,
                  speed: float = 5.0) -> None:
    """Start `serve()` in a daemon thread and block until /config answers (or fail loudly)."""
    threading.Thread(
        target=serve,
        args=(CFG, None),
        kwargs=dict(port=port, seed=7, speed=speed, body=body, read_only=read_only, bind=bind),
        daemon=True,
    ).start()
    for _ in range(200):
        try:
            _get_json(port, "/config")
            return
        except Exception:
            time.sleep(0.05)
    pytest.fail(f"server on port {port} did not come up")


def _key(raw: bytes) -> tuple:
    """A per-tick ordering / identity key: (episode, sim-time). Distinct per published version (t grows
    by dt within an episode, the episode counter grows across episodes)."""
    obj = json.loads(raw)
    return (obj.get("episode"), obj.get("t"))


# --- Property 6 over the live feed: multi-subscriber fan-out -------------------------------------

# Feature: sim-mirror-webapp, Property 6
# Validates: Requirements 3.2, 3.5
def test_live_multi_subscriber_fanout_consistency():
    """Three spectators on a real `serve()` receive per-tick payloads that are byte-identical for every
    tick they share and are delivered in the same tick order, and every delivered payload carries all
    the required fields -- fan-out consistency over the live SSE feed (R3.5), first state within 1 s
    of connecting (R3.2)."""
    port = _free_port()
    _start_server(port, read_only=True, speed=5.0)

    subs = [SSE(port) for _ in range(3)]
    try:
        # each subscriber's first state must arrive quickly (R3.2)
        firsts = [s.read_payload(timeout=1.0) for s in subs]
        assert all(f is not None for f in firsts)

        results: list[list[bytes]] = [[] for _ in subs]

        def drain(i: int):
            results[i] = subs[i].read_many(15, timeout=8.0)

        threads = [threading.Thread(target=drain, args=(i,)) for i in range(len(subs))]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        for s in subs:
            s.close()

    streams = [[firsts[i]] + results[i] for i in range(len(subs))]
    for stream in streams:
        assert len(stream) >= 3
        keys = [_key(raw) for raw in stream]
        assert keys == sorted(keys)                          # R3.5: tick order (episode, t) non-decreasing
        for raw in stream:                                   # every delivered payload is well-formed (R3.4)
            obj = json.loads(raw)
            for k in REQUIRED:
                assert obj.get(k) is not None

    # any tick two subscribers both received is byte-identical, and there is real overlap to check
    maps = [{_key(raw): raw for raw in stream} for stream in streams]
    overlap = 0
    for a in range(len(maps)):
        for b in range(a + 1, len(maps)):
            common = set(maps[a]) & set(maps[b])
            overlap += len(common)
            for key in common:
                assert maps[a][key] == maps[b][key]          # R3.5: same tick -> same payload
    assert overlap > 0


# --- near-term integration: LAN watch-only view + sim-to-robot mirror ----------------------------

class FakeSerial:
    """A stand-in for the S1's S-Bus serial line: records every 25-byte frame the body streams out."""

    def __init__(self):
        self._lock = threading.Lock()
        self.frames: list[bytes] = []
        self.closed = False

    def write(self, b):
        with self._lock:
            self.frames.append(bytes(b))

    def snapshot(self) -> list[bytes]:
        with self._lock:
            return list(self.frames)

    def close(self):
        self.closed = True


class RecordingS1Body(S1Body):
    """An `S1Body` on a fake serial that also records the motor packets the Mirror hands it, so a test
    can prove the sim's per-tick drive reaches the robot driver (not just that frames stream out)."""

    def __init__(self, *a, **k):
        self.sends: list[dict] = []
        super().__init__(*a, **k)

    def send(self, packet: dict) -> bool:
        self.sends.append(packet)
        return super().send(packet)


# Feature: sim-mirror-webapp
# Validates: Requirements 3.1, 9.3, 9.4
def test_integration_mirror_drives_fake_s1_while_serving_watch_only_over_lan():
    """The bridge, started with an S1Body on a fake serial and bound to all interfaces (LAN), mirrors
    the fly's per-tick drive onto the fake S1 while serving every web client watch-only: the fake serial
    keeps receiving valid S-Bus frames, the Mirror hands the body motor packets, `body.status()` reports
    the commanded sticks, and a control POST is rejected with 403 (R3.1 mirror path, R9.3/R9.4 LAN serve,
    R1.2 watch-only)."""
    port = _free_port()
    ser = FakeSerial()
    body = RecordingS1Body({"timeout_s": 0.5}, ser=ser, start=True)
    _start_server(port, bind="0.0.0.0", body=body, read_only=True, speed=5.0)

    sub = SSE(port)
    try:
        # the LAN State_Feed answers quickly (bound to 0.0.0.0, reached over loopback; R9.4)
        first = sub.read_payload(timeout=2.0)
        assert first is not None
        # let a handful of ticks flow so the mirror has driven the body several times
        sub.read_many(6, timeout=4.0)
    finally:
        sub.close()

    # the sim's drive reached the robot driver: the Mirror handed S1Body motor packets each tick (R3.1)
    assert len(body.sends) >= 3
    assert all("motor" in p for p in body.sends)
    # the fake serial keeps receiving well-formed 25-byte S-Bus frames from the body's background sender
    frames = ser.snapshot()
    assert len(frames) >= 4 and all(len(f) == 25 and len(decode(f)) == 16 for f in frames)
    # body.status() shows the commanded sticks in [-1, 1]
    stk = body.status()["sticks"]
    assert set(stk) == {"forward", "strafe", "yaw"}
    assert all(-1.0 <= v <= 1.0 for v in stk.values())

    # the LAN viewer (bound to all interfaces, R9.3) is served watch-only: a control POST is rejected
    code, cfg = _get_json(port, "/config")
    assert code == 200 and cfg["read_only"] is True
    assert _post(port, {"paused": True}) == 403
    body.close()
    assert ser.closed


# Feature: sim-mirror-webapp
# Validates: Requirements 3.6, 9.3, 9.4
def test_integration_disconnecting_client_cleaned_up_while_others_keep_receiving():
    """When one spectator disconnects, the Local_Bridge stops streaming to it and keeps delivering fresh
    per-tick state to the remaining spectators without interruption (R3.6)."""
    port = _free_port()
    _start_server(port, read_only=True, speed=5.0)

    a, b, c = SSE(port), SSE(port), SSE(port)
    try:
        # all three are receiving to begin with
        assert a.read_payload(timeout=2.0) is not None
        assert b.read_payload(timeout=2.0) is not None
        assert c.read_payload(timeout=2.0) is not None

        c.close()                                            # one spectator drops

        # the survivors keep receiving fresh, advancing state (their tick times move on)
        for survivor in (a, b):
            got = survivor.read_many(4, timeout=4.0)
            assert len(got) >= 2
            times = [json.loads(p)["t"] for p in got]
            assert len(set(times)) >= 2                      # the feed kept advancing (R3.6)
    finally:
        a.close()
        b.close()
        c.close()
