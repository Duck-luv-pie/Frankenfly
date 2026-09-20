"""Publisher isolation + resilience tests for the sim-mirror-webapp (fake relay, no real socket).

The outbound publish client (`hunt_gpu/publish.py`) is a daemon thread fed from the shared state
`box{version, payload}` + `Condition`. It opens an *outbound* connection to the relay and pushes the
newest per-tick payload. These tests drive it against an in-process **fake relay** (a connection
object exposing `.send`/`.close` that records frames, supplied via the injectable `connect=` factory) so
nothing touches a real socket.

Property 7 (Publish isolation): for any sequence of state-box updates the Publisher reads only
`box["payload"]` (and the `box["version"]` counter it waits on) and holds no reference to `S1Body`; and
for any publish failure the mirror path and local SSE serving are unaffected. We assert the first half
with a `RecordingBox` (a dict subclass that records every key the Publisher reads/writes, plus an
`S1Body`-like tripwire value that must never be read) and by checking every transmitted frame is exactly
one of the produced payloads in order. We assert the second half by running an independent concurrent
counter (standing in for the S1Body send loop / local serving) against a failing fake relay and showing
it keeps advancing while the Publisher isolates its exceptions and records the failure.

The unit tests cover: frames published in tick order; a down relay causing discard-not-queue (states
produced while disconnected are dropped -- after reconnect only the newest version is sent, not a
backlog); reconnect/backoff and failure recording (`status().failures` increments, `last_error` set);
giving up after `max_attempts`; and a concurrent fake S1Body left untouched by publisher failures.

Feature: sim-mirror-webapp, Property 7
Validates: Requirements 11.6, 11.7, 8.4, 12.8.
"""
import threading
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from companion_brain.hunt_gpu.publish import Publisher

# Property 7 requires >= 100 iterations; threads + sleeps mean we relax Hypothesis' timing checks.
PBT = settings(max_examples=120, deadline=None,
               suppress_health_check=[HealthCheck.too_slow])

FAST = dict(backoff=0.01)   # short backoff so reconnect tests stay quick


# --- fakes ---------------------------------------------------------------------------------------

class FakeConn:
    """An in-process fake relay connection: records the frames it is sent, never opens a socket."""

    def __init__(self, relay):
        self.relay = relay
        self.closed = False

    def send(self, msg):
        if self.relay.fail_send:
            raise ConnectionError("fake relay send failed")
        with self.relay.lock:
            self.relay.frames.append(msg)

    def close(self):
        self.closed = True


class FakeRelay:
    """A controllable fake relay used as the Publisher's `connect(url, token) -> conn` factory.

    `block_connect` / `fail_connects` make `connect` raise (relay unreachable); `fail_send` makes an
    established connection raise on `.send`. All frames the Publisher transmits land in `frames`."""

    def __init__(self, fail_connects=0, fail_send=False, block_connect=False):
        self.frames = []
        self.conns = []
        self.connect_calls = 0
        self.fail_connects = fail_connects
        self.fail_send = fail_send
        self.block_connect = block_connect
        self.lock = threading.Lock()

    def connect(self, url, token):
        with self.lock:
            self.connect_calls += 1
            n = self.connect_calls
            blocked = self.block_connect
        if blocked or n <= self.fail_connects:
            raise ConnectionError(f"fake relay unreachable (attempt {n})")
        conn = FakeConn(self)
        with self.lock:
            self.conns.append(conn)
        return conn

    def frame_list(self):
        with self.lock:
            return list(self.frames)


class FakeS1Body:
    """An independent, concurrent 'robot sender' loop. The Publisher must never touch it; its counter
    stands in for the mirror path / local serving that a publish failure must leave unaffected."""

    def __init__(self, interval=0.002):
        self.sent = 0
        self._interval = interval
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._t.start()

    def _loop(self):
        while not self._stop.is_set():
            self.sent += 1
            time.sleep(self._interval)

    def stop(self):
        self._stop.set()
        self._t.join(timeout=1.0)

    @property
    def alive(self):
        return self._t.is_alive()


class RecordingBox(dict):
    """A state box that records every key the Publisher reads (`__getitem__`) or mutates
    (`__setitem__`). Tests bump the box via `raw_set` (bypassing the recorder) so `reads`/`writes`
    reflect the Publisher alone."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.reads = []
        self.writes = []
        self._lock = threading.Lock()

    def __getitem__(self, key):
        with self._lock:
            self.reads.append(key)
        return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            self.writes.append(key)
        super().__setitem__(key, value)

    def raw_set(self, key, value):
        super().__setitem__(key, value)

    def read_keys(self):
        with self._lock:
            return set(self.reads)


# --- helpers -------------------------------------------------------------------------------------

def wait_until(pred, timeout=2.0, interval=0.005):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def new_box(**extra):
    box = {"version": 0, "payload": b"{}"}
    box.update(extra)
    return box


def bump(box, cond, payload, *, raw=False):
    """Publish a new tick into the box (mirrors serve()'s publish step) and wake the Publisher."""
    with cond:
        version = box["version"] + 1
        if raw:
            box.raw_set("version", version)
            box.raw_set("payload", payload)
        else:
            box["version"] = version
            box["payload"] = payload
        cond.notify_all()


def is_subsequence(sub, whole):
    it = iter(whole)
    return all(item in it for item in sub)


def run_publisher(box, cond, relay, **kw):
    pub = Publisher(box, cond, "wss://relay.example/ingest", token="t",
                    connect=relay.connect, **{**FAST, **kw})
    pub.start()
    return pub


# --- Property 7: Publish isolation ---------------------------------------------------------------

# Feature: sim-mirror-webapp, Property 7: Publish isolation
# Validates: Requirements 11.7, 8.4
@PBT
@given(
    values=st.lists(st.integers(min_value=0, max_value=99999), min_size=0, max_size=5, unique=True),
    relay_down=st.booleans(),
)
def test_property7_publish_isolation(values, relay_down):
    """For any state-box update sequence: the Publisher reads only `payload`/`version` and never
    mutates the box nor touches the S1Body-like tripwire; every transmitted frame is exactly one of
    the produced payloads, in order; and any publish failure leaves an independent concurrent sender
    (the mirror path / local serving) advancing while the Publisher isolates and records the failure."""
    payloads = [b'{"tick":%d}' % v for v in values]
    tripwire = object()   # an S1Body-like reference that must never be read
    # the Publisher starts from seen=-1, so on connect it first transmits whatever version-0 payload
    # is present; give it a distinct initial value and treat it as the head of the transmittable stream.
    initial = b'{"tick":-1}'
    box = RecordingBox(version=0, payload=initial, s1body=tripwire)
    cond = threading.Condition()

    relay = FakeRelay(block_connect=relay_down)
    body = FakeS1Body()
    body.start()
    body_start = body.sent

    pub = run_publisher(box, cond, relay, max_attempts=None)
    try:
        for p in payloads:
            bump(box, cond, p, raw=True)
            time.sleep(0.01)      # give the daemon a chance to transmit this tick
        time.sleep(0.05)          # let the last frame(s) settle
    finally:
        pub.stop()
        pub.join(timeout=2.0)
        body.stop()

    # --- isolation of the box: only payload/version read, nothing written, S1Body never touched ---
    assert box.read_keys() <= {"payload", "version"}, f"unexpected box reads: {box.read_keys()}"
    assert box.writes == [], f"Publisher mutated the box: {box.writes}"
    assert box["s1body"] is tripwire  # untouched

    frames = relay.frame_list()
    # the universe of what could legitimately be transmitted: the initial payload then the produced
    # ticks, in order. Discard-while-connected means frames are a subsequence of this -- never anything
    # else, never reordered.
    transmittable = [initial.decode()] + [p.decode() for p in payloads]
    if relay_down:
        # a publish failure transmits nothing and is recorded; serving is unaffected
        assert frames == []
        assert pub.status()["failures"] >= 1
        assert pub.status()["last_error"] is not None
    else:
        # every transmitted frame is exactly one produced payload, in order (discard, never reorder)
        assert all(f in transmittable for f in frames), f"frame not among payloads: {frames} vs {transmittable}"
        assert is_subsequence(frames, transmittable), f"frames out of order: {frames} vs {transmittable}"

    # the independent concurrent sender kept advancing regardless of publish outcome (R11.7 / R8.4)
    assert body.sent > body_start


# --- unit tests ----------------------------------------------------------------------------------

# Validates: Requirements 11.5, 11.7
def test_frames_published_in_tick_order():
    """With a healthy relay, per-tick payloads are transmitted in tick order."""
    box = new_box(payload=b"init")
    cond = threading.Condition()
    relay = FakeRelay()
    pub = run_publisher(box, cond, relay)
    try:
        # the Publisher transmits the initial version-0 payload on connect; wait for it, then feed ticks
        assert wait_until(lambda: pub.sent >= 1)
        for p in (b"a", b"b", b"c"):
            bump(box, cond, p)
            time.sleep(0.02)
        assert wait_until(lambda: pub.sent >= 4)
    finally:
        pub.stop()
        pub.join(timeout=2.0)

    assert relay.frame_list() == ["init", "a", "b", "c"]
    assert pub.status()["connected"] is False  # closed on stop
    assert pub.status()["failures"] == 0


# Validates: Requirements 11.6 (discard, do not queue)
def test_down_relay_discards_not_queue():
    """States produced while the relay is down are dropped; after reconnect only the newest version is
    sent -- there is no backlog replay."""
    box = new_box()
    cond = threading.Condition()
    relay = FakeRelay(block_connect=True)
    pub = run_publisher(box, cond, relay)
    try:
        # five ticks produced while disconnected -- all but the newest must be discarded
        for i in range(1, 6):
            bump(box, cond, b'{"v":%d}' % i)
            time.sleep(0.01)
        assert wait_until(lambda: pub.status()["failures"] >= 1)  # confirm it really failed to connect
        relay.block_connect = False                               # relay comes back
        assert wait_until(lambda: pub.sent >= 1)
        time.sleep(0.05)                                          # allow any (wrongly) queued frames
    finally:
        pub.stop()
        pub.join(timeout=2.0)

    frames = relay.frame_list()
    assert frames == ['{"v":5}'], f"expected only the newest tick, got a backlog: {frames}"


# Validates: Requirements 11.6, 12.8 (reconnect + failure recording)
def test_reconnect_backoff_and_failure_recording():
    """A relay that refuses the first few connects is retried; each failure is recorded in status()
    and the newest state is delivered once the relay accepts."""
    box = new_box()
    cond = threading.Condition()
    relay = FakeRelay(fail_connects=3)
    pub = run_publisher(box, cond, relay, max_attempts=None)
    try:
        bump(box, cond, b"hello")
        assert wait_until(lambda: pub.sent >= 1, timeout=3.0)
    finally:
        pub.stop()
        pub.join(timeout=2.0)

    status = pub.status()
    assert relay.connect_calls >= 4          # 3 refused + at least 1 accepted
    assert status["failures"] >= 3           # each refused connect recorded
    assert status["last_error"] is not None
    assert "ConnectionError" in status["last_error"]
    assert relay.frame_list() == ["hello"]


# Validates: Requirements 11.6 (up to N attempts, then give up; serving continues)
def test_gives_up_after_max_attempts():
    """When the relay never accepts, the Publisher stops after max_attempts consecutive failures rather
    than looping forever, having recorded each failure."""
    box = new_box()
    cond = threading.Condition()
    relay = FakeRelay(block_connect=True)
    pub = run_publisher(box, cond, relay, max_attempts=3)
    try:
        bump(box, cond, b"x")
        assert wait_until(lambda: not pub.is_alive(), timeout=3.0), "Publisher did not give up"
    finally:
        pub.stop()
        pub.join(timeout=2.0)

    assert pub.status()["failures"] == 3
    assert pub.status()["last_error"] is not None


# Validates: Requirements 11.7, 8.4 (concurrent S1Body unaffected by publish failures)
def test_concurrent_s1body_unaffected_by_publish_failures():
    """An independent concurrent S1Body send loop keeps running (and the Publisher holds no reference
    to it) even while the relay is unreachable and the Publisher is failing/backing off."""
    box = new_box()
    cond = threading.Condition()
    relay = FakeRelay(block_connect=True)
    body = FakeS1Body()
    body.start()
    start = body.sent

    pub = run_publisher(box, cond, relay, max_attempts=None)
    try:
        bump(box, cond, b"data")
        assert wait_until(lambda: pub.status()["failures"] >= 2, timeout=3.0)
        mid = body.sent
        assert wait_until(lambda: body.sent > mid, timeout=1.0)  # still advancing
        assert body.alive and pub.is_alive()
    finally:
        pub.stop()
        pub.join(timeout=2.0)
        body.stop()

    assert body.sent > start
    assert pub.status()["failures"] >= 2


# Validates: Requirements 11.7 (send failure isolated, recorded, serving continues)
def test_send_failure_is_isolated_and_recorded():
    """If an established connection raises on send, the Publisher records the failure and keeps running
    (reconnects) rather than crashing the thread."""
    box = new_box()
    cond = threading.Condition()
    relay = FakeRelay(fail_send=True)
    pub = run_publisher(box, cond, relay, max_attempts=None)
    try:
        bump(box, cond, b"boom")
        assert wait_until(lambda: pub.status()["failures"] >= 1)
        assert pub.is_alive()
    finally:
        pub.stop()
        pub.join(timeout=2.0)

    assert relay.frame_list() == []
    assert pub.status()["last_error"] is not None
