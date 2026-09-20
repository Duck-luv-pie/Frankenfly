"""Relay fan-out, auth, capacity, and robot-isolation tests for the sim-mirror-webapp.

The relay (`companion_brain/relay/server.py`) is a standalone websocket fan-out server: it accepts
exactly one authenticated publisher on ``/ingest`` and rebroadcasts each per-tick state to many
read-only browser spectators on ``/events``. Its transport (the ``websockets`` library) is imported
*lazily* inside ``serve_forever`` only, so the ``Relay`` decision logic is unit-testable against
in-process **fake connections** with no real socket and no ``websockets`` install.

Fakes:
- ``FakeConn`` -- an async connection: ``async def send`` records frames; async iteration
  (``__aiter__``/``__anext__``) yields a queued list of incoming messages then stops. Used to drive
  ``handle_ingest`` / ``handle_subscriber`` / ``broadcast``.
- ``TrackingConn`` -- a ``FakeConn`` that records every public attribute the relay touches, so we can
  prove behaviourally that the fan-out only ever calls ``conn.send`` (Property 4, relay side).
- ``FakeHandshakeConn`` / ``FakeRequest`` -- stand-ins for the websockets handshake hook: ``respond``
  records ``(status, text)`` and returns a non-``None`` sentinel (a rejection); ``FakeRequest`` carries
  ``.path`` and a ``.headers`` dict. Used to drive ``process_request`` (401 / 503).

Async tests are driven with ``asyncio.run`` (no ``pytest-asyncio`` dependency).

Feature: sim-mirror-webapp, Property 4
Validates: Requirements 12.2, 12.4, 12.5, 12.7, 8.4.
"""
import ast
import asyncio
import inspect
from http import HTTPStatus

from companion_brain.relay import server as relay_server
from companion_brain.relay.server import EVENTS_PATH, INGEST_PATH, Relay


# --- fakes ---------------------------------------------------------------------------------------

class FakeConn:
    """An in-process fake connection: `send` records frames; async iteration yields queued messages."""

    def __init__(self, incoming=()):
        self.incoming = list(incoming)
        self.sent = []

    async def send(self, msg):
        self.sent.append(msg)

    def __aiter__(self):
        self._it = iter(self.incoming)
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


class TrackingConn:
    """A fake connection that records every *public* attribute the relay accesses on it, so a test can
    assert the fan-out only ever reaches `conn.send` and never any other (robot-ish) surface. Internal
    bookkeeping lives under leading-underscore names, which are excluded from the access record."""

    def __init__(self, incoming=()):
        object.__setattr__(self, "_incoming", list(incoming))
        object.__setattr__(self, "_sent", [])
        object.__setattr__(self, "_accessed", set())

    def __getattribute__(self, name):
        if not name.startswith("_"):
            object.__getattribute__(self, "_accessed").add(name)
        return object.__getattribute__(self, name)

    async def send(self, msg):
        object.__getattribute__(self, "_sent").append(msg)

    def __aiter__(self):
        object.__setattr__(self, "_it", iter(object.__getattribute__(self, "_incoming")))
        return self

    async def __anext__(self):
        try:
            return next(object.__getattribute__(self, "_it"))
        except StopIteration:
            raise StopAsyncIteration


class FakeHandshakeConn:
    """Stand-in for the websockets handshake `connection`: `respond` records the rejection and returns
    a non-`None` sentinel (which, from `process_request`, means the connection is refused)."""

    def __init__(self):
        self.responses = []

    def respond(self, status, text):
        self.responses.append((status, text))
        return ("HTTP_RESPONSE", status, text)


class FakeRequest:
    """Stand-in for the websockets handshake `request`: a path and a headers dict."""

    def __init__(self, path, headers=None):
        self.path = path
        self.headers = dict(headers or {})


# --- fan-out: identical payloads to every subscriber (R12.3, R12.4) ------------------------------

def test_publish_fans_out_identical_payloads_to_all_subscribers():
    """A single publisher stream is rebroadcast to every connected subscriber: each receives the exact
    same payloads in the same tick order, and the relay records the latest and the count."""
    relay = Relay(token=None)
    subs = [FakeConn() for _ in range(4)]
    for s in subs:
        relay.add_subscriber(s)

    msgs = [b'{"t":1}', b'{"t":2}', b'{"t":3}']
    publisher = FakeConn(incoming=msgs)
    asyncio.run(relay.handle_ingest(publisher))

    for s in subs:
        assert s.sent == msgs           # identical payload, identical order, to every subscriber
    assert relay.latest == msgs[-1]
    assert relay.received == 3


def test_broadcast_drops_only_the_dead_subscriber():
    """A send failure on one subscriber drops just that one; the others still receive the payload."""
    relay = Relay(token=None)
    good_a, good_b = FakeConn(), FakeConn()

    class DeadConn(FakeConn):
        async def send(self, msg):
            raise ConnectionError("dead subscriber")

    dead = DeadConn()
    for s in (good_a, dead, good_b):
        relay.add_subscriber(s)

    asyncio.run(relay.broadcast(b'{"t":9}'))

    assert good_a.sent == [b'{"t":9}'] and good_b.sent == [b'{"t":9}']
    assert dead not in relay.subs       # the failed subscriber was discarded
    assert good_a in relay.subs and good_b in relay.subs


# --- auth: unauthenticated ingest rejected with 401 (R12.2) --------------------------------------

def test_unauthenticated_ingest_rejected_with_401():
    """A publisher on /ingest without a valid Bearer token is rejected with 401 at handshake, and no
    per-tick state is received; a valid token is accepted (process_request returns None)."""
    relay = Relay(token="secret")

    # missing Authorization header -> 401
    c_missing = FakeHandshakeConn()
    r = relay.process_request(c_missing, FakeRequest(INGEST_PATH))
    assert r is not None                                  # non-None == rejected
    assert c_missing.responses[-1][0] == HTTPStatus.UNAUTHORIZED

    # wrong token -> 401
    c_wrong = FakeHandshakeConn()
    r = relay.process_request(c_wrong, FakeRequest(INGEST_PATH, {"Authorization": "Bearer nope"}))
    assert r is not None
    assert c_wrong.responses[-1][0] == HTTPStatus.UNAUTHORIZED

    # malformed scheme -> 401
    c_bad = FakeHandshakeConn()
    r = relay.process_request(c_bad, FakeRequest(INGEST_PATH, {"Authorization": "secret"}))
    assert r is not None
    assert c_bad.responses[-1][0] == HTTPStatus.UNAUTHORIZED

    # valid token -> accepted (None), nothing rejected
    c_ok = FakeHandshakeConn()
    r = relay.process_request(c_ok, FakeRequest(INGEST_PATH, {"Authorization": "Bearer secret"}))
    assert r is None
    assert c_ok.responses == []

    # rejections counted; no state ever received while rejecting
    assert relay.rejected_ingest == 3
    assert relay.latest is None and relay.received == 0


# --- capacity: subscriber past max rejected with 503, existing keep receiving (R12.4, R12.5) -----

def test_subscriber_past_capacity_rejected_with_503_existing_keep_receiving():
    """A spectator connecting past max_subscribers is rejected with 503, while the already-connected
    spectators keep receiving broadcasts."""
    relay = Relay(token=None, max_subscribers=3)
    existing = [FakeConn() for _ in range(3)]
    for s in existing:
        relay.add_subscriber(s)
    assert relay.at_capacity()

    # the over-capacity connection is refused with 503
    over = FakeHandshakeConn()
    r = relay.process_request(over, FakeRequest(EVENTS_PATH))
    assert r is not None
    assert over.responses[-1][0] == HTTPStatus.SERVICE_UNAVAILABLE
    assert relay.rejected_subscribers == 1

    # the existing subscribers are unaffected and still receive the fan-out
    asyncio.run(relay.broadcast(b'{"still":"serving"}'))
    for s in existing:
        assert s.sent == [b'{"still":"serving"}']


def test_subscriber_under_capacity_is_accepted():
    """Below max_subscribers, an /events connection is accepted (process_request returns None)."""
    relay = Relay(token=None, max_subscribers=3)
    relay.add_subscriber(FakeConn())
    under = FakeHandshakeConn()
    assert relay.process_request(under, FakeRequest(EVENTS_PATH)) is None
    assert under.responses == []
    assert relay.rejected_subscribers == 0


# --- isolation: subscriber messages never forwarded to ingest (R12.7, R8.4) ----------------------

def test_subscriber_messages_never_forwarded_to_ingest():
    """Anything a spectator sends is drained and discarded: it never becomes ingest state, is never
    rebroadcast, and there is no path back toward the publisher / robot."""
    relay = Relay(token=None)
    other = FakeConn()
    relay.add_subscriber(other)              # a second spectator that must never see forwarded input

    spammer = FakeConn(incoming=[b'{"motor":{"forward":1}}', b'{"sticks":[1,2,3,4]}', b'hi'])
    asyncio.run(relay.handle_subscriber(spammer))

    # subscriber input did not become state, was not counted as ingest, and was not fanned out
    assert relay.latest is None
    assert relay.received == 0
    assert other.sent == []                  # nothing the spammer sent reached another subscriber
    assert spammer.sent == []                # no latest to replay, so the spammer got nothing back
    assert spammer not in relay.subs         # cleaned up on disconnect


def test_subscriber_gets_latest_replay_then_input_is_discarded():
    """On connect a spectator is replayed the latest payload; its own subsequent messages are then
    discarded (not counted, not rebroadcast)."""
    relay = Relay(token=None)
    relay.record_ingest(b'{"replay":1}')     # a publisher tick arrived before this spectator connected
    conn = FakeConn(incoming=[b'ignored-input'])

    asyncio.run(relay.handle_subscriber(conn))

    assert conn.sent == [b'{"replay":1}']    # only the replayed latest, never an echo of its own input
    assert relay.received == 1               # subscriber input did not bump the ingest counter
    assert relay.latest == b'{"replay":1}'


# --- Property 4 (relay side): robot isolation invariant ------------------------------------------
# Feature: sim-mirror-webapp, Property 4
# Validates: Requirements 8.4, 12.7

def _forbidden_token(text):
    """True if `text` names a robot surface the relay must never import/reference."""
    low = text.lower()
    return ("body" in low) or ("s1body" in low) or ("mirror" in low) or (".s1" in low) or low == "s1"


def test_property4_relay_imports_nothing_robot_structural():
    """Structural invariant: neither relay/server.py nor the relay package imports or references
    anything from body / s1 / S1Body / Mirror. There is simply no name in the module through which a
    robot could be reached (R8.4, R12.7)."""
    for module in (relay_server, __import__("companion_brain.relay", fromlist=["x"])):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert not _forbidden_token(alias.name), f"forbidden import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert not _forbidden_token(mod), f"forbidden from-import module: {mod}"
                for alias in node.names:
                    assert alias.name not in ("S1Body", "Mirror"), f"forbidden imported name: {alias.name}"
            elif isinstance(node, ast.Name):
                assert node.id not in ("S1Body", "Mirror"), f"forbidden name reference: {node.id}"
            elif isinstance(node, ast.Attribute):
                assert node.attr not in ("S1Body", "Mirror"), f"forbidden attribute reference: {node.attr}"

    # the loaded module exposes no robot symbol as an attribute
    assert not hasattr(relay_server, "S1Body")
    assert not hasattr(relay_server, "Mirror")


def test_property4_relay_holds_no_robot_reference():
    """A Relay instance holds only fan-out bookkeeping (token, capacity, latest payload, subscriber
    set, counters) -- no robot object, no writer that could reach an S1."""
    relay = Relay(token=None)
    assert set(vars(relay)) == {
        "token", "max_subscribers", "latest", "subs",
        "received", "rejected_ingest", "rejected_subscribers",
    }
    # none of the held values is a robot-ish object with a stick/S-Bus surface
    for value in vars(relay).values():
        assert not hasattr(value, "motor_to_sticks")
        assert not hasattr(value, "channels")


def test_property4_broadcast_only_calls_send_behavioural():
    """Behavioural invariant: `broadcast` reaches its subscribers only through `conn.send` -- it never
    touches any other attribute on the connection (nothing robot-ish, no back-channel)."""
    relay = Relay(token=None)
    subs = [TrackingConn() for _ in range(3)]
    for s in subs:
        relay.add_subscriber(s)

    asyncio.run(relay.broadcast(b'{"t":1}'))

    for s in subs:
        assert s._accessed <= {"send"}, f"broadcast touched more than send: {s._accessed}"


def test_property4_handle_subscriber_only_calls_send_behavioural():
    """Behavioural invariant: serving a spectator (replay latest, then drain-and-discard its input)
    reaches the connection only through `conn.send` and async iteration -- never any robot surface."""
    relay = Relay(token=None)
    relay.record_ingest(b'{"latest":1}')
    conn = TrackingConn(incoming=[b'a', b'b', b'c'])

    asyncio.run(relay.handle_subscriber(conn))

    assert conn._accessed <= {"send"}, f"handle_subscriber touched more than send: {conn._accessed}"
    assert conn._sent == [b'{"latest":1}']   # only the replayed latest; its own input was discarded
