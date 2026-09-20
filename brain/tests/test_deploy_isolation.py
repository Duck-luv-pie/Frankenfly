"""Outbound-only + structural network-isolation tests for the sim-mirror-webapp (task 9.2).

These tests exercise the *deployed* posture: the Local_Bridge must expose no non-loopback listening
socket, its optional local viewer must bind only to ``127.0.0.1``, and the publish client must be
outbound-only. Two independent properties are asserted:

Property 8 (Outbound-only in deployed mode): for any deployed-mode startup configuration,
``main._deploy_kwargs`` maps a publish/deploy config to ``bind="127.0.0.1"`` (loopback only) unless
the operator explicitly forces ``lan``; the HTTP server bound with that address binds only the
loopback interface (never ``0.0.0.0``) while still accepting a loopback connection; and the Publisher
reaches the relay only through an injected *outbound* connect factory, opening no listening socket.

Property 4 (Robot isolation, structural): ``relay/server.py`` and ``hunt_gpu/publish.py`` import
nothing from ``body`` / ``s1`` and reference no ``S1Body`` / ``Mirror`` name -- there is no symbol in
either module through which the relay or the publish path could reach the physical robot.

The deployed-bind check binds a throwaway ``ThreadingHTTPServer`` exactly the way ``serve()`` does
(``ThreadingHTTPServer((bind, 0), Handler)``) so the loopback-only semantics are tested against the
real socket layer, on an ephemeral port, with no simulation loop. The Publisher is driven against an
in-process fake connect factory (no real socket).

Feature: sim-mirror-webapp, Property 4, Property 8
Validates: Requirements 13.5, 14.1, 14.2, 14.3, 14.4, 14.5, 8.4.
"""
import argparse
import ast
import inspect
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from companion_brain import main as cbmain
from companion_brain.hunt_gpu import publish as publish_mod
from companion_brain.hunt_gpu.publish import Publisher
from companion_brain.relay import server as relay_server


# --- helpers -------------------------------------------------------------------------------------

def _args(*, bind=None, publish=None, publish_token=None, remote=False, relay_url=None):
    """A minimal argparse.Namespace with exactly the fields `_deploy_kwargs` reads."""
    return argparse.Namespace(bind=bind, publish=publish, publish_token=publish_token,
                              remote=remote, relay_url=relay_url)


class _Quiet(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


def _serve_like(bind):
    """Bind a ThreadingHTTPServer on an ephemeral port exactly as `serve()` does, for `bind`."""
    return ThreadingHTTPServer((bind, 0), _Quiet)


# =================================================================================================
# Property 8: Outbound-only in deployed mode
# Feature: sim-mirror-webapp, Property 8
# Validates: Requirements 14.1, 14.3, 14.4
# =================================================================================================

# Property 8 requires >= 100 representative configs.
PBT = settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])

_URLS = st.sampled_from([
    "wss://relay.example/ingest",
    "wss://relay.example:8443/ingest",
    "ws://127.0.0.1:9000/ingest",
    "wss://host/relay",              # no /ingest segment
])
_RELAY = st.one_of(st.none(), st.sampled_from(["wss://r.example/events", "wss://other/feed"]))


@given(publish=_URLS, token=st.one_of(st.none(), st.text(min_size=1, max_size=12)),
       remote=st.booleans(), relay_url=_RELAY)
@PBT
def test_property8_publish_config_binds_loopback_only(publish, token, remote, relay_url):
    """For any deployed (publish-enabled) config with `--bind` left unset, `_deploy_kwargs` selects the
    loopback address 127.0.0.1 -- the deployed machine opens no non-loopback listener (R14.1/14.3)."""
    kw = cbmain._deploy_kwargs(_args(bind=None, publish=publish, publish_token=token,
                                     remote=remote, relay_url=relay_url))
    assert kw["bind"] == "127.0.0.1"
    assert kw["publish_url"] == publish            # the outbound publish URL is threaded through
    assert kw["publish_token"] == token
    assert kw["feed"] == ("remote" if remote else "local")
    # a remote page must have a subscriber URL: explicit --relay-url, else derived from --publish
    assert kw["relay_url"] == (relay_url or cbmain._relay_subscribe_url(publish))


@given(bind=st.sampled_from(["lan", "local"]),
       publish=st.one_of(st.none(), _URLS),
       remote=st.booleans())
@PBT
def test_property8_explicit_bind_is_honored(bind, publish, remote):
    """An explicit `--bind` always wins: `local` -> 127.0.0.1, `lan` -> 0.0.0.0, regardless of whether
    a publish URL is set. Deployed operators can only *widen* to LAN by asking for it explicitly."""
    kw = cbmain._deploy_kwargs(_args(bind=bind, publish=publish, remote=remote))
    assert kw["bind"] == ("127.0.0.1" if bind == "local" else "0.0.0.0")


@given(remote=st.booleans())
@PBT
def test_property8_non_deploy_defaults_to_lan(remote):
    """Without a publish URL and without an explicit bind, the default is LAN (0.0.0.0) so nearby
    devices can watch the local view -- deployed loopback-only kicks in only when publishing."""
    kw = cbmain._deploy_kwargs(_args(bind=None, publish=None, remote=remote))
    assert kw["bind"] == "0.0.0.0"


def test_property8_deployed_server_binds_loopback_not_wildcard():
    """A server bound the way `serve()` binds it with the deployed address (127.0.0.1) is bound to the
    loopback interface only -- never the 0.0.0.0 wildcard -- while still accepting a loopback client."""
    server = _serve_like("127.0.0.1")
    try:
        host, port = server.socket.getsockname()[:2]
        assert host == "127.0.0.1"                 # bound to loopback, not the 0.0.0.0 wildcard
        assert host != "0.0.0.0"
        threading.Thread(target=server.serve_forever, daemon=True).start()
        # a loopback connection is accepted ...
        with socket.create_connection(("127.0.0.1", port), timeout=2) as s:
            s.sendall(b"GET / HTTP/1.0\r\n\r\n")
            assert b"200" in s.recv(64)
    finally:
        server.shutdown()
        server.server_close()


def test_property8_lan_server_binds_wildcard_for_contrast():
    """Contrast case: the LAN bind uses the 0.0.0.0 wildcard, confirming the two modes are distinct and
    that loopback-only is a real, checkable property rather than the only possible outcome."""
    server = _serve_like("0.0.0.0")
    try:
        assert server.socket.getsockname()[0] == "0.0.0.0"
    finally:
        server.server_close()


def test_property8_publisher_is_outbound_only():
    """The Publisher reaches the relay only through its injected *outbound* connect factory: it calls
    connect(url, token) and sends the newest payload; it opens no listening socket."""
    calls = []

    class FakeConn:
        def __init__(self):
            self.sent = []

        def send(self, msg):
            self.sent.append(msg)

        def close(self):
            pass

    conn = FakeConn()

    def fake_connect(url, token):
        calls.append((url, token))     # records that we *dialed out* to the relay
        return conn

    cond = threading.Condition()
    box = {"version": 0, "payload": b'{"t":0}'}
    pub = Publisher(box, cond, "wss://relay.example/ingest", token="tok",
                    backoff=0.01, connect=fake_connect)
    pub.start()
    try:
        for v in range(1, 4):
            with cond:
                box["version"] = v
                box["payload"] = ('{"t":%d}' % v).encode()
                cond.notify_all()
            time.sleep(0.02)
    finally:
        pub.stop()
        pub.join(timeout=2)

    assert calls and calls[0] == ("wss://relay.example/ingest", "tok")   # outbound dial-out
    assert conn.sent                                                     # payloads pushed outbound
    # every frame is one of the produced payloads (incl. the initial version-0 box state) -- nothing
    # arrives inbound to be served; the publisher is a pure sender.
    produced = {'{"t":0}', '{"t":1}', '{"t":2}', '{"t":3}'}
    assert all(f in produced for f in conn.sent)
    assert '{"t":3}' in conn.sent                                        # the newest tick made it out


def test_property8_publisher_opens_no_listening_socket_structural():
    """Structural: the publish module never binds/listens/accepts and never imports a server framework
    -- it is a pure outbound client, so it cannot expose an inbound port (R14.1/14.2)."""
    tree = ast.parse(inspect.getsource(publish_mod))
    for node in ast.walk(tree):
        # no server-socket verbs anywhere in the module
        if isinstance(node, ast.Attribute):
            assert node.attr not in ("bind", "listen", "accept"), f"server-socket call: {node.attr}"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id not in ("bind", "listen", "accept")
        # no import of a listening-server framework
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not _names_a_server(alias.name), f"server framework imported: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            assert not _names_a_server(node.module or ""), f"server framework imported: {node.module}"


def _names_a_server(mod: str) -> bool:
    low = mod.lower()
    return any(tok in low for tok in ("socketserver", "http.server", "wsgiref", ".server"))


# =================================================================================================
# Property 4: Robot isolation invariant (structural, deploy-side modules)
# Feature: sim-mirror-webapp, Property 4
# Validates: Requirements 8.4
# =================================================================================================

def _forbidden_token(text: str) -> bool:
    """True if `text` names a robot surface the deploy-side modules must never import/reference."""
    low = text.lower()
    return ("s1body" in low) or ("mirror" in low) or (".s1" in low) \
        or low == "s1" or low.endswith(".body") or low == "body" or "body.s1" in low


def test_property4_deploy_modules_import_nothing_robot_structural():
    """Structural invariant: neither `hunt_gpu/publish.py` nor `relay/server.py` imports or references
    anything from body / s1 / S1Body / Mirror. There is no name in either module through which the
    relay or the publish path could reach the physical robot (R8.4)."""
    for module in (publish_mod, relay_server):
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

    # neither loaded module exposes a robot symbol as an attribute
    for module in (publish_mod, relay_server):
        assert not hasattr(module, "S1Body")
        assert not hasattr(module, "Mirror")


def test_property4_publisher_instance_holds_no_robot_reference():
    """A Publisher instance holds only outbound-publish bookkeeping (box/cond/url/token/policy/counters)
    -- no robot object, no writer that could reach an S1. Its held values expose no S-Bus surface."""
    pub = Publisher({"version": 0, "payload": b"{}"}, threading.Condition(),
                    "wss://relay.example/ingest", token=None)
    for value in vars(pub).values():
        assert not hasattr(value, "motor_to_sticks")
        assert not hasattr(value, "channels")
        assert not hasattr(value, "send_sbus")
