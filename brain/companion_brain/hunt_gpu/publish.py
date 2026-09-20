"""The outbound publish client (see the sim-mirror-webapp design, "hunt_gpu/publish.py -- outbound
publish client"). A daemon thread that reads the already-serialized state `box["payload"]` and pushes
it, one message per tick, over an *outbound* websocket to the relay.

Two safety properties are structural here, not incidental:

- **No robot path.** The `Publisher` is fed from the shared state `box`; it reads only `box["payload"]`
  and `box["version"]` and holds no reference to `S1Body` or the `Mirror`. The relay therefore has no
  way to influence the physical robot (R8.4).
- **Outbound only.** It *connects out* to the relay and never opens a listening socket, so the deployed
  machine exposes no inbound internet-facing port (R14.1, R14.2).

Resilience follows the design: on any failure it applies backoff and *discards* the states produced
while it was disconnected (only the newest payload for the current version is ever sent -- nothing is
queued, R11.6), records the failure, and isolates its exceptions so the mirror and the local SSE serving
carry on untouched (R11.7). The reconnect policy is a parameter; the deployed default retries
indefinitely at 5 s (R12.8), while callers can cap it at, e.g., 5 attempts for the R11.6 policy.
"""
from __future__ import annotations

import threading
import time
from collections.abc import Sequence


def _default_connect(url: str, token: str | None):
    """The real outbound connector: a synchronous websocket client (websockets.sync) so the Publisher
    stays a plain daemon thread. Returns a connection object exposing `.send(str)` and `.close()`.
    Tests inject their own `connect` and never touch a real socket."""
    from websockets.sync.client import connect as ws_connect

    headers = {"Authorization": f"Bearer {token}"} if token else None
    # outbound connection to the relay; open_timeout bounds a hung connect so backoff can kick in
    return ws_connect(url, additional_headers=headers, open_timeout=10)


class Publisher(threading.Thread):
    """Push per-tick state from the shared `box` to the relay over an outbound websocket.

    Args:
        box:   the shared state dict created in `serve` (`{"version", "payload", ...}`); read-only here.
        cond:  the `threading.Condition` notified on each tick; the Publisher waits on the version bump.
        url:   the relay ingest URL (wss://... in deployment).
        token: the bearer token presented as `Authorization: Bearer <token>` (R12.1).
        backoff: reconnect delay policy -- a constant seconds value (default 5.0, the deployed policy),
                 a sequence of per-attempt delays, or a callable `attempt -> seconds`.
        max_attempts: cap on *consecutive* reconnect attempts before giving up; None (default) retries
                 indefinitely at `backoff` (R12.8). Pass 5 for the R11.6 "up to 5 attempts" policy.
        connect: an injectable factory `connect(url, token) -> conn` (conn has `.send`/`.close`);
                 defaults to a real websocket connector so tests can supply a fake relay.
    """

    def __init__(self, box, cond, url, token=None, *, backoff=5.0, max_attempts=None,
                 connect=None, name="publisher"):
        super().__init__(daemon=True, name=name)
        self.box = box
        self.cond = cond
        self.url = url
        self.token = token
        self._backoff = backoff
        self.max_attempts = max_attempts
        self._connect = connect if connect is not None else _default_connect
        self._stopped = threading.Event()
        self._poll_timeout = 5.0                 # bound the version wait so stop() stays responsive
        # observable state (R11.6 "record an indication that publishing failed"); read via status()
        self.connected = False
        self.sent = 0
        self.failures = 0
        self.last_error: str | None = None

    # ---- reconnect policy -------------------------------------------------
    def _delay(self, attempt: int) -> float:
        """Seconds to wait before reconnect attempt `attempt` (1-based)."""
        b = self._backoff
        if callable(b):
            return max(0.0, float(b(attempt)))
        if isinstance(b, Sequence) and not isinstance(b, (str, bytes)):
            if not b:
                return 5.0
            return max(0.0, float(b[min(attempt - 1, len(b) - 1)]))
        return max(0.0, float(b))

    def _record_failure(self, exc: Exception) -> None:
        self.failures += 1
        self.last_error = f"{type(exc).__name__}: {exc}"
        print(f"[publish] outbound relay publish failed ({self.failures}x), "
              f"discarding state while disconnected: {self.last_error}", flush=True)

    # ---- lifecycle --------------------------------------------------------
    def stop(self) -> None:
        """Ask the thread to exit (also wakes it from the Condition wait)."""
        self._stopped.set()
        with self.cond:
            self.cond.notify_all()

    def status(self) -> dict:
        """A snapshot for logging / observability -- never references S1Body."""
        return {"connected": self.connected, "sent": self.sent,
                "failures": self.failures, "last_error": self.last_error}

    def run(self) -> None:
        # `seen` tracks the last version we sent. It is NOT reset on reconnect, so after a drop we resume
        # from whatever the newest payload is -- the states produced while disconnected are simply skipped
        # (discard, not queue: R11.6).
        seen = -1
        attempt = 0
        while not self._stopped.is_set():
            conn = None
            try:
                conn = self._connect(self.url, self.token)          # outbound only (R14.1)
                self.connected = True
                attempt = 0                                          # a clean connection resets the backoff
                while not self._stopped.is_set():
                    with self.cond:
                        self.cond.wait_for(
                            lambda: self.box["version"] != seen or self._stopped.is_set(),
                            timeout=self._poll_timeout,
                        )
                        if self._stopped.is_set():
                            break
                        if self.box["version"] == seen:
                            continue                                 # spurious/timeout wake, nothing new
                        seen = self.box["version"]
                        payload = self.box["payload"]                # read the serialized payload ONLY
                    # send the newest payload for this version; a bytes payload goes out as UTF-8 text so
                    # the feed stays JSON text end to end (browsers/relay read text frames).
                    conn.send(payload.decode() if isinstance(payload, (bytes, bytearray)) else payload)
                    self.sent += 1
            except Exception as exc:                                 # any failure is isolated here (R11.7)
                self.connected = False
                self._record_failure(exc)
                attempt += 1
                if self.max_attempts is not None and attempt >= self.max_attempts:
                    print(f"[publish] giving up after {attempt} attempts; local serving continues",
                          flush=True)
                    break
                # wait out the backoff, but wake immediately on stop()
                if self._stopped.wait(self._delay(attempt)):
                    break
            finally:
                self.connected = False
                if conn is not None:
                    try:
                        conn.close()
                    except Exception:
                        pass
