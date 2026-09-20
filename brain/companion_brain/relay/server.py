"""The relay fan-out server (see the sim-mirror-webapp design, "relay/server.py -- fan-out server").

A **standalone** websocket server, deployable to a cheap public host with no dependency on the
simulation. It accepts exactly one authenticated publisher on ``/ingest`` and fans the per-tick state
out to many read-only browser spectators on ``/events``.

Three safety properties are structural here, not incidental:

- **No robot path.** This module imports nothing from ``body``/``S1Body`` or the ``Mirror``. The
  ``Relay`` holds only ``latest`` (the most recent payload) and a set of subscriber connections. No
  code path forwards anything toward a robot -- there is simply no robot to reach (R8.4, R12.7).
- **Auth on ingest only.** The publisher must present ``Authorization: Bearer <token>`` matching the
  configured token, otherwise the connection is rejected with 401 and no state is received (R12.1,
  R12.2). Subscribers are anonymous, read-only spectators (R13.1, R12.6).
- **Read-only fan-out.** Anything a subscriber sends is drained and discarded; it is never forwarded
  to the ingest side or anywhere else (R12.7).

Capacity: the relay serves at least ``max_subscribers`` (default 500) concurrent spectators and
rejects any connection past that with 503 while continuing to serve the ones already connected
(R12.4, R12.5).

Testability: the ``Relay`` decision logic (token check, capacity guard, subscriber bookkeeping, and
the broadcast fan-out) is pure and unit-testable against injected fake connections. The websockets
library is imported *lazily* inside ``run``/``serve_forever`` only, so this module imports cleanly and
the ``Relay`` class can be tested without a websockets install or a real socket -- mirroring how
``hunt_gpu/publish.py`` keeps its transport dependency lazy.
"""
from __future__ import annotations

import argparse
import os
from http import HTTPStatus

# Endpoint paths. Kept as module constants so tests and the front-end config agree.
INGEST_PATH = "/ingest"
EVENTS_PATH = "/events"

DEFAULT_MAX_SUBSCRIBERS = 500
DEFAULT_PORT = 8700


class Relay:
    """Fan-out state from a single authenticated publisher to many read-only subscribers.

    Args:
        token: the bearer token the publisher must present on ``/ingest`` (R12.1/2). When ``None`` or
            empty, ingest authentication is effectively open -- intended only for local testing; a real
            deployment always sets a token.
        max_subscribers: the capacity guard; connections past this are rejected with 503 (R12.4/5).
    """

    def __init__(self, token: str | None = None, max_subscribers: int = DEFAULT_MAX_SUBSCRIBERS):
        self.token = token
        self.max_subscribers = int(max_subscribers)
        # the most recent payload received from the publisher; sent to a subscriber on connect so it
        # renders immediately instead of waiting for the next tick. `None` until the first ingest.
        self.latest = None
        # the live set of subscriber connections. A `set` gives O(1) add/discard and dedupe.
        self.subs: set = set()
        # observability counters (never reference any robot state)
        self.received = 0
        self.rejected_ingest = 0
        self.rejected_subscribers = 0

    # ---- pure decision logic (unit-testable, no I/O) ----------------------
    def authorized(self, auth_header: str | None) -> bool:
        """True iff ``auth_header`` is a ``Bearer`` token matching the configured token (R12.1/2).

        A relay configured without a token accepts any publisher (local-testing convenience)."""
        if not self.token:
            return True
        if not auth_header:
            return False
        parts = auth_header.split(" ", 1)
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False
        return parts[1].strip() == self.token

    def at_capacity(self) -> bool:
        """True when the subscriber set has reached ``max_subscribers`` (R12.4/5)."""
        return len(self.subs) >= self.max_subscribers

    def add_subscriber(self, conn) -> None:
        self.subs.add(conn)

    def remove_subscriber(self, conn) -> None:
        self.subs.discard(conn)

    def record_ingest(self, msg) -> None:
        """Store the newest payload so it can be fanned out and replayed to fresh subscribers."""
        self.latest = msg
        self.received += 1

    async def broadcast(self, msg) -> None:
        """Send ``msg`` to every current subscriber (R12.3). A send failure just drops that one
        subscriber (its connection is dead) and never interrupts the others."""
        for conn in list(self.subs):
            try:
                await conn.send(msg)
            except Exception:
                self.subs.discard(conn)

    # ---- connection handlers (transport-agnostic; work on any conn with async send/iterate) ----
    async def handle_ingest(self, conn) -> None:
        """The single-publisher stream: store each message and fan it out (R12.1, R12.3).

        Authorization is enforced at handshake by ``process_request`` before this runs, so any
        connection reaching here is already a valid publisher."""
        async for msg in conn:
            self.record_ingest(msg)
            await self.broadcast(msg)

    async def handle_subscriber(self, conn) -> None:
        """A read-only spectator: replay the latest payload, then drain and DISCARD anything the
        spectator sends (R12.6, R12.7). Nothing a subscriber sends is ever forwarded."""
        self.add_subscriber(conn)
        try:
            if self.latest is not None:
                await conn.send(self.latest)
            async for _ in conn:
                pass  # R12.7: discard subscriber input; there is no path back to ingest/robot
        finally:
            self.remove_subscriber(conn)

    # ---- websockets glue (lazy transport) ---------------------------------
    def process_request(self, connection, request):
        """websockets handshake hook. Rejects bad publishers (401) and over-capacity subscribers (503)
        before the connection is upgraded; returns ``None`` to let a valid connection proceed."""
        path = request.path.split("?", 1)[0]
        if path == INGEST_PATH:
            auth = request.headers.get("Authorization")
            if not self.authorized(auth):
                self.rejected_ingest += 1  # R12.2: reject, receive no state
                return connection.respond(HTTPStatus.UNAUTHORIZED, "invalid or missing token\n")
            return None
        if path == EVENTS_PATH:
            if self.at_capacity():
                self.rejected_subscribers += 1  # R12.5: reject, keep serving existing subscribers
                return connection.respond(HTTPStatus.SERVICE_UNAVAILABLE, "relay at capacity\n")
            return None
        return connection.respond(HTTPStatus.NOT_FOUND, "not found\n")

    async def handler(self, connection) -> None:
        """Dispatch an accepted connection to the ingest or subscriber loop by request path."""
        path = connection.request.path.split("?", 1)[0]
        if path == INGEST_PATH:
            await self.handle_ingest(connection)
        elif path == EVENTS_PATH:
            await self.handle_subscriber(connection)
        # any other path was already rejected in process_request

    async def serve_forever(self, host: str, port: int) -> None:
        """Run the websocket server until cancelled. Imports websockets lazily so the module stays
        importable (and the pure logic testable) without the dependency installed."""
        from websockets.asyncio.server import serve as ws_serve

        async with ws_serve(self.handler, host, port, process_request=self.process_request) as server:
            print(f"[relay] listening on ws://{host}:{port}  "
                  f"(ingest={INGEST_PATH}, events={EVENTS_PATH}, max_subscribers={self.max_subscribers})",
                  flush=True)
            await server.serve_forever()


def run(host: str = "0.0.0.0", port: int = DEFAULT_PORT, token: str | None = None,
        max_subscribers: int = DEFAULT_MAX_SUBSCRIBERS) -> None:
    """Launch a standalone relay. Blocks until interrupted."""
    import asyncio

    relay = Relay(token=token, max_subscribers=max_subscribers)
    try:
        asyncio.run(relay.serve_forever(host, port))
    except KeyboardInterrupt:
        print("[relay] shutting down", flush=True)


def main(argv=None) -> None:
    """CLI entry point. The token defaults to the ``RELAY_TOKEN`` environment variable so it need not
    appear on the command line in a deployment."""
    parser = argparse.ArgumentParser(
        prog="companion-relay",
        description="Standalone sim-mirror relay: one authenticated publisher fans out to many "
                    "read-only spectators. Has no path to the robot.")
    parser.add_argument("--host", default="0.0.0.0",
                        help="interface to bind (default: 0.0.0.0, all interfaces)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"port to listen on (default: {DEFAULT_PORT})")
    parser.add_argument("--token", default=os.environ.get("RELAY_TOKEN"),
                        help="bearer token required on /ingest (default: $RELAY_TOKEN)")
    parser.add_argument("--max-subscribers", type=int, default=DEFAULT_MAX_SUBSCRIBERS,
                        help=f"spectator capacity before 503 (default: {DEFAULT_MAX_SUBSCRIBERS})")
    args = parser.parse_args(argv)
    if not args.token:
        print("[relay] WARNING: no token set (--token or $RELAY_TOKEN); ingest is unauthenticated. "
              "Set a token before deploying publicly.", flush=True)
    run(host=args.host, port=args.port, token=args.token, max_subscribers=args.max_subscribers)


if __name__ == "__main__":
    main()
