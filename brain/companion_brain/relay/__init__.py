"""The relay fan-out package (see the sim-mirror-webapp design, "relay/server.py -- fan-out server").

This package is a *standalone*, deployable-to-a-cheap-host component. It deliberately imports nothing
from the simulation, the body/`S1Body` driver, or the `Mirror`: the relay has no communication path,
direct or indirect, to the physical robot (R8.4, R12.7). It only receives per-tick state from a single
authenticated publisher and rebroadcasts it to many read-only browser spectators.
"""
