# Two modes: chase vs. mirror

The fly's brain and the RoboMaster S1 can be wired together two ways. Both end at the
same place — `S1Body.send()` streaming S-Bus stick frames to the S1 at ~70 Hz — they
just differ in what *produces* the drive.

```
Mode 1 (chase):   camera + PIR ─► brain ─► drive ─► Mirror/decode ─► S1Body ─► S-Bus ─► S1
Mode 2 (mirror):  simulation   ─► fly drive ─► Mirror ────────────► S1Body ─► S-Bus ─► S1
                                     └─► state feed ─► web viewer (spectators, read-only)
```

## Mode 1 — robot chases real humans
The robot senses the real world and reacts to it.

- Real camera + PIR feed the connectome/trained brain (`companion run`).
- The brain emits a `drive = [forward, turn]`; the body layer turns it into S-Bus sticks.
- The robot pursues people it actually sees.

Needs: S1 + Raspberry Pi + S-Bus wiring **and** the camera/PIR sensing.

## Mode 2 — simulated fly mirrored onto the robot
The robot is a puppet that reproduces the simulated fly's movement.

- The GPU sim runs one room of the hunt arena in real time (`companion hunt-gpu --watch`).
- Each tick, the fly's `drive` goes through the **Mirror** → S-Bus → S1, so the physical
  robot mirrors the sim fly. The robot senses nothing itself.
- The same per-tick state is served as a web page (3-D room, senses, drive, neurons).
- Optionally the state is published outward to a **relay** so remote people can watch a
  deployed website.

Needs: S1 + Raspberry Pi + S-Bus wiring. No camera/PIR. The web/relay part is watch-only
and is **not** required to drive the robot.

## What already works (this branch)
- **Mirror** (`hunt_gpu/mirror.py`): brain-agnostic drive→stick conversion — clamps to
  `[-1,1]`, maps `[0,0]` to neutral, applies calibrated constants, and caps forward/yaw to
  configurable limits (`--v-max`, `--w-max`, `--sign-yaw`). No S1 configured → sends nothing.
- **Safety** (`body/s1.py`, `hunt_gpu/viewer.py`): E-stop, paused-centering, a configurable
  failsafe timeout (sticks center if the tick stream stops), and out-of-range command rejection.
- **Read-only web viewer** (`ui/hunt_gpu.html`, `serve`/`Handler`): spectators get the view but
  no controls (control POSTs return 403); `/config` advertises read-only + feed source; no
  inbound web request can reach `S1Body`.
- **Deploy path** (`hunt_gpu/publish.py`, `relay/server.py`): an outbound-only publisher pushes
  state to a standalone relay (token auth, capacity guard) that fans out to many browsers. The
  relay imports nothing from the robot code — it structurally cannot move the S1.
- **Tests**: Properties 1–8 covered with a fake serial and a fake relay (no hardware needed).

## Key safety invariant
In Mode 2 the robot is always driven by the **local** machine running the sim (the Pi), never
by the public website. The relay/website is a viewer only and is isolated from the robot by
construction.

## Running it
```bash
# from brain/ ,  uv at ~/.local/bin/uv

# Mode 2, sim + viewer only (any machine, no hardware):
uv run companion hunt-gpu --watch --open --no-public

# Mode 2 on the robot (run on the Pi wired to the S1):
uv run companion hunt-gpu --watch --s1 /dev/ttyAMA0 --v-max 0.5 --w-max 45
```
Prop the S1 up (wheels off the ground) for the first hardware run to check direction and
scaling before it can drive into anything; flip `--sign-yaw inverted` if it turns the wrong way.
