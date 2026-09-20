# Runbook

Everything in this repo, what starts it, and how you know it worked. If you are picking this up cold,
read this and nothing else.

There are **two brains** in here and that is on purpose:

| | `brain/` (Ducks) | `flybrain-rover/` (Taka) |
|---|---|---|
| runs on | the Raspberry Pi, as a systemd service | a laptop |
| detector | NanoDet ONNX via OpenCV | YOLO11n via ultralytics |
| brain | 8,503-neuron hunter subcircuit | 15,000-neuron male-CNS connectome, 2.33 M synapses |
| job | **the one wired to the robot** | **the one with the science**: lesions, measured latencies, the viewer |

They share `companion_brain.body.s1` (the S-Bus driver) and `companion_brain.senses.camera`.
`flybrain-rover/scripts/robot_bridge.py` imports both at runtime, so **changing `S1Body`,
`channels()` or `motor_to_sticks()` breaks the other stack silently**. There is a check for it below.

---

## 1. The simulation and the viewer (no hardware, start here)

```bash
cd brain
uv run companion hunt-gpu --watch --open --no-public
```
Serves on **http://localhost:8601**. `--no-public` lets you drive from the page; without it every
viewer is read-only and POSTs return 403.

**Our brain in the same viewer**, which is what the lesion demo uses:
```bash
cd flybrain-rover
.venv/bin/python scripts/viz_adapter.py --ui ../brain/companion_brain/ui
```
The LOBOTOMIZE and STOP buttons in the bottom left are added as the page is served, so Ducks's file is
never edited. They grey out for spectators.

**Working looks like:** neurons visible as a grey field with spikes lighting up, a fly that turns
toward the people, and the two buttons responding.

## 2. The judge-facing demo, dry run

```bash
cd flybrain-rover
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run
```
`--dry-run` means no wheel command leaves the laptop. Hotkeys are in `docs/controls.md`. The three
that matter:

- **`8`** lobotomize — forward collapses and it starts sweeping
- **`9`** wake up
- **`0`** stop / resume the wheels, brain still running

**Working looks like** this, and these are measured, not impressions:

| | forward | turn |
|---|---|---|
| intact, person in view | +0.278 | locked on |
| lobotomized | +0.039 | swinging 1.30, alternating sides |
| stopped | 0.000 | 0.000 |

## 3. The robot

**Do not skip the bench test.** Prop the S1 up with the wheels off the ground for any first run, and
check the turn direction before it can drive into anything. `--sign-yaw inverted` flips it.

### 3a. Neutral check, no brain
```bash
cd brain
uv run python ../tools/s1_sbus.py --port /dev/ttyUSB0 --neutral
```

### 3b. Mirror mode — the sim drives the robot, no camera involved
The lowest-risk way to get the brain onto the chassis, because nothing has to be detected.
```bash
cd brain
uv run companion hunt-gpu --watch --s1 /dev/ttyAMA0 --v-max 0.5 --w-max 45
```

### 3c. Chase mode — the real thing, on the Pi
From the Mac, on the Pi's `companion` hotspot:
```bash
tools/pi_push.sh http://10.42.0.157:4747/video on udp     # [CAM_URL] [on|off] [udp|usb|serial]
tools/pi_check.sh                                          # one-shot diagnosis if it misbehaves
```
Watch it at **http://10.42.0.1:8601**. `S1_MODE` picks the S-Bus path: `udp` through the body ESP32 over
Wi-Fi, `usb` over its cable at 460800, `serial` straight out of a port.

### 3d. Our brain on the chassis
```bash
# on the Pi
python3 flybrain-rover/scripts/pi_s1_relay.py            # UDP :4310 -> S1Body
# on the laptop
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --s1-udp <pi>:4310 --source 0
```
**Never run this without doing 3a first.** The first key to press is `0`, because its correct
behaviour is the wheels not moving.

## 4. The badge

```bash
cd flybrain-rover
.venv/bin/python badge/export_badge_circuit.py
.venv/bin/python badge/build_app.py
```
Then paste `badge/flybadge_app.lua` into the badge IDE. Three traps that have each cost real time:

- battery switch **off** first, and use a **data** cable, not a charge-only one
- the port is named **USB JTAG/serial debug unit**
- if the app opens and jumps straight back to the launcher, press **Reboot** in the IDE, not `reload` —
  `heap_kb` is only read on a reboot
- **never enable `badge.radio`.** It panics the device with this circuit loaded. See
  `badge/SDK_NOTES.md` for the console trace.

## 5. The website

`docs/index.html` is the FlyBadge install page. It goes live via GitHub Pages: Settings → Pages →
Deploy from a branch → `/docs`. DNS and the apex A records are in `flybrain-rover/site/README.md`.
Needs repo admin, which is Ducks.

The live viewer stays local this weekend. Senthil's relay (`companion-relay`) is merged and tested but
not deployed, and `websockets` is an optional extra (`uv sync --extra relay`) if anyone picks it up.

---

## Before you push anything

```bash
# our tests
cd flybrain-rover && .venv/bin/python -m pytest tests -q          # 219 passed

# Ducks's and Senthil's
cd brain && uv run pytest -q                                      # 76 passed, 11 skipped
                                                                  # (skips are torch/connectome/numba, all environmental)

# the cross-stack coupling: robot_bridge imports this at runtime
cd flybrain-rover && .venv/bin/python -c "
import sys, importlib; sys.path.insert(0, '../brain')
m = importlib.import_module('companion_brain.body.s1')
print(m.motor_to_sticks({'forward': .5, 'turn': .2}, m.DEFAULTS))"
```

**Syncing our repo into this one** is a script now, not a hand-copy:
```bash
cd flybrain-rover
bash scripts/export_to_hunting_fly.sh --dry-run
bash scripts/export_to_hunting_fly.sh --into ../hunting-fly-integration
```
It refuses to run if anything secret-shaped is in the payload, and it will tell you that
`checkpoints/demo_brain.pt` needs `git add -f`.

## Branches

`main` is an **orphan history** — no common ancestor with anything else. It was only ever a
publication branch holding a hand-copy of `flybrain-rover/`. Do not merge it; that path is 23 add/add
conflicts over files nobody edited twice. `integration` is the real trunk: `s1-fly-brain` plus
Senthil's `sim-mirror-webapp` plus the grafted `flybrain-rover/`. Every pre-merge tip is tagged
`pre-merge/*`.
