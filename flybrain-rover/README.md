# FlyBrain Rover

A real fruit fly's wiring diagram, running live as spiking neurons, driving a robot that finds a person in the room. We take the male *Drosophila* connectome (Berg et al., *Cell*, Sept 2026, served on neuPrint as `male-cns:v1.0`), cut out the roughly 15,000-neuron subcircuit that runs from the eye to the descending neurons (RESULTS.md 1), and simulate every synapse as a leaky integrate-and-fire cell with the constants from Shiu et al. 2024. A camera frame becomes current on the fly's eye neurons; the descending neurons, the ones that would drive its legs, are read out as wheel commands. Nothing in between is programmed. Where we do train, it's the fly's own dopamine-gated plasticity on existing synapses, never a new network.

## How it works, in 90 seconds

camera &rarr; `brain/retina.py` (24 angular columns) &rarr; `brain/senses.py` (current onto the eye neurons) &rarr; `brain/lif.py` (batched leaky integrate-and-fire on the real wiring) &rarr; `brain/motor.py` (descending neurons &rarr; forward and turn) &rarr; wheels.

Two more pieces read the same brain but are siblings, not links in this chain: the badge (`badge/`, a 150-neuron escape circuit running on a microcontroller) and the voice agent (`scripts/talk.py`, an ElevenLabs agent that reads live brain state and presses the demo's own hotkeys). Close either one and the robot drives identically.

Two things in that chain are ours, not the connectome's, and we label them as such in code. Forward speed is read from the whole descending population, because the eye never drives the two textbook forward-walking neurons at any working gain (RESULTS.md 6). The search behaviour used when nobody is in view (`brain/explore.py`) is an internal drive we added, not something we found in the wiring. The steering, which is the actual claim, is untouched connectome.

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python data/pull_connectome.py     # neuPrint -> data/brain.npz, no account or token needed
python -m pytest tests -q          # 248 tests
python scripts/reproduce.py --quick   # 8 headline claims, about 12 s
```

Four ways to run the demo, quoted from [DEMO.md](DEMO.md):

```bash
# terminal 1: the RoboMaster SDK daemon, its own Python 3.8 process
.venv38/bin/python scripts/robot_daemon.py --conn ap

# terminal 2: the brain, driving the robot through the daemon
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --robot-daemon 127.0.0.1:9500

# no robot at all: brain and camera only, wheel commands printed instead of sent
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run

# wheels over the S1's S-Bus receiver pins through an ESP32, no SDK, no gimbal needed
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --s1-sbus /dev/tty.usbserial-XXXX --local-eye --source 0 --dry-run   # check signs
```

In the demo window: **1** baseline, **2** remove the tracking neurons, **3** restore, **4** wipe the learning, **5** restore it, **6** wiring shuffle on and off, **7** delete a quarter of the tracking population, **space** stop.

## Results

Six headline numbers; every one of them, with its full protocol, is in [RESULTS.md](RESULTS.md).

| Result | Condition |
|---|---|
| Untrained, turns toward a person: **75%** of trials | 64 arenas, 2 s, people frozen (RESULTS.md 2) |
| Untrained, reaches the person: **97.7%** of trials | 128 arenas, 20 s (RESULTS.md 2) |
| Remove the tracking neurons (LC10a, 275 cells): turn-toward falls to **0%** | 256 arenas, 2 s, people frozen (RESULTS.md 3) |
| Shuffle the wiring, same neurons and degrees, random targets: turn-toward falls to **0%** | 256 arenas, untrained (RESULTS.md 3) |
| Learning's effect: time to contact **3.23 s** vs **4.25 s** untrained, same 0.977 contact rate | stage A, 128 arenas, 20 s, seed 1000 (RESULTS.md 5) |
| Camera to wheel command: **15 ms** median | dry run, detector on GPU (RESULTS.md 7) |

The robot has never been driven on the real chassis. Every row above the latency one is arena simulation, and the latency row is a dry run against a fake or bench-tested link, not a live drive. [DEMO.md](DEMO.md) has the full two-minute script and every fallback.

## Team and roles

- **Ducks**: hardware, the RoboMaster S1, the ESP32 links, the Three.js viewer.
- **Taka**: this half of the repo, brain, senses, motor mapping, arena, training, robot bridge, the badge, sponsor tracks.
- **Max, Senthil**: build and demo support.

Ducks's full Companion project (FlyWire brain, ESP32-CAM, ESP32 body) lives on the `s1-fly-brain` branch of the same repo; we import his S1 and camera drivers rather than duplicating them.

## Citations

- Berg et al. 2026, "Sexual dimorphism in the complete connectome of the Drosophila male CNS," *Cell*, [doi.org/10.1016/j.cell.2026.08.015](https://doi.org/10.1016/j.cell.2026.08.015). Connectome data, CC-BY via [neuPrint](https://neuprint.janelia.org) (`male-cns:v1.0`).
- Shiu et al. 2024, "A leaky integrate-and-fire computational model based on the connectome of the entire adult Drosophila brain," *Nature*, [doi.org/10.1038/s41586-024-07763-9](https://doi.org/10.1038/s41586-024-07763-9). LIF constants and the neurotransmitter-to-sign rule.
- Eon Systems, [`fly-brain`](https://github.com/eonsystemspbc/fly-brain), GPL-2.0-or-later. LIF engine reference.
- [`ilyaosovskoi/connectome-pilot`](https://github.com/ilyaosovskoi/connectome-pilot), MIT. Three-factor plasticity reference.
- [`Imperol3/flybrain`](https://github.com/Imperol3/flybrain), license not stated in the repo. Looming and escape pipeline reference.
- Ultralytics, [`yolo11n`](https://github.com/ultralytics/ultralytics), AGPL-3.0 (or an Ultralytics Enterprise license). Person detector.

Also: Nern et al. 2025 and the Reiser lab eyemap documentation (lobula column coordinates for the retinotopy); Keles and Frye 2017 (LC11 small-object tuning); Ribeiro et al. 2018 and Hindmarsh Sten et al. 2021 (LC10a and courtship tracking); Rayshubskiy et al. 2020 (DNa02 steering); Bidaye et al. 2020 (DNp09 forward walking); Marin et al. 2020 (thermosensory glomeruli); neuprint-python, PyTorch, OpenCV, the DJI RoboMaster SDK, Three.js.
