# Companion

A desk robot whose behavior comes from the real fruit-fly brain.

The [FlyWire](https://flywire.ai) connectome (Princeton + Google, release v783) maps all
139,255 neurons and ~2.7 M connections of an adult *Drosophila* brain. Shiu et al. (Nature, 2024)
showed that running this wiring diagram as a leaky integrate-and-fire (LIF) network predicts
real fly behavior. Companion runs that network on a Mac, feeds it what a fly's senses would report
from a camera and a motion sensor, and expresses the fly's motor commands through two round
displays (eyes) and a speaker.

```
ESP32-CAM ──MJPEG /stream (WiFi)──▶ Mac brain ──UDP JSON 20 Hz──▶ ESP32 DevKit "body"
                                     ▲    │                         ├─ 2× GC9A01 eyes (SPI)
   PIR ──GPIO──▶ body ──UDP {pir}────┘    │                         ├─ DFPlayer (UART2)
                                          └─ --dry-body (no robot)  └─ arm stub (future)
```

| Directory | What it is |
|---|---|
| `brain/` | Python: connectome download, circuit pruning, real-time LIF simulation, camera → fly-vision features, descending-neuron readout, UDP link to the body |
| `firmware/body/` | PlatformIO project for the ESP32 DevKit: eyes, DFPlayer audio, PIR, WiFi link |
| `firmware/cam/` | PlatformIO project for the ESP32-CAM: MJPEG stream server |
| `firmware/sbus-bridge/` | PlatformIO project: a spare ESP32 as the S-Bus inverter between the Pi and a RoboMaster S1 |
| `docs/` | [architecture](docs/architecture.md), [neuron map](docs/neuron-map.md), [wiring](docs/wiring.md) |
| `hardware/` | [bill of materials](hardware/bom.md) |
| `tools/` | `make_sounds.py` synthesizes the fly sounds for the SD card |
| `brain/companion_brain/ui/` | live dashboard; `fly.js` is the Fly / Body Lab procedural *Drosophila* rig (36 joints), `vendor/` holds Three.js r180 (MIT) |
| `assets/sounds/` | SD card layout for the DFPlayer |

## Quick start

Brain (Mac, needs Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/)):

```sh
cd brain
uv sync                                   # creates .venv with dependencies
uv run companion download                 # ~135 MB of connectome + annotations
uv run companion prune                    # builds the real-time circuit (cached)
uv run companion bench                    # checks the circuit runs faster than real time
uv run companion test-gf                  # looming neurons → Giant Fiber escape sanity test
uv run companion experiment                              # two-odor conditioning, headless neural readout (--no-learn = control)
uv run companion batch --runs 6 --control --save         # 6 flies + 6 controls through the arena assay in parallel; saves the best synapses
uv run companion hunt --episodes 6                       # hunt arena: the fly as a vehicle among walking humans (heat + vision)
uv run companion hunt --train --generations 20           # CMA-ES + mushroom-body learning on all cores -> data/cache/hunter.npz
uv sync --extra gpu && uv run companion hunt-gpu --train  # the hunt on a GPU: thousands of rooms as tensors, PPO -> data/cache/hunter_gpu.pt
uv run companion run --sim-camera 0 --dry-body --open   # no robot yet: Mac webcam + live dashboard
uv run companion run                                      # ESP32-CAM + real robot (dashboard too)
```

The live brain adds a spontaneous-activity floor, spike-frequency adaptation and an inhibitory
closure to the reference model so the fly is never silent (see `docs/architecture.md`); the
fly in the dashboard is moved only by continuous motor channels decoded from descending neurons.

Every `run` serves a live dashboard at <http://localhost:8600> (`--open` launches it, `--no-ui`
disables it). The dashboard is the fly's world: a table with a grape (smell, sugar), a flower
(pollen → grooming), a water drop (humidity), a neighbour fly, and a window on which your webcam
plays. The fly sees this world through its own eyes (an 80×60 retina rendered from its head is
what the brain's optic lobe gets), walks, flies, feeds and grooms on its descending neurons, and
the page also shows the two robot eyes, the MaleCNS neuron atlas (real neuron skeletons and cell
bodies from the Fly / Neural Atlas) lit by the simulated spikes, motor channels, readout rates and
the cell types firing right now. The atlas data lives in `brain/data/atlas/male-cns` (git-ignored;
copy it from the Fly / Neural Atlas project's `public/data/male-cns`).

The fly also learns: its mushroom body runs dopamine-gated plasticity (toggle on the dashboard).
The table holds a grape (rewarded with sugar) and a lemon (never rewarded), and a second world,
the two-odor arena, runs the classic choice assay with a before/after preference index. See
`docs/architecture.md`, "Learning".

A third world is the **hunt arena**: a meter-scale room with the walking humans of the Fly / People
Lab (their GLB exports live in `brain/companion_brain/ui/models/humans/`), where the fly is a 32 cm
ground vehicle driven by its descending neurons that must find and touch a person using heat (the
arista's hot cells) and vision, rewarded with sugar and dopamine, punished on a timeout. `companion
hunt --train` trains many flies in parallel (mushroom-body plasticity plus CMA-ES over a few gains)
and saves the trained fly's synapses and gains; `run --load-weights` puts it on the dashboard. See
`docs/architecture.md`, "Hunting". `companion hunt-gpu` is the same hunt built for a GPU: the room as
a batch of tensors (thousands of episodes stepped together, same physics and seeds) and a compact
fly-shaped network (hot cells → Kenyon cells → MBONs, retinal columns → pursuit, a central-complex
GRU) trained by PPO, rewarded for picking a person, hitting them fast and then staying on them as they
walk away (`brain/companion_brain/hunt_gpu/`). The humans are the other player: runners faster than the fly,
trained by PPO in the same rollouts to keep away from it, who tire and must rest so a tracking fly can catch up;
`hunt-gpu --watch --open` shows the chase live in the 3-D room. `companion hunt-brain` does the same for the real fly: the
hunter subcircuit of the connectome (8.5k neurons) runs as a differentiable rate model in the arena, its synapse gains and
thresholds learn by PPO, and the result loads back into the spiking brain (`--watch` shows the neurons lighting up as it hunts).

Firmware (needs [PlatformIO](https://platformio.org/) CLI, `uv tool install platformio`):

```sh
cp firmware/secrets.example.ini firmware/secrets.ini   # put WiFi SSID/password in it
cd firmware/cam  && pio run -t upload && pio device monitor
cd firmware/body && pio run -t upload && pio device monitor
```

Then follow the bring-up order in [docs/wiring.md](docs/wiring.md).

## Credits and data

- FlyWire connectome v783: Dorkenwald et al., Nature 2024; Schlegel et al., Nature 2024.
  Annotations from [flyconnectome/flywire_annotations](https://github.com/flyconnectome/flywire_annotations).
- LIF model and model-ready connectivity tables: Shiu et al., Nature 2024,
  [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- The NumPy exponential-Euler formulation follows [flypoke](https://github.com/vshapenko/flypoke) (MIT).

Connectome data is downloaded at first run and is not stored in this repo.
