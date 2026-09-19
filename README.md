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
| `docs/` | [architecture](docs/architecture.md), [neuron map](docs/neuron-map.md), [wiring](docs/wiring.md) |
| `hardware/` | [bill of materials](hardware/bom.md) |
| `tools/` | `make_sounds.py` synthesizes the fly sounds for the SD card |
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
uv run companion run --sim-camera 0 --dry-body --open   # no robot yet: Mac webcam + live dashboard
uv run companion run                                      # ESP32-CAM + real robot (dashboard too)
```

Every `run` serves a live dashboard at <http://localhost:8600> (`--open` launches it, `--no-ui`
disables it): the camera with what the fly's eye detects, an animated fly acting out the
descending neurons, the two robot eyes, a map of all simulated neurons flashing as they spike,
readout rates, behavior scores and the cell types firing right now.

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
