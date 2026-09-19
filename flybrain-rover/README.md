# FlyBrain Rover

A real fruit fly's wiring diagram, running live as spiking neurons, driving a robot that finds a person in the room. Nothing in between is programmed, and nothing is trained from scratch.

We take the male *Drosophila* CNS connectome (Berg et al., *Cell*, September 2026; served on neuPrint as `male-cns:v1.0`), cut out the 15,000-neuron subcircuit that runs from the eye to the descending neurons, and simulate all 2,334,959 signed synapses as leaky integrate-and-fire cells with the constants from Shiu et al. 2024. A camera frame becomes current on the fly's visual projection neurons. The descending neurons, the ones that would drive its legs, are read out as wheel commands.

**Untrained, straight out of the electron microscope, it turns toward a person in 75% of trials and reaches them 97.7% of the time.** Learning, where we use it, is the fly's own: reward and punishment are delivered onto its dopamine neurons, and only existing synapses change strength.

The point of the demo is not that a robot follows you. A three-line controller does that better. The point is that this one can be broken in named, predictable ways:

- Remove LC10a, the 275 cells a male fly uses to track a mate: tracking goes to **0%**.
- Keep every neuron, every synapse and every neuron's exact number of connections, and randomize only *which cell connects to which*: the eye still fires at 70 Hz, the steering neurons go from **158 Hz to zero**.
- Remove the looming detectors: the escape reflex vanishes (giant fibre 130 Hz to 0.1 Hz) while steering is untouched.
- Remove one of the two steering neurons: it biases to the other side, which we predicted before running it.

Every number here, with its protocol, is in **[RESULTS.md](RESULTS.md)**. Check them yourself in under a minute with `python scripts/reproduce.py --quick`.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python data/pull_connectome.py     # neuPrint -> data/brain.npz, no account or token needed
python -m pytest tests -q          # 125 tests
python scripts/reproduce.py --quick

# the demo on a laptop webcam, no robot, wheel commands printed instead of sent
python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run
```

In the demo window: **1** baseline, **2** remove the tracking neurons, **3** restore, **4** wipe the learning, **5** restore it, **6** wiring shuffle on and off, **7** delete a quarter of the tracking population, **space** stop.

Driving a real RoboMaster S1, eyes and wheels chosen independently:

```bash
# wheels over the S1's S-Bus receiver pins through an ESP32, eye from any camera
python scripts/demo.py --checkpoint checkpoints/demo_brain.pt \
  --s1-sbus /dev/cu.usbserial-XXXX --local-eye --source 0 --dry-run
# --source also takes a stream URL (ESP32-CAM, phone IP camera) or a video file
```

[DEMO.md](DEMO.md) is the runbook: setup, a two-minute script, and what to do when something fails.

## How it fits together

| | |
|---|---|
| `data/pull_connectome.py` | neuPrint to `data/brain.npz`: ids, types, signed weights, named neuron groups |
| `brain/lif.py` | batched spiking simulation, three engines (sparse, dense, event-driven) that agree spike for spike; lesion and restore |
| `brain/retina.py`, `brain/senses.py` | camera or ground truth to 24 angular columns to current on the eye neurons |
| `brain/retinotopy.py` | each eye neuron gets the retina column its real lobula dendrites look at |
| `brain/motor.py` | descending neurons to forward and turn |
| `brain/explore.py` | search as an internal state, labelled non-sensory, gated off the moment the eye sees anyone |
| `env/arena.py` | batched 2-D world: people who walk, contact events, reward |
| `learn/` | three-factor dopamine-gated plasticity and evolution strategies, both under Dale's law and hard bounds |
| `scripts/why_silent.py` | why a descending neuron does not fire: strips its inhibition source by source |
| `scripts/dropout.py` | how much of the connectome can be deleted before the behaviour goes |
| `scripts/export_timeseries.py` | the recordings into TimescaleDB, TigerData Cloud or a local SQLite file |
| `scripts/` | milestones, evaluation, lesion tables, pruning, benchmarks, replay, the robot bridge and the demo |

## What we engineered, and say so

Forward speed is read from the descending population as a whole, because the eye does not drive the two classic forward-walking neurons in this connectome; the anatomical paths exist but carry no signal ([RESULTS.md](RESULTS.md) section 5). The search behaviour is an internal drive we added, not something we found in the wiring. The steering, which is the claim, is untouched connectome.

## Credits

- Berg et al. 2026, "Sexual dimorphism in the complete connectome of the Drosophila male CNS", *Cell*. Data CC-BY via [neuPrint](https://neuprint.janelia.org).
- Shiu et al. 2024, *Nature*: leaky integrate-and-fire constants and the neurotransmitter-to-sign rule.
- Nern et al. 2025 and the Reiser lab eyemap documentation: lobula column coordinates for the retinotopy.
- Keles and Frye 2017 (LC11 small-object tuning), Ribeiro et al. 2018 and Hindmarsh Sten et al. 2021 (LC10a and courtship tracking), Rayshubskiy et al. 2020 (DNa02 steering), Bidaye et al. 2020 (DNp09), Marin et al. 2020 (thermosensory glomeruli).
- Eon Systems `fly-brain` (GPL-2.0-or-later), `ilyaosovskoi/connectome-pilot`, `Imperol3/flybrain` as references.
- neuprint-python, PyTorch, Ultralytics YOLO11n, OpenCV, the DJI RoboMaster SDK, Three.js.
