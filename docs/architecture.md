# Architecture

## The idea

A fruit fly's brain has clearly identified **input** neurons (what its eyes and antennae report)
and **output** neurons (descending neurons, the commands sent down to the body). The FlyWire
connectome tells us how every neuron in between is wired. So:

1. Pretend the robot's sensors are the fly's sense organs and stimulate the *real* input neurons.
2. Run the *real* wiring as a spiking network (leaky integrate-and-fire, Shiu et al. 2024).
3. Read the *real* descending neurons and express whatever they say with eyes and sound.

Nothing in the robot's behavior is scripted; the mapping tables in
`brain/configs/default.yaml` only say which neurons each sensor talks to and which neurons each
actuator listens to. See [neuron-map.md](neuron-map.md) for the tables.

## Data flow

```
                  ┌───────────────────────── Mac: brain (Python) ──────────────────────────┐
ESP32-CAM ─MJPEG─▶│ camera.py ─▶ optic_lobe.py ─▶ features_to_rates ─▶ LIF network (numba) │
                  │   80x60 gray   loom/object/bar   LC4, LPLC2, LC11...   29k neurons      │
ESP32 body ─UDP──▶│ link.poll  ─▶ PIR burst ───────▶ JO wind neurons        0.1 ms steps     │
                  │                                        │                                 │
                  │              decode.py ◀── rates() ◀───┘  GF, DNp09, MDN, DNa02, pC1... │
                  │                 │ state + modulators                                     │
                  │              eyes.py ─▶ link.send ── UDP JSON 20 Hz ──▶ ESP32 body      │
                  └────────────────────────────────────────────────────────────────────────┘
                                                                              │
                                                       eyes.cpp (2x GC9A01)  audio.cpp (DFPlayer)
```

## Brain: what runs and how fast

| Piece | Detail |
|---|---|
| Neurons | FlyWire v783, 138,639 neurons in the Shiu et al. model tables |
| Connections | 2.70 M with ≥ 5 synapses (weaker pairs dropped, as in the paper) |
| Pruned circuit | neurons within 2 synaptic hops forward of the input groups **and** 2 hops backward of the readout groups: **29,115 neurons, 817,620 connections** |
| Model | LIF: rest −52 mV, threshold −45 mV, τ_m 20 ms, alpha synapse τ 5 ms, refractory 2.2 ms, delay 1.8 ms, 0.275 mV per synapse × count × sign (ACh +, GABA/Glu −), dt 0.1 ms |
| Stimulation | driven neurons spike as a Poisson process at 0–150 Hz (the reference model's Poisson input is super-threshold, so this is equivalent) |
| Speed | numba kernel, ~2.7× real time on an Apple Silicon laptop; 3 hops (95k neurons) runs at ~0.96× and `--full` (all neurons) slower still |
| Pacing | `runner.py` advances the brain in 10 ms chunks to match the wall clock; if it falls > 1 s behind it skips ahead rather than lagging forever |

Why no spontaneous activity: the reference model has none, so with no sensory input the network
is silent. Idle and sleep states are therefore decided host-side from the motion-energy timer,
not from the brain.

## Senses

`optic_lobe.py` is a deliberately simple stand-in for the fly's optic lobe (which alone is 77k
neurons and not simulated). Per hemifield of the 80×60 frame:

- **loom_fast**: an already-visible moving region that grows quickly → LC4 + LPLC2 → Giant Fiber
- **loom_slow**: low-passed growth → LC16 → backward walking
- **small_object** (+ azimuth/elevation): small dark moving blob → LC11 + LC10a; azimuth is
  also used directly for gaze
- **bar**: vertical edge energy that moved → LC12 + LC15
- **motion_energy**: for the sleep/wake timer only

The PIR sensor's rising edge drives the wind-sensitive Johnston's organ neurons (JO-C/E) at
100 Hz for 500 ms: something warm arrived and moved the air.

## Readout

`decode.py` averages each readout group's rate over a 100 ms window, scales it by a reference
rate to a 0–1 score, and picks the highest-priority behavior whose score is above its threshold
(with a hold time so states don't flicker). Left/right rate differences give a lateral bias used
for gaze. Modulators: MBON approach − avoid → valence (iris tint), octopamine → arousal (pupil).

## Body protocol

UDP JSON, brain → body on port 4210 at 20 Hz; body → brain on port 4211 every 500 ms and on
each PIR edge. Field definitions are in [wiring.md](wiring.md#network-protocol). The body eases
toward each target at 30 fps and falls back to a local idle animation after 2 s without packets,
and widens the eyes on its own when the PIR fires (a fast local reflex, like the Giant Fiber).

## Adding arms later

`arms.l` / `arms.r` are already in every packet (currently the steering lateral bias and 0).
Map MDN/DNa/DNg readouts to servo targets in `eyes.py`'s neighbor `body/arms.py`, and drive
servos from `firmware/body` using the `armL`/`armR` fields already parsed in `link.cpp`.
