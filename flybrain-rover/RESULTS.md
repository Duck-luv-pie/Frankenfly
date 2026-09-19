# RESULTS

Every number this project claims, with the protocol that produced it. Anything not listed here is not a result.

Unless stated otherwise: `data/brain.npz` (15,000 neurons, 2,334,959 signed synapses from neuPrint `male-cns:v1.0`), leaky integrate-and-fire with Shiu et al. 2024 constants, global gain 0.05, sensory amplitude `amp_track` 20, turn = k_t (DNa02_R − DNa02_L), forward = k_f × mean firing rate over all 241 descending neurons (k_f 0.3), retinotopy from real lobula column footprints. "Turn-toward" means the absolute bearing to the nearest person fell by more than 0.02 rad over the episode; "advance" means the distance fell by more than 0.05 m.

Reproduce the headline checks on your own machine in under a minute: `python scripts/reproduce.py --quick` (all eight passed here in 12 s).

## 1. The circuit is alive and lateralized

| check | protocol | result |
|---|---|---|
| M1, the map | neuPrint pull, every named group non-empty | 15,000 neurons, 2,334,959 synapses (1.87M excitatory, 0.46M inhibitory) |
| M2, signal reaches the motor | drive LC10a_L only, 300 ms, gain sweep 0.005 to 0.275 as one batch | passes for gain 0.025 to 0.15; at 0.05, DNa02_L 126 Hz and DNa02_R 0 Hz. Shiu's whole-brain 0.275 seizes this cut |
| M2c, laterality | one person at 0, −30 and +30 degrees, 2 m, 300 ms, 16 arenas | centred: DNa02 0/0 and turn 0.00. Left 30: 174/0, turn −1.00. Right 30: 0/179, turn +1.00 |

## 2. Behaviour before any learning

| check | protocol | result |
|---|---|---|
| M3, turn toward a person | stage A, 64 arenas, 2 s, people frozen | **75%** of arenas turn toward; mean bearing 0.361 → 0.124 rad |
| M3b, close the distance | same | **100%** of arenas advance; 1.84 m → 0.66 m |
| lesion baseline | 256 arenas, seed 2000, 2 s | 78% turn-toward |
| front contact, stage A | 128 arenas, 20 s, seed 1000, people walking | **0.977** contact, 4.25 s to first contact, 76% of the episode in contact |
| trailing a walking person | 32 arenas, 15 s, walkers 1.0–1.4 m/s with no pauses, scored over the last 10 s | in view **100%**, within 1 m **82%**, mean distance 1.02 m |
| search from out of view | stage B, 128 arenas, 20 s, 1–8 people starting outside the camera, exploratory state on | **0.742** contact, 91% sight a person, first sight **2.46 s** |

## 3. The controls: it is the wiring, not the parts

256 arenas, seed 2000, 2 s, people frozen. Each lesion zeroes every synapse into and out of the named cells and is restored afterwards.

| condition | turn-toward | notes |
|---|---|---|
| intact circuit | **78%** | |
| remove LC10a, both sides (275 of 15,000 cells) | **0%** | the population a male fly uses to track a mate |
| shuffle the wiring, degrees preserved | **0%** untrained, 11% trained | same neurons, same in- and out-degree per neuron, same signed weights, random targets |
| remove LC4 and LPLC2 (looming) | 75% | steering survives; the giant fibre under a looming person goes **130 Hz → 0.1 Hz** |
| remove one steering neuron (DNa02 left) | 25% | signed turn biases right (+0.055 against +0.002 intact), as predicted before running it |

Live on the demo brain, one keypress, person at the left of frame, 45 frames at 30 Hz:

| | LC10a L/R (the eye) | DNa02 L/R (steering) | turn |
|---|---|---|---|
| real connectome | 66.6 / 0.0 | **158.4** / 0.0 | −1.00 |
| wiring shuffled | 69.9 / 0.0 | **0.0** / 0.0 | 0.00 |

The eye is untouched by the shuffle and nothing arrives at the steering neurons: the information is in which cell connects to which.

Deleting LC10a a quarter at a time, same protocol: DNa02 **158.4, 137.5, 83.4, 40.3, 0.0 Hz** at 0, 25, 50, 75, 100% removed, with the turn command holding at full until three quarters are gone. Restoring returns 158.4 Hz and −1.00 exactly.

## 4. Learning, using the fly's own dopamine neurons

Reward and punishment are delivered as current onto PAM and PPL1; plasticity is confined to existing synapses under Dale's law with bounds of 3× the original magnitude, plus homeostatic scaling toward 150 Hz. No new network is trained at any point.

Fixed evaluation set, stage A, 128 arenas, 20 s, seed 1000, no learning during evaluation:

| brain | front contact | time to contact | sustained contact | DNa02 mean |
|---|---|---|---|---|
| untrained | 0.977 | 4.25 s | 0.762 | 51 Hz |
| demo brain (three-factor, 512 arenas, 21 generations) | 0.977 | **3.23 s** | **0.817** | 156 Hz |

Learning bought speed and persistence, not the behaviour. Lobotomizing the demo brain back to the raw connectome and restoring it is bit-identical; learning had changed 178,445 of 2,334,959 synapses.

Negative results we are not hiding: four overnight biological training runs were flat, because a rover that cannot advance never earns reward, and because the Kenyon cells, the fly's actual learning site, are silent in this task, so mushroom-body plasticity had nothing to act on. An earlier run with a non-biological tonic "walking" current did raise contact but saturated DNa02 at 300 Hz; we do not call that learning.

## 5. Anatomy that carries no signal

The shortest paths from the eye to the classic forward-walking neurons exist: LC10a reaches DNp09 in one to three hops (550 source-target pairs) and DNa01 in two to three. 739 of the 795 neurons on those paths were already in the circuit; forcing in the remaining 56 changed nothing measurable. With a centred person at 2 m, DNp09 and DNa01 stay below 2 Hz at every non-seizing gain. We therefore read forward speed from the descending population as a whole, and say so.

## 6. Speed

| measurement | protocol | result |
|---|---|---|
| brain step | batch 1, CPU, event-driven engine, M2 Pro | **0.68 ms** per 1 ms of simulated brain time (14 ms per camera frame at 20 substeps) |
| pruned circuit | 4,104 neurons, 408,114 synapses, behaviour intact (77% turn-toward, 100% advance) | 0.50 ms per step; only 1.4× faster, because an event-driven engine never touches a silent neuron |
| person detector | yolo11n, 320 px | 29 ms on the laptop GPU, 83 ms on its CPU |
| camera to wheel command | dry run, network camera, detector on GPU | **15 ms median**, 24 ms at the 95th percentile. Budget was 100 ms |
| training | A100, 512 arenas, event engine | 3.4 ms per step, about 1.1 min per 20 s generation |

## 7. Sensors and interfaces

The retina encoder was checked against ground truth on 50 synthetic pinhole-projected frames: boxes from a detector and boxes computed from true positions agree within one of 24 angular columns, and the pixel-to-azimuth round trip is accurate to 0.5 degrees. The simulated heat sensors are lateralized correctly but reach no descending neuron at a stable gain, so they only bias the search direction.

125 automated tests cover the three simulation engines against each other spike for spike, arena geometry and reward rules, Dale's law and weight bounds under both learning rules, the lobotomize round trip, and the retina agreement above: `python -m pytest tests -q`.
