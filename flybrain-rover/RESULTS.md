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

## 4. How much of the wiring is load-bearing

Synapses deleted, neurons left intact, then the turn-toward test re-run (32 arenas, 2 s, people frozen). Two deletion orders, the same counts.

| deleted | at random | weakest synapses first |
|---|---|---|
| 0% | 81% | 75% |
| 25% | 78% | 78% |
| 50% | 66% | 78% |
| 75% | **0%** | 72% |
| 90% | 0% | 72% |
| 95% | 0% | **72%** |
| 99% | 0% | 47% |

**Throw away the weakest 95% of the connections, 2.2 million of them, and the behaviour is intact. Throw away the same number at random and it is gone by 75%.** The steering rides on roughly a hundred thousand strong synapses, and the rest of the graph is, for this task, ballast. Repeat runs of an identical condition differ by about 5 points (the retina noise is redrawn), which is far below the effect.

### The smallest circuit that still finds a person

Taking that seriously: keep the strongest 5% of synapses (`scripts/minimal_circuit.py`), then keep only the neurons that lie on a path from the eye to a descending neuron (`scripts/prune_brain.py`). What is left is a real brain file every script here accepts.

| | neurons | synapses | turn-toward | advance | per camera frame |
|---|---|---|---|---|---|
| full circuit | 15,000 | 2,334,959 | 75% | 100% | 14 ms |
| strongest 5% of synapses | 13,670 | 119,743 | 75% | 100% | — |
| **+ only what lies between eye and motor** | **2,211** | **10,387** | **81%** | **100%** | **3 ms** |

**0.4% of the original wiring, and it behaves the same.** It also fails in the same named ways, which is what makes it the same circuit rather than a lucky one (64 arenas, seed 2000, 2 s): intact 73%, LC10a removed **0%**, wiring shuffled **2%**, DNa02 left removed 28% with a rightward bias of +0.048, looming cells removed 70% with the giant fibre going 74.9 Hz to 0.0 under a looming person.

The same circuit in **pure numpy**, no PyTorch at all (`edge/flybrain_mini.py`, one file, about 200 lines), is bit-identical to the reference implementation, spike for spike over 300 steps, and five times faster at batch one, where a tensor framework is mostly per-operation overhead:

| runtime | full circuit (15,000 neurons) | minimal circuit (2,211 neurons) |
|---|---|---|
| PyTorch, event engine | 14 ms per camera frame | 3 ms |
| numpy only | **2.5 ms** | **0.6 ms**, a 1,553 Hz loop |

So the brain needs neither a GPU nor a deep-learning framework, and a Raspberry Pi would run even the full circuit in roughly 10 ms per frame. At 0.15 ms per millisecond of brain time, the PyTorch version runs at 335 Hz on one laptop core, which puts it comfortably in real time on a Raspberry Pi and makes the detector, not the brain, the only obstacle to putting the whole loop on the robot. The mushroom body does not survive this cut (Kenyon cells and MBONs are silent in this task and drop out), so the tiny circuit is a runtime artifact, not something to train on.

## 5. Learning, using the fly's own dopamine neurons

Reward and punishment are delivered as current onto PAM and PPL1; plasticity is confined to existing synapses under Dale's law with bounds of 3× the original magnitude, plus homeostatic scaling toward 150 Hz. No new network is trained at any point.

Fixed evaluation set, stage A, 128 arenas, 20 s, seed 1000, no learning during evaluation:

| brain | front contact | time to contact | sustained contact | DNa02 mean |
|---|---|---|---|---|
| untrained | 0.977 | 4.25 s | 0.762 | 51 Hz |
| demo brain (three-factor, 512 arenas, 21 generations) | 0.977 | **3.23 s** | **0.817** | 156 Hz |

Learning bought speed and persistence, not the behaviour. Lobotomizing the demo brain back to the raw connectome and restoring it is bit-identical; learning had changed 178,445 of 2,334,959 synapses.

Negative results we are not hiding: four overnight biological training runs were flat, because a rover that cannot advance never earns reward, and because the Kenyon cells, the fly's actual learning site, are silent in this task, so mushroom-body plasticity had nothing to act on. An earlier run with a non-biological tonic "walking" current did raise contact but saturated DNa02 at 300 Hz; we do not call that learning.

## 6. Two ways for a descending neuron to be silent

This began as our biggest problem, that the eye does not drive the forward-walking neurons, and turned into the most interesting thing we found. `scripts/why_silent.py` drives the eye, then strips the inhibitory inputs onto a chosen descending neuron one source at a time, strongest first, and watches what happens.

**DNa10 is held silent on purpose.** It receives 801 synapses straight from LC10a, the largest direct eye-to-descending projection in the circuit, and never fires: 0.0 Hz at every drive from 1.5 to 80 mV per step, while the eye itself saturates at 326 Hz. It is not weakly wired. The same eye signal also drives AOTU041, AOTU063_a and AOTU063_b, which place 1,201 inhibitory synapses on it. Peel those off and it comes straight up:

| inhibitory sources removed (cumulative) | DNa10 |
|---|---|
| none | 0.0 Hz |
| AOTU063_b | 0.0 Hz |
| + AOTU041 | 32.5 Hz |
| + AOTU063_a | 91.0 Hz |
| + AOTU042 | 106.8 Hz |

The connectome is running feedforward inhibition that gates this pathway off while the tracking pathway is active. That fits the published picture: DNa10 sits downstream of LC10d and drives object *avoidance*, and a male fly tracking a female has no business triggering avoidance.

**DNp09 and DNa01 are silent for the opposite reason.** Removing every inhibitory input onto them, eight sources each, leaves both at exactly 0.0 Hz. Nothing is sitting on them; the excitation never arrives. Meanwhile **DNa02**, which receives no direct LC10a synapse at all, reaches 126 Hz through a three-hop path, and the shortest anatomical paths to DNp09 and DNa01 do exist (550 source-target pairs within three hops; 739 of the 795 neurons on them were already in the circuit, and forcing in the remaining 56 changed nothing).

Anatomy is not drive, and there are at least two distinguishable ways for that to be true. This is also why we read forward speed from the descending population rather than from DNa01 and DNp09: not a workaround we chose, a fact we measured.

## 7. Speed

| measurement | protocol | result |
|---|---|---|
| brain step | batch 1, CPU, event-driven engine, M2 Pro | **0.68 ms** per 1 ms of simulated brain time (14 ms per camera frame at 20 substeps) |
| pruned circuit | 4,104 neurons, 408,114 synapses, behaviour intact (77% turn-toward, 100% advance) | 0.50 ms per step; only 1.4× faster, because an event-driven engine never touches a silent neuron |
| person detector | yolo11n, 320 px | 29 ms on the laptop GPU, 83 ms on its CPU |
| camera to wheel command | dry run, network camera, detector on GPU | **15 ms median**, 24 ms at the 95th percentile. Budget was 100 ms |
| training | A100, 512 arenas, event engine | 3.4 ms per step, about 1.1 min per 20 s generation |

## 8. Sensors and interfaces

The retina encoder was checked against ground truth on 50 synthetic pinhole-projected frames: boxes from a detector and boxes computed from true positions agree within one of 24 angular columns, and the pixel-to-azimuth round trip is accurate to 0.5 degrees. The simulated heat sensors are lateralized correctly but reach no descending neuron at a stable gain, so they only bias the search direction.

125 automated tests cover the three simulation engines against each other spike for spike, arena geometry and reward rules, Dale's law and weight bounds under both learning rules, the lobotomize round trip, and the retina agreement above: `python -m pytest tests -q`.
