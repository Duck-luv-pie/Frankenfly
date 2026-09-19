# FlyBrain Rover

## Inspiration

Hack the North 2026 rewards physical, memorable demos, something a judge can walk up to in five minutes at the table (a robot arm, a motorized chess board, a Nerf gun as a controller). We wanted the strangest version we could pull off: a real fruit fly's brain, wired the way evolution wired it, driving a robot around the room.

The male Drosophila CNS connectome (Berg et al., Cell, Sept 3 2026: 166,700 neurons, about 125 million synapses, served on neuPrint as male-cns:v1.0) went viral the week it came out, people wired it to play Doom, Mario 64, Beat Saber. Cool, but still a game on a screen. We wanted it driving a body that finds, tracks, and touches an actual person, nothing hand-coded in between. If the robot turns toward you, it's because a real male fly uses those exact neurons to track a mate, not because we told it to.

## What it does

We take about 15,000 neurons, the full connectome's eye-to-descending subcircuit, 2,334,959 signed synapses (`data/brain.npz`), and simulate them as leaky integrate-and-fire neurons with Shiu et al. 2024's constants. A camera frame becomes current on the fly's visual projection neurons (the LC populations); its descending neurons, which would normally drive its legs, get read out as forward/turn commands for a DJI RoboMaster S1. Nothing in the middle is programmed.

Untrained, the brain turns toward a person in 75 to 78% of arena episodes from wiring alone (M3, 64 envs, stage A, 2 s, humans frozen: 72 to 77% across an amp_track sweep, 75% with the retinotopy-corrected default; 256-env lesion baseline 77.7%; STATUS.md). Lesion controls, the part we're proudest of, prove it's the wiring (all four: brain_v2.npz, 256 envs, seed 2000, stage A, 2 s, humans frozen):

- LC10a out (used to track a female): turning drops to 0%.
- Wiring shuffled (same neurons, degrees, and signs, reconnected at random): turning also drops to 0%.
- LC4 + LPLC2 out (looming detectors): escape response vanishes (giant-fiber under looming: 106.9 to 0.0 Hz) while steering stays at 75%.
- One DNa02 out: hard rightward bias.

Two of those controls run live on the robot, on a keypress, which is the part we would ask a judge to watch. Press one key and every one of the 2,334,959 synapses keeps its sign, its strength and its neuron's exact number of inputs and outputs, but its target is randomized. The eye carries on firing at 70 Hz and the steering neurons go from 158 Hz to zero: the information was in the map, not the parts. Press another key and the tracking population is deleted a quarter at a time; the steering drive falls 158, 137, 83, 40, 0 Hz while the turn command holds until three quarters are gone, then collapses. Restoring gives back 158.4 Hz and a full turn, exactly.

This is also our answer to the obvious question, why not just train a neural net. A net does this too, and a three-line hand-coded controller does it better. But a net with random weights does nothing at all, and you cannot lesion a named cell type out of one and predict the result. We predicted that removing the left steering neuron would bias the turn right, and it did. A net cannot be wrong that way, so it cannot be right that way either.

Learning, where we do it, uses the fly's own mechanism: reward and punishment land on its dopamine neurons (PAM reward, PPL1 punishment), gating plasticity at existing synapses under Dale's law, hard bounds, and homeostatic scaling toward 150 Hz. We never train a new network; only existing synapses change strength.

## How we built it

Pipeline: camera to `brain/retina.py` (24 angular columns) to `brain/senses.py` (current into LC eye neurons) to `brain/lif.py` (batched LIF on the real wiring) to `brain/motor.py` (descending neurons to forward/turn) to wheels, all batched torch, no Python loops over environments. `env/arena.py` is the batched 2D world of humans, walls, contact events, and reward.

The eye alone doesn't drive forward motion: no eye input reaches DNa01 or DNp09 (the textbook forward-walking neurons) at any non-seizing gain, even after forcing every neuron on the shortest anatomical path (1-3 hops) into a second circuit, brain_v2 (739 of 795 were already there; nothing changed). The path exists on paper but carries no functional drive, so forward speed became an explicit read-out: k_f times the mean firing rate across all 241 descending neurons in the circuit (132 types). The read-out is ours; every rate feeding it is the connectome's. With it, the untrained brain reaches a 0.969 front-contact rate in stage A over 20 s (64 envs, DN_ALL, index mapping), closing distance 1.84 m to 0.66 m in 2 s in 100% of envs (64 envs, seed 0, k_f 0.3).

We also fixed a retinotopy bug: indexing LC columns by database ID made a centred person veer hard right (turn +0.97). Real lobula column positions from neuPrint, mapped via the Nern 2025 / Reiser lab eyemap conventions, fixed it: turn 0.00 at centre, turn-toward held at 75%, final bearing tightened (0.124 vs 0.185 rad).

For search: a labelled, non-sensory exploratory state, tonic DNa01 current plus a DNa02 saccade generator bursting left/right every 1.5 s, heat-biased, gated off within 20 ms of retina presence. Stage B contact rate with it on is 0.742 (128 envs, 20 s), first sight at a mean of 2.46 s.

Robot: yolo11n at 29 ms/frame on the Mac's GPU, brain about 10 ms/frame on an event-driven LIF engine we wrote, dry-run latency a median 41 ms under load. The RoboMaster SDK only ships Python 3.6-3.8 wheels, so it runs as its own daemon process, streaming a live feed over websocket for Ducks's Three.js viz.

## Challenges we ran into

Forward drive was the real fight: short paths from eye to forward neurons exist anatomically but carried nothing functionally, and we had to prove that rather than assume it.

Our first four biological training runs came back flat: three-factor learning on eye-to-descending synapses, three-factor on Kenyon-cell-to-MBON synapses (the fly's actual learning site), and evolution strategies on two other synapse subsets. No trend in contact rate, weights barely moved. Without forward drive a rover only turns, so every "contact" was a human walking into a stationary robot. The Kenyon-cell run moved weights by essentially zero (|Δw| under 5e-7): nothing reaches the mushroom body at a stable rate, so no eligibility trace formed.

Before that read-out, we tried a non-biological fallback: constant "walking" current into DNa01, labelled as such, off on contact. Contact rate rose, but the reward balance was broken (the no-contact penalty never crossed threshold to fire PPL1, so learning only potentiated), and DNa02 climbed to 240-320 Hz, against its own ~330 Hz refractory ceiling: a saturated circuit, not a tuned one. We don't call that learning.

## Accomplishments that we're proud of

The lesion table. Plenty of connectome demos show a trick working. We show the same trick failing on command when you remove the exact circuit responsible, at 0% for both the real lesion and a degree-preserving random shuffle of the same wiring. That negative control isn't something anyone else doing connectome demos has shown.

We're also proud every number here traces back to a specific brain file, environment count, seed, and episode length in our logs. Nothing is "it worked, trust us."

## What we learned

A path existing anatomically doesn't mean it carries drive: 739 of 795 neurons on the shortest eye-to-forward path were already in our circuit, forcing the rest in changed nothing.

Reward-gated plasticity is only as good as the reward signal behind it: an unbalanced dopamine channel saturates a circuit instead of tuning it, and the two look similar in a five-minute demo if you're not watching firing rates.

There are two different ways a neuron can be silent, and you can tell them apart. Chasing down why the eye never drove the forward-walking neurons, we built a tool that strips the inhibition off a descending neuron one source at a time. DNa10, which receives the single largest direct projection from the tracking cells (801 synapses), never fires at any drive, and comes up to 91 Hz the moment two inhibitory populations are removed: the connectome is actively gating it off with feedforward inhibition, which fits its published role as an avoidance pathway a tracking fly should not be triggering. DNp09 and DNa01 fail the other way: strip every inhibitory input and they still sit at exactly 0.0 Hz, because no excitation ever arrives. That is why we read forward speed from the whole descending population, and it is a measurement rather than a workaround.

Pruning a spiking circuit does not speed it up the way you expect. We cut the runtime brain to the neurons that matter (sensory populations, every descending neuron, and the bridge between them): 4,104 neurons and 408,114 synapses, 27% and 17% of the original, with behaviour intact (77% turn-toward against 75%, 100% advance). It runs 1.4x faster, not 4x, because an event-driven engine never touches a silent neuron in the first place. Cost is spikes times out-degree, not neuron count.

connectome-pilot's own numbers show a 3-line hand-coded controller can beat this same connectome at a similar task, and we're not claiming otherwise. The point was always that real behavior falls out of real wiring, with nothing programmed in the middle.

## What's next

Today's runs (brain.npz, `tfA_both_dnall` on the Mac plus the box queue at 512 envs) hold DNa02 near its 150 Hz target instead of saturating, contact 0.97 to 0.99 (STATUS.md, 2026-09-18 morning session). On the fixed evaluation set (stage A, 128 envs, 20 s, seed 1000) the locked demo brain, checkpoint gen 20 of that box run, reaches the person in 3.23 s versus 4.25 s untrained and holds front contact for 82% of the episode versus 76%, at the same 0.977 contact rate, with DNa02 at 156 Hz (STATUS.md, M6). Learning bought speed and persistence, not the behaviour itself, and we say that.

We also measured whether the brain could live on the robot instead of a laptop: 0.50 ms per millisecond of brain time at batch one on an M2 Pro CPU, which puts a Raspberry Pi 5 at roughly 30 to 50 ms per camera frame. The brain fits on a Pi in plain Python, no C rewrite. The detector is what does not: 83 ms on this laptop's CPU means 150 to 250 ms on a Pi without an accelerator.

After that: get the chain onto the actual RoboMaster hardware (everything above is dry-run or arena-simulated), confirm `chassis.drive_speed`'s turn sign, and measure the real camera's field of view instead of assuming spec. If there's time before Saturday's 2:00 PM sponsor-track lock, wire the exploratory search state into the live demo instead of starting with a person already in view.

## Built with

Python, PyTorch (event-driven sparse LIF engine), neuPrint (Janelia), Ultralytics YOLO11n, OpenCV, DJI RoboMaster SDK and an S-Bus link through an ESP32, Three.js (live viewer over websockets), Matplotlib, Vast.ai A100 (training only; the control loop never leaves the laptop).

## Credits

- Berg et al. 2026, "Sexual dimorphism in the complete connectome of the Drosophila male CNS," Cell. Connectome data, CC-BY via neuPrint (male-cns:v1.0).
- Shiu et al. 2024, Nature. LIF constants, neurotransmitter-to-sign rule.
- Eon Systems `fly-brain` (GPL-2.0-or-later). LIF engine reference.
- `ilyaosovskoi/connectome-pilot`. Three-factor plasticity reference.
- `Imperol3/flybrain`. Looming/escape pipeline reference.
- Keles and Frye, 2017. LC11 small-object tuning.
- Rayshubskiy et al., 2020 (bioRxiv). DNa02 ipsilateral turning.
- Marin et al., 2020, Current Biology. VP2/VP3 thermosensory glomeruli.
- Bidaye et al., 2020, Neuron. DNp09 forward walking.
- Nern 2025 / Reiser lab eyemap documentation. Lobula column coordinates.
- neuprint-python (Janelia), Ultralytics, DJI RoboMaster SDK.

Team: Ducks (hardware, RoboMaster S1, the Three.js arena and visualization), Max, Senthil, and Taka (this repo: brain, senses, motor mapping, arena, training, robot bridge).
