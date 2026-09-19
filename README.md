# Hunting Fly

A real fruit fly brain, taught to find a person and stay on them, driving a small robot.

The brain is the [FlyWire](https://flywire.ai) connectome of an adult *Drosophila*: every neuron and synapse, with their signs and synapse counts. A subcircuit of about 8,500 neurons and 245,000 synapses is cut from it and run as spiking neurons. A camera and a PIR motion sensor are written onto the fly's own sensory neurons; motor commands are read out from its descending neurons, the ones that carry commands from brain to body, and drive an RC vehicle. Nothing about the wiring is invented, and training is only allowed to tune what a real brain could tune: synapse strengths within bounds, neuron thresholds, and sensory and readout gains.

The task: find a walking person in a room, go up to them, and trail them for as long as possible, facing them. The robot is meant to attach something to the person, and that takes time, so tagging them is not enough.

## What the fly senses and does

| sense | model |
|---|---|
| eyes | a 150 degree field of view split into 24 retinal columns, 6 m range; a person lights up the columns their shoulders cover, plus a motion channel |
| warmth | an HC-SR501 PIR: one bit, high for 2 s after a warm body moves inside a 110 degree, 6 m cone; blind to a standing person |
| body | its own speed, its last drive commands, time elapsed |

The columns drive the fly's visual projection neurons (LC10a and LC11 for small objects, LC12 and LC15 for motion), each side of the brain seeing its own hemifield. The PIR drives the hot cells. A person filling the view drives LC9, the visual input the connectome wires most heavily onto DNp09, the freezing neuron, which is the vehicle's brake.

The vehicle is a car: forward drive from the descending population, braked by DNp09; turn from the DNa01 and DNa02 steering neurons, read as a normalized left-minus-right contrast so a person straight ahead is a small correction and a person off to one side is a hard turn. Search, pursuit and braking are all wiring that exists in the real fly.

## How it was trained

Thousands of simulated rooms run in parallel on a GPU. The spiking circuit is replaced, for training only, by a differentiable mean-field version fitted to its measured responses, so gradients can flow through 8,500 neurons. Recurrent PPO nudges the learnable parameters after every 32 ticks, backpropagating through four ticks of neural dynamics, with an exploration floor so the policy keeps trying new things.

The reward is built around the physical goal:

- a bonus for the first touch, faster is better
- a per-tick payment for trailing the locked person, within 0.6 m and facing them, ramping up over 5 s of unbroken trailing
- a smaller ramped payment for head-on nose contact
- a charge for having them behind
- a heavy one-off charge for losing them within 2 s of a touch
- half of all episodes begin already at contact, so the follow is practised directly

The learned values are saved by FlyWire neuron ID and written back onto the spiking brain, so the trained fly runs as live spikes in a 3-D viewer and in the robot's runtime.

## What it took to get there

The first trained brain could touch a person 98% of the time but trailed them only 4% of the time afterward. Each fix came from tracing the brain tick by tick after a touch rather than guessing:

1. **It never slowed down.** The proximity signal was wired to looming neurons that never reach DNp09 in the connectome. Probing every visual projection type found LC9, the one that does. Rerouting proximity there took trailing from 4% to 31% before any retraining.
2. **It spun away at contact.** The two DNa02 steering neurons form a lopsided winner-take-all: a person straight ahead drives the left one 37 times harder than the right. A normalized contrast readout replaced the raw difference.
3. **It braked while following.** The freeze channel's background activity left a fifth of the brake on. Its dead zone became learnable.
4. **It stopped exploring.** PPO had shrunk its noise to nothing; a floor fixed that.
5. **It lost people who darted across its nose.** Every remaining loss happened at full steering lock. That is physics, so the turn rate was raised to 3.5 rad/s.
6. **The world was made honest.** Runners became people who walk about and ignore the fly, the warmth lobes became the PIR, and the collision became the vehicle's actual body.

Held-out results on 256 fresh rooms, driven deterministically, against people walking about:

| brain | first touch | time to touch | trailing | longest trail | head-on contact | longest contact |
|---|---|---|---|---|---|---|
| before | 93% | 13.6 s | 2.4% | 0.5 s | 1% | 0.2 s |
| after | 100% | 3.5 s | 66% | 22.7 s | 60% | 11.6 s |

A lobotomy control in the viewer severs the 17,101 synapses leaving the sensory neurons. The eyes still light up, the steering neurons stay at rest, and the fly stops hunting: the circuit is doing the work.

## What is left

The fly still loses a person about 0.7 times per episode at very close range, mostly by sliding around someone who has paused. Before the brain runs on the robot, the camera's true field of view and the car's real turn rate need to go into the configuration so the simulation matches the hardware.

## The spiking brain and the robot bridge (`flybrain-rover/`, Taka)

A second brain lives in [`flybrain-rover/`](flybrain-rover/): the **male Drosophila CNS connectome** (Berg et al., Cell, Sept 2026, neuPrint `male-cns:v1.0`), a 15,000-neuron eye-to-descending subcircuit with 2.33 M signed synapses, run as leaky integrate-and-fire spiking neurons at 1 ms with Shiu et al. 2024 constants, no surrogate. It turns toward a person in 75 to 78 % of arena episodes **untrained**, reaches them 97.7 % of the time once forward drive is read from the descending population, and trails a walking person within 1 m 87 to 92 % of the time. Learning uses the fly's own dopamine neurons (PAM reward, PPL1 punishment) gating plasticity on existing synapses only, with homeostatic scaling; the trained checkpoint (`flybrain-rover/checkpoints/demo_brain.pt`) closes in 3.2 s instead of 4.3 s. Lesion controls: remove LC10a and tracking drops to 0 %; shuffle the wiring with degrees preserved and it drops to 0 %; remove the looming neurons and the escape response vanishes while steering stays. Retinotopy comes from each neuron's real lobula column footprint. The same directory holds the RoboMaster S1 path (`scripts/robot_daemon.py` for the SDK, `scripts/demo.py` for the demo with lesion and lobotomize hotkeys, live websocket feed for the viewer), the evaluation harness, and `STATUS.md` with every number and its provenance. Start with `flybrain-rover/README.md`; the connectome file regenerates with `python data/pull_connectome.py` (no token needed).
