# Hunting Fly

**We ran 15,000 neurons of a real fruit fly's brain on a robot. Untrained, it finds a person 97.7% of the time. Randomize which cell connects to which and it stops dead.**

## Inspiration

Hunting Fly is a robot that hunts a person using nothing but the wiring diagram of a real fruit fly's brain.

That diagram was published this month by Janelia and Cambridge: every neuron and every synapse in a male fruit fly's central nervous system, 166,700 cells and more than 124 million connections, free to download (Berg et al., *Cell*, September 3 2026, on neuPrint as `male-cns:v1.0`). Within days people had it playing Doom, Mario 64, Beat Saber. All of it on a screen.

A fly's best-studied behaviours are not screen behaviours. It tracks, it chases, it dodges your hand. Those belong in a body. So we took the part of the map running from the eyes to the cells that command the legs, ran it as spiking neurons, and wired it to a robot's wheels.

Then we spent most of the weekend trying to prove it was fake.

## What it does

A camera looks at you. A detector finds you in the frame. The edges of that box become angles, the angles become 24 columns spanning the fly's field of view, and those columns become current injected into named cells: **LC10a**, the neurons a male fly uses to track a mate, and **LC4 and LPLC2**, the looming detectors that fire when something rushes at it.

15,000 neurons and 2,334,959 signed synapses then run on the real wiring, one millisecond at a time. On the far side we read the descending neurons, the fly's only channel from brain to body. Turning is the difference between left and right **DNa02**, which is how a real fly steers.

There is no line anywhere that says turn toward a person.

Straight from the connectome, with no learning of any kind:

- turns toward a person in **75%** of arenas (64 arenas, 2 s, people frozen)
- reaches and touches them **97.7%** of the time (128 arenas, 20 s, seed 1000, people walking)
- trailing a walking person, keeps them **in view 100%** of the time and stays within 1 m for **82%** of it
- **15 ms** median from camera frame to wheel command, 24 ms at the 95th percentile. Our budget was 100 ms

### Four lesion controls, live on hotkeys

This is the part we care about. A robot that follows you is a contest we lose: three lines of Python beat this connectome at the task. What three lines of Python cannot do is fail on command in a named way, predicted in advance. So we hand the judge the keyboard.

**Cut the tracking neurons.** Remove LC10a, 275 cells out of 15,000, and turn-toward goes from 78% to **0%** (256 arenas, seed 2000). No other eye population reaches the steering neurons.

**Cut them a quarter at a time.** Four presses and the steering drive falls **158.4, 137.5, 83.4, 40.3, 0.0 Hz**, with the turn command holding at full until three quarters are gone. It degrades in proportion. It does not switch off. Restore and it returns to 158.4 Hz exactly.

**Cut the looming detectors instead.** Remove LC4 and LPLC2 and steering is untouched at 75%, while the giant fibre, the fly's escape trigger, drops **130 Hz to 0.1**. Two channels, separately breakable.

**Shuffle the wiring.** Keep every neuron, every synapse, every weight, every sign, and each cell's exact in-degree and out-degree. Randomize only *which cell connects to which*. The eye keeps firing at **69.9 Hz**. Steering goes to **0.0**, and the robot sits still while you walk past it.

That last one is why we believe the claim. Same parts, same amount of wiring, different map, no behaviour. A random network this size does not do this. This specific map does.

### The same connectome fits in your pocket

Everyone here is carrying an ESP32-C3, so we put **150 real connectome neurons and 900 measured synapses** on the badge: LC4 and LPLC2 converging on the giant fibre, the fly's escape reflex, as fixed-point LIF at 100 Hz, verified spike for spike against the Python reference. Swat it and six LEDs flash white. Press **B** and it lesions the wiring out of the eye: the eye keeps firing, nothing reaches the giant fibre, nothing flashes. The whole argument of this project, in 150 neurons in your hand.

It fits because of *how* that circuit is wired. LC4 makes 71 direct synapses onto the giant fibre, monosynaptic. LC10a makes **zero** direct synapses onto DNa02 and every signal takes an obligate two-stage relay. Escape is built for speed, steering is built for computation, and that is why a microcontroller runs a credible escape reflex while the rover needs all 15,000 neurons to turn toward you.

You can also talk to it: an ElevenLabs agent whose entire knowledge of the world is eight tools that read the live brain and press these same controls. An interpreter with a microphone, never in the control loop.

## How we built it

**The map.** Pulled from neuPrint by cell type, expanded one synaptic hop so the real interneurons between eye and motor are present, capped at 15,000. Synapse counts set the weights, predicted neurotransmitter sets the sign. Leaky integrate-and-fire with the constants from Shiu et al. 2024, in three interchangeable engines that agree spike for spike, so the same brain trains on a GPU and drives the robot on a CPU.

**The eye.** One encoder with two front ends. In training it computes the 24 columns analytically from ground truth, so hundreds of arenas run in parallel with no rendering at all. On the robot it computes identical columns from the detector's boxes. The brain never sees a pixel and cannot tell which world it is in. Each LC neuron also gets the retina column matching its real dendritic position in the lobula; before that, a person standing dead ahead drove the right steering neuron at 68 Hz and the left at 0, which on a robot means it swerves past you.

**Learning, the fly's way.** We never train a separate network. Only synapses that already exist change, gated by dopamine the way the fly does it: contact fires the PAM reward neurons, failure fires PPL1. Dale's law holds, bounds hold, no connection is ever created.

**The body.** Camera to detector to retina to brain to descending neurons to wheels on a RoboMaster S1, with a 300 ms watchdog, an E-stop and a speed cap. The whole control loop is local. Nothing in it touches a network.

## Challenges we ran into

**The fly's forward-walking neurons never fired.** DNa01 and DNp09 are the textbook forward locomotion cells and they sat at exactly 0.0 Hz no matter what the eye did. We wrote a tool that strips inhibition off a target neuron, strongest source first, and found two different reasons for silence. DNa10 receives 801 synapses directly from LC10a, the largest direct eye-to-descending projection in the circuit, and is actively held shut by feedforward inhibition driven by that same eye signal; peel two sources off and it jumps to 91 Hz, which matches its published role as an avoidance pathway a fly chasing a mate has no business firing. But DNp09 and DNa01 stay at exactly 0.0 even with every inhibitory input removed, and forcing all 56 missing neurons from the shortest anatomical paths into the circuit changed nothing. No excitation ever arrives.

So we read forward speed from the descending population. That is our engineering decision and we label it as ours everywhere, but it rests on a measurement rather than a guess.

**A prediction the data refuted.** The anterior optic tubercle and lateral accessory lobe carry 80.6% of the bottleneck weight between eye and steering, and they are the pathway the courtship literature already describes. It looks load-bearing. We predicted that lesioning the tubercle would kill tracking. Lesion it and tracking is still **61%**, against 71% for the same number of randomly chosen cells. A connectome tells you where the wiring went; a lesion tells you what it is for. Here they disagree, and we report the disagreement.

**Four biological training runs came out flat.** The mushroom body, the fly's actual learning centre, does nothing in this task: the Kenyon cells never fire, so dopamine-gated plasticity has nothing to act on. An earlier run with a tonic "walking" current did raise contact, and it also pinned steering at 300 Hz. That is a saturated circuit, not a tuned one, so we threw it out, labelled it non-biological in the code, and left it off by default.

## Accomplishments that we're proud of

The shuffle control. Everything else in this project follows from being able to run it in front of you, on a keypress, and have it fail the way we said it would.

Second: **247 passing tests**, and `python scripts/reproduce.py --quick` re-derives the eight headline claims on your machine in 12 seconds. Every number on this page carries its brain file, arena count, seed and episode length.

## What we learned

**Most of the connectome is ballast for this behaviour.** Delete the weakest 95% of the synapses, 2.2 million of them, and turn-toward is still 72%. Delete the same number at random and it is 0% by the time you have removed three quarters.

**The behaviour fits in 2,211 neurons.** Keep the strongest 5% of synapses, then keep only what lies on a path from eye to motor: 2,211 neurons and 10,387 synapses, **0.4% of the original wiring**, 81% turn-toward, and every lesion signature reproduced. That last part is what makes it the same circuit rather than a lucky one.

**It needs no GPU and no framework.** The whole thing rewritten as one 200-line numpy file is bit-identical to the reference spike for spike over 300 steps, and five times faster at batch one: 2.5 ms per camera frame for the full circuit, 0.6 ms for the minimal one.

**What is ours and what is the fly's.** The steering is the connectome's, untrained. Training bought speed, not behaviour: 4.25 s to contact down to 3.23 s at the same 0.977 contact rate, changing 178,445 of 2,334,959 synapses, and lobotomising it back to the raw connectome is bit-identical. The retina encoding, the forward read-out and the exploratory search state are our engineering, and we draw that line ourselves rather than wait for a judge to draw it.

A three-line hand-coded controller beats the connectome at this task, and that is the point. We were not trying to build a good line-follower. We were testing whether real wiring produces goal-directed behaviour with no training. The lesions say it does.

Every motion figure here is from simulation or dry run. The brain has not yet driven a real chassis.

## What's next

Legs. The published map includes the ventral nerve cord, the fly's spinal cord, so the real leg motor neurons are sitting in the file we already downloaded. We read descending neurons and let two wheels stand in for six legs. Closing that gap is the obvious next thing, and the map is already on disk.

## Built with

python, pytorch, numpy, neuprint, connectome, spiking-neural-network, computational-neuroscience, drosophila, opencv, ultralytics, yolo, dji-robomaster, esp32, esp32-c3, lua, sbus, three.js, websockets, timescaledb, elevenlabs, pytest

## Credits and data

- Male CNS connectome: Berg et al., "Sexual dimorphism in the complete *Drosophila* male central nervous system connectome," *Cell*, September 2026. neuPrint `male-cns:v1.0` (HHMI Janelia and the University of Cambridge).
- LIF model and model-ready connectivity: Shiu et al., *Nature* 2024, [philshiu/Drosophila_brain_model](https://github.com/philshiu/Drosophila_brain_model) (MIT).
- FlyWire connectome v783: Dorkenwald et al., *Nature* 2024. NumPy exponential-Euler formulation follows [flypoke](https://github.com/vshapenko/flypoke) (MIT). Person detection: Ultralytics YOLO11n. Viewer: Three.js r180 (MIT).
