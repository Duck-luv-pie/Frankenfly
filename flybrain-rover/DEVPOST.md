# FlyBrain Rover

## Inspiration

Hack the North rewards a demo you can walk up to. The male *Drosophila* connectome (Berg et al., *Cell*, Sept 2026, on neuPrint as male-cns:v1.0) went viral the week it dropped, wired to play Doom and Mario 64. We wanted it driving a body instead: finding and touching a real person, nothing hand-coded in between.

## What it does

We simulate the connectome's eye-to-descending subcircuit, 15,000 neurons and 2,334,959 signed synapses (RESULTS.md 1), as leaky integrate-and-fire neurons with Shiu et al. 2024's constants. A camera frame becomes current on the fly's visual neurons; its descending neurons, which would drive its legs, are read out as wheel commands. Untrained, it turns toward a person in 75% of trials and reaches them 97.7% of the time (RESULTS.md 2).

The demo is two live controls, not the chase. Randomize which cell connects to which, keeping every sign, strength and degree, and the eye still fires at 70 Hz while steering drops from 158 Hz to zero (RESULTS.md 3). Delete the tracking population a quarter at a time and steering fades 158, 137, 83, 40, 0 Hz while the turn holds until three quarters are gone (RESULTS.md 3). Remove the looming detectors instead and the escape reflex vanishes (130 to 0.1 Hz) while steering stays at 75% (RESULTS.md 3).

## How we built it

Camera to `brain/retina.py` (24 columns) to `brain/senses.py` (current onto eye neurons) to `brain/lif.py` (batched LIF on the real wiring, three engines agreeing spike for spike) to `brain/motor.py`. `env/arena.py` is the batched world of people, walls and contact events.

The eye never drives the textbook forward-walking neurons, even after forcing every neuron on the shortest anatomical path into the circuit. So forward speed is our read-out, not the connectome's: the mean rate across all 241 descending neurons (RESULTS.md 6). Every rate feeding it is real; the read-out is ours.

Robot: yolo11n on the laptop GPU, our event-driven LIF engine, a live feed for Ducks's Three.js viewer.

## Challenges we ran into

Forward drive was the real fight: a path exists anatomically from eye to forward neurons but carries no functional signal, and we had to prove that, not assume it.

Four overnight training runs came back flat (RESULTS.md 5): no trend, weights barely moving, because a rover that can't advance never earns reward, and the Kenyon cells, the fly's real learning site, are silent in this task.

A non-biological fallback (constant current into a forward neuron) raised contact but saturated steering at 300 Hz (RESULTS.md 5): a circuit at its ceiling, not a tuned one. We don't call that learning.

## Accomplishments that we're proud of

The lesion table: plenty of connectome demos show a trick working; we show it failing on command when you remove the exact circuit responsible, 0% for both the real lesion and a degree-preserving random shuffle of the same wiring (RESULTS.md 3). Every number here traces to a brain file, environment count, seed and episode length in our logs. Nothing is "it worked, trust us."

## What we learned

A path existing anatomically doesn't mean it carries drive (RESULTS.md 6). DNa10 gets 801 direct synapses from the tracking cells and never fires until two inhibitory populations are lifted off it, then jumps to 91 Hz: the connectome is gating it off, matching its published role as an avoidance pathway.

Anatomy isn't causation. LC10a and its steering neuron share no direct synapse; every signal runs through the anterior optic tubercle and lateral accessory lobe, holding 80.6% of the bottleneck weight (RESULTS.md 4b), which reads as load-bearing. It isn't: lesioning the tubercle leaves tracking at 61% against 71% for a same-size random lesion (RESULTS.md 4b). The figure shows where the weight sits, not what the behaviour needs.

We also rewrote the simulation in one 200-line numpy file, bit-identical spike for spike and five times faster at batch one (RESULTS.md 4): no GPU needed.

## What's next

The robot has not been driven by this brain: the chain runs end to end only in dry run and arena simulation, 15 ms median camera to command (RESULTS.md 7). Getting it onto the real chassis is next.

After that: wire in the exploratory search state we built (labelled non-sensory, off by default) instead of starting with a person in view, and rebalance the dopamine channel so training stops saturating steering.

## Built with

Python, PyTorch, neuPrint (Janelia), Ultralytics YOLO11n, OpenCV, DJI RoboMaster SDK via an ESP32 S-Bus link, Three.js, Vast.ai A100 for training only, never in the control loop.

## Sponsor prize paragraphs (paste one under each prize on Devpost)

**MLH: Best Use of ElevenLabs.** You can talk to the fly. An ElevenLabs conversational agent whose only
knowledge of the world is eight client tools that read the live brain and press the demo's own controls: ask
what it sees and it answers from its retina columns with a bearing and distance; tell it to remove its eye and
it lesions LC10a, reads its own steering neurons, and reports the rates going from 251 Hz to zero while the eye
still fires. The language model is an interpreter with a microphone, not part of the control loop; the brain runs
identically with it closed. Ten narration lines generated with eleven_v3 and cached are the no-network fallback.
`scripts/talk.py`, `scripts/voice.py`.

**MLH: Best Use of Tiger Data.** The robot's brain is a 50 Hz time series: 25 population firing rates, 24 retina
columns, the motor command and every spike in a 512-cell sample, about 60,000 rows per 15-second episode. Frames
stream from the demo into TimescaleDB hypertables as they happen (COPY, one batch per second), a continuous
aggregate `rates_1s` rolls them up per second with the newest second computed on read, compression policies keep
a free-tier instance sufficient, and a wall chart polls the aggregate once a second. Four recorded episodes are
loaded; the lesioned one shows DNa02 flat at zero while the giant fibre still fires at 84.5 Hz, which is the
whole argument, queryable. `scripts/export_timeseries.py`, `scripts/live_chart.py`.

**MLH: Best Domain Name from GoDaddy Registry.** The domain does a job: it installs the fly's escape circuit on
any Hack the North badge. One page with the circuit running live in the browser, a Copy-the-app button carrying
the whole 16 KB single-file app, and the four badge-IDE steps. Every attendee has the same badge, so the domain
is the distribution point for putting a fly brain on a thousand of them. `site/index.html`.
