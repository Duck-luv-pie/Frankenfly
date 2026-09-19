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
| Pruned circuit | neurons within 2 synaptic hops forward of the input groups **and** 2 hops backward of the readout groups, plus the inhibitory closure: **46,072 neurons, 1,241,045 connections** (visual, wind, smell, taste, touch and humidity inputs) |
| Model | LIF: rest −52 mV, threshold −45 mV, τ_m 20 ms, alpha synapse τ 5 ms, refractory 2.2 ms, delay 1.8 ms, 0.275 mV per synapse × count × sign (ACh +, GABA/Glu −), dt 0.1 ms |
| Stimulation | driven neurons spike as a Poisson process at 0–150 Hz (the reference model's Poisson input is super-threshold, so this is equivalent) |
| Speed | numba kernel, ~1.7× real time on an Apple Silicon laptop; `--full` (all neurons) ~0.55× |
| Pacing | `runner.py` advances the brain in 10 ms chunks to match the wall clock; if it falls > 1 s behind it skips ahead rather than lagging forever |

### Making it a living brain (deviations from the reference model)

The reference model has **no spontaneous activity**: with no sensory input every neuron is
silent, and a silent brain cannot walk, groom or explore. Three additions, all configurable in
`lif` / `prune` / `decode` of `default.yaml`, turn it into a brain that is always doing something
and that the senses *modulate* rather than *create*:

| Addition | Config | Why |
|---|---|---|
| Spontaneous firing floor, 1 Hz Poisson on brain-intrinsic neurons only (central, descending, ascending, centrifugal, endocrine) | `lif.background_hz`, `lif.background_classes` | A real brain is never silent. Sensory and visual-projection neurons are excluded so they fire only when the camera or PIR drives them; hundreds of looming cells converge on the Giant Fiber and even 0.5 Hz of noise on them fires it. |
| Spike-frequency adaptation, 2 mV per spike decaying with τ = 200 ms | `lif.adapt_mv`, `lif.tau_adapt_ms` (also `lif.depress_u` for optional synaptic depression) | The plain LIF has no rate limit except the refractory period, so a self-exciting clique of antennal-lobe local neurons runs away to ~400 Hz and drags the rest along. 2 mV is the compromise found by sweeping: it keeps resting descending activity low and the looming → Giant Fiber, dust → grooming and sugar → proboscis pathways all working, while 5 mV or any synaptic depression silences the polysynaptic sugar pathway. The olfactory clique still saturates; it is documented, not hidden. |
| Inhibitory closure in pruning: also keep inhibitory neurons with ≥ 20 synapses both from and into the circuit | `prune.inhibitory_closure_min_synapses` | Path pruning keeps excitatory chains but drops the inhibitory interneurons hanging off them; the full brain keeps the Giant Fiber at 0 Hz at rest while the unclosed pruned circuit had it at 25 Hz. Adds ~7k neurons (36k total). |

| Antennal-lobe local neurons forced inhibitory | `lif.force_inhibitory_classes: [ALLN]` | The connectivity table signs synapses by predicted transmitter, which makes 62% of the synapses of antennal-lobe local neurons excitatory although these cells are overwhelmingly GABAergic or glutamatergic. They form a self-exciting clique (the 400 Hz hubs), projection neurons sit at 82 Hz, Kenyon cells fire at 12 Hz and every odor looks the same. With the correction: mean resting rate 0.44 Hz, no hub above ~50 Hz, projection neurons at 1 Hz responding to odor, Kenyon cells silent at rest with ~9% responding to an odor and near-zero overlap between odors. |

`companion test-gf` and the reference test in `tests/` switch these off and reproduce the
paper's behavior (silent brain, Giant Fiber at ~170 Hz on looming).

## Learning: the mushroom body

The fly's reinforcement-learning circuit is simulated with its own rule (`sim/plasticity.py`,
config `learning`, toggle from the dashboard, `--no-learn`, or `learning.enabled`):

- **Three-factor depression.** A Kenyon-cell → MBON synapse is weakened when its Kenyon cell
  fired above its own resting rate within the last ~2 s (eligibility trace) *and* dopamine
  arrives in that MBON's compartment. Weights floor at 10% and relax back over ~15 minutes.
- **Compartments from the wiring.** For each MBON, its dopamine signal is the rate (above a slow
  resting estimate) of the PAM or PPL1 neurons that synapse directly onto it. Reward DANs (PAM)
  innervate the MBONs that drive avoidance, punishment DANs (PPL1) the MBONs that drive approach
  (Aso et al. 2014b), so the valence readout uses those wiring-defined sets: reward pairing
  weakens avoidance of the current smell.
- **Reward delivery.** Sugar at the grape drives the sugar gustatory neurons *and* the PAM
  dopamine neurons directly (the model's wiring does not carry sugar to PAM; in vivo the reward
  signal also arrives through nutrient sensing). Punishment can be wired the same way
  (`learning.punish`) but nothing in the world delivers it by default.
- **KC → MBON gain ×3.** The LIF's MBON odor responses are ~0.5 Hz, far below the tens of Hz
  measured in vivo; scaling these synapses gives ~10 Hz responses that learning can abolish.
- **Behavior.** The learned MBON valence steers the fly: approach turns toward the stronger-
  smelling antenna and speeds up, avoidance turns away (`learning.valence_steering`).

**The two-odor arena (second world).** The world panel switches between the *table* and a
*two-odor arena*: a plain 24 × 9 mm chamber with the grape's scent from one end and the lemon's
from the other, scent plumes drawn on the floor, nothing else to see. "Run assay" runs the classic
choice test on the live brain: a 60 s naive test counting time spent on each half, then training
with a sugar drop at the grape end until the fly has fed for 12 s (or 90 s), then a 60 s test.
The preference index PI = (t_A − t_B)/(t_A + t_B) is shown before and after; switching plasticity
off first gives the control. Learned MBON valence reaches behavior through the valence-steering
coupling, so the trained fly turns toward and speeds up in the grape's plume.

**Training many flies at once.** `companion batch --runs N --control --save` runs the same arena
assay headless (`sim/arena.py` is a port of the world's kinematics) with one brain per worker
process: N flies with learning on and N with it off, on as many cores as you have. Each assay is
about 3 minutes of brain time; a 14-core Mac runs 12 of them in about 4.5 minutes of wall time,
so a day of fly-time takes an hour. It prints every fly's naive → trained preference index and
the group means, and `--save` stores the best fly's learned KC→MBON synapses so the live fly can
start already trained: `companion run --load-weights data/cache/learned_weights.npz`. Steering
in both worlds is klinotaxis on the learned value: the fly turns when its mushroom-body valence is
falling and runs when it is rising, plus a small left/right antenna comparison.

**The two-odor experiment (headless, neural readout).** The world has two smells: the grape (odor A, glomeruli DM1/DM4/
VA2/DM2) with sugar, and a lemon (odor B, glomeruli VA6/DL1/VM4/DM3/DL4) that is never rewarded.
`companion experiment` runs it headless: probe both odors, train (A + sugar, then B alone), probe
again. Result: odor A's drive onto avoidance-MBONs is abolished (≈ 8.6 → 0.1 Hz) while its drive
onto approach-MBONs and odor B's responses are unchanged, and the synapses from A-cells fall to
≈ 0.13 of naive vs ≈ 0.3 for B-cells (B shares some cells). `--no-learn` is the control. In the
live world the Learning panel shows dopamine, compartment weights and eligible Kenyon cells, and
"probe odors" replays both smells on a scratch copy of the brain.


### Reading a noisy brain: baselines and z-scores

Because every readout group now has a resting rate, behaviors are measured as deviations from
the fly's own baseline. At startup the brain runs with no senses (3 s warm-up, then 6 s), and the
mean and std of every readout group's rate is recorded per side and per rate window (cached in
`data/cache/baseline_*.json`). Every readout is then a z-score, mapped through a dead zone of one
sigma; a slow drift (τ = 120 s) re-centers the means on the running brain. Tiny groups (the Giant
Fiber is 2 neurons) use longer windows so a couple of chance spikes cannot label a behavior.

**Continuous motor channels are what move the fly.** `decode.motor.channels` maps descending
activity to smoothed 0..1 channels every frame with no thresholds: forward walking from the
whole descending population (674 neurons), turning from DNa01/DNa02 right minus left, backward
from MDN, grooming from DNg11/12, wing threat, one-wing song, landing, freezing, and the Giant
Fiber jump. The behavior state label (priority + hold) survives only as a caption and for the
robot's eyes and sounds. Nothing in the fly display is scripted: idle is whatever the descending
neurons happen to be doing.

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

## The fly's world (closed loop)

The dashboard is not just a display: it is the world the fly lives in, and the fly perceives it
through its own senses. `ui/fly3d.js` builds a table with a grape, a flower with pollen, a water
drop, pebbles, a leaf, a second (scripted) fly, and a window onto the human world on which the
webcam plays. So you are in the fly's world too, seen through the window.

| Sense | How the world computes it | Neurons driven | What the brain does with it |
|---|---|---|---|
| Vision | a 95° camera on the fly's head renders an 80×60 **retina** at 10 Hz, posted to `/retina`; the optic lobe processes exactly these pixels (the webcam is only a fallback when no page is open) | LC4/LPLC2 looming, LC16, LC11/LC10a small objects, LC12/15 bars | escape, tracking, freezing, landing |
| Efference copy | the world reports `self_motion` (walking speed, turning, flight); visual drive is scaled by 1 − 0.8·self_motion | | the fly's own movement moves the whole retina and must not read as looming (Kim et al. 2015) |
| Smell | fruit-ester concentration at each antenna, 1/(1+(d/4 mm)²), scaled by hunger (Root et al. 2011) | `ORN_fruit` = ORN_DM1, DM4, VA2, DM2, per side | antennal lobe → lateral horn / mushroom body → whatever it does |
| Taste | standing at the split in the grape's skin, scaled by hunger | `GRN_sugar` = sugar/water gustatory neurons | proboscis + ingestion motor neurons (`MN_proboscis`) → **feed** channel → proboscis extends, satiety rises |
| Touch | walking into the flower dusts the fly; dust decays only by grooming (`groom` channel) | `BM_eye` (interommatidial bristles), `BM_head` | `DN_groom` = DNg35, DNg84, DNg15, DNge132, DNg87, the descending neurons the connectome shows receiving the strongest direct bristle input (DNg11/12 are not reached), plus DNg11/12 |
| Humidity | 1/(1+(d/2.5 mm)²) from the water drop | `HRN_moist` = HRN_VP4 | logged; no behavior mapped yet |
| Wind / presence | PIR on the robot body | `JO_wind` | startle, grooming |

The webcam is a toggle ("webcam on" in the Fly's eyes panel, `--no-webcam` to start closed).
Closing it releases the camera device, the window in the fly's world shows drawn curtains, and
the brain gets no webcam frames; the fly's own retina keeps working as long as the page is
visible (a hidden tab stops rendering, and then the fly is blind until it is shown again).

Body state that lives in the world, not the brain: satiety (rises while feeding, decays over
~4 min and scales smell and taste), dust (pollen on the body), flight (a Giant Fiber takeoff
starts a flight that lasts until the landing neurons DNp07/DNp10 fire or ~9 s pass; the retina
sees optic flow the whole time, so landing is closed-loop too).

Everything else the fly does is decoded from descending neurons as described above; the only
scripted animal on the table is the neighbour fly, which is scenery.

## Live dashboard

`companion_brain/ui/server.py` is a stdlib HTTP server started by `run` (port 8600). It serves
`ui/index.html`, `/circuit.json` once (normalized FAFB x/y position, role and cell type of every
simulated neuron), `/frame.jpg` (latest webcam frame, which the world paints on its window),
`/atlas/*` (the MaleCNS neuron atlas data), `/events`, a server-sent-events stream at the body rate carrying the body packet plus behavior
scores, motor channels, modulators, sensory features, world senses, readout rates, the indices of
neurons that spiked since the last event, and the cell types with the most spikes; and it accepts
`POST /retina` (raw 80×60 gray frame from the fly's eyes) and `POST /world` (odor per antenna,
sugar, dust, moist, satiety, self-motion). The page is plain HTML/canvas plus vendored Three.js.

## The brain view: MaleCNS atlas lit by FlyWire spikes

The Brain panel renders the Fly / Neural Atlas data (`brain/data/atlas/male-cns`, exported by
that project's `prepare-neurons.py` from the MaleCNS v1.0 connectome, HHMI Janelia × Google
Research, CC BY 4.0): 165k catalogued neurons with 140k cell-body positions and 1,508 complete
skeletons, drawn with the atlas viewer's shader (an activity texture indexed by catalogue slot,
`ui/atlas.js` adapted from its `neural.js`, `ui/atlas-data.js` copied from `neural-data.js`).

The simulation runs FlyWire (a female brain); the atlas is a male CNS. Neuron ids do not
correspond, so `ui/atlas_map.py` maps each simulated neuron to a MaleCNS neuron of the same
**cell type and side** (round-robin, preferring neurons with a bundled skeleton), falling back
to the same type on any side, then to a subtype-prefix match (pC1a → pC1_*, KCab → KCab-*).
About 71% of the simulated neurons map; the rest are optic-lobe types the two nomenclatures
name differently and are simply not lit. Every spike from a mapped neuron is a flash on its
counterpart's cell body and, when bundled, its whole skeleton. What you see is therefore the
real anatomy of the fly's nervous system with our simulated activity painted onto it by
identity, not a rendering of the simulated neurons' own positions.

## Hunting: the fly as a vehicle among humans

The third world (`hunt arena` on the dashboard, `companion hunt` headless) is a meter-scale room
ported from the Fly / People Lab arena of the fly project, with its four walking humans (the GLB
exports in `ui/models/humans/`, patrolling seeded ellipses; `sim/hunt_arena.py` uses a bit-exact
port of that project's seeded generator, so a training seed shows the same layout on the page).
The fly is a 32 cm ground vehicle: the descending forward / backward / turn channels set its speed
and heading rate, like an RC car. It cannot walk or fly (the jump channel is ignored, legs are
tucked, wings folded). Touching a person delivers sugar + reward dopamine for 2 s; running out of
time (45 s) delivers bitter + punishment dopamine (`GRN_bitter`, `DAN_punish`) and the episode ends.

**The hunter's rules** (config `hunt`): a person is a solid cylinder of radius 0.20 m, and a touch
counts only when the vehicle's nose (16 cm ahead of its centre) reaches the skin; brushing past
with the flank does not. Head-on contact (within 30°) earns the full reward, a glancing contact
half (`side_reward`, in both the sugar drive and the score). The hunter never escapes: the looming
features that recruit the Giant Fiber and LC16 are zeroed (`hunt.overrides`), the freeze channel
does not brake it and there is no reverse; it only drives forward and turns. Its eye is wide
(150°, like a fly's), and the object drive is retinotopic (`senses.features.small_object.azimuth_weight`:
a person far out in a hemifield drives that side harder than one near the midline, so steering is
proportional rather than bang-bang).

Two senses, both real neuron groups:

| Sense | World model | Neurons | What the brain does with it |
|---|---|---|---|
| Heat | two sensors on the head looking 45° left / right (cosine lobes), each summing every person's warmth 1/(1+(d/3 m)²) | `TRN_hot` = `TRN_VP2`, the arista's hot cells, 3/4 per side | antennal lobe VP2 glomerulus → projection neurons → lateral horn and mushroom body |
| Vision | each person in the 150° field of view is a small dark moving object and a vertical bar on its hemifield (computed analytically in training; on the dashboard it is the real retina) | `LC11`, `LC10a`, `LC12`, `LC15` | the innate pursuit circuit: an object on the left fires the left DNa02 at ~90 Hz |

**The innate hunter.** Before any training the connectome already chases: the LC10a → AOTU → DNa02
pathway turns the fly toward a seen object and the descending population fires with it (forward
drive). Measured on the untrained brain: 83% of 45 s episodes end in a touch, mean time to touch
~22 s. Note the sign: DNa02 drives an *ipsilateral* turn (Rayshubskiy et al. 2020) and the turn
channel is right minus left, so in this world +turn = turn right (`hunt.turn_sign`); the other two
worlds' convention would make the reflex steer away.

**What learns.** Two things, both the fly's own:

1. *Mushroom-body plasticity.* A touch arrives while the Kenyon cells that encoded the last seconds
   are still eligible, so their synapses onto avoidance MBONs are depressed and that situation gains
   approach valence, which steers the vehicle (valence × heat gradient, and klinotaxis on the valence
   trend, as in the other worlds). Only heat reaches the mushroom body: the hot cells drive the VP2
   projection neurons at ~90 Hz and 303 Kenyon cells receive their synapses, but one or two PNs per
   glomerulus cannot fire a KC at the model's 0.275 mV per synapse (0 KC spikes even at 400 Hz of
   hot-cell drive). `lif.synapse_gains` scales the VP-PN → KC synapses ×4 (the same kind of
   calibration as `learning.kc_mbon_gain`): ~3% of KCs respond to heat, the left- and right-heat
   sets are disjoint, rest stays silent and odor responses are unchanged. The LC visual neurons reach
   no Kenyon cell within two hops, so vision drives the innate pursuit only; the fly learns *warmth*.
2. *Gains around the wiring.* `sim/hunt.py` runs CMA-ES (dependency-free, `CMAES`) over 15 scalars
   (`PARAMS`: sensory gains, heat lobe and range, efference copy, steering ratio, readout thresholds
   and smoothing, plasticity rate and time constants). Every candidate is a fresh brain that lives
   through N episodes with plasticity on; its fitness is the hunting score of the later episodes, so
   what is optimized is a fly that *learns* to hunt. Score = touches (head-on 1, glancing 0.5) +
   0.5 × how early + 0.3 × fraction of time facing someone + 0.3 × path efficiency (straight-line
   distance over distance driven).

```sh
uv run companion hunt --episodes 6                       # one fly, learning on, prints each episode
uv run companion hunt --episodes 6 --no-learn            # control
uv run companion hunt --scripted                         # a hand-written hunter: the arena's reference
uv run companion hunt --train --generations 20 --episodes 6   # CMA-ES, all cores; writes data/cache/hunter_gains.json
uv run companion hunt --load data/cache/hunter.npz --no-learn --episodes 10   # evaluate the trained fly
uv run companion run --load-weights data/cache/hunter.npz --open              # watch it on the dashboard (hunt arena)
```

**Result (2026-09-15, 20 generations × 12 flies × 6 episodes, 68 min on 10 cores).** Best search
score 1.71; the final fly touched someone in 23 of 24 training episodes. On ten held-out layouts
with learning frozen:

| Fly | Touches | Mean time to touch | Time facing someone |
|---|---|---|---|
| untrained brain, default gains | 10/10 | 18.0 s | 22% |
| tuned gains, naive synapses | 10/10 | 10.4 s | 44% |
| trained fly (tuned gains + learned synapses) | 10/10 | 10.6 s | 46% |

So the tuning halves the time to touch and doubles the time spent facing a person, and it does so
through the innate circuit: CMA-ES drove `learning.valence_steering` to 0 and the efference copy
to 0, and made the steering readout fast and sensitive (turn `z_ref` 1.9, `z0` 0, 100 ms
smoothing, wide 80° heat lobes with a short 1.1 m range). The mushroom body does learn from the
touches (reward-compartment KC→MBON weights fall to 0.89) but the optimizer found its valence
signal unhelpful for steering, so the learned synapses add nothing measurable on top of the gains.
Honest reading: this fly hunts with its eyes and its pursuit reflex; its memory of warmth is
formed but not used.

**Second run (2026-09-16, the hunter's rules above: nose contact, no escape, forward only, 150° eye;
40 generations × 14 flies × 8 episodes, 158 min on 12 cores, warm-started from the first run's
gains).** Best search score 1.95; the final fly touched in 30 of 30 training episodes, 26 head-on.
On twelve held-out layouts with learning frozen:

| Fly | Touches | Head-on | Mean time to touch | Facing | Path efficiency |
|---|---|---|---|---|---|
| untrained brain, default gains | 11/12 | 4 | 15.6 s | 31% | 0.36 |
| first hunter's gains, under the new rules | 12/12 | 9 | 8.4 s | 46% | 0.64 |
| second hunter (this run) | 12/12 | 9 | 8.4 s | 42% | 0.63 |

The new rules and the removal of escape are what made the fly a hunter; the second search found
the same plateau as the first (its gains: forward `z_ref` 0.4 so it drives at the slightest
descending activity, turn `z_ref` 3.6 with `turn_gain` 0.88, heat lobes 41°, a small retinotopic
weighting of 0.18, still no valence steering). Roughly a quarter of touches are glancing: the
vehicle arrives at a walking person from the side because the pursuit reflex steers at the person's
current bearing, not where they will be.

`--train` ends by training one fly for `--final-episodes` with the best gains and saving
`data/cache/hunter.npz`: the learned KC→MBON synapses + the tuned gains + its record. That file is
the trained fly; `run --load-weights` applies the gains before building the brain and loads the
synapses. Each candidate costs about `episodes × 45 s` of brain time at ~1.6× real time, so a
population of 12 on 12 cores is ~3 min per generation.

### The hunt on a GPU (`companion hunt-gpu`)

A spinoff of the hunt for training at scale, in `brain/companion_brain/hunt_gpu/` (needs the `gpu`
extra: `uv sync --extra gpu` installs PyTorch; CUDA, Apple MPS and CPU all work, `--device` picks).
The connectome trainer above runs one 139k-neuron brain per core at ~1.6× real time, so a
generation of 12 flies costs minutes. Here the room is a batch of tensors and the fly a compact
network, so thousands of rooms advance one tick per kernel launch:

- **`arena.py` — `BatchArena`.** A vectorized port of `sim/hunt_arena.py`: the same room, walls,
  solid people on their seeded patrol ellipses (the same mulberry32 layouts, so a seed shows the
  same people and start pose on the dashboard), the same RC-car motion (forward and turn drive, no
  walking, no flight), the same nose-contact rule and head-on / glancing distinction. The tests
  drive the two arenas step for step and they agree to a millimetre. An episode ends the instant of
  a touch or at the 45 s timeout; the 2 s sugar / bitter phases are not needed, the reward is a
  number. What the fly senses: the two hot-cell lobes (heat left / right, exactly as on the CPU),
  `vision_bins` retinotopic columns across the 150° field with an object channel (angular width of
  a person) and a motion channel per column, like the LC columns, and its body (speed, previous
  drives, time elapsed). Rooms can hold a random 1–8 people per episode (`people_min` / `people_max`).
- **Senses have a range** (`hunt_gpu.senses`): a person is seen only within 6 m and felt only
  within 5 m (the warmth fades over the last metre); beyond that the fly's inputs are zero. When
  nobody is in range the fly is paid for every new 1 m cell of the room it drives into and charged
  for standing still (`reward.explore`, `reward.idle`), so it searches rather than freezes; the
  progress and facing terms below apply only to a person it can sense, and losing sight of the
  target lets it switch to anyone else it senses for free.
- **The reward** (config `hunt_gpu.reward`) is written for "pick a target, hit it fast, then stay on
  it": the first touch pays 10 (× 0.5 if glancing) plus 10 × the fraction of the episode still left;
  until then each tick pays the metres closed on the *chosen* target (the person nearest at the
  start, changed only when someone else is a metre closer, each change costing 0.5) minus a small
  time cost, plus a small bonus for keeping the target ahead. The first touch locks the target and
  the episode runs the whole 45 s clock (`hunt_gpu.track`): every tick with that person's skin
  within 0.6 m of the nose and within 45° of straight ahead pays 0.05, every renewed contact pays 2
  (at most once per 2 s), letting them get more than 2.5 m away costs 0.02 per tick, and the
  progress term keeps paying for closing on them as they walk off. A 45 s episode without a touch
  costs 5. `track.enabled: false` restores the end-at-first-touch episode of the CPU arena. The CPU
  hunting score is computed from the first-touch statistics, so GPU and connectome flies are
  compared on one scale; `track` (fraction of the time after the first touch spent close behind
  the locked person) and `touches` are reported alongside.
- **The humans** (config `hunt_gpu.humans`) are the other player. `patrol` keeps the CPU arena's
  seeded ellipse walkers (the fly trained against them learned their orbits, which is why they are
  no longer the default). `flee` is a scripted reflex: run straight away from the fly at 2.0 m/s,
  faster than the fly's 1.6, blending in a turn toward the middle near a wall. `learn` (the default)
  makes every person a runner driven by `EvaderNet`, one small feed-forward policy shared by all
  people over each runner's own-frame view (the fly's distance, bearing, speed and heading, its own
  speed and yaw, the four walls, the nearest other person, whether the fly is locked on it, the
  clock). Runners are rewarded per tick for distance kept from the fly and punished when caught
  (the one touched), tracked or touched again, so their reward mirrors the fly's; PPO trains them in
  the same rollouts as the fly, with their own optimizer (every living person at every tick is one
  sample). The fly, slower, has to corner or intercept; the runners have to learn the room. Runners
  tire (`humans.stamina`): sprinting above half speed empties the tank in 4 s and an exhausted
  runner must stand and rest until it is full again (3 s), so a fly that keeps tracking catches up
  while its person rests. A first run without stamina showed why this matters: the runners learned
  to keep away within 100 updates and the fly's catch rate fell below 1%, leaving it nothing to
  learn tracking from. A second run with stamina alone showed the runners learning to pace
  themselves below the sprint threshold and never resting, so on top of stamina every runner is
  made to stop now and then whatever its policy says (`humans.pause`: every 4–12 s, for 2.5 s), the
  player stopping to let the fly catch up. Stamina and the rest state are part of what a runner
  senses, so a learned runner can plan its sprints. Runners mind the fly only when it is within
  `alarm_m` (4 m, with hysteresis out to 6 m); otherwise they stroll between random waypoints at
  0.8 m/s, and their policy is trained only on the ticks they were minding it. A version without
  this had every runner cowering in the corner furthest from the fly for the whole episode. In adversarial training the saved pair is the one at the end of the
  run (an early "best" fly would only be one that met untrained runners). The trained runners are
  saved next to the fly as `evaders_gpu.pt` and picked up automatically by `--load` (or given with
  `--humans PT`; `--humans patrol|flee` selects the scripted kinds).
- **`brain.py` — `HunterNet`**, shaped like the fly's hunting circuit: hot cells → projection
  neurons → a fixed random Kenyon-cell expansion with k-winners-take-all (a ~3% sparse code) →
  MBON features; the retinal columns → two convolutions over azimuth (the optic glomeruli) → a
  visual vector; a GRU as the central complex (heading memory, target commitment); a policy head
  (forward and turn, Gaussian with a learned spread) and a value head, with a direct sensory →
  motor shortcut like the innate LC10a → DNa02 pursuit so the fly starts near a reflex. ~115k
  parameters. `hunter_gpu.pt` holds the weights, the sizes and the observation layout.
- **`ppo.py` — recurrent PPO.** Each update rolls every room forward `steps` ticks (the hidden
  state runs on across episode boundaries; a new episode announces itself in the body inputs),
  computes GAE, then replays whole sequences through the fused GRU in minibatches with the clipped
  surrogate, value clipping and an entropy bonus. Finished episodes are logged with the hunting
  score; the best checkpoint by touch rate then score is `hunter_gpu.pt`, the latest
  `hunter_gpu_last.pt`, the curve `hunt_gpu_train.jsonl`.
- **`bridge.py`** drives the CPU arena with a saved network (`--cpu-check`), the proof that the
  batched room is the real one, and turns the network into a `policy(arena) -> motor dict` in the
  brain's convention.
- **`viewer.py` + `ui/hunt_gpu.html` — the live view** (`--watch`, <http://localhost:8601>). One
  room of the batch arena runs at real time with the saved fly (or the scripted hunter), and the
  page draws the dashboard's 3-D room: the four GLB humans walking, the Body Lab fly rig as the
  32 cm vehicle with its 150° field of view on the floor, its path, the line to its chosen target
  and a green ring on the person it is locked on. The panel shows what the fly senses each tick
  (heat left / right, the 24 retinal columns with motion tint), its forward / turn drive, the
  running reward, a chase camera (or free orbit), pause / next episode / playback speed, and the
  last episodes' first-touch time, tracking fraction and contacts. An inset in the corner is what
  the fly sees: the room rendered from a camera on its head with its field of view, and its
  sensing range drawn as fog. The **lobotomize** button swaps in a brain of the same shape that
  never learned anything (it twitches at random); pressing it again puts the trained brain back.
  Nothing is deleted, the two networks sit side by side and the button chooses which one drives.

```sh
cd brain && uv sync --extra gpu
uv run companion hunt-gpu                                  # the scripted hunter through 256 seeded rooms at once (arena check)
uv run companion hunt-gpu --train                          # PPO: 2048 rooms x 64 ticks per update, 300 updates; the fly vs runners learning to evade -> data/cache/hunter_gpu.pt + evaders_gpu.pt
uv run companion hunt-gpu --train --humans flee            # ... vs the scripted runners only (or --humans patrol: the old walkers)
uv run companion hunt-gpu --train --envs 8192 --device cuda --updates 600   # a real GPU
uv run companion hunt-gpu --load data/cache/hunter_gpu.pt --episodes 512    # evaluate vs the trained runners next to it (mean actions, no sensor noise)
uv run companion hunt-gpu --load data/cache/hunter_gpu.pt --humans flee     # ... vs the scripted runners
uv run companion hunt-gpu --load data/cache/hunter_gpu.pt --cpu-check -v    # the same fly through the CPU arena, same seeds
uv run companion hunt-gpu --load data/cache/hunter_gpu.pt --watch --open     # watch it hunt, live, in the 3-D room
```

**Results (2026-09-16).** Against the patrol walkers (120 updates, 2.9 min): 100% first touches at
3.46 s, tracking the locked person 90% of the time afterwards with 19 contacts per 45 s episode,
score 2.00 (the scripted hunter: 85% head-on, 31% tracking, 1.84); the same fly through the CPU
arena on the same seeds scored 1.94. Against the runners, with the ranged senses, the search
reward, the alarm radius and the forced stops (fly and runners trained together from scratch,
300 updates, 39 M ticks, 12 min), on 512 held-out layouts (seed 777, mean actions, no sensor
noise), the whole 45 s clock:

| Hunter | Humans | First touch | Time to first touch | Tracking afterwards | Contacts / episode | Humans' mean distance | Nobody in range |
|---|---|---|---|---|---|---|---|
| scripted hunter | scripted runners | 100% | 7.2 s | 27% | 5.3 | 1.4 m | 2% |
| scripted hunter | trained runners | 99% | 13.1 s | 12% | 2.5 | 2.2 m | 4% |
| trained fly | scripted runners | 100% | 7.0 s | 53% | 9.4 | 1.1 m | 1% |
| trained fly | trained runners | 99% | 15.1 s | 25% | 4.4 | 1.6 m | 1% |

The fly now finds everyone (nobody in range only 1% of the time, and it stands still for 4% of
that), catches 99% of the trained runners and tracks them a quarter of the time after the first
touch (half the time against scripted runners, twice the scripted hunter's contacts). The runners
trained under these rules are only a little harder than the scripted ones: allowed to react only
within 4 m and made to stop every few seconds, they cannot hide, which is the point. Earlier
rounds are worth remembering: with unlimited evasion the runners won outright (fly under 1%); with
stamina alone they paced themselves and never rested; with unlimited senses and no alarm radius
they huddled in the far corners and the fly waited for their pauses instead of hunting.
Throughput was 45–90k ticks/s on the Apple GPU with 2048 rooms; CPU ~6k; every tick is a handful
of small kernels, so a CUDA GPU with 8k+ rooms is where the batch pays off. The saved fly is a
network, not connectome synapses: `run --load-weights` takes `hunter.npz`, not `hunter_gpu.pt`.

### The real fly's brain on the GPU (`companion hunt-brain`)

Everything above trains a network shaped like the fly. `companion hunt-brain` trains the fly: the
hunter subcircuit of the FlyWire connectome, run as a batched rate model in the same arena, with the
learned changes written back into the spiking brain so `companion hunt --load`, `companion run
--load-weights` and the live view run it as spikes. Code in `hunt_gpu/connectome.py` and
`hunt_gpu/brain_train.py`, config `hunt_gpu.brain`.

- **The subcircuit** (`HunterCircuit`): from the pruned circuit, every neuron within 2 synapses
  downstream of the hunter's senses (LC10a, LC11, LC12, LC15, the arista's hot cells) and 1 synapse
  upstream of its descending readouts (DNa01, DNa02, the descending population, DNp09, MDN), plus
  those groups themselves and all connections among them: 8,503 neurons, 244,545 synapses (4,444
  central, 2,507 visual projection, 1,299 descending). Wiring, signs and synapse counts are the
  connectome's; the runner's VP → KC gain applies as in the spiking brain.
- **The rate model** (`RateBrain`) is the mean field of `sim/lif.py`: a neuron's input is the sum
  of its presynaptic rates times the signed weights times τ_syn; its rate follows the LIF f-I curve
  1/(t_ref + τ_m ln(g/(g − 7 mV))) gated by a soft threshold (the spiking brain's Poisson noise
  fires neurons slightly below threshold), with the mean of the spike-frequency adaptation, the
  1 Hz spontaneous floor of central neurons, and stimulated sensory neurons at their drive rate.
  Two constants (soft threshold 2.5 mV, synaptic scale 0.75) were fitted against the spiking
  subcircuit's responses: a left object at 150 Hz drives the left DNa02 to 154 Hz (spiking: 107)
  and leaves the right one silent (both), rest is near the 1 Hz floor (both). Senses and readout
  are the real ones: the arena's heat and vision become Poisson rates on the same neuron groups as
  `sim/hunt.py`, and forward / backward / turn are the z-scored descending rates of
  `body/decode.py` against a resting baseline whose stds are the spiking subcircuit's.
- **What learns**, all of it legal for the spiking brain: a gain on every synapse (0.1–10, sign
  kept), a threshold offset on every neuron (±4 mV), and the sensory gains and decoder thresholds
  that `sim/hunt.py` already tunes (the CMA-ES fly's values are the starting point). A penalty keeps
  gains and thresholds near the connectome's. The policy is the decoded drive plus Gaussian
  exploration; a small critic over the observation is training scaffolding. PPO backpropagates
  through the brain one arena tick at a time (the neuron state is detached between ticks): 512
  rooms × 32 ticks per update, about 17 s on the Apple GPU. On Apple's MPS the gradient of a sparse
  matmul never reaches its values, so the gradient passes there use a dense weight matrix built once
  per minibatch as a leaf; elsewhere the sparse form.
- **Back into spikes**: `hunter_brain.npz` holds the gains keyed by presynaptic and postsynaptic
  FlyWire root id and the offsets by root id, so `apply_to_runner` writes them into any
  `BrainRunner` (the subcircuit or the whole pruned circuit). `--spiking` evaluates the subcircuit
  as LIF neurons in the batch arena (one room at a time), `--watch` shows it hunting live with its
  neurons lighting up (frontal view, gold for the visual inputs, orange for the hot cells, green for
  the descending neurons), `--check` prints spiking against rate-model responses, and the lobotomy
  button swaps in the same neurons without the learned gains.

**Status (2026-09-16).** First run: 150 updates × 512 rooms against the benchmark's trained runners
(frozen), 44 min, starting from the CMA-ES fly's gains. The rate-model fly then catches 95% of the
trained runners (mean first touch 15.3 s) and 100% of the scripted ones (8.1 s); as spiking
neurons (the subcircuit, learned gains applied) it caught the scripted runner in 6 of 6 held-out
episodes (mean 13.4 s) and, loaded into the whole 48k-neuron pruned circuit through `companion
hunt`, 3 of 3. So the real fly's brain hunts, and its neurons can be watched doing it. What it does
not yet do is track: 3–4% of the time after the first touch against the network fly's 25%, and
when it senses nobody it stands still (the descending population at rest decodes to zero forward
drive, by construction of the z-score). The second run makes the forward dead zone learnable (a
negative `z0` gives a resting forward drive, so the fly walks when it sees nothing) and
backpropagates through two ticks; it could not run on the development Mac (26 GB of swap from a
virtual machine and Chrome on 24 GB of RAM paged the trainer out; small evaluations still run).

A tick-by-tick trace of the trained brain after its first touch explains the 4%: the fly hits the
person head-on at full speed, the solid-body contact shoves it sideways while its steering flips
between hard left and hard right every half second (the DNa02 pair saturates at 340–430 Hz and the
tuned turn readout is bang-bang), and within a second the person is behind it, where neither the
150° eye nor the forward-looking heat lobes sense anyone. The innate circuit then chases whoever
else is in view: two contacts per episode, rarely with the same person. Nothing in the subcircuit
remembers which person it touched, so tracking an individual through that moment needs either
learned braking at contact (the freeze channel, DNp09, brakes the vehicle on the CPU arena when
`hunt.freeze_brakes` is on; the batch arena would need the same) and a smoother turn readout, both
reachable by the training above, or a memory the network fly has (its GRU) and this subcircuit does
not. Hand-setting a negative forward dead zone on the trained brain changed nothing (tracking
3.4–4.3% for z0 from 0 to −2), nor did any hand-set decoder (smoothing 70–500 ms, turn gain 0.4–1,
forward z_ref 0.4–3: tracking 2–4% throughout), which confirms the loss is at contact and inside
the circuit, not in the readout. `hunt-brain` now lets the freeze channel brake the vehicle
(`hunt.freeze_brakes`, saved with the brain so the CPU arena does the same; `--no-brake` turns it
off) so the next training run can learn to stop at contact.

Two more things were tried by hand on the trained brain, both without effect on tracking (2–4%
either way, 32–128 held-out episodes against the trained runners): a proportional instead of
bang-bang turn readout (turn `z_ref` 15–40, smoothing 70–250 ms), and a brake reflex through the
fly's own looming detectors (the arena now drives LC4 / LPLC2 with the growth and the size of a
person's image, `hunt.loom_gain`, `hunt.near_deg`; with their synapses onto DNp09 ×8 the freeze
neuron does answer, 19 Hz against 5 at rest). The brake either fires during the approach, so the
fly stops short and never touches, or not at all; a brake that engages only in contact and a turn
that stays put would have to be learned together, which is what the searches below are for.

**Gradient-free training** (`hunt-brain --evolve`, `hunt_gpu/brain_es.py`): where backpropagation
cannot run, CMA-ES (the same optimizer as the connectome fly's gains in `sim/hunt.py`) searches the
12 sensory / decoder tunables plus a gain per synapse class of the subcircuit (presynaptic class →
postsynaptic class, the hunter groups and super classes: 105 classes), scored on 16 whole hunts per
candidate with a fitness that weighs tracking first. It needs only forward passes of the rate model
(a few hundred MB, about 8 min per generation of 18 on the CPU). The class gains multiply the
per-synapse gains and are folded into them on save, so the result is an ordinary `hunter_brain.npz`.
Two rounds were run (20 generations at 16 episodes per candidate, then 14 at 48 with tracking
weighted four times). Their best candidates reported 12–23% tracking on their own evaluation
episodes, but re-evaluated on 128 held-out episodes they track 4–6%, like the starting brain: with
tracking so variable from episode to episode (a fly parked on a resting runner scores 0.8, most
score 0), the best of 18 candidates is a lucky draw, and the search chases noise. A third round fixed
that: every candidate of every generation is scored on the same 48 episodes (common random
numbers), a candidate is accepted as the best only after a re-score on 64 other seeds, and the
looming / proximity drive was in the searched vector. Its first generation already produced a
brain that tracks: as a rate model on 128 fresh seeds, 31% of the time after the first touch
against the trained runners (first touch 75%, 2.8 contacts) and 43% against the scripted ones
(81%, 3.3 contacts), above the benchmark's 25%. Its recipe is readable in the gains: a weak
looming and proximity drive into the freeze neuron (`loom_gain` 0.2, `near_gain` 0.33 past 53°)
that brakes the vehicle at contact, a contrast turn readout, stronger heat and bar gains. As
spiking neurons the same brain tracks 9% against the trained runners and 17% against the scripted
ones (92% first touches, 12 episodes each): a real gain over the 4% before, and a real
rate-to-spike gap, so the search now also scores each new best as spiking neurons and keeps the
spiking-best separately (`hunter_brain_spiking.npz`). That brain is `data/cache/hunter_brain_track.npz`.
Two corrections followed. The fitness now scores tracking over all episodes (tracking fraction ×
touch rate, with a penalty below 85% touches), because the previous form let a fly that rarely
catches anyone but parks on the one it catches win (12–29% touches, 46% tracking). And the rate
model's decoder can add the spiking brain's resting jitter to its window means
(`RateBrain.readout_noise`, the per-group stds of the LIF baseline), which the search uses, so it
cannot exploit a precision the spiking neurons do not have.

With those, the search reached the benchmark as a rate model in two generations (98% first
touches, 24% tracking on held-out seeds) and kept climbing (100%, 37% by generation 6). The
rate-to-spike gap then turned out to be mostly a bug: the spiking hunter's motor output never
applied the freeze brake that the rate model and the CPU arena apply, so every spiking evaluation
above ran without the very mechanism the search had found. With the brake wired, the
generation-2 brain as spiking neurons (16 episodes each, seed 4242) tracks its person **35%** of
the time against the trained runners (75% first touches, 2.4 contacts) and **65%** against the
scripted ones (81%, 7.5 contacts), above the benchmark's 25% and 53%; its first-touch rate is
what remains below the benchmark's 99%. (Spiking evaluations were then made reproducible: the
runners' waypoints and pauses are seeded from the episode seed.) The generation-6 brain as spiking
neurons: 88% first touches, 25% tracking, 2.4 contacts against the trained runners (16 episodes);
with its brake threshold doubled, 67–71% first touches, 37–45% tracking, 4 contacts (24 episodes,
two seeds) and 69% / 66% / 8.4 against the scripted runners. The brake trades first touches for
tracking, and the first touches it costs are lost while nobody is in range: the spiking fly then
stands still about 70% of the time, because the freeze neuron's resting jitter reaches the brake.
The brake's dead zone is now a tunable (`freeze.z0`) for that reason, but the real cause was
elsewhere: the spiking decoder slowly re-centres its resting baseline on the running brain
(`decode.baseline_drift_s`, 120 s, a homeostasis the rate model does not have), so after a chase
the descending population reads as below rest and the forward drive drops to zero. With that
drift off, the generation-6 brain as spiking neurons (32 fresh seeds) catches 97% of the trained
runners (12.4 s, 17% tracking, 3.0 contacts) and 100% of the scripted ones (8.0 s, 51% tracking,
8.8 contacts), standing still 14% of its blind time; saved brains now carry `baseline_drift_s: 0`.

**Result (2026-09-19).** A sweep of the brake with drift off found the operating point: brake
threshold `freeze.z_ref` 20 with dead zone `freeze.z0` 3 on the generation-6 brain, saved as
`data/cache/hunter_brain.npz`. As spiking LIF neurons (the 8,503-neuron hunter subcircuit with
the learned gains, the real decoder), 32 held-out episodes per row, the whole 45 s clock:

| Hunter | Humans | First touch | Time to first touch | Tracking afterwards | Contacts / episode |
|---|---|---|---|---|---|
| benchmark: the network fly | trained runners | 99% | 15.1 s | 25% | 4.4 |
| **the fly's brain, spiking** (seed 9001) | trained runners | 100% | 11.9 s | 25% | 3.7 |
| **the fly's brain, spiking** (seed 12345) | trained runners | 97% | 12.7 s | 24% | 3.0 |
| benchmark: the network fly | scripted runners | 100% | 7.0 s | 53% | 9.4 |
| **the fly's brain, spiking** (seed 9001) | scripted runners | 100% | 7.6 s | 56% | 9.7 |

The real fly's brain now matches the benchmark: the same first-touch rate, earlier first touches,
the same tracking fraction, slightly fewer contacts against the trained runners and slightly more
against the scripted ones. What learned is a gain on each of 244,545 synapses (through 75 synapse
classes), the sensory gains and the decoder thresholds; the wiring is the connectome's.
`hunt-brain --watch --load data/cache/hunter_brain.npz` shows the neurons doing it, and
`companion run --load-weights data/cache/hunter_brain.npz` puts the same brain, in the whole
48k-neuron pruned circuit, on the dashboard with the atlas lit by its spikes.

```sh
uv run companion hunt-brain --check                                   # the rate model against the spiking subcircuit
uv run companion hunt-brain --train --gains data/cache/hunter.npz --humans data/cache/evaders_gpu.pt   # vs the benchmark's runners (frozen) -> data/cache/hunter_brain.npz
uv run companion hunt-brain --load data/cache/hunter_brain.npz --episodes 256          # the rate model, one batch
uv run companion hunt-brain --load data/cache/hunter_brain.npz --spiking --episodes 16 # as spiking neurons
uv run companion hunt-brain --load data/cache/hunter_brain.npz --watch --open          # watch the neurons
uv run companion hunt --load data/cache/hunter_brain.npz --no-learn --episodes 6       # the whole pruned circuit, CPU arena
```

## Adding arms later

`arms.l` / `arms.r` are already in every packet (currently the steering lateral bias and 0).
Map MDN/DNa/DNg readouts to servo targets in `eyes.py`'s neighbor `body/arms.py`, and drive
servos from `firmware/body` using the `armL`/`armR` fields already parsed in `link.cpp`.
