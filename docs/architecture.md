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

**The two-odor experiment.** The world has two smells: the grape (odor A, glomeruli DM1/DM4/
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

## Adding arms later

`arms.l` / `arms.r` are already in every packet (currently the steering lateral bias and 0).
Map MDN/DNa/DNg readouts to servo targets in `eyes.py`'s neighbor `body/arms.py`, and drive
servos from `firmware/body` using the `armL`/`armR` fields already parsed in `link.cpp`.
