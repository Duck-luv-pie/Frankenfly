# SUMMARY — complete state of the project, Saturday 10:50

For the team. `RESULTS.md` is the evidence, `DEMO.md` the runbook, `HANDOFF.md` the quickstart, `DEVPOST.md` the submission story.

Everything known, built, measured, blocked and decided, in one file, after an overnight session and a morning of autonomous work.

---

## 1. Where it stands in one paragraph

A real male-*Drosophila* connectome subcircuit, 15,000 neurons and 2,334,959 signed synapses, runs as spiking neurons and drives a robot toward a person. Untrained, straight from the connectome, it turns toward a person in 75% of trials and reaches them 97.7% of the time. Learning uses the fly's own dopamine neurons and buys speed, not the behaviour. The demo's centrepiece is not the chase but two live controls: a wiring shuffle that keeps every neuron and connection count but randomizes the map, and a graded deletion of the tracking population. Along the way we found three things about the connectome that are results rather than features. The software is done, tested, documented and public. The robot is wired but has never been driven for real, which is the single biggest gap.

## 2. What was found (the science)

### 2.1 The behaviour is in the wiring, not the parts
Untrained, no learning of any kind: turns toward a person in **75%** of arenas (64 arenas, 2 s, people frozen), advances in **100%**, front contact **0.977** (128 arenas, 20 s), trails a walking person within 1 m for **87%** of the time and never loses them (32 arenas, walkers at 1.0–1.4 m/s).

Controls, 256 arenas, seed 2000:

| condition | turn-toward |
|---|---|
| intact | 78% |
| remove LC10a (275 of 15,000 cells) | **0%** |
| shuffle the wiring, every degree preserved | **0%** untrained, 11% trained |
| remove LC4 + LPLC2 (looming) | 75%, giant fibre 130 → 0.1 Hz |
| remove one DNa02 | 25%, rightward bias, predicted before running |

Live on the demo brain, one keypress: real wiring gives LC10a 66.6 Hz and DNa02 158.4 Hz; shuffled gives LC10a **69.9 Hz** and DNa02 **0.0 Hz**. The eye is untouched; the map is what carries the signal.

### 2.2 Two different ways for a descending neuron to be silent
`scripts/why_silent.py` drives the eye and strips the inhibition off a target neuron, strongest source first.

- **DNa10** receives 801 synapses straight from LC10a, the largest direct eye-to-descending projection in the circuit, and never fires at any drive from 1.5 to 80 mV per step. Remove AOTU041 and AOTU063 and it goes to **91 Hz**; remove the top four sources and **107 Hz**. It is actively gated off by feedforward inhibition driven by the same eye signal. That matches its published role as an *avoidance* pathway, which a fly tracking a mate should not be triggering.
- **DNp09 and DNa01**, the classic forward-walking neurons, stay at exactly **0.0 Hz** with every inhibitory input removed. No excitation ever arrives. Meanwhile DNa02, with **zero** direct LC10a synapses, reaches 126 Hz through a three-hop path.

This closed our longest-standing open question and justifies reading forward speed from the descending population: a measurement, not a workaround.

### 2.3 Most of the connectome is ballast for this behaviour
Synapses deleted, neurons untouched, 32 arenas:

| deleted | at random | weakest first |
|---|---|---|
| 50% | 66% | 78% |
| 75% | **0%** | 72% |
| 95% | 0% | **72%** |
| 99% | 0% | 47% |

The weakest 2.2 million synapses can all go. The same count removed at random kills it between 50 and 75%.

### 2.4 The behaviour fits in 2,211 neurons
Keep the strongest 5% of synapses, then keep only what lies between eye and motor: **2,211 neurons and 10,387 synapses, 0.4% of the original wiring**, turn-toward **81%**, advance 100%. Every lesion signature reproduces (LC10a out 0%, shuffled 2%, DNa02_L out 28% with the same rightward bias, looming cells out 70% with the giant fibre going 74.9 → 0.0 Hz). Caveat: the mushroom body does not survive this cut, so it is a runtime circuit, not a training one.

### 2.5 It needs no GPU and no framework
`edge/flybrain_mini.py`, one ~200-line numpy file, is **bit-identical** to the PyTorch implementation (0 of 300 steps differ, once matched to float32) and five times faster at batch one:

| runtime | full circuit | minimal circuit |
|---|---|---|
| PyTorch event engine | 14 ms/frame | 3 ms |
| numpy only | **2.5 ms** | **0.6 ms** (1,553 Hz) |

A Raspberry Pi would run even the full circuit at roughly 10 ms per frame.

## 3. What is built

**Brain and simulation.** `brain/lif.py` (three engines: sparse CSR, dense, event-driven, agreeing spike for spike; lesion and restore; mutable weights), `brain/retina.py`, `brain/senses.py`, `brain/retinotopy.py` (each LC neuron gets the retina column of its real lobula dendrites), `brain/motor.py`, `brain/explore.py` (search as an internal state, labelled non-sensory), `brain/plastic.py` (lobotomize and restore).

**World and learning.** `env/arena.py` (batched 2-D world, people who walk, contact events, reward), `learn/three_factor.py` (dopamine-gated plasticity with homeostatic scaling), `learn/es.py`, `learn/plastic.py` (masks). Dale's law and bounds enforced and tested.

**Data.** `data/pull_connectome.py` (neuPrint over cypher, no token needed), `data/find_paths.py`. Brain files: `brain.npz` (main, 15,000), `brain_v2.npz` (path neurons forced in, behaves identically), `brain_pruned.npz` (4,104), `brain_minimal.npz` (13,670 at 5% synapses), `brain_tiny.npz` (2,211).

**Experiments.** `scripts/milestones.py` (m1, m2, m2b, m2c, m3, m3b), `evaluate.py`, `ablations.py`, `why_silent.py`, `dropout.py`, `minimal_circuit.py`, `prune_brain.py`, `bench_brain.py`, `probe.py`, `probe_dns.py`, `paths.py`, `reproduce.py` (eight headline claims in 12 s), `lock_demo.py`.

**Robot and demo.** `scripts/robot_bridge.py` (any camera to wheels, two wheel links, watchdog, E-stop, latency log, live websocket feed, survives injected failures in every subsystem), `robot_daemon.py` (DJI SDK isolated in Python 3.8), `demo.py` (hotkeys 1–7), `viz_adapter.py` (drives Ducks's Three.js viewer with our brain), `dump_replay.py`, `replay/viewer.html` (standalone player), `export_timeseries.py`.

**Artifacts.** `checkpoints/demo_brain.pt` (the locked demo brain), four replay episodes, five figures in `devpost/`, 127 passing tests.

## 4. What works right now

From `~/Downloads/filess/flybrain-rover`:

```bash
.venv/bin/python -m pytest tests -q                      # 127 pass
.venv/bin/python scripts/reproduce.py --quick            # 8/8 claims in 12 s
.venv/bin/python edge/flybrain_mini.py --selftest --demo # numpy brain, identical to the reference
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --source 0 --show --dry-run
.venv/bin/python scripts/viz_adapter.py --ui ../hunting-fly-s1/brain/companion_brain/ui \
  --replay replay/episode_7_trained.json --port 8601
```

Demo hotkeys: **1** baseline, **2** remove LC10a, **3** restore, **4** wipe learning, **5** restore it, **6** wiring shuffle, **7** delete a quarter of LC10a, **space** stop.

## 5. The robot

Two independent halves. **Wheels**: either Ducks's S-Bus link (laptop USB → ESP32 inverter → the S1's receiver pins, no SDK, no gimbal needed) via `--s1-sbus <port>`, or the DJI SDK via `--robot-daemon`. **Eyes**: `--source` takes a webcam index, a stream URL (ESP32-CAM, phone IP camera) or a video file.

```bash
.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt \
  --s1-sbus /dev/cu.usbserial-XXXX --local-eye --source 0 --dry-run   # check the turn sign first
```

**Blocked on hardware, not software:** the S1's gimbal, camera and blaster boards do not answer the SDK (chassis and vision do). Battery was at 5%. Charge it and reseat the gimbal cable; if it stays dead, the S-Bus path plus any other camera is enough. The end-to-end chain has only been run in dry-run and against a fake daemon; **the real chassis has never been driven.** First real test must be `--dry-run` with the robot on a box, wheels off the floor, checking the printed turn flips sign as you move left and right.

Measured latency, dry run: **15 ms median** camera to command (detector 8–11 ms on the laptop GPU, brain 4 ms), against a 100 ms budget.

## 6. Devpost

`DEVPOST_FIELDS.md` has every field filled in: a 185-character elevator pitch, name options with character counts, 22 "built with" tags, the repo link, gallery order with captions, and a 60-second video shot list with voiceover. `DEVPOST.md` is the story, about 1,900 words, nine headings, every number carrying its protocol.

**Needs a human:** the project name (team decision, the repo is `hunting-fly`), the video recording, adding the team on the form, and selecting tracks.

## 7. Tracks

Finalists is the target. Of the sponsor list, most have hard requirements we do not meet (QNX needs their OS, Baseten needs a model served there, Bracket Bot needs their hardware, the OpenAI and Gemini prizes need an LLM we do not use). The one that genuinely fits is **MLH Tiger Data**: `scripts/export_timeseries.py` already writes episodes or the live robot feed into TimescaleDB or TigerData Cloud, about 60,000 rows per 15-second episode, with hypertables and a query-back report. The only missing piece is a free Tiger Cloud DSN in `$TIMESERIES_DSN`. MongoDB Atlas and ElevenLabs are possible but unbuilt. **Tracks lock 2:00 PM Saturday.** 

## 8. Repositories

Shared repo `github.com/Duck-luv-pie/hunting-fly`. Our half is `flybrain-rover/` on `main`, 15 commits. Ducks's full Companion project (FlyWire brain, ESP32-CAM, ESP32 body, the S1 over S-Bus) is the `s1-fly-brain` branch, checked out locally at `../hunting-fly-s1` so we import his S1 and camera drivers rather than rewriting them. His `brain/` and `tools/` have never been touched by us, verified byte-identical on every push.

The public repo carries the product only: code, tests, README, RESULTS, DEMO, DEVPOST, HANDOFF, figures, replay viewer. Detailed working notes and submission planning stay on Taka's machine; this file is the shared picture.

## 9. Decisions already made

- Forward speed is read from the whole descending population, k_f 0.3. Not a fallback; the alternative is measured to carry no signal.
- Gain 0.05, amp_track 20, retinotopy "rank", turn = k_t (DNa02_R − DNa02_L). Changing these silently invalidates every number above.
- Search is an internal exploratory state, labelled non-sensory, off by default, available with `--explore`.
- The tonic "walking current" fallback stays in the code, off, superseded by the population read-out.
- The demo leads with the controls; the chase is the readout. No swatter needed.
- Stage B search is not part of the scripted demo.
- Local git only for the internal notes; the shared repo is public.

## 10. Open items

1. **Charge the S1 battery, plug in the ESP32 bridge, run the dry-run sign check.** This is the last untested link.
2. **Project name** for Devpost.
3. **Record the 60-second video** (shot list is written).
4. **Tracks by 2 PM**, and if Tiger Data is wanted, create a free Tiger Cloud account and export the DSN.
5. Training ran on a rented GPU box that is now shut down. Every checkpoint we need is local, including the one the demo uses.

## 11. Honesty lines that must survive editing

- Untrained wiring does the steering. Training bought 4.25 → 3.23 s to contact and sustained contact 0.76 → 0.82, not the behaviour.
- The forward read-out and the search state are our engineering, and we say so.
- Four overnight biological training runs were flat; the Kenyon cells, the fly's real learning site, are silent in this task.
- An earlier run with a tonic current raised contact but saturated DNa02 at 300 Hz. That is not learning and we do not call it that.
- A three-line hand-coded controller beats the connectome at this task, by connectome-pilot's own numbers. The claim is emergence from real wiring, not superiority.
- The 2,211-neuron circuit drops the mushroom body, so it cannot be used for the learning story.
