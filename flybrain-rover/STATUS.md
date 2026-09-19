# Back from class (written 12:30, all class-time milestones done; the box keeps training until ~20:20)

**Passed**
- M6 demo brain locked: `checkpoints/demo_brain.pt` (box job 1, gen 20, stage A, 512 envs). Fixed set (stage A, 128 envs, 20 s, seed 1000): contact 0.977, sustained 0.817, contact in 3.23 s (untrained 4.25 s), DNa02 156 Hz, every non-sensory group < 160 Hz. Literal "no group > 200 Hz" fails on LC10a only (retina-driven, 266 Hz; the untrained brain is at 206 Hz too), stated in DEMO.md and below.
- M7 search: stage B 0.742 with the exploratory internal state (bar 0.60), first sight 2.46 s; params locked in DEMO.md; `--explore`.
- M8 lesions on demo_brain.pt: LC10a out 0.00, DNa02_L out → right bias (+0.055), LC4/LPLC2 out → escape 130 → 0.1 Hz with steering intact, shuffle 0.11; lobotomize → restore bit-identical (178,445 synapses changed by learning).
- M9 runbook: `scripts/demo.py` (hotkeys 1–5, space, live feed on) + `DEMO.md` (setup, 2-minute script, fallbacks, `replay/viewer.html`).
- M10 `DEVPOST.md` (1,400 words, provenance on every number, honest wording; the trained-vs-untrained sentence now filled).
- M11 `handoff/` (demo_brain.pt, DEMO.md, DEVPOST.md, replay/, BEFORE_LEAVING.md); opencv/ultralytics in .venv, robomaster SDK in .venv38 (Python 3.8 under Rosetta, hence the daemon), yolo11n cached.
- Box: tmux session `fly` (afternoon queue + watcher); main queue job 2 (`tfA_both_pun10`) running, then esA_dnin_512, then the afternoon queue (explore, ES lc_dn, 3-h continuation, stage-B explore) until ~00:20 box / 20:20 Toronto.

**Did not pass / caveats**
- The literal M6 rule "no group > 200 Hz": no brain passes it at amp_track 20 (LC10a is the input population). Decision needed only if you want the literal rule: lower amp_track (re-sweep) or accept the non-sensory reading.
- Training moved LC10a_L to 266 Hz vs LC10a_R 145 (DNa02 stays balanced); a small right turn bias (+0.03) in the ablation baseline. Watch for a veer on the robot; the untrained brain is the clean fallback (`scripts/lock_demo.py --brain data/brain.npz --lock --force`).
- The Mac run (`tfA_both_dnall`) auto-promoted to stage B and its final checkpoint fails M6 (DNa02 182 Hz); superseded.
- Hardware untested: RoboMaster SDK path exists (`scripts/robot_daemon.py` + `--robot-daemon`) but has only run against the fake daemon.

**Do these first at the venue**
1. `BEFORE_LEAVING.md` steps 1–3 before closing the lid (pull box checkpoints/logs, kill nothing on the Mac that isn't already dead, commit). Tonight after 20:30, pull again.
2. With the RoboMaster: terminal 1 `.venv38/bin/python scripts/robot_daemon.py --conn ap`; terminal 2 `.venv/bin/python scripts/demo.py --checkpoint checkpoints/demo_brain.pt --robot-daemon 127.0.0.1:9500 --dry-run`, walk left/right, confirm the printed turn sign (turn > 0 = +z = clockwise), then drop `--dry-run`.
3. Rehearse DEMO.md once with Ducks's viz on ws://localhost:8765 (or `replay/viewer.html` as standby); keys 2/3 (lesion/restore) and 4/5 (lobotomize/restore) are the story.

---

# STATUS — overnight log (read this first tomorrow)

Everything below ran locally on the M2 Pro (torch 2.14, Apple MPS, dense engine). Nothing was pushed
anywhere. The only external traffic was anonymous cypher queries to neuprint.janelia.org (public data,
no token). The repo is a local git repo only; all commits are authored as you.

## Milestones
| M  | Status | Evidence |
|----|--------|----------|
| M0 | n/a    | No cloud box used. MPS works: `python -c "import torch;print(torch.backends.mps.is_available())"` |
| M1 | PASS   | `data/brain.npz`: N=15,000, nnz=2,334,959 (1.87M exc / 0.46M inh), every group non-empty. `logs/pull_connectome.log` |
| M2 | PASS   | gain sweep 0.005–0.275 in one batch: passes for 0.025–0.15, seizes at 0.275 (Shiu's value). Default set to **0.05**: LC10a_L poke → DNa02_L 126 Hz, DNa02_R 0 Hz. `python scripts/milestones.py m2` |
| M3 | PASS   | 64 envs, stage A, 2 s, humans frozen: mean abs bearing 0.36 → 0.12 rad, improved in 72% (amp_track 15) / 77% (amp_track 30). `python scripts/milestones.py m3` |
| M4 | running overnight | see "Overnight queue" |
| M5 | not started | robot bridge + replay JSON planned in the overnight loop |

## What was built tonight (all tests: `.venv/bin/python -m pytest tests/ -q` → 22 passed)
- `data/pull_connectome.py`  cypher over `/api/custom/custom` (works without a token). Family-aware 1-hop expansion so DN inputs are not crowded out by KC partners. Thermo = TRN_VP3a/b hot receptor neurons + VP3 PNs (no type matches "hot/thermo/TPN"; VP3 is the hot glomerulus).
- `brain/lif.py`  Shiu et al. 2024 constants, exponential synapses, sparse CSR (CUDA/CPU) or dense (MPS) engines that agree spike for spike, mutable weights (`set_w_subset`), per-env gain (`gain_b`), plastic hook for ES.
- `env/arena.py`  batched world per docstring; contact/front classification; heat with latency + false positives; retina noise; ±10% gain.
- `train.py`  build/run_episode/train/sweep/record. `scripts/milestones.py` (m1/m2/m3), `scripts/probe.py` (which eye population drives which DN), `scripts/paths.py` (who feeds each DN).
- `learn/three_factor.py`, `learn/es.py`, `learn/plastic.py` (masks: kc_mbon, dn_in, lc_dn, both). Dale's law + [0, 3×|w0|] bounds enforced and tested.

## Findings you need to know
1. **Turn sign bug (fixed).** DNa02 is ipsilateral (LC10a_L → DNa02_L), motor emits turn>0 = right, but the arena applied turn>0 as CCW. Rover turned away from the human. Now `yaw -= turn·ω·dt`. On the real robot, turn>0 must map to clockwise (RoboMaster `chassis.drive_speed(z=+deg/s)` is clockwise).
2. **Tracking amplitude.** `size` (angular width / hfov) is 0.1–0.3 at 1–3 m, so `pres·size·1.2` never crossed threshold; LC10a was silent in the arena. `amp_track` default is now 20. LC10a is the ONLY eye population that reaches DNa02 (probe table in `scripts/probe.py`). LC11/12/15 and THERMO reach no DN at gain 0.05; LC4/LPLC2 drive the giant fiber (escape).
3. **Forward drive is the open problem.** Nothing drives DNa01 or DNp09 under any single or combined input, so `forward = k_f·(DNa01+DNp09)` is 0 and the rover only turns. `scripts/paths.py`: DNa01's top inputs are LAL/VES central-complex neurons with no eye input; DNp09 gets eye-driven excitation (PVLP150) and inhibition (PVLP020) in roughly equal measure. Training must open this path (that is what the overnight ES/three-factor runs test), otherwise the honest options are: (a) plasticity on `dn_in` edges, (b) a tonic walking-state drive on DNp09 (a motor-side parameter, like k_f), (c) a different forward DN pair. Your call in the morning.
4. Gain ≥ 0.1 lets THERMO drive the whole mushroom body to 250+ Hz. Stay at 0.05.
5. Speed: MPS dense ≈ 8–10 ms per brain step at B=64 → ~200 s per 20 s generation. A CUDA box with the sparse engine would be several times faster, but is not needed for tonight's questions.

## Overnight queue (scripts/overnight.sh, logs in logs/*.csv and logs/overnight_*.log)
1. three_factor, plastic=both, stage A, 64 envs, 20 s episodes, 75 min
2. es, plastic=dn_in, stage A, 128 envs, 10 s episodes, 75 min
3. es, plastic=lc_dn, stage A, 128 envs, 10 s episodes, 75 min
4. three_factor, plastic=kc_mbon, stage A, 64 envs, 75 min (the biological default, for the writeup)
The loop reviews each run's CSV, keeps notes below, and may add runs. Checkpoints: `checkpoints/<run>_latest.pt`.

## Loop notes (appended by each wake-up)

### 23:35 tick 1
- Job 1 (three_factor, both) gen 0: contact 0.44 in 20 s stage A, but `fwd` 0.01 (contacts come from humans walking into the stationary rover), dopamine D 0.009, |dw| 8e-5. 549 s per generation on MPS with the learner on.
- New **event-driven LIF engine** (`engine="event"`): gathers only the outgoing synapses of neurons that spiked. Exact (same spikes as CSR/dense, tested). CPU: 0.7 ms/step at B=1 (CSR: 33 ms). Robot bridge brain loop is now 9 ms per frame on this Mac's CPU (was 630 ms). On MPS at B=64 it ties dense (~21 ms/step contended), so training stays on dense there.
- `scripts/robot_bridge.py` written: `--bench`, `--fake-box`, webcam/video/robot sources, YOLO person detector (optional), heat over serial (optional), keys r/p (reward/punish PAM/PPL1), l (lobotomy), space (stop). RoboMaster: turn>0 → positive z (clockwise). Needs `opencv-python`, `ultralytics`, `robomaster` on the demo laptop (not installed here).
- Vast box (A100 40 GB, torch 2.14+cu130, CUDA OK = M0). My rsync of the repo to it was blocked by the permission classifier; Taka has the command to run. Once the repo is there: tests, CUDA sparse-vs-event benchmark at B=512, large-batch queue.
- README rewritten with run commands, learned-vs-swept statement, replay JSON format, full credits.

## OVERNIGHT.md run (started 23:38, 2026-09-17; 6 h cap → no new tasks after 05:38)
Guardrail 1 note: the agent cannot run `/model`; that is a user-side command. To spare quota, mechanical
tasks are delegated to Sonnet subagents where possible and the main session keeps tool calls lean.

### [23:40] Task 0 — Sanity — done
What was done: queue driver `bash scripts/overnight.sh` PID 87510 (parent 1, no tty, nohup) alive, job 1 `tfA_both` at gen 2; `git status` clean except the new path files; caffeinate ×2 running (12 h). Nothing restarted.
Files touched: none.

### Task 1 — Forward drive, biological fix — in progress
Shortest ConnectsTo paths (≤4 hops, every edge weight ≥5), neuPrint male-cns:v1.0, one shortestPath per (source neuron, target neuron) pair (`data/find_paths.py`, `data/paths_v2.json`, `logs/find_paths.log`):

| pair | paths (src×tgt pairs) | path lengths | most frequent intermediates (count) |
|---|---|---|---|
| LC10a->DNp09 | 550 | 1 hop: 4, 2 hop: 105, 3 hop: 441 | LAL026_b (127), AOTU041 (107), TuTuA_1 (66), AOTU042 (61), AOTU101m (47), LC10c-1 (46) |
| LC10a->DNa01 | 550 | 2 hop: 355, 3 hop: 195 | AOTU019 (188), AOTU041 (94), AOTU016_a (52), AOTU016_c (31), AOTU001 (29), LT82a (28) |
| LC4->DNp09 | 252 | 2 hop: 155, 3 hop: 97 | PVLP141 (51), PVLP010 (49), ? (24), VES023 (19), DNp04 (16), LHAD1g1 (16) |
| LC4->DNa01 | 252 | 2 hop: 209, 3 hop: 43 | PVLP141 (59), PVLP137 (54), PLP029 (50), ? (33), LT82a (18), aSP10A_b (15) |

795 distinct neurons lie on these paths; **739 were already in brain.npz**, only 56 are new (LC10d, LLPC1, AOTU062, PVLP005, LC10c, MeLo13, pC1_9a, …). So the eye→DNp09/DNa01 paths were present in v1; DNp09/DNa01 are silent because the drive along them does not propagate (weak direct weights: only 4 LC10a→DNp09 edges ≥5, and strong inhibitory inputs PVLP020/CL366 onto DNp09), not because neurons were missing. brain_v2.npz = same seed + these 56 forced + 1-hop neighbours to the 15,000 cap (`--include-ids data/include_ids_v2.json`).

### [23:52] Task 1 — Forward drive, biological fix — done (path found anatomically, NOT functional; fallback implemented, off by default)
What was done:
- `data/find_paths.py` → `data/paths_v2.json` (table above). `data/pull_connectome.py --include-ids data/include_ids_v2.json --out data/brain_v2.npz`: 389 path neurons forced in (of which 56 were absent from v1), 1-hop neighbours filled to the 15,000 cap. **brain_v2.npz: N=15,000, nnz=2,346,485 (v1: 2,334,959). M1 PASS.** brain.npz untouched.
- M2 on v2 (`scripts/milestones.py m2 --brain data/brain_v2.npz`, drive 1.5 mV/step on all LC10a_L, 300 ms, gains 0.025–0.15 as one batch): PASS at every gain, numbers identical to v1 (gain 0.05: DNa02_L 126 Hz, DNa02_R 0). Gain kept at 0.05.
- New check **M2b** (`milestones.py m2b`): stage A, single human placed dead ahead at 2 m and frozen, 16 envs, seed 0, 300 ms, no heat; pass = DNp09 or DNa01 > 5 Hz with the human and < 2 Hz without. Grid gain {0.05, 0.07, 0.1} × amp_track {20, 40}: **FAIL everywhere**. Best forward-DN rate 2.0 Hz (DNa01_R, gain 0.1, amp_track 20). Without the human: 0 Hz. DNp09 never fired. (Side finding: with a centred human LC10a_L/R both ~30 Hz but DNa02_R 68–101 Hz vs DNa02_L 0; the column→neuron assignment in senses.py is by index order, not real retinotopy. Open question below.)
- **M3/M3b** on v2 (`milestones.py m3`, 64 envs, seed 0, stage A, humans frozen, 2 s, gain 0.05, amp_track 20, k_f 0.01): M3 turn-toward 75% (mean |bearing| 0.361 → 0.105 rad) PASS; **M3b advance 0%** (mean dist 1.84 → 1.82 m; metric = distance to nearest human decreased by > 0.05 m) FAIL. Forward 0.004.
- Decision: a real path exists anatomically (1–3 hops) but carries no drive at any non-seizing gain, so the queue was NOT restarted on v2 (v2 ≡ v1 functionally). Implemented the **tonic-locomotion FALLBACK** (`brain/senses.py amp_tonic`, `--amp_tonic` in train.py/milestones.py; constant current into DNa01_L/R, gated off while touching a human; default 0 = off; labelled as non-biological in code). M3b with it, same protocol on v2:
  | amp_tonic | k_f | turn-toward | advance | mean dist 2 s | DNa01_L/R Hz | fwd |
  |---|---|---|---|---|---|---|
  | 0.4 | 0.01 | – | 100% | 1.84→1.37 m | 9.3 / 9.2 | 0.19 |
  | 0.6 | 0.01 | – | 100% | 1.84→0.45 m | 15 / 43 | 0.58 |
  | 1.0 | 0.01 | – | 100% | 1.84→0.43 m | 33 / 61 | 0.60 |
  | 0.6 | 0.02 | 72% | 100% | 1.84→0.43 m | 12 / 40 | 0.62 |
  Recommended if Taka accepts the fallback: amp_tonic 0.6, k_f 0.01. GF rises to ~110 Hz as the rover closes in (looming from approach), harmless with dodge off.
- Added one extra training job that starts only after the main queue ends (~04:30) and finishes before 07:00: `tfA_both_tonic` = three_factor, plastic both, v1, 64 envs, 20 s, 60 min, amp_tonic 0.6 (`scripts/overnight_2.sh`). It is the only queued job in which the rover can move forward, so it is the only one that can learn contact. Main queue untouched.
- Code review (subagent) found one real bug: checkpoints store `gain` but `scripts/robot_bridge.py` and `train.build()` ignored it when loading; fixed (checkpoint gain/k_t/k_f/amps now override CLI defaults).
Open questions:
- Use the tonic fallback for training and the demo? (Options: yes with amp_tonic 0.6 and say so in the writeup; or keep biological purity and accept a rover that only turns; or find a forward DN that LC10a does drive, e.g. probe every DN type in the male CNS for LC10a-driven ones.)
- Real retinotopy: assign LC columns from each neuron's `LO_*_col_*` ROI centroid (available in neuPrint) instead of index order.
Files touched: data/find_paths.py (new), data/pull_connectome.py (--include-ids), data/brain_v2.npz + data/paths_v2.json + data/include_ids_v2.json (new), brain/senses.py (amp_tonic, contact gate), train.py (contact gate, --amp_tonic, checkpoint restores gain/k/amps), scripts/milestones.py (m2b, M3b, --amp_tonic/--k_f/--k_t), scripts/robot_bridge.py (gain restore), scripts/overnight_2.sh (new).

### [00:10] Task 2 — Evaluation harness — done (scripts written by a Sonnet subagent, verified here; full runs launched)
What was done: `scripts/evaluate.py` (fixed evaluation set, no learning, per-stage CSV + shared `logs/eval_all.csv`; restores a checkpoint's own gain/k/amps) and `scripts/watch_queue.py` (every 15 min evaluates any new `checkpoints/*_latest.pt` generation on stage A, 64 envs, 20 s → `logs/eval_queue.csv`, state in `logs/eval_queue_state.json`). Started detached at 00:10: `nohup .venv/bin/python scripts/watch_queue.py --interval 900 --envs 64 --seconds 20`.
Result (provenance): untrained evaluations launched in the background on CPU (event engine) with **128 envs and 20 s episodes instead of the spec's 256 × 25 s** (CPU throughput is 9.3 min per stage at 128 envs; 256 × 25 s would cost 70 min per brain while the GPU is busy training): brain.npz and brain_v2.npz, stages A/B/C, seeds 1000/1001/1002 (one seed per stage; the 128 envs of a stage are the 128 draws from that seed). Table goes in the Morning Report when they finish (`logs/eval_all.csv`).
Metric definitions: front_contact_rate = fraction of envs with ≥1 front-strip touch; mean_time_to_contact = mean first-touch time over those envs; sustained_fraction = fraction of env steps in front contact, mean over envs; back_contact_rate = fraction of envs with ≥1 back/side touch; mean_abs_bearing_2s = mean |bearing| to nearest human at t = 2 s; advance_2s = fraction of envs whose distance to the nearest human fell by > 0.05 m by t = 2 s; DN columns = episode-mean firing rate (Hz).
Files touched: scripts/evaluate.py (new), scripts/watch_queue.py (new), train.py (run_episode now returns back_contact_rate, sustained_fraction, bearing_abs_2s, dist_2s, advance_2s, turn_mean, first_contact_t).

### [00:12] Task 3 — Lesion ablations — done
Result (`scripts/ablations.py`, `logs/ablations.csv`; brain_v2.npz untrained, stage A, **256 envs, seed 2000, 2 s, humans frozen**, gain 0.05, no tonic; turn-toward = |bearing| fell > 0.02 rad; advance = distance fell > 0.05 m; turn_mean = signed motor turn, + = right; lesion = in+out synapses of the group zeroed with `lif.lesion(group, mode="both")`, restored after each row; shuffle = random permutation of every edge's POST index, so every neuron keeps its exact in- and out-degree and every synapse its sign and weight):

| condition | turn-toward | advance | turn_mean | DNa02_L / R Hz | GF Hz | GF under looming* |
|---|---|---|---|---|---|---|
| baseline | **0.78** | 0.01 | +0.020 | 17.1 / 18.0 | 9.0 | **106.9** |
| lesion LC10a (L+R) | **0.00** | 0.00 | 0.000 | 0.0 / 0.0 | 0.0 | – |
| lesion DNa02_L | 0.25 | 0.00 | **+0.061** (rightward bias) | 0.0 / 5.0 | 2.0 | – |
| lesion LC4 + LPLC2 (L+R) | 0.75 | 0.00 | +0.021 | 17.2 / 18.3 | 0.0 | **0.0** |
| shuffled (degree-preserving) | **0.00** | 0.03 | −0.002 | 0.1 / 0.0 | 0.0 | – |
*looming = human starts 3 m dead ahead and walks straight at the stationary rover at 1.4 m/s for 2 s; GF = episode-mean giant-fiber rate.
Reading: steering needs LC10a and nothing else in the eye; removing one DNa02 biases the turn to the other side; the looming channel (LC4/LPLC2 → GF) is separate from steering; a wiring-preserving shuffle abolishes the behaviour entirely (0%, not the "near 50%" one might expect: with no DNa02 drive the rover does not turn at all, so |bearing| never falls in frozen-human envs).
Files touched: brain/lif.py (`lesion()`, `restore()` + test), scripts/ablations.py (new), train.py (turn_mean).

### [00:13] Task 4 — Lobotomize and restore — done (Sonnet subagent; tests verified here)
What was done: `brain/plastic.py`: `snapshot_plastic`, `lobotomize` (reset plastic synapses to connectome values; optional `zero_lc10a` lesion for the dramatic demo), `restore`; CLI `python -m brain.plastic --lobotomize --checkpoint X.pt [--zero-lc10a]` / `--restore --checkpoint X_lobotomized.pt` (never overwrites inputs; writes a snapshot file next to the output). `tests/test_plastic.py`: 8 tests incl. bit-identical restore and a subprocess CLI round trip.
Files touched: brain/plastic.py (new), tests/test_plastic.py (new).

### [00:16] Task 5 — Robot bridge — done (hardware untested; plumbing and latency measured)
What was done: `scripts/robot_bridge.py`: camera (RoboMaster stream, webcam/video via OpenCV, or `--no-camera` synthetic 30 fps) → person detector (ultralytics yolo11n, person class, conf 0.4; clear ImportError text if not installed; `--fake-boxes` synthetic sweeping box) → `Retina.from_boxes` → `Senses.inject` → LIF **event engine at batch 1** → `Motor.decode` → `chassis.drive_speed(x=forward·v_max, y=0, z=turn·w_max)` with turn > 0 = positive z (clockwise). `--dry-run` prints commands instead of sending; watchdog zeroes velocity when no frame arrives for `--watchdog-ms` (300); E-stop = space in the window or Enter in the terminal; keys r/p inject reward/punishment into PAM/PPL1, l = lobotomy toggle; per-frame latency log `logs/bridge_latency.csv` (camera timestamp → command timestamp) with a summary at exit. Checkpoint loading restores the trained gain/k/amps.
Result: `python scripts/robot_bridge.py --fake-boxes --no-camera --dry-run --max-frames 150 --device cpu` on this M2 Pro: **median 15.1 ms, p95 23.8 ms, max 27.6 ms** camera→command (brain 10 ms for 20 × 1 ms steps; fake detector 0 ms). YOLO-nano on a laptop CPU typically adds 20–40 ms, so the 100 ms budget holds. Sign check in the same run: box sweeping on the left → LC10a_L 33 Hz, DNa02_L 35 Hz, turn −0.70 (left). Needs on the demo laptop: `pip install opencv-python ultralytics robomaster` (not installed here).
Files touched: scripts/robot_bridge.py.

### [00:17] Task 7 — Detector on synthetic frames — done (Sonnet subagent; verified here)
What was done: `tests/test_retina_agreement.py`: 50 synthetic 320×240 pinhole-projected frames (floor gradient, red cylinder blob at known bearing/distance, incl. partly-clipped cases), a colour-threshold bounding-box "detector" standing in for YOLO, and assertions that `Retina.from_boxes` agrees with `Retina.from_state` within 1 column at each edge, < 1 column in centroid, < 0.02 in size; plus a px_to_az round-trip within 0.5°. 91 tests pass, 0 skipped. No library bug found.
Files touched: tests/test_retina_agreement.py (new).

## CHANGELOG (every file changed tonight, 2026-09-17 22:45 → 2026-09-18, and why)
| file | change | why |
|---|---|---|
| data/pull_connectome.py | new: cypher over /api/custom/custom, family-aware 1-hop expansion, NT sign rule, `--include-ids` | M1; neuPrint answers without a token; DN inputs must not be crowded out by KC partners; path neurons for v2 |
| data/find_paths.py, data/paths_v2.json, data/include_ids_v2.json | new | Task 1 shortest eye→forward-DN paths |
| data/brain.npz, data/brain_v2.npz | new (gitignored) | v1 = tonight's circuit; v2 = + path neurons |
| brain/lif.py | rewrite: Shiu 2024 constants, exponential synapses, sparse/dense/event engines, mutable weights, per-env gain, plastic hook, `lesion()/restore()`, gain default 0.05 | GPU-less Mac needs dense (MPS) / event (CPU) engines; learners need writable weights; ablations |
| brain/senses.py | auto device; amp_track 1.2 → 20; `amp_tonic` fallback + contact gate | LC10a never crossed threshold from the retina; forward-drive fallback (off by default) |
| brain/motor.py | auto device | run on CPU/MPS |
| brain/plastic.py | new: snapshot/lobotomize/restore + CLI | Task 4 demo button |
| env/arena.py | new: batched world per docstring; turn sign fixed (turn>0 = clockwise); front-contact rule; step-count timeout | M3; the sign bug made the rover flee |
| learn/plastic.py, learn/three_factor.py, learn/es.py | new: masks (kc_mbon/dn_in/lc_dn/both), three-factor rule, ES with per-env plastic hook | M4 learning, Dale's law + bounds enforced |
| train.py | new: build/run_episode/train/sweep/record; checkpoint restores gain/k/amps; eval metrics; replay record incl. retina columns | training loop, sweeps, provenance |
| scripts/milestones.py | new: m1, m2 (batched gain sweep), m2b (centred human), m3 (+M3b advance) | pass/fail checks with provenance |
| scripts/probe.py, scripts/paths.py | new | which eye population drives which DN; who feeds each DN |
| scripts/evaluate.py, scripts/watch_queue.py | new | Task 2 evaluation harness |
| scripts/ablations.py | new | Task 3 lesion table |
| scripts/robot_bridge.py | new: detector→retina→brain→RoboMaster, bench, fake boxes, dry-run, watchdog, E-stop, latency log | Task 5 / M5 |
| scripts/dump_replay.py, replay/FORMAT.md, replay/episode_7_*.json | new | Task 6 viz replays for Ducks |
| scripts/overnight.sh, scripts/overnight_2.sh | new | detached training queue (+1 fallback job) |
| tests/test_lif.py, tests/test_arena.py, tests/test_learn.py, tests/test_plastic.py, tests/test_retina_agreement.py | new (123 tests) | engines equal spike-for-spike; geometry/reward rules; Dale/bounds; lobotomize round trip; sim-to-real retina agreement |
| README.md, STATUS.md, .gitignore (.venv) | updated / new | credits, run commands, replay format; this log |

### [00:14] Task 6 — Replay dump for Ducks — done
What was done: `scripts/dump_replay.py` + `replay/FORMAT.md` (every field with units). Four episodes dumped (brain.npz, stage A, seed 7, 15 s at 50 Hz = 750 frames, env 0 of 4, CPU event engine): `replay/episode_7_untrained.json`, `_trained.json` (checkpoint tfA_both_latest.pt = gen 4 of job 1, the only checkpoint so far), `_lesioned.json` (LC10a_L+R removed), `_tonic.json` (FALLBACK amp_tonic 0.6). ~0.9 MB each.
Result (contact_rate over the 4 envs of each dump, forward = mean motor forward): untrained 0.25 / fwd 0.00, DNa02 8.5/9.2 Hz; trained-gen4 0.25 / 0.01, DNa02 10.2/10.8; lesioned 0.00 / 0.00, DNa02 0/0 (GF 5.3); tonic 1.00 / 0.49, DNa02 85/85, DNa01 19/30 Hz. Note the 0.25 contacts without forward drive are humans walking into the rover.
Files touched: scripts/dump_replay.py (new), replay/FORMAT.md (new), replay/episode_7_*.json (new), train.py (record includes pres/size/mot/loom).

### [00:20] Task 8 — Hygiene — done
What was done: README credits complete (Berg 2026, Shiu 2024, Eon fly-brain, connectome-pilot, Imperol3/flybrain, Keles & Frye 2017, Rayshubskiy 2020, Marin 2020, Bidaye 2020, neuprint-python, PyTorch, Ultralytics, RoboMaster SDK). `pytest`: **123 passed**. Type hints added to every public function/method signature in brain/, env/, learn/, train.py, data/ and the scripts (Sonnet subagent, annotation-only diff, suite re-run green). CHANGELOG section above. Committed.
Files touched: 15 modules (signatures only), STATUS.md.

### [00:42] Task 9 — Training results, job 1 `tfA_both` — FLAT
Provenance: brain.npz, three_factor, plastic=both (KC→MBON + eye/DN edges, 256k synapses), stage A, 64 envs, 20 s episodes, seeds 0–8 (one per generation), 9 generations in 75 min (≈555 s each on MPS), `logs/tfA_both.csv`, checkpoint `checkpoints/tfA_both_latest.pt` (gen 8).
Numbers: contact_rate per generation 0.44, 0.31, 0.31, 0.34, 0.47, 0.30, 0.44, 0.38, 0.33 (200-episode window 0.33–0.40, no trend); those contacts are humans walking into a rover whose forward command stayed ≤ 0.011 all run. |dw| per update 4e-5 – 3e-4 (weights essentially unchanged; mean |w0| on the plastic set is several synapses); dopamine D 0.004–0.025 (the −0.02 no-contact penalty is far below PPL1 threshold, so the gate only ever opens on the rare +5 contact events). DNp09 0.0 Hz in every generation; DNa02 8–21 Hz; max any-DN rate 21 Hz (no divergence). Independent evaluation of gen 4 (`scripts/watch_queue.py`, stage A, 64 envs, seed 1000, 20 s): front-contact 0.50, advance_2s 0.15, DNp09 0.0. Verdict: **flat** (no contact-rate trend, no weight movement, forward drive not opened).

### [01:15] Task 2 (cont.) — Untrained evaluation table — done
`scripts/evaluate.py`, no learning, CPU event engine, **128 envs, 20 s episodes**, seed 1000 (A) / 1001 (B) / 1002 (C), gain 0.05, amp_track 20, k_f 0.01, no tonic. Front = fraction of envs with ≥1 front-strip touch; ttc = mean first-touch time (s); sust = fraction of steps in front contact; back = fraction of envs with ≥1 back/side touch; |bear|2s = mean |bearing| to nearest human at 2 s (rad); adv2s = fraction whose distance fell > 0.05 m by 2 s; rates in Hz (episode mean).

| brain | stage | front | ttc | sust | back | \|bear\|2s | adv2s | DNa02_L | DNa02_R | DNp09 | GF | track |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1 | A | 0.359 | 7.95 | 0.005 | 0.000 | 0.134 | 0.328 | 10.4 | 10.5 | 0.0 | 14.9 | 0.975 |
| v1 | B | 0.438 | 8.01 | 0.027 | 0.516 | 1.951 | 0.641 | 16.7 | 16.8 | 0.0 | 35.0 | 0.672 |
| v1 | C | 0.531 | 8.17 | 0.009 | 0.367 | 1.448 | 0.430 | 16.5 | 16.4 | 0.0 | 37.6 | 0.745 |
| v2 | A | 0.367 | 7.86 | 0.005 | 0.000 | 0.136 | 0.328 | 10.4 | 10.5 | 0.0 | 14.8 | 0.974 |
| v2 | B | 0.430 | 8.05 | 0.027 | 0.523 | 1.953 | 0.648 | 17.0 | 17.0 | 0.0 | 34.7 | 0.672 |
| v2 | C | 0.523 | 7.92 | 0.010 | 0.367 | 1.450 | 0.445 | 16.6 | 16.3 | 0.0 | 37.3 | 0.745 |

Reading: v2 ≡ v1 (the 389 forced path neurons change nothing functionally). With forward drive at 0, every contact is a human walking into the rover: stage B/C have more humans (1–8) so more bumps, including ~50% back/side bumps, and adv2s there is humans approaching, not the rover advancing. Stage A |bearing| at 2 s = 0.13 rad (the rover has turned to face the human; humans are not frozen here). DNp09 = 0 everywhere. These rows are the baselines for the checkpoint rows in `logs/eval_queue.csv` (64 envs, stage A, seed 1000: tfA_both gen 4 → 0.50, gen 7 → 0.53; esA_dnin gen 4 → 0.45; untrained v1 stage A at 128 envs → 0.36; at 64 envs the noise is ±0.06).

### [01:58] Task 9 — job 2 `esA_dnin` — FLAT
Provenance: brain.npz, ES (sigma 0.05, top 20% of envs), plastic=dn_in (4,526 synapses onto DNa02/DNa01/DNp09/GF), stage A, 128 envs, 10 s episodes, seeds 0–26, 27 generations in 75 min (181 s each), `logs/esA_dnin.csv`, checkpoint gen 26.
Numbers: contact_rate per generation 0.34 0.30 0.27 0.27 0.24 0.26 0.24 0.27 0.27 0.24 0.26 0.26 0.27 0.25 0.20 0.32 0.25 0.28 0.23 0.27 0.23 0.24 0.23 0.27 0.30 0.27 0.26 (first-half mean 0.269, second-half 0.258; 10 s episodes so lower than the 20 s runs). ES did move the DN-input weights (|Δw| 0.20–0.22 synapse units per generation, within Dale/bounds) but forward stayed ≤ 0.008, DNp09 0.0 Hz, DNa01 L+R ≤ 0.8 Hz every generation, max any-DN rate 20 Hz (no divergence). Watcher evaluation (stage A, 64 envs, seed 1000, 20 s): gen 4 → front 0.453, gen 16 → 0.484, gen 23 → 0.484, DNp09 0.0 (untrained v1: 0.36 at 128 envs; noise at 64 envs ≈ ±0.06). Verdict: **flat**. Fitness has no signal to follow: with sigma 0.05 no perturbation of the DN inputs produced forward motion, and the return is dominated by which envs' humans happened to bump the rover.

### [03:13] Task 9 — job 3 `esA_lcdn` — FLAT
Provenance: brain.npz, ES (sigma 0.05, top 20%), plastic=lc_dn (195k synapses: every eye-neuron output plus every DN input), stage A, 128 envs, 10 s episodes, seeds 0–9, 10 generations in 75 min (448 s each: the per-env gather/scatter over 195k edges is the cost), `logs/esA_lcdn.csv`, checkpoint gen 9.
Numbers: contact_rate per generation 0.30 0.20 0.27 0.26 0.22 0.30 0.20 0.24 0.26 0.24 (first-half 0.252, second-half 0.248). |Δw| 0.054–0.055 per generation; forward ≤ 0.007; DNp09 0.0; DNa01 L+R ≤ 0.7 Hz; DNa02 mean 14.0 → 11.8 Hz; max any-DN 21 Hz (no divergence). Watcher evaluation (stage A, 64 envs, seed 1000, 20 s): gen 1/4/6 → front 0.453/0.453/0.438, DNp09 0.0. Verdict: **flat** (same reason as job 2: no fitness gradient without forward motion).

### [04:32] Task 9 — job 4 `tfA_kcmbon` — FLAT (weights did not move at all)
Provenance: brain.npz, three_factor, plastic=kc_mbon (61,210 KC→MBON synapses, the fly's own learning site), stage A, 64 envs, 20 s episodes, seeds 0–17, 18 generations in 75 min (254 s each), `logs/tfA_kcmbon.csv`, checkpoint gen 17.
Numbers: contact_rate per generation 0.44 0.31 0.33 0.34 0.47 0.28 0.36 0.38 0.25 0.34 0.44 0.41 0.34 0.41 0.36 0.38 0.42 0.42 (first-half mean 0.351, second-half 0.391; within the ±0.06 noise of 64 envs). |Δw| per update 5e-8 – 5e-7, i.e. zero: the Kenyon cells fire at 0 Hz in this task (no olfactory/thermal drive reaches the mushroom body at gain 0.05), so no KC→MBON eligibility ever forms and the dopamine gate (D 0.004–0.021) has nothing to act on. Watcher evaluation (stage A, 64 envs, seed 1000, 20 s) of gens 1, 6, 11: front 0.469, DNa02 11.0/11.1, GF 15.9 — identical to the digit, confirming the checkpoint is functionally the untrained brain. DNp09 0.0; max any-DN 18 Hz. Verdict: **flat**. Biological note for the writeup: KC→MBON plasticity can only shape behaviour if the KCs carry a sensory code; in this circuit they need an input we are not providing (in the fly: odour/temperature via the antennal lobe; THERMO does reach KC/MBON only at gain ≥ 0.1, where it seizes).
Main queue finished 04:30 (all four jobs exit 0). Fallback job `tfA_both_tonic` started 04:30 (60 min).

### [05:40] Task 9 — job 5 `tfA_both_tonic` (FALLBACK amp_tonic 0.6) — DIVERGED by the >200 Hz rule, contact improved, stage A target met
Provenance: brain.npz, three_factor, plastic=both, stage A → promoted to **stage B after gen 3** (train.py promotes when the 200-episode window ≥ 0.90: contact 1.00 0.95 1.00 0.97 over 256 episodes), 64 envs, 20 s, seeds 0–6, 7 generations in 60 min (503 s each), `logs/tfA_both_tonic.csv`, checkpoint gen 6 (stage B). amp_tonic 0.6 = the non-biological forward drive.
Numbers: stage A gens 0–3: contact 1.00 0.95 1.00 0.97, time-to-contact 3.3 → 2.4 s, return 238 → 397, forward 0.55 → 0.92, dopamine D 0.45–0.73 (every contact fires PAM hard), |Δw| 4.4e-3 → 1.7e-3 per update, **DNa02 240 → 320 Hz, GF 68 → 117 Hz** (DNa02 at the refractory ceiling ≈ 330 Hz). Stage B gens 4–6: contact 0.06 0.11 0.11, track_fraction 0.06–0.11 (humans start out of view; the rover drives straight at forward 0.99 and never finds them: heat drives no DN, see probe), DNa02 20–35 Hz.
Verification of the gen-6 checkpoint (32 envs, 5 s, stage A, seeds 1000 and 4, both MPS-dense and CPU-event): contact 0.94 / track 0.97–1.00 / forward 0.96 / DNa02 ≈ 272 / 270 Hz, DNa01 81–95 Hz, GF 118–131 Hz, identical across engines and seeds (no engine sensitivity). Same protocol untrained + tonic 0.6: contact 0.72, DNa02 63/63, forward 0.64. Watcher (64 envs, 20 s, seed 1000): gen 0 → front 0.969, sustained 0.585, adv2s 0.94; gen 4 → 0.984, sustained 0.80.
Verdict: **diverged** by the stated rule (rates > 200 Hz) even though contact improved 0.72 → 0.94 and the stage A target was met: the three-factor rule potentiated eye→DN synapses under a huge, unbalanced dopamine signal (+5 per contact reaches PAM; the −0.02 no-contact penalty never reaches PPL1 threshold, so PPL1 never depresses anything) until DNa02 saturated at the 3×|w0| bound. The behaviour still "works" (charge forward, coarse steering) but it is a saturated circuit, not a tuned one. Do not present this as "the fly learned to hunt" without saying so.
Open questions (need a human): rebalance the dopamine channel (scale PPL1 punishment so −0.02/step is visible, or lower dopamine amp / eta) — allowed within the existing rule; a homeostatic/weight-decay term would be a new rule (guardrail 4) — ask Taka.

# MORNING REPORT (written 05:40, 2026-09-18; run started 23:38)

## 1. Where the project stands
The whole pipeline exists and is tested end to end on this Mac: the real male-CNS subcircuit (15,000 neurons, 2.33M signed synapses) runs as a batched LIF with three interchangeable engines, the retina/senses/motor mapping is calibrated, the batched arena works, both learning rules are implemented with Dale's law and bounds enforced, and there is an evaluation harness, lesion controls, a lobotomize/restore CLI, replay files for the viz, and a robot bridge that meets the latency budget. M1, M2, M3 pass on the connectome alone. **The untrained fly brain turns toward a human (75–78% of envs) purely from wiring, and lesions prove it (LC10a out → 0%, shuffled wiring → 0%, LC4/LPLC2 out → looming response gone, steering intact).** What the wiring does NOT give us is forward drive: no eye input reaches DNa01/DNp09 at any non-seizing gain, even after forcing every neuron on the shortest anatomical paths into the circuit (brain_v2). All four biological training jobs are flat because a rover that cannot advance never earns reward. With a clearly-labelled tonic "walking drive" fallback the rover advances, reaches the stage A target in 4 generations, and then saturates its steering neurons under the unbalanced dopamine signal. Stage B (search) fails because heat reaches no descending neuron either.

## 2. Forward drive: NOT fixed biologically; fallback implemented (off by default); decision needed
Shortest ConnectsTo paths (≤4 hops, weight ≥5), neuPrint male-cns:v1.0 (`data/paths_v2.json`):
| pair | paths | lengths | top intermediates |
|---|---|---|---|
| LC10a→DNp09 | 550 | 1 hop: 4, 2: 105, 3: 441 | LAL026_b, AOTU041, TuTuA_1, AOTU042, AOTU101m, LC10c-1 |
| LC10a→DNa01 | 550 | 2: 355, 3: 195 | AOTU019, AOTU041, AOTU016_a/c, AOTU001, LT82a |
| LC4→DNp09 | 252 | 2: 155, 3: 97 | PVLP141, PVLP010, VES023, DNp04, LHAD1g1 |
| LC4→DNa01 | 252 | 2: 209, 3: 43 | PVLP141, PVLP137, PLP029, LT82a, aSP10A_b |
739 of the 795 path neurons were already in brain.npz. brain_v2.npz (all path neurons forced) behaves identically to v1: M2b (centred human 2 m) forward-DN rate ≤ 2 Hz at gains 0.05/0.07/0.1 and amp_track 20/40; M3b advance 0%. The direct/2-hop excitation onto DNp09 is weak (4 direct LC10a edges of 6–16 synapses) and DNp09's strongest inputs (PVLP020, CL366) are inhibitory and themselves eye-driven. Fallback: `--amp_tonic 0.6` (constant current into DNa01_L/R, off on contact) → M3b advance 100%, turn-toward 72–75%, contact in 15 s in 4/4 replay envs.

## 3. Evaluation tables (no learning during evaluation; CPU event engine; gain 0.05, amp_track 20, k_f 0.01)
Untrained brains — 128 envs, 20 s, seeds 1000/1001/1002 (A/B/C):
| run / checkpoint | stage | envs | tonic | front | ttc s | sust | back | |bear|2s | adv2s | DNa02_L | DNa02_R | DNp09 | GF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v1_untrained | A | 128 | 0.0 | 0.359 | 7.954 | 0.005 | 0.0 | 0.134 | 0.328 | 10.4 | 10.5 | 0.0 | 14.9 |
| v1_untrained | B | 128 | 0.0 | 0.438 | 8.007 | 0.027 | 0.516 | 1.951 | 0.641 | 16.7 | 16.8 | 0.0 | 35.0 |
| v1_untrained | C | 128 | 0.0 | 0.531 | 8.172 | 0.009 | 0.367 | 1.448 | 0.43 | 16.5 | 16.4 | 0.0 | 37.6 |
| v2_untrained | A | 128 | 0.0 | 0.367 | 7.863 | 0.005 | 0.0 | 0.136 | 0.328 | 10.4 | 10.5 | 0.0 | 14.8 |
| v2_untrained | B | 128 | 0.0 | 0.43 | 8.051 | 0.027 | 0.523 | 1.953 | 0.648 | 17.0 | 17.0 | 0.0 | 34.7 |
| v2_untrained | C | 128 | 0.0 | 0.523 | 7.92 | 0.01 | 0.367 | 1.45 | 0.445 | 16.6 | 16.3 | 0.0 | 37.3 |

Checkpoints (watcher: stage A, 64 envs, seed 1000, 20 s; each checkpoint restores its own gain/k/amps):
| run / checkpoint | stage | envs | tonic | front | ttc s | sust | back | |bear|2s | adv2s | DNa02_L | DNa02_R | DNp09 | GF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tfA_both_gen0004 | A | 64 | 0.0 | 0.5 | 6.702 | 0.006 | 0.0 | 0.15 | 0.359 | 14.5 | 14.7 | 0.0 | 18.2 |
| tfA_both_gen0007 | A | 64 | 0.0 | 0.531 | 6.653 | 0.006 | 0.0 | 0.141 | 0.359 | 21.1 | 21.6 | 0.0 | 22.6 |
| esA_dnin_gen0004 | A | 64 | 0.0 | 0.453 | 6.323 | 0.004 | 0.0 | 0.151 | 0.359 | 10.9 | 11.1 | 0.0 | 15.9 |
| tfA_both_gen0008 | A | 64 | 0.0 | 0.531 | 6.691 | 0.008 | 0.0 | 0.151 | 0.359 | 22.0 | 22.7 | 0.0 | 23.6 |
| esA_dnin_gen0016 | A | 64 | 0.0 | 0.484 | 6.5 | 0.004 | 0.0 | 0.15 | 0.359 | 11.5 | 11.6 | 0.0 | 16.1 |
| esA_dnin_gen0023 | A | 64 | 0.0 | 0.484 | 6.478 | 0.005 | 0.0 | 0.15 | 0.359 | 11.4 | 11.5 | 0.0 | 15.7 |
| esA_dnin_gen0026 | A | 64 | 0.0 | 0.484 | 6.473 | 0.004 | 0.0 | 0.145 | 0.359 | 11.3 | 11.3 | 0.0 | 16.0 |
| esA_lcdn_gen0001 | A | 64 | 0.0 | 0.453 | 6.148 | 0.004 | 0.0 | 0.145 | 0.359 | 10.8 | 10.9 | 0.0 | 16.0 |
| esA_lcdn_gen0004 | A | 64 | 0.0 | 0.453 | 5.837 | 0.004 | 0.0 | 0.149 | 0.359 | 11.0 | 11.1 | 0.0 | 16.1 |
| esA_lcdn_gen0006 | A | 64 | 0.0 | 0.438 | 5.809 | 0.005 | 0.0 | 0.148 | 0.359 | 10.8 | 10.9 | 0.0 | 16.0 |
| esA_lcdn_gen0009 | A | 64 | 0.0 | 0.469 | 6.633 | 0.007 | 0.0 | 0.15 | 0.359 | 10.9 | 11.1 | 0.0 | 15.8 |
| tfA_kcmbon_gen0001 | A | 64 | 0.0 | 0.469 | 6.158 | 0.004 | 0.0 | 0.143 | 0.359 | 11.0 | 11.1 | 0.0 | 15.9 |
| tfA_kcmbon_gen0006 | A | 64 | 0.0 | 0.469 | 6.158 | 0.004 | 0.0 | 0.143 | 0.359 | 11.0 | 11.1 | 0.0 | 15.9 |
| tfA_kcmbon_gen0011 | A | 64 | 0.0 | 0.469 | 6.158 | 0.004 | 0.0 | 0.143 | 0.359 | 11.0 | 11.1 | 0.0 | 15.9 |
| tfA_kcmbon_gen0015 | A | 64 | 0.0 | 0.469 | 6.158 | 0.004 | 0.0 | 0.144 | 0.359 | 11.1 | 11.2 | 0.0 | 15.9 |
| tfA_both_tonic_gen0000 | A | 64 | 0.6 | 0.969 | 2.113 | 0.585 | 0.0 | 0.266 | 0.938 | 307.6 | 302.2 | 0.0 | 105.9 |
| tfA_kcmbon_gen0017 | A | 64 | 0.0 | 0.484 | 6.508 | 0.004 | 0.0 | 0.144 | 0.359 | 11.1 | 11.2 | 0.0 | 16.0 |
| tfA_both_tonic_gen0004 | A | 64 | 0.6 | 0.984 | 2.023 | 0.798 | 0.0 | 0.232 | 0.969 | 312.8 | 307.0 | 0.0 | 109.7 |
Column meanings: front = envs with ≥1 front touch; ttc = mean first-touch time; sust = fraction of steps in front contact; back = envs with ≥1 back/side touch; |bear|2s = mean |bearing| at 2 s (rad); adv2s = envs whose distance fell >0.05 m by 2 s; rates in Hz. Without tonic, every contact is a human walking into the rover (forward ≈ 0); stage B/C bumps include ~50% back/side.

## 4. Lesion table (brain_v2 untrained, stage A, 256 envs, seed 2000, 2 s, humans frozen; `logs/ablations.csv`)
| condition | turn-toward | advance | turn_mean | DNa02_L | DNa02_R | GF | GF looming |
|---|---|---|---|---|---|---|---|
| baseline | 0.777 | 0.008 | 0.02 | 17.1 | 18.0 | 9.0 | 106.9 |
| lesion LC10a (L+R) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |  |
| lesion DNa02_L | 0.246 | 0.0 | 0.061 | 0.0 | 5.0 | 2.0 |  |
| lesion LC4+LPLC2 | 0.754 | 0.0 | 0.021 | 17.2 | 18.3 | 0.0 | 0.0 |
| shuffled (degree-preserving) | 0.0 | 0.031 | -0.002 | 0.1 | 0.0 | 0.0 |  |
GF looming = giant-fiber rate while a human walks at the stationary rover from 3 m at 1.4 m/s.

## 5. Training verdicts (all brain.npz, stage A)
- `tfA_both` (three-factor, KC→MBON + eye/DN, 64 envs, 20 s, 9 gens): **flat** — contact 0.31–0.47 with no trend (window 0.33–0.40), |Δw| ≤ 3e-4, DNp09 0, forward ≤ 0.011.
- `esA_dnin` (ES σ 0.05 on 4.5k DN-input synapses, 128 envs, 10 s, 27 gens): **flat** — contact first/second half 0.269/0.258, |Δw| 0.2/gen but forward ≤ 0.008, DNp09 0.
- `esA_lcdn` (ES σ 0.05 on 195k eye/DN synapses, 128 envs, 10 s, 10 gens): **flat** — 0.252/0.248, forward ≤ 0.007, DNp09 0.
- `tfA_kcmbon` (three-factor on 61k KC→MBON, 64 envs, 20 s, 18 gens): **flat** — |Δw| ≤ 5e-7 (Kenyon cells are silent in this task, no eligibility ever forms); checkpoint evaluations identical to the digit.
- `tfA_both_tonic` (three-factor both + FALLBACK tonic 0.6, 64 envs, 20 s, 7 gens): **diverged by the >200 Hz rule, contact improved** — stage A window 0.98 → promoted to B after 4 gens (M4 met, with the fallback); DNa02 240–320 Hz, GF ~110 Hz; gen-6 checkpoint contact 0.94 vs 0.72 untrained+tonic (5 s, 32 envs); stage B 0.06–0.11 (no search behaviour: heat drives no DN).

## 6. Ready for Friday
- Robot bridge `scripts/robot_bridge.py`: full chain, dry-run, watchdog (300 ms), E-stop (space / Enter), latency CSV. Measured on this Mac (fake boxes, no camera, CPU event engine): **median 15 ms, p95 24 ms, max 28 ms** camera→command; brain 10 ms per frame. Untested on hardware; needs `pip install opencv-python ultralytics robomaster` on the demo laptop. Turn>0 → positive z (clockwise).
- Replays for Ducks: `replay/episode_7_{untrained,trained,lesioned,tonic}.json` + `replay/FORMAT.md` (every field, units, rendering hints). 750 frames each at 50 Hz, ~0.9 MB.
- Lobotomize CLI: `python -m brain.plastic --lobotomize --checkpoint X.pt [--zero-lc10a]` / `--restore`; bit-identical round trip tested. Demo buttons in the bridge: l (lobotomy), r/p (reward/punish).
- Tests: `pytest tests/ -q` → 123 passed. `git log` has 12 commits; nothing pushed anywhere.
- Checkpoints: `checkpoints/{tfA_both,esA_dnin,esA_lcdn,tfA_kcmbon,tfA_both_tonic}_latest.pt` (+ numbered every 10 gens). watch_queue.py is still running (PID in `pgrep -f watch_queue`), harmless; kill it when you start new training.

## 7. Do these first
1. Decide on forward drive (question A below). If you accept the fallback, set `--amp_tonic 0.6` as the default in train.py/bridge and say so in the writeup; if not, the demo is a "turns to face you" demo, which the lesion table already makes honest and strong.
2. Fix the dopamine balance before any more training (question B): the −0.02 penalty never reaches PPL1, so learning is potentiation-only and saturates DNa02. Then re-run `tfA_both_tonic` for 1 h and watch DNa02 stay < 150 Hz.
3. On the demo laptop: `pip install opencv-python ultralytics robomaster`, then `python scripts/robot_bridge.py --fake-boxes --no-camera --dry-run` (latency), then `--source 0 --show` (webcam), then `--robot --dry-run`, then `--robot`.

## 8. Decisions only you can make
- **A. Forward drive.** (1) Use the tonic fallback (honest, labelled, works: advance 100%, M4 met); (2) keep biological purity: rover only turns; or (3) spend Friday morning probing every DN type in the male CNS for one that LC10a actually drives (`scripts/probe.py` + one cypher over all descending_neuron types), and swap it into motor.py if found.
- **B. Dopamine balance / stability.** Allowed within the current rule: scale the punishment channel so −0.02/step is visible to PPL1 (e.g. `amp` per unit reward asymmetric), lower `eta`, lower `dopamine_norm`. Not allowed without your OK (new rule): homeostatic weight decay or a rate-based brake. Which?
- **C. Retinotopy.** senses.py assigns LC neurons to columns by index order; neuPrint has each neuron's lobula column ROIs, so a real retinotopic map is a 1–2 h job. The symptom today: a centred human drives DNa02_R but not DNa02_L. Worth it before the robot demo?
- **D. Search (stage B / M5).** THERMO reaches nothing at gain 0.05 and seizes the mushroom body at 0.1. Options: route heat into LC10a-like drive (non-biological), or accept that the demo starts with the human in view.
- **E. Vast box.** It refused connections at 00:12 (instance gone). Everything ran on the Mac; nothing is on the box. Rent again only if you want 512-env sweeps on Friday.

## 2026-09-18 morning session (Taka's decisions: C retinotopy first, A fallback → superseded, B dopamine fix, D skip search, E no Vast)

### [10:05] Forward drive from population descending activity — WORKS, replaces the tonic fallback
Hypothesis (Taka): forward = k_f × mean rate over ALL descending neurons in the subcircuit, not DNa01+DNp09.
Done: `DN_ALL` group added to brain.npz and brain_v2.npz (241 neurons, 132 DN types incl. GF; original saved as data/brain_v1_backup.npz). `brain/motor.py` forward_source="dn_all" (default) or "dna01_dnp09" (old); checkpoints now store forward_source; old checkpoints load with the old read-out.
Result (brain.npz, gain 0.05, amp_track 20, **amp_tonic 0**, 64 envs, seed 0, stage A, humans frozen, 2 s; turn-toward = |bearing| fell > 0.02 rad; advance = distance fell > 0.05 m):
| k_f | turn-toward | advance | mean dist 2 s | fwd | DNa02 L/R | GF |
|---|---|---|---|---|---|---|
| 0.01 (old read-out DNa01+DNp09) | 75% | 0% | 1.84→1.82 m | 0.00 | 19/19 | 10 |
| 0.3 | **75%** | **100%** | 1.84→0.66 m | 0.79 | 60/60 | 77 |
| 0.6 | 72% | 100% | 1.84→0.46 m (touch range) | 0.91 | 69/68 | 97 |
| 1.0 | 70% | 100% | 1.84→0.45 m | 0.93 | 71/70 | 100 |
| 2.0 | 69% | 100% | 1.84→0.45 m | 0.95 | 74/73 | 100 |
M2b-pop (centred human 2 m, 16 envs, 300 ms): DN_ALL mean 0.49 Hz with the human vs 0.00 without (the static signal is almost all DNa02_R). During the approach the population mean rises to 10.95 Hz spread over many types: DNb05 12% (161 Hz), DNp01/GF 7%, DNp103 5%, DNp04 5%, DNa02 5%, DNg40 5%, DNg108 5%, DNp06 4%, DNg111 4%, DNp02/DNp11/DNp03 3–4% each (looming-driven descending activity as the human looms). Removing GF from the read-out changes little (advance 100%, 1.87→0.54 m, turn-toward 69%), so it is not a GF artefact. **Default k_f = 0.3.** Yes, the rover now closes distance with the human in view, from wiring alone.
Honest framing for the writeup: the read-out choice "descending population activity = go" is ours; every rate in it is the connectome's. The tonic fallback (`amp_tonic`) stays in the code, off, for the record.
Also from option 3 (`scripts/probe_dns.py`, `logs/probe_dns_LC10a_L.log`): LC10a_L drives, ipsilaterally, DNa02 126 Hz, DNae002 62, DNa06 58, DNg111 45, DNa03 39, DNa16 37, DNg75 34, DNa13 27, DNg97_R 24, DNge026 17. Whole-CNS direct LC10a→DN targets: DNa10 (54 LC10a neurons, 718 syn; literature: downstream of LC10d, mediates object AVOIDANCE, so not a pursuit DN), DNa16, DNp11, DNp63, DNp09 (4 neurons, 43 syn), DNd05.
Files touched: data/brain.npz, data/brain_v2.npz (DN_ALL key), brain/motor.py, train.py, scripts/milestones.py, scripts/robot_bridge.py, scripts/probe_dns.py (new).

### [10:15] C — Real retinotopy for the LC populations — done (default: rank, anterior_sign +1)
What was done: `scripts/pull_lc_columns.py` → `data/lc_columns.json` (1,352 LC neurons; per-neuron postsynapse-weighted centroid in lobula hex space from neuPrint roiInfo `LO_{L,R}_col_{hex1}_{hex2}`). Hex→visual axes from the Reiser-lab eyemap docs (hex1→q, hex2→p, origin (18,19), v = p+q vertical, horizon axis q−p): H = hex1−hex2+1, V = hex1+hex2−37. L/R statistics match (same convention both sides). `brain/retinotopy.py` assigns columns; `Senses(col_of=...)`; `train.build(retinotopy=, anterior_sign=)`; `--retinotopy rank|angle|index`, `--anterior-sign`; checkpoints record it (old checkpoints load as "index"). Robot bridge uses it too.
Anterior sign: not in the docs. +H taken as anterior (LC10a biased to +H) and confirmed behaviourally below (−1 breaks turning).
Result (`milestones.py m2c`, brain.npz, 16 envs, human at 2 m frozen, 300 ms, gain 0.05, amp_track 20; and `m3`, 64 envs, 2 s, humans frozen, k_f 0.3, amp_tonic 0):
| mapping | centre DNa02 L/R | centre turn | left30 DNa02 L/R | right30 L/R | M3 turn-toward | final |bearing| | M3b advance | dist 2 s |
|---|---|---|---|---|---|---|---|---|
| index (old, bodyId order) | 0.0 / 67.8 | **+0.97 (veers right)** | 92 / 0 | 0 / 105 | 75% | 0.185 rad | 100% | 0.66 m |
| rank, +H anterior (new default) | 0.0 / 0.0 | 0.00 | 174 / 0 | 0 / 179 | **75%** | **0.124 rad** | 100% | 0.76 m |
| rank, −H anterior | 89 / 104 | +0.30 | 53 / 0 | 0 / 28 | 31% | 0.435 rad | 100% | 0.65 m |
The demo-visible veer is gone (centre symmetric), lateral responses doubled, aim improved, oscillation halved (|turn| 0.22 vs 0.39). Behavioural note: the most anterior LC10a neurons do not drive DNa02, so a perfectly centred static human at 2 m gives no turn and no forward drive until it moves or looms (in M3 the rover still closes 100% because starts are off-centre and looming takes over). Angle mode (absolute 5.3°/column, out-of-FOV neurons silent) is implemented but untested; ~2/3 of LC11/12/15 fall outside the camera in that mode.
Files touched: brain/retinotopy.py (new), scripts/pull_lc_columns.py (new), data/lc_columns.json (new), brain/senses.py, train.py, scripts/milestones.py (m2c, --retinotopy), scripts/robot_bridge.py.

### [10:38] Compute moved to the Vast A100 (Taka's decision); Mac keeps only the demo/bridge
Box: A100-SXM4 40 GB, torch 2.14+cu130 (last night's venv survived the restart), repo synced by Taka with `scripts/sync_to_box.sh`. On the box: 124 tests pass; M2 PASS on CUDA (identical numbers); **event engine on CUDA at 512 envs: 3.42 ms per brain step → 1.1 min per 20 s generation** (CSR 13.3 ms; the Mac took ~9 min per generation at 64 envs). M3/M3b at 512 envs: 75% turn-toward (|bearing| 0.342→0.120 rad), 100% advance (1.94→0.88 m). Queue `scripts/vast_queue.sh` started 10:35 (box clock 14:35): tfA_both_512 → tfA_both_pun10 → esA_dnin_512, 60 min each, plus `watch_queue.py` (256 envs, stage A, every 10 min). Pull results: `rsync -az -e "ssh -p 30714" root@150.136.39.147:~/flybrain-rover/checkpoints/ checkpoints_box/` (and logs/).
Mac run `tfA_both_dnall` (started 10:16) continues to its 60 min end as the first demo-brain candidate; gen 0: contact 0.97, forward 0.89, DNa02 141 Hz (scaling holds it under 150), GF 147, |Δw| 3.3e-3, D 0.72.

### [10:50] Task 4 — Robot bridge bring-up on the Mac — done (dry; hardware pending)
- Packages: `opencv-python 5.0`, `ultralytics 8.4` in `.venv` (Python 3.12). The DJI `robomaster` SDK only ships cp36–cp38 wheels, so it lives in `.venv38` (uv-managed **x86_64** Python 3.8 under Rosetta; `arch x86_64`). Hence a two-process design: `scripts/robot_daemon.py` (py3.8, the only process touching the SDK; JPEG frames out / `V x y z` commands in over localhost TCP; watchdog 300 ms; `--fake` mode) ↔ `scripts/robot_bridge.py --robot-daemon 127.0.0.1:9500` (py3.12, brain). Tested end to end with the fake daemon: frames flow, dry-run latency **median 41 ms, p95 66 ms, max 95 ms** with the CPU saturated by two evaluations and a training run (brain alone was 10 ms idle, 35 ms under that load).
- Detector: yolo11n person class, imgsz 320: **29 ms/frame on MPS, 83 ms on CPU** (loaded machine). Bridge now puts the detector on the GPU by default (`--yolo-device`), brain on the CPU event engine. Expected demo loop: ~30 ms detector + ~10 ms brain + JPEG hop ≈ 50–60 ms.
- Still to do with the real robot: `--source robot`/daemon `--conn ap`, verify `chassis.drive_speed` sign (turn > 0 → +z clockwise), measure real camera hfov, walk-left/right test (`--show`).
- Exploratory state (`--explore`, brain/explore.py): implemented, internal drive labelled non-sensory; stage B evaluation running (result below when it lands).

### [10:55] Box queue restarted with the event-sparse learner and --no-promote
- `learn/three_factor.py` eligibility is now event-sparse (only edges of neurons that spiked; exact, tested against the dense reference). Learner overhead ~4× lower on CPU; on the A100 the dense version made a 512-env generation take 288 s vs 66 s of pure simulation.
- Caveat found: with 512 envs one generation is already ≥ 200 episodes, so `train.py` promoted the first box job to stage B after a single 0.93 generation (its 3 stage-B generations: contact 0.52–0.53, track 0.42–0.44, no explorer). Added `--no-promote`; all three box jobs use it so they stay in stage A (demo brain). That first attempt is kept as `logs/train_tfA_both_512_dense_learner.log` / `checkpoints/tfA_both_512_dense_learner_latest.pt` on the box.
- Box queue now: tfA_both_512 → tfA_both_pun10 → esA_dnin_512 (60 min each, 512 envs, stage A, event engine), watcher 256 envs every 10 min.

### [11:00] Task 6 — Live brain feed for the viz — done
`scripts/robot_bridge.py --viz-ws 8765 --viz-jsonl logs/live.jsonl`: one JSON object per camera frame with the replay/FORMAT.md fields (t since start, forward, turn, boxes, heat, pres/size/mot, loom, rates for 25 groups incl. DN_ALL, spike raster over the same fixed 512-neuron sample, exploring, lobotomy). Websocket broadcast (`websockets`, latest frame to each client) plus a tail-able JSONL. Tested with a client while the bridge ran fake boxes with --explore: frames at camera rate, latency budget unaffected (median 30 ms). Ducks binds ws://<laptop>:8765 in Three.js.

### [11:08] Search as an exploratory internal state — done, stage B PASS (0.742 > 0.60)
Implemented `brain/explore.py` ("exploratory state, internal drive, not sensory"): after 0.5 s with no retina presence, tonic 0.4 mV/step into DNa01_L/R plus a saccade generator (300 ms ± 30% bursts of 1.0 mV/step into DNa02_L then DNa02_R, alternating every 1.5 s ± 30%; bursts biased toward the warmer heat side when the two heat sensors disagree); gated off within one env step (20 ms) of any presence. `--explore` in train.py, evaluate.py, robot_bridge.py; arena now records time to first sight.
Result (`scripts/evaluate.py --stages B --explore`, brain.npz untrained, DN_ALL forward k_f 0.3, retinotopy rank, gain 0.05, **128 envs, 20 s, seed 1000**, humans 1–8 start OUT of view, CPU event engine):
| condition | front contact | acquire rate | time to first sight | time to first contact | sustained | back contact | track | DNa02 L/R |
|---|---|---|---|---|---|---|---|---|
| explore ON | **0.742** | 0.906 | **2.46 s** | 6.0 s | 0.497 | 0.391 | 0.638 | 35.7 / 35.3 |
| explore OFF (v1_untrained_dnall, 64 envs, seed 1001, index mapping) | 0.469 | – | – | 4.7 s | 0.269 | 0.516 | 0.368 | 38.7 / 37.8 |
Reading: the saccades sweep the camera and the tonic drive creeps forward, so 91% of envs see a human within a few seconds and pursuit takes over; back/side bumps (39%) are humans walking into the rover while it turns. With the DN_ALL read-out the tonic DNa01 current adds little forward speed by itself (2 of 241 neurons); the saccade bursts contribute (DNa02 activity is in DN_ALL), which gives curved search paths. Heat bias is active but heat reaches no DN, so it only steers the saccade choice. Demo default stays "person in view" (TODAY.md decision); `--explore` is available for the search moment if Taka wants it.
Also from the same evaluation batch: untrained brain.npz with DN_ALL forward (index mapping, 64 envs, 20 s): stage A front contact **0.969**, ttc 2.5 s, sustained 0.84, DNa02 117/116 Hz, advance 0.80; stage C 0.688. So M4's 90% stage-A bar is met by the UNTRAINED brain once forward drive exists; training can only add speed/robustness, and the writeup must say so.

## Class-time run (Taka away 11:20 → 15:30; box trains unattended until ~20:20)

### [11:22] Box hardened — done
Box (150.136.39.147:30714): main queue `scripts/vast_queue.sh` (nohup, parent PID 1, started 14:54 box = 10:54 Toronto; job 1 `tfA_both_512` at gen 6: contact 0.97, DNa02 ~155 Hz, forward 0.92, 146–157 s/generation) continues to ~17:54 box, then `tmux` session **fly** takes over: window `queue2` runs `scripts/vast_queue_2.sh` under nohup (waits for the main queue, then tfA_both_explore 60 min → esA_lcdn_512 60 → tfA_both_long 180 (continues tfA_both_512) → tfB_explore 90 (stage B with the explorer) ≈ done 00:20 box = 20:20 Toronto); window `watcher` runs `watch_queue.py` (256 envs, stage A, every 10 min) under nohup. Verified: `tmux ls` → fly: 2 windows; processes have parent 1 or the tmux server; `nvidia-smi` 100% util, 25 GB used. A stray duplicate afternoon queue from an interrupted ssh was killed (PID 2758). No Mac job runs past 15:30 except evaluations started before it.
Files touched: scripts/vast_queue_2.sh (new, also on the box), scripts/lock_demo.py (new: M6 criteria, locks checkpoints/demo_brain.pt and DEMO.md), train.py (run_episode returns group_rates/group_peaks for every named group), scripts/ablations.py (--checkpoint; shuffle uses the trained weights).

### [11:25] M7 Search — PASS, parameters locked in DEMO.md
Stage B contact 0.742 ≥ 0.60 with the exploratory state (128 envs, 20 s, seed 1000, brain.npz untrained; details in the 11:08 entry), so no sweep. Parameters (onset 0.5 s, tonic 0.4, saccade 1.0 mV/step, 300 ms ± 30 %, period 1.5 s ± 30 %, heat-biased) written to DEMO.md "Search" section; `--explore` flag on demo.py/bridge.

### [11:26] M9 Demo runbook — done (hardware step pending)
`scripts/demo.py`: the bridge with hotkeys 1 baseline / 2 lesion LC10a / 3 restore / 4 lobotomize / 5 restore plasticity / space E-stop, live feed always on (ws://localhost:8765 + logs/demo_live.jsonl). Hotkeys work from the OpenCV window and as typed digits in the terminal; dry-tested (fake boxes, no camera): keys 2, 4, 1 executed, wheels commands printed, latency 14 ms median. `DEMO.md`: setup (daemon in .venv38 + demo.py), a 2-minute script (time / presenter line / key / what the audience sees), honesty lines, fallbacks (E-stop + dry-run narration; `replay/viewer.html` standalone player; detector misses; latency). Files: scripts/demo.py (new), scripts/robot_bridge.py (hotkey hooks, main(argv, hotkeys)), DEMO.md (new), handoff/BEFORE_LEAVING.md (new).

### [11:35] M8 Lesion table on the trained checkpoint — PASS (preview on the box's gen-11 checkpoint; re-run on demo_brain.pt below when locked)
Protocol: `scripts/ablations.py --checkpoint`, brain.npz, box GPU (event engine), **256 envs, seed 2000, 2 s, humans frozen**, DN_ALL forward k_f 0.3, retinotopy rank. Checkpoint = tfA_both_512 gen 11 (three-factor both, 512 envs, stage A, ~30 min of training). Same protocol untrained in the second column set.
| condition | trained: turn-toward / advance / turn_mean / DNa02 L,R / GF | untrained: turn-toward / advance / turn_mean / DNa02 L,R / GF |
|---|---|---|
| baseline | 0.71 / 0.99 / +0.029 / 65, 70 / 103 | 0.75 / 0.99 / +0.002 / 16.5, 16.5 / 63 |
| lesion LC10a L+R | **0.00** / 0.99 / +0.006 / 0.0, 0.3 / 28 | **0.00** / 0.99 / 0.000 / 0, 0 / 19 |
| lesion DNa02_L | 0.23 / 0.99 / **+0.054 (right bias)** / 0, 6.0 / 71 | 0.32 / 0.99 / +0.050 / 0, 3.6 / 40 |
| lesion LC4+LPLC2 | 0.71 / 0.99 / +0.025 / 50, 54 / 0.2 | 0.72 / 0.99 / +0.002 / 12.5, 12.4 / 0.0 |
| looming GF (3 m approach at 1.4 m/s): baseline → lesioned | **130.2 → 0.1 Hz** | 105.3 → 0.0 Hz |
| shuffled (degree-preserving) | 0.11 / 0.74 / +0.014 / 11, 12 / 2.3 | 0.00 / 0.38 / −0.008 / 0.6, 0.2 / 0.0 |
Reading: all three predicted lesion effects hold on the trained brain. Training raised DNa02 drive 4× (65–70 vs 16.5 Hz) without improving 2-s aiming (0.71 vs 0.75, within noise), and the shuffled-wiring control is weaker on trained weights (0.11 vs 0.00: potentiated synapses spread over random targets still produce some descending drive; "advance" under shuffle is looming-driven population activity, not tracking). Lobotomize → restore on this checkpoint: **bit-identical** (`python -m brain.plastic`), and the trained brain differs from the connectome on 178,388 of 2,334,959 synapses.

### [11:38] M10 Devpost draft — done
`DEVPOST.md` (Sonnet subagent from CONTEXT.md §2/3/5/6/7 + STATUS.md numbers; reviewed): Inspiration, What it does, How we built it, Challenges, Accomplishments, What we learned, What's next, Built with, Credits. 1,400 words, every number with provenance, no em dashes, no "learned to hunt", the DN_ALL read-out labelled as our choice, the flat overnight runs and the saturated fallback run described as such, connectome-pilot's "a 3-line controller can beat it" acknowledged. One `[TODO: pending demo_brain.pt evaluation]` in "What's next" to be replaced with the M6 number. Copied to handoff/.

### [12:02] M6 Demo brain — evaluation (fixed set: stage A, 128 envs, 20 s, seed 1000, no learning, `scripts/lock_demo.py`)
| candidate | front contact | sustained | ttc | back | DNa02 mean | highest group (episode mean) | verdict vs literal criteria |
|---|---|---|---|---|---|---|---|
| Mac `tfA_both_dnall_latest` (gen 6, promoted to stage B after gen 3) | 0.961 | 0.759 | 4.16 s | 0.000 | **182.4** | LC10a_L 246.6 | FAIL (DNa02 > 160, LC10a > 200) |
| Box `tfA_both_512_latest` (gen 12, stage A, sparse learner, --no-promote) | **0.977** | **0.817** | **3.24 s** | 0.008 | 155.7 | LC10a_L 266.0 (LC10a_R 143.9) | passes contact/sustained/DNa02; LC10a > 200 |
| Untrained brain.npz (same read-out/retinotopy) | 0.977 | 0.762 | 4.25 s | 0.000 | 51.0 | LC10a_R 206.4 | passes contact/sustained/DNa02; LC10a > 200 |
Finding: the literal "no group > 200 Hz" cannot be met by any brain at amp_track 20, because LC10a is the retina-driven input population and fires 200–270 Hz whenever the human fills the view (the untrained brain violates it too, by 6 Hz). Every non-sensory group is below 200 Hz in the box and untrained brains (DNa02 ≤ 158, GF ≤ 143, DNa01 ≤ 56; THERMO 150–200 is also sensory, driven by the heat sensors). Training's measurable effect on this set: time to contact 4.25 → 3.24 s and sustained contact 0.762 → 0.817 at equal contact rate; it also made LC10a_L (266) fire more than LC10a_R (144), a side asymmetry to keep an eye on (DNa02 L/R stayed balanced: 157.7 / 153.7). Evaluating box gen 0 and gen 10 next, then locking the best non-saturated generation with the 200 Hz rule applied to non-sensory groups, stated as such.

### [12:30] M6 Demo brain — LOCKED: `checkpoints/demo_brain.pt` = box `tfA_both_512_gen0020.pt` (job 1 final, stage A, 512 envs, 21 generations, three-factor "both", DN_ALL forward, retinotopy rank, PPL1 ×5, eta 3e-4, scaling 150 Hz)
Full fixed-set table (stage A, 128 envs, 20 s, seed 1000, no learning, CPU event engine; `logs/m6/*.log`):
| candidate | contact | sustained | ttc | back | DNa02 mean | highest non-sensory group | LC10a L / R |
|---|---|---|---|---|---|---|---|
| untrained brain.npz | 0.977 | 0.762 | 4.25 s | 0.000 | 51.0 | GF 101.5 | 197.5 / 206.4 |
| box gen 0 | 0.953 | 0.743 | 4.40 s | 0.000 | 144.2 | DNa02_R 144.4 | 164 / 227 |
| box gen 10 | 0.977 | 0.816 | 3.29 s | 0.008 | 153.2 | DNa02_L 155.1 | 266 / 144 |
| box gen 12 | 0.977 | 0.817 | 3.24 s | 0.008 | 155.7 | DNa02_L 157.7 | 266 / 144 |
| **box gen 20 (locked)** | **0.977** | **0.817** | **3.23 s** | 0.008 | **156.1** | DNa02_L 157.8 | 266 / 145 |
| Mac gen 6 (stage-B trained) | 0.961 | 0.759 | 4.16 s | 0.000 | 182.4 | fails | 247 / – |
Verdict, said plainly: the literal criterion "no group > 200 Hz" fails for EVERY brain including the untrained one, because LC10a is the retina-driven input population (amp_track 20) and fires 200–270 Hz whenever the human fills the view. With the 200 Hz rule applied to non-sensory groups (everything that is not LC*/LPLC2/THERMO) gen 20 passes all four criteria; gens 10–20 are a plateau, so the final checkpoint was chosen. Gen 0 is the "best literal" candidate (LC10a_L 164) but still fails on LC10a_R (227) and is worse on every behavioural number. Caveat recorded: training potentiated LC10a_L (266 Hz) more than LC10a_R (145) while DNa02 stayed balanced (157.8/154.3); the ablation baseline shows a small right turn bias (turn_mean +0.029 vs +0.002 untrained). What learning bought on this set: time to contact 4.25 → 3.23 s and sustained contact 0.762 → 0.817 at equal contact rate. `DEMO.md` "Demo brain" section written with the exact commands; the checkpoint carries its evaluation, source and git commit.

### [12:32] M8 Lesion table on demo_brain.pt — PASS (box GPU on the identical weights `tfA_both_512_gen0020.pt`, brain.npz, 256 envs, seed 2000, 2 s, humans frozen; `logs/ablations_demo_brain.csv` on the box)
| condition | turn-toward | advance | turn_mean | DNa02 L / R | GF |
|---|---|---|---|---|---|
| baseline | 0.71 (gen 11 preview; gen 20 row in the box CSV) | 0.99 | +0.029 | 65 / 70 | 103 |
| lesion LC10a L+R | **0.00** | 0.99 | +0.006 | 0.0 / 0.3 | 28 |
| lesion DNa02_L | 0.23 | 0.99 | **+0.055 (right bias)** | 0 / 6.0 | 70 |
| lesion LC4+LPLC2 | 0.72 | 0.99 | +0.025 | 50.5 / 54.7 | 0.1 |
| looming (3 m approach, 1.4 m/s): GF baseline → lesioned | **129.7 → 0.1 Hz** | | | | |
| shuffled (degree-preserving, trained weights) | 0.11 | 0.73 | −0.010 | 12.1 / 11.7 | 2.1 |
Lobotomize → restore on demo_brain.pt (`python -m brain.plastic`): **bit-identical**; learning changed 178,445 of 2,334,959 synapses.

### [12:34] M11 Leave-ready — done
Packages: `.venv` has opencv-python 5.0 + ultralytics 8.4; `.venv38` (x86_64 Python 3.8 under Rosetta) has the robomaster SDK + cv2 4.10; `yolo11n.pt` cached in the repo root (bridge ran). `handoff/`: demo_brain.pt, DEMO.md, DEVPOST.md, replay/ (episode_7_untrained/trained/lesioned/tonic.json regenerated with today's brain; trained = demo_brain.pt; FORMAT.md; viewer.html), BEFORE_LEAVING.md (rsync box checkpoints/logs down, kill Mac watchers, commit; box stays alive until tonight's pull). No Mac training runs; the only Mac processes are done.

### [23:20 Fri] Robot-readiness check: chassis lag and turn rate (Taka's concern: "the real robot won't turn as simply")
Facts: the S1 has mecanum wheels and `chassis.drive_speed(x, y, z)` takes forward speed and rotation rate separately (the SDK mixes the wheels), so it rotates in place and drives forward at once; the sim's differential drive is the conservative case. Real differences: (1) gimbal mode: the camera must be locked to the chassis (CHASSIS_LEAD), otherwise it stays pointed at the wall while the body turns; `scripts/robot_daemon.py` now sets it after recentring. (2) inertia + Wi-Fi camera latency: modelled as a first-order velocity lag `Arena(motor_tau=...)`. (3) turn-rate scale: sim 143°/s at turn=1, bridge default 90°/s. (4) z sign: must be verified on the robot (turn > 0 → +z clockwise).
Test (`checkpoints/demo_brain.pt`, 64 envs, seed 0, 2 s, humans frozen; turn-toward = |bearing| fell > 0.02 rad; advance = dist fell > 0.05 m):
| chassis lag tau | omega_max | turn-toward | final |bearing| | advance | dist 1.87 m → |
|---|---|---|---|---|---|
| 0 (sim default) | 143°/s | 0.73 | 0.163 rad | 1.00 | 0.68 m |
| 0.2 s | 143°/s | 0.72 | 0.188 | 1.00 | 0.69 m |
| 0.4 s | 143°/s | 0.70 | 0.191 | 1.00 | 0.78 m |
| 0.2 s | 90°/s | 0.72 | 0.178 | 1.00 | 0.72 m |
| 0.4 s | 90°/s | 0.72 | 0.183 | 1.00 | 0.80 m |
| 0.4 s | 57°/s | 0.72 | 0.175 | 1.00 | 0.81 m |
| 0.8 s | 90°/s | 0.67 | 0.205 | 1.00 | 0.94 m |
Reading: the loop closes through the world, so lag and a slower turn rate only slow the approach a little; tracking holds to 0.4 s of lag (a heavy chassis) and only drops to 0.67 at 0.8 s. Start the robot at w_max 90°/s, v_max 0.5 m/s. Files: scripts/robot_daemon.py (gimbal mode), env/arena.py (motor_tau).

### [01:10 Sat] Trailing test (does it follow a walking person?) — YES
`/tmp/trail.py` (kept in logs/trail_test.log): stage A, humans walking continuously (no pauses, new heading every 2–6 s), rover v_max = 1.15 × walker speed, 32 envs, 15 s, scored over the last 10 s, CPU event engine.
| brain | walkers | in view | within 1 m | within 2 m | mean dist | front-touching | lost (<50 % in view) |
|---|---|---|---|---|---|---|---|
| demo_brain.pt | 1.0–1.4 m/s (seed 11) | 100 % | **87 %** | 89 % | 0.78 m | **86 %** | 0 % |
| untrained | 1.0–1.4 m/s (seed 11) | 100 % | 82 % | 85 % | 1.02 m | 81 % | 0 % |
| demo_brain.pt | 1.4 m/s (seed 12) | 97 % | 92 % | 92 % | 0.75 m | 91 % | 3 % |
Reading: once acquired, the rover stays glued to a walking person (touching most of the time, never losing them at 1.0–1.4 m/s); training's contribution is a tighter follow (0.78 vs 1.02 m). Ducks's earlier sim reported 19 % tracking; ours is not the same metric, but the behaviour is clearly there.

### [01:15 Sat] Robot bring-up, attempt 1 — waiting for the robot's Wi-Fi
Taka: S1, SSID RMEP-21adf5 / 12341234, "do not move it". Scan from the laptop (30 networks incl. HackTheNorth, eduroam, bracketbot) shows no RMEP-21adf5: robot off, out of range, or its connection switch on router mode instead of direct-connect. A watcher joins the network as soon as it appears; after that only read-only SDK calls (version, modules, battery, one camera frame). No drive_speed, no gimbal commands until Taka is back.
- 01:40: `RMEP-21adf5` appeared (−40 dBm, WPA2 Personal, ch 165) but the Mac could not associate with password 12341234 (macOS −3958 then −3912, then the network dropped out of the scan twice). Either the sticker password differs or the robot is cycling its AP. A watcher keeps retrying and logs the error codes; the SDK probe stays read-only and runs only once the Mac holds a 192.168.2.x address. Nothing has been sent to the robot.
- 04:25: Mac joined the robot AP (192.168.2.30, robot 192.168.2.1 pings at 40 ms). SDK 0.1.1.61 connection request to the proxy 192.168.2.1:30030 times out (5 s) with local IP auto and explicit, conn_type ap and sta; macOS firewall off; no TCP SDK port open on the robot (40923 plaintext, 40921 video, 20020). So the robot's SDK service is not up: either the RoboMaster app is holding the robot, or the unit still needs activation / firmware update through the app (loaner units usually do), or SDK mode is off. Nothing was moved. Waiting on Taka: disconnect the app, activate/update via the app, confirm EP vs S1 (SSID prefix RMEP suggests EP).
- 04:40: **SDK connected** (0.2 s, firmware 01.01.1150, sn 3JKDH6C0012Z43) after Taka activated the robot in the app and rejoined the AP. Modules: chassis, gimbal, camera, battery, led, blaster, vision, sensor, armor → an S1. Chassis attitude readable (yaw 17.8°, pitch −8.3°). **Battery 30 % → charge it.** Camera stream start timed out on the first probe (cmdset 0x3f cmdid 0xd2 to receiver 0100); retrying with defaults. Nothing moved.
- 04:50: SDK 0.1.1.68 has no macOS wheel; transplanted its 35 pure-Python files onto a copy of the venv (`.venv38b`, macOS decoder kept, deps netifaces/netaddr/myqr added). Same camera failure: `ProtoStreamCtrl` (0x3f/0xd2) to receiver 0100 times out → the camera/gimbal board is not answering, not an SDK protocol issue. Likely the RoboMaster app holding the stream, or the board needing a power-cycle after the firmware update. Watcher retries the read-only stream test every 60 s. Backup implemented: daemon `--no-camera` (wheels only) + bridge `--local-eye --source 0` (laptop webcam as the eye). SDK gotcha fixed: conn_type must be an interned string (`sys.intern("ap")`), the SDK compares with `is`.
- 04:56: **Gimbal assembly offline.** `get_version()` per board: chassis 00.00.88.87 and vision 00.18.88.87 answer; gimbal (0x0400), camera (0x0100) and blaster (0x2300) time out. The camera sits on the gimbal, so no SDK version can fetch video until the gimbal talks to the intelligent controller: check the gimbal cable / the app's gimbal error / whether the gimbal firmware update finished. Backup path (laptop webcam as the eye, robot wheels via the daemon) being verified in dry run.
- 05:20 Sat: **Code is on Ducks's repo main**: https://github.com/Duck-luv-pie/hunting-fly as `flybrain-rover/` (5 incremental commits f4d4605…4ef0289, README section added; his `brain/` and `tools/` untouched, verified byte-identical; our 125 tests pass in the merged tree; his own tests fail on his tree because `companion_brain.config`/`data` are not pushed yet, not because of us). PR #1 closed as superseded. Excluded from git: `data/*.npz` (regenerate), checkpoints except `demo_brain.pt`, logs.
- 05:50 Sat: `scripts/viz_adapter.py` drives Ducks's Three.js viewer with our brain (his SSE protocol reproduced; brain map from `data/neuron_positions.json`, 14,878/15,000 soma positions; replay or live source). Verified: `/brain.json`, `/who`, 3 SSE ticks with every field his page reads, models served. His `fly.js` and .glb files are not in his pushed `ui/` yet (stub served meanwhile). Robot: back on with battery 12 %; gimbal/camera/blaster boards still silent → charging; re-probe watcher armed.
