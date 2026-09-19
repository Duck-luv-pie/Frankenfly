# FlyBrain Rover
A real fruit fly connectome (male CNS v1.0, Janelia/Google, Sept 2026) driving a RoboMaster S1 to find and touch humans. Behavior emerges from real neural wiring; learning uses the fly's own dopamine-gated plasticity.

Pipeline: camera -> `brain/retina.py` (24 angular columns) -> `brain/senses.py` (current into LC eye neurons) -> `brain/lif.py` (real wiring, leaky integrate-and-fire) -> `brain/motor.py` (descending neurons -> forward/turn) -> wheels.

## Run
```
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python data/pull_connectome.py          # neuPrint -> data/brain.npz (works without a token; .env token is optional)
python -m pytest tests/ -q               # 124 tests: LIF engines, arena, learning rules, lobotomize, retina sim-to-real
python scripts/milestones.py m1          # groups non-empty, N/nnz in range
python scripts/milestones.py m2          # batched gain sweep; DNa02 laterality
python scripts/milestones.py m3          # untrained brain turns toward (M3) and advances on (M3b) a human
python scripts/milestones.py m2b         # centred human at 2 m: forward-DN rates with vs without a human
python scripts/milestones.py m2c         # lateral symmetry: human at 0 / -30 / +30 deg -> DNa02 L/R and turn sign
python scripts/ablations.py              # lesion table (LC10a, DNa02_L, LC4+LPLC2, shuffled)
python scripts/evaluate.py --brain data/brain.npz --checkpoint checkpoints/X_latest.pt   # fixed evaluation set
python scripts/probe.py                  # which eye population drives which descending neuron
python scripts/paths.py                  # who feeds each descending neuron (why a DN is silent)
python train.py --stage A --envs 64 --learn three_factor --plastic both --minutes 60
python train.py --stage A --envs 128 --learn es --plastic dn_in --seconds 10 --minutes 60
python train.py --sweep gain 0.01:0.2:10 --envs 80
python train.py --record logs/replay.json --envs 4 --seconds 10     # env-0 replay for the Three.js viz
python scripts/robot_bridge.py --bench   # brain-loop latency on this laptop
python scripts/robot_bridge.py --source 0 --show --fake-box          # webcam plumbing test, no robot
python scripts/robot_bridge.py --robot --checkpoint checkpoints/<run>_latest.pt
```
Status, findings and the overnight training log: `STATUS.md`. Plan: `PLAN.md`. Rules: `CLAUDE.md`.

## Read-outs and mappings (2026-09-18)
- **Forward** = k_f (0.3) × mean firing rate over `DN_ALL`, every descending neuron in the subcircuit (241 neurons, 132 types). Nothing in the eye drives DNa01/DNp09 alone; the population does move (mostly looming-driven DN types as the target grows). `--k_f`, `Motor(forward_source=...)`.
- **Turn** = k_t (0.02) × (DNa02_R − DNa02_L); turn > 0 = right = clockwise = positive z on the RoboMaster.
- **Retinotopy**: each LC neuron gets the retina column of its real lobula dendritic footprint (`brain/retinotopy.py`, from neuPrint column ROIs via `scripts/pull_lc_columns.py`). Default `--retinotopy rank`; `index` is the old bodyId-order spread, kept for A/B tests.
- **Dopamine**: reward → PAM, punishment → PPL1 with a ×5 channel weight (`--pun_gain`); three-factor eta 3e-4 plus homeostatic synaptic scaling toward 150 Hz (`--r_target`, `--eta_h`).
- `--amp_tonic` (constant walking current into DNa01) is a labelled fallback, off by default, superseded by the DN_ALL read-out.

## What is and is not learned
No new neural network anywhere. The only parameters that change are the weights of existing connectome
synapses (fixed sparsity mask, sign fixed by the presynaptic neurotransmitter), via
`learn/three_factor.py` (eligibility traces gated by the fly's own PAM/PPL1 dopamine neurons) or
`learn/es.py` (evolution strategies over per-synapse scale factors). Sensory transduction gains
(`brain/senses.py` amplitudes), the global synaptic gain and the motor read-out gains are swept, not learned.

## Replay JSON (for the arena viz)
`train.py --record out.json` writes `{"meta": {...}, "frames": [...]}`. Each frame: `t`, `rxy`, `ryaw`
(CCW-positive), `hxy` list, `hr` list, `forward`, `turn` (+ = right), `reward`, `heat` [L, R],
`rates` (Hz per named group), `spikes` (indices into `meta.viz_neurons`, a fixed random sample of 512
neurons, spike count summed over the 20 ms env step). `meta.arena_L` is the square arena side in metres.

## Credits
- Berg et al. 2026, "Sexual dimorphism in the complete connectome of the Drosophila male CNS", Cell. Data CC-BY via neuPrint (`male-cns:v1.0`, https://neuprint.janelia.org).
- Shiu et al. 2024, "A Drosophila computational brain model reveals sensorimotor processing", Nature. LIF constants (tau 20 ms, -52/-45/-52 mV, 2.2 ms refractory, 5 ms synapse, 0.275 mV per synapse) and the NT-to-sign rule.
- Eon Systems `fly-brain` (GPL-2.0-or-later). LIF engine reference.
- `ilyaosovskoi/connectome-pilot`. Three-factor plasticity reference.
- `Imperol3/flybrain`. Looming/escape pipeline reference.
- Keles and Frye 2017. LC11 small-object tuning (why the 180-turn happens).
- Rayshubskiy et al. 2020 (bioRxiv). DNa02 drives ipsilateral turning; matches what we measure (LC10a_L -> DNa02_L).
- Marin et al. 2020, Current Biology. VP2/VP3 thermosensory glomeruli (VP3 = hot), used to pick the THERMO group.
- Bidaye et al. 2020, Neuron. DNp09 (P9) forward walking.
- neuprint-python (Janelia), PyTorch, Ultralytics YOLO (person detector on the robot), DJI RoboMaster SDK.
