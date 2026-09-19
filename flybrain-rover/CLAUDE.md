# FlyBrain Rover — project context for Claude Code

Read this first on every session, then CONTEXT.md (the why, the team, the science, the demo), then TODAY.md (the plan), then STATUS.md (results so far). CLAUDE.md is rules; the others are context and plan.

## What this is
A real fruit fly brain drives a wheeled robot. Male Drosophila CNS connectome (Berg et al., Cell, Sept 3 2026; neuPrint male-cns:v1.0), ~15,000-neuron eye-to-descending subcircuit, leaky integrate-and-fire on the real wiring, camera translated into current on the fly's eye neurons, descending neurons read as wheel commands, on a RoboMaster S1 that turns toward and reaches a person. Nothing in between is programmed. Learning uses the fly's own dopamine-gated plasticity on existing synapses. Hack the North 2026, Sept 18 to 20; sponsor tracks lock 2:00 PM Saturday.

## Non-negotiable rules
1. No new neural networks. Only existing connectome synapses may change (fixed sparsity mask), only via learn/three_factor.py or learn/es.py, plus Taka-approved homeostatic scaling on the plastic set. If you are writing an MLP policy, stop.
2. Dale's law: excitatory weights stay >= 0, inhibitory <= 0. Bounds [0, 3x |w0|]. Clip after every update.
3. The brain never sees pixels. Sensory input enters only through brain/retina.py columns and brain/senses.py injection. Training uses Retina.from_state; the robot uses Retina.from_boxes; identical tensors.
4. Everything batched, torch only, B environments as the leading dim. No Python loops over environments.
5. The control loop is local (laptop to robot). Never route it through the cloud. Under 100 ms camera to wheels.
6. Secrets only in .env. Never commit tokens or keys. Local git only; never push, publish, or send anything anywhere.
7. Honesty in every number: report brain file, envs, seeds, episode length, and the metric definition. Label non-biological mechanisms (amp_tonic) as such in code and STATUS.md. Never describe a saturated circuit (rates above 200 Hz) as "learned".
8. Quota: use the cheaper model for mechanical work, delegate grunt tasks to subagents, no open-ended loops. Taka needs Claude for the whole weekend.
9. Cite every external repo and paper in README.

## Units and conventions
meters, radians, seconds. Azimuth + = robot's RIGHT (image x). Yaw CCW positive in world frame. turn > 0 = clockwise = positive z in chassis.drive_speed (the arena applies yaw -= turn * omega * dt). Env dt 0.02 s; brain dt 0.001 s, 20 substeps per env step. Camera: 320x240, hfov 98.43, vfov 82, pinhole, 0.231 m high, turret at neutral. Gain 0.05. amp_track 20.

## What exists (all tested, 123 tests)
- data/pull_connectome.py (cypher over neuPrint, no token needed, --include-ids), data/find_paths.py
- brain/lif.py: Shiu 2024 constants, sparse CSR (CUDA/CPU), dense (MPS), event (CPU, batch 1, 0.7 ms/step) engines, spike-identical; lesion()/restore(); mutable plastic weights
- brain/retina.py, brain/senses.py (size-tuned LC injection, amp_tonic fallback off by default), brain/motor.py, brain/plastic.py (lobotomize/restore CLI)
- env/arena.py (batched world, front-contact reward, randomization), learn/three_factor.py, learn/es.py, learn/plastic.py (masks)
- train.py, scripts/milestones.py (m1, m2, m2b, m3 + M3b), scripts/evaluate.py, scripts/watch_queue.py, scripts/ablations.py, scripts/probe.py, scripts/paths.py
- scripts/robot_bridge.py (camera -> yolo11n -> retina -> brain -> RoboMaster, dry-run, watchdog, E-stop, latency log), scripts/dump_replay.py + replay/FORMAT.md

## Named neuron groups (keys in brain.npz)
LC10a_L/R, LC11_L/R, LC12_L/R, LC15_L/R, LC4_L/R, LPLC2_L/R, THERMO_L/R, DNa02_L/R, DNa01_L/R, DNp09, GF, PAM, PPL1, KC, MBON. To add today: DN_ALL (TODAY.md task 1), col_of_<group> retinotopy (task 2).

## Milestones
M1 PASS (15,000 neurons, 2.33M synapses). M2 PASS (LC10a_L -> DNa02_L 126 Hz, R 0 Hz at gain 0.05). M3 PASS (untrained turn-toward 72 to 78%). M3b advance: FAIL from wiring, PASS with amp_tonic 0.6. M4 met only with the fallback and a saturated circuit; needs TODAY.md task 3. M5 (robot on hardware) pending the RoboMaster.

## Do not
- render pixels inside training
- hardcode neuron IDs; load groups from brain.npz
- reward proximity or aura; reward only on events; never reward back or side contact
- re-open decisions listed in TODAY.md
- start jobs that run past the time Taka leaves for the shuttle
