# PLAN — who does what, step by step

Three actors: **Taka** (you), **Claude Code** (writes/runs code in this repo), **Claude chat** (design, unblocking, reading results, redesign).

## GPU decision
Use **Vast.ai now** (A100 40GB, 99.5%+ uptime, ~$0.6-1.2/hr; Ducks's runs cost about $1 each). File the **AWS** GPU quota request in parallel (Service Quotas -> EC2 -> "Running On-Demand G and VT instances" and "P instances" -> request 8+ vCPUs); if approved during the weekend, switch to a g6e.xlarge (L40S) or p4d and it is free on your credits. Keep checkpoints as plain files in `checkpoints/`; moving boxes is one `scp`. See scripts/gpu_setup.md.

## Tonight (Thu) — target M3
| Step | Who | Done when |
| --- | --- | --- |
| Rent A100 on Vast, ssh in, clone repo, `pip install -r requirements.txt` | Taka | M0 |
| Get neuPrint token (neuprint.janelia.org -> login -> Account), put in .env | Taka | token works |
| Implement data/pull_connectome.py from its docstring; run it | Claude Code | M1 |
| Sanity-check brain.npz group sizes; paste them to Claude chat | Taka | groups look right |
| Run `python -m brain.lif`; sweep global gain until M2 passes | Claude Code | M2 |
| Implement env/arena.py from its docstring; run untrained brain | Claude Code | M3 |
| Send Ducks the turret-lock / left-right-heat message | Taka | he replies |
| If M3 fails by 2 am: stop, sleep, bring findings to Waterloo | Taka | |

## Friday — training + hardware
| Step | Who |
| --- | --- |
| Claim RoboMaster S1 the moment hardware requests open; grab a Pi 5 as backup compute | Ducks/Taka |
| learn/three_factor.py; Stage A training running in background | Claude Code |
| Detector (ultralytics yolo11n, person class) on Ducks's sim frames; if it misses, fine-tune on 200 GT-labelled frames | Claude Code |
| scripts/robot_bridge.py: camera -> retina.from_boxes -> brain -> motor -> `robomaster` SDK chassis.drive_speed | Claude Code + Ducks |
| Match sim hfov to the real S1 camera once measured | Taka |
| Lock tracks list by Sat 2 PM: Finalists, MLH Tiger Data, MLH MongoDB, MLH ElevenLabs (+Baseten only if serving a model) | Taka |

## Saturday
Stage B/C training, brain viz replay into Ducks's Three.js, lobotomize/restore button, ES fallback if three-factor is flat, demo rehearsal x5. Foam swatter.

## Sunday morning
Freeze code, record backup video of a clean run, Devpost writeup citing every library.

## What to paste to Claude chat when stuck
1. group sizes from brain.npz  2. firing-rate table per group under a fixed stimulus  3. the last 50 lines of train.py's CSV  4. the exact traceback
