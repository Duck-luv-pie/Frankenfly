# BEFORE_LEAVING.md — 5 minutes before you close the lid

The box keeps training until ~20:20 Toronto without you. The Mac only needs to carry the demo files.

1. **Pull the box's checkpoints and logs** (30 s):
   ```
   cd ~/Downloads/filess/flybrain-rover
   rsync -az -e "ssh -p 30714" root@150.136.39.147:~/flybrain-rover/checkpoints/ checkpoints_box/
   rsync -az -e "ssh -p 30714" root@150.136.39.147:~/flybrain-rover/logs/ logs_box/
   ```
   Do this again at the venue tonight (~20:30) to get the finished afternoon queue (tfA_both_explore, esA_lcdn_512, tfA_both_long, tfB_explore).
2. **Kill anything still running on the Mac** (evaluations are fine to leave; nothing should be training):
   ```
   pkill -f "scripts/watch_queue.py"; pkill -f "train.py"; pgrep -fl "evaluate.py|train.py|watch_queue" || echo clean
   ```
3. **Commit** (local only, never push):
   ```
   git add -A && git commit -m "leaving for Waterloo"
   ```
4. **Check the handoff folder** has: `demo_brain.pt`, `DEMO.md`, `DEVPOST.md`, `replay/` (episode JSONs, FORMAT.md, viewer.html). If `demo_brain.pt` is missing, `scripts/lock_demo.py --checkpoint checkpoints_box/tfA_both_512_latest.pt --lock` makes it (10 min on the Mac CPU) or use the untrained brain (`--brain data/brain.npz --lock --force`), which already scores 0.969 contact.
5. **Box stays alive**: do NOT destroy the Vast instance until you have pulled tonight's checkpoints. `tmux attach -t fly` on the box shows the queue; `tail -f ~/flybrain-rover/logs/queue.log`.
6. At the venue: `pip install` is done in `.venv` (opencv, ultralytics) and `.venv38` (robomaster); `yolo11n.pt` is cached in the repo root. First command after unpacking the RoboMaster: `.venv38/bin/python scripts/robot_daemon.py --conn ap`, then `DEMO.md` step 3.
