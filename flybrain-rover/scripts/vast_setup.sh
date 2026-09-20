#!/bin/bash
# Run ON the Vast box after sync_to_box.sh. Bare CUDA image assumed (python3, no torch).
set -e
cd ~/flybrain-rover
apt-get install -y -qq python3-venv python3-pip tmux > /tmp/apt.log 2>&1 || true
[ -d .venv ] || python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q torch numpy pandas tqdm neuprint-python python-dotenv pytest scipy requests
.venv/bin/python -c "import torch;print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
.venv/bin/python -m pytest tests/ -q 2>&1 | tail -1
# CUDA engine check: sparse CSR vs event must agree, and M2 must pass on the GPU
.venv/bin/python scripts/milestones.py m2 --device cuda --engine sparse --gains 0.05 2>&1 | tail -2
.venv/bin/python scripts/milestones.py m3 --device cuda --engine sparse --envs 512 2>&1 | grep -E "M3|M3b|DN rates"
echo "box ready. start a queue with:  nohup bash scripts/vast_queue.sh > logs/vast_queue.log 2>&1 &"
