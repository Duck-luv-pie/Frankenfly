# GPU setup

## Vast.ai (tonight)
1. vast.ai -> sign up -> load $10 (minimum).
2. Search: GPU = A100 (40 GB is enough), reliability >= 99.5%, disk >= 50 GB, image = pytorch/pytorch (CUDA 12.x). Rent.
3. Instances -> SSH button -> copy the ssh command.
4. `ssh ...` then `git clone <repo> && cd flybrain-rover && pip install -r requirements.txt && cp .env.example .env` and paste the neuPrint token.
5. `python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`
6. Run long jobs under `tmux` so a dropped SSH does not kill training. Sync checkpoints: `scp -r <box>:flybrain-rover/checkpoints ./`
Cost: about $1/hr. Destroy the instance when done; you are billed while it exists.

## AWS (if quota approved)
1. Service Quotas -> EC2 -> request "Running On-Demand G and VT instances" (>= 8 vCPU) and "P instances". Takes 1-2 days.
2. Launch g6e.xlarge (L40S, 48 GB) with the AWS Deep Learning AMI (PyTorch). p4d.24xlarge for 8xA100 if you want massive sweeps.
3. Same clone/install steps. Credits cover it. Stop the instance when idle.

## Rule
The control loop for the real robot NEVER runs on either box. Cloud is for training and sweeps only. The brain that drives the robot runs on a laptop next to it.
