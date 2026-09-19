"""
reproduce.py -- check the headline claims on this machine, in about three minutes.

    python scripts/reproduce.py                 # the untrained connectome
    python scripts/reproduce.py --checkpoint checkpoints/demo_brain.pt
    python scripts/reproduce.py --quick         # fewer arenas, about one minute

Runs the checks behind RESULTS.md sections 1 to 3 and prints what it measured next to what we
published, so a reader can see agreement or catch us being wrong. Needs data/brain.npz; if it is
missing, run `python data/pull_connectome.py` first (no account or token required).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from brain.lif import LIF, load_brain  # noqa: E402
from train import build, run_episode  # noqa: E402

OK, BAD = "  ok ", " OFF "


class Report:
    def __init__(self):
        self.rows, self.t0 = [], time.time()

    def add(self, claim, published, measured, ok):
        self.rows.append((claim, published, measured, ok))
        print(f"[{OK if ok else BAD}] {claim:<44s} published {published:<22s} measured {measured}", flush=True)

    def table(self):
        n = sum(1 for r in self.rows if r[3])
        print(f"\n{n}/{len(self.rows)} checks match the published numbers, in {time.time() - self.t0:.0f} s.")
        if n < len(self.rows):
            print("A mismatch is not automatically a bug: arena draws are seeded, but thread count and\n"
                  "floating-point order differ between machines. Large gaps are worth reporting.")
        return n == len(self.rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain", default="data/brain.npz")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--engine", default="event")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    envs = 16 if a.quick else 64
    r = Report()
    print(f"reproducing on {a.device} / {a.engine} engine, {envs} arenas\n")

    # ---- 1. the map
    d, N, groups = load_brain(a.brain)
    nnz = int(d["W_values"].shape[0])
    r.add("neurons and synapses in the circuit", "15,000 / 2,334,959", f"{N:,} / {nnz:,}",
          N == 15000 and abs(nnz - 2334959) < 50000)

    # ---- 2. signal reaches the motor, and only on the driven side
    lif = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1,
              device=a.device, engine=a.engine, g=0.05)
    g = {k: v.to(lif.device) for k, v in groups.items()}
    I = torch.zeros(1, N, device=lif.device)
    I[:, g["LC10a_L"]] = 1.5
    for _ in range(300):
        lif.step(I)
    l, rr = lif.rates(g["DNa02_L"]).item(), lif.rates(g["DNa02_R"]).item()
    r.add("drive the left eye only -> steering", "126 Hz left, 0 Hz right", f"{l:.0f} Hz left, {rr:.0f} Hz right",
          l > 80 and rr < 5)

    # ---- 3. untrained behaviour in the arena
    parts = build(envs, device=a.device, engine=a.engine, stage="A", brain_path=a.brain, seed=0,
                  episode_s=2.0, checkpoint=a.checkpoint)
    st = run_episode(parts, seconds=2.0, seed=0, stage="A", freeze_humans=True)
    b0, b1 = st["bearing_abs_start"], st["bearing_abs_end"]
    d0, d1 = st["dist_start"], st["dist_end"]
    toward = (b1 < b0 - 0.02).float().mean().item()
    advance = (d1 < d0 - 0.05).float().mean().item()
    r.add("turns toward a person, untrained", "75%", f"{100 * toward:.0f}%", toward > 0.65)
    r.add("closes the distance, untrained", "100%", f"{100 * advance:.0f}%", advance > 0.85)

    # ---- 4. the controls
    lif2, motor, senses, retina = parts["lif"], parts["motor"], parts["senses"], parts["retina"]
    grp = parts["groups"]

    def drive_left_person(brain_lif):
        """One person filling part of the left view, straight through the retina, 45 frames."""
        brain_lif.reset(); retina.reset()
        box = torch.tensor([[[60.0, 30.0, 110.0, 239.0]]], device=brain_lif.device)
        valid = torch.ones(1, 1, dtype=torch.bool, device=brain_lif.device)
        for _ in range(45):
            out = retina.from_boxes(box, valid, 1 / 30)
            cur = senses.inject(out, torch.zeros(1, 2, device=brain_lif.device))
            for _ in range(20):
                brain_lif.step(cur)
        return (brain_lif.rates(grp["LC10a_L"]).mean().item(),
                brain_lif.rates(grp["DNa02_L"]).mean().item())

    single = LIF(torch.as_tensor(d["W_indices"]), torch.as_tensor(d["W_values"]), N, 1,
                 device=a.device, engine=a.engine, g=parts["lif"].g)
    if a.checkpoint:
        single.set_w(parts["lif"].w.detach().clone())
    save_b, parts["lif"] = parts["lif"], single
    senses.N = N
    eye_real, dn_real = drive_left_person(single)

    post, pre = single.post_idx.cpu(), single.pre_idx.cpu()
    perm = torch.randperm(post.numel(), generator=torch.Generator().manual_seed(0))
    shuf = LIF(torch.stack([post[perm], pre]), single.w.detach().cpu(), N, 1,
               device=a.device, engine=a.engine, g=single.g)
    eye_sh, dn_sh = drive_left_person(shuf)
    parts["lif"] = save_b
    r.add("shuffled wiring: the eye still fires", "about 70 Hz", f"{eye_sh:.0f} Hz", eye_sh > 0.5 * eye_real)
    r.add("shuffled wiring: steering goes silent", "0 Hz", f"{dn_sh:.0f} Hz (real {dn_real:.0f})",
          dn_sh < 0.1 * max(dn_real, 1e-6))

    snap = single.lesion(torch.cat([grp["LC10a_L"], grp["LC10a_R"]]), mode="both")
    _, dn_les = drive_left_person(single)
    single.restore(snap)
    _, dn_back = drive_left_person(single)
    r.add("remove LC10a: steering goes silent", "0 Hz", f"{dn_les:.0f} Hz", dn_les < 1.0)
    r.add("restore LC10a: identical again", f"{dn_real:.1f} Hz", f"{dn_back:.1f} Hz", abs(dn_back - dn_real) < 0.05)

    sys.exit(0 if r.table() else 1)


if __name__ == "__main__":
    main()
