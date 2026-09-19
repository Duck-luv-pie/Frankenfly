"""A trained GPU hunter driving the CPU arena (`sim/hunt_arena.py`), tick by tick.

The CPU arena is what the dashboard's hunt world and the connectome trainer use, so this is the
cross-check that the batched arena is the same room: the same seed gives the same layout, the same
motor convention moves the vehicle the same way, and a fly trained on the GPU should hunt just as
well here. It also turns a saved network into a `policy(senses) -> motor` callable in the brain's
motor-dict convention (`forward`, `backward`, `turn`, `freeze`)."""
from __future__ import annotations

import torch

from ..sim.hunt_arena import HuntArena
from .arena import BatchArena, sense
from .brain import HunterNet


class CpuHunter:
    """Wraps a HunterNet so it can be stepped against a `HuntArena` on the CPU."""

    def __init__(self, cfg, net: HunterNet):
        self.net = net.to("cpu").eval()
        self.geo = BatchArena(cfg.hunt, cfg.hunt_gpu, 1, "cpu")     # sensor geometry (lobes, columns), one room
        self.h = self.net.initial_state(1, "cpu")
        self.prev = torch.zeros(1, 2)

    def reset(self) -> None:
        self.h = self.net.initial_state(1, "cpu")
        self.prev = torch.zeros(1, 2)

    def observe(self, arena: HuntArena) -> torch.Tensor:
        n = len(arena.people)
        P = self.geo.alive.shape[1]
        d = torch.zeros(1, P); b = torch.zeros(1, P); alive = torch.zeros(1, P, dtype=torch.bool); ps = torch.zeros(1, P)
        for i, p in enumerate(arena.people):
            di, bi = arena.bearing(p.x, p.z)
            d[0, i], b[0, i], alive[0, i], ps[0, i] = di, bi, True, p.speed
        speed = torch.tensor([arena.speed], dtype=torch.float32)
        heat, vis = sense(d, b, alive, ps, speed, self.geo)
        body = torch.tensor([[arena.speed / self.geo.vmax, float(self.prev[0, 0]), float(self.prev[0, 1]), arena.t / self.geo.episode_s]])
        return torch.cat([heat, vis.reshape(1, -1), body], dim=1)

    @torch.no_grad()
    def __call__(self, arena: HuntArena) -> dict:
        a, _, _, self.h = self.net.act(self.observe(arena), self.h, deterministic=True)
        a = a.clamp(-1, 1)
        self.prev = a
        fwd, turn = float(a[0, 0]), float(a[0, 1])
        return {"forward": max(0.0, fwd), "backward": max(0.0, -fwd), "turn": turn, "freeze": 0.0}


def run_cpu(cfg, net: HunterNet, seeds: list[int], verbose: bool = False) -> list[dict]:
    """The network hunts through the CPU arena once per seed. Returns `HuntArena.result()` records."""
    hunter = CpuHunter(cfg, net)
    h = dict(cfg.hunt)
    h["_valence_steering"] = 0.0                       # no mushroom-body steering: the network is the whole brain
    arena = HuntArena(h, seeds[0])
    tick = float(h.get("tick_s", 0.05))
    out = []
    for seed in seeds:
        arena.reset(seed)
        hunter.reset()
        while not arena.done:
            arena.sense()
            arena.step(hunter(arena), 0.0, tick)
        res = arena.result()
        out.append(res)
        if verbose:
            print(f"  seed {seed:10d}: {('TOUCH at %5.1f s %s' % (res['t_touch'], 'head-on' if res['frontal'] else 'glancing')) if res['touched'] else 'timeout'} "
                  f"| facing {res['facing']:.0%} | closest {res['min_dist']:.2f} m", flush=True)
    return out
