"""Training the real fly's brain in the batch arena, and running it as spiking neurons afterwards.

`BrainTrainer` is the PPO trainer of `ppo.py` with the fly replaced by `RateBrain` (the hunter
subcircuit of the connectome as a rate model): the policy mean is the decoded forward / turn drive of
its descending neurons, the exploration noise a Gaussian on those channels, and the critic a small
network over the observation (training scaffolding, not part of the fly). Gradients flow through the
brain one arena tick at a time (truncated backpropagation: the neuron state is detached between
ticks), which keeps the memory of an 8k-neuron batch of 512 rooms within reach; the humans learn as
in `ppo.py`. What is learned is written back with `RateBrain.save`.

`SpikingHunter` runs the same subcircuit as spiking LIF neurons (`sim/lif.py`) with the learned gains
applied, driven by the batch arena's senses through the same sensory mapping and read out by the real
`Decoder`, so the trained fly can be evaluated and watched as spikes."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .arena import BatchArena, pick_device
from .brain import EvaderNet, build_evader
from .connectome import HunterCircuit, RateBrain, apply_to_runner, load_brain
from .ppo import Trainer, gae, summarize


def lif_std(cfg, hc: HunterCircuit, seconds: float = 6.0, verbose: bool = True) -> tuple[dict, dict]:
    """The spiking subcircuit's resting baseline (mean and std per readout group and side, 300 ms windows),
    the std of which the rate model's decoder uses. Returns (std dict {group: {side: hz}}, full baseline)."""
    from ..sim.runner import BrainRunner
    r = BrainRunner(hc.circuit, cfg, seed=0)
    base = r.calibrate(seconds, [300.0, 500.0], warmup_s=3.0, verbose=verbose)
    w = str(int(float(cfg.decode.motor.channels.forward.get("window_ms", cfg.decode.window_ms))))
    std = {g: {s: base[w][g][s]["std"] for s in ("left", "right", "all")} for g in hc.readouts if g in base[w]}
    return std, base


def build_brain(cfg, hops=(2, 1), gains: dict | None = None, device="cpu", verbose: bool = True, calibrate: bool = True) -> tuple[HunterCircuit, RateBrain]:
    from ..data.prune import load_or_build
    c = load_or_build(cfg, verbose=verbose)
    hc = HunterCircuit(c, cfg, hops=hops)
    if verbose:
        sm = hc.summary()
        print(f"[brain] hunter subcircuit: {sm['neurons']:,} neurons, {sm['synapses']:,} synapses (hops {sm['hops']}); classes {sm['classes']}", flush=True)
    brain = RateBrain(hc, cfg, gains, dict(cfg.hunt_gpu.get("brain", {}))).to(device)
    arena = BatchArena(cfg.hunt, cfg.hunt_gpu, 1, device)
    brain.bind_arena(arena.spec, arena.bin_c, arena.fov, arena.vmax)
    if calibrate:
        std, _ = lif_std(cfg, hc, verbose=verbose)
        base = brain.calibrate(std=std)
        if verbose:
            print("[brain] rate-model resting baseline (mean, std Hz): " + ", ".join(f"{g}/{s}={m}±{sd}" for (g, s), (m, sd) in base.items() if s != "all" and g in ("DNa01", "DNa02", "DN_all")), flush=True)
    return hc, brain


class Critic(nn.Module):
    def __init__(self, obs_size: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)[..., 0]


class BrainTrainer(Trainer):
    """PPO with the connectome's rate model as the fly. Reuses the runners' learning from `Trainer`."""

    def __init__(self, cfg, n_envs: int | None = None, device: str = "auto", seed: int = 0, out_dir: Path | None = None,
                 hops=(2, 1), gains: dict | None = None, resume: str | None = None):
        g = dict(cfg.hunt_gpu)
        bc = dict(g.get("brain", {}))
        self.cfg, self.g = cfg, g
        self.p = {**dict(g.get("ppo", {})), **dict(bc.get("ppo", {}))}
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.arena = BatchArena(cfg.hunt, g, n_envs or int(bc.get("envs", 512)), self.device, seed=seed)
        self.hc, self.brain = build_brain(cfg, hops=hops, gains=gains, device=self.device)
        if resume:
            meta = self.brain.load_params(resume)
            print(f"[brain] resumed the brain from {resume} ({meta})", flush=True)
        self.net = self.brain                                     # what Trainer's bookkeeping calls the policy
        self.critic = Critic(self.arena.spec.size).to(self.device)
        lr = float(self.p.get("lr", 3e-4))
        brain_params = [self.brain.u_gain, self.brain.u_th]
        head_params = [p for n, p in self.brain.named_parameters() if n.startswith("p_") or n == "log_std"]
        self.opt = torch.optim.Adam([{"params": brain_params, "lr": float(bc.get("lr_brain", 1e-2))},
                                     {"params": head_params, "lr": float(bc.get("lr_head", 3e-3))}], eps=1e-5)
        self.copt = torch.optim.Adam(self.critic.parameters(), lr=lr, eps=1e-5)
        self.reg = float(bc.get("reg", 1e-2))
        self.bptt = max(1, int(bc.get("bptt_ticks", 1)))          # ticks of backpropagation through the neurons before the state is detached
        hcfg = dict(g.get("humans", {}))
        self.adversarial = self.arena.human_mode == "learn" and not bool(hcfg.get("frozen", False))
        self.hp = {**dict(g.get("ppo", {})), **dict(hcfg.get("ppo", {}))}
        self.hnet = build_evader(self.arena.hspec, hcfg.get("net", {}), float(self.hp.get("log_std_init", -0.5))).to(self.device) if self.arena.human_mode == "learn" else None
        self.hopt = torch.optim.Adam(self.hnet.parameters(), lr=float(self.hp.get("lr", 3e-4)), eps=1e-5) if self.adversarial else None
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.st = self.brain.initial_state(self.arena.B)
        self.obs = self.arena.observe()
        self.update, self.total_steps, self.best = 0, 0, {"touch_rate": -1.0, "score": -1.0}
        self.recent: list[dict] = []

    # ---- rollout: the brain's state is carried, and its start-of-rollout copy kept for the replay ----
    @torch.no_grad()
    def rollout(self, T: int) -> dict:
        B, D, P, dev = self.arena.B, self.arena.spec.size, self.arena.px.shape[1], self.device
        obs = torch.zeros(T, B, D, device=dev)
        act = torch.zeros(T, B, 2, device=dev)
        logp, val, rew = torch.zeros(T, B, device=dev), torch.zeros(T, B, device=dev), torch.zeros(T, B, device=dev)
        done = torch.zeros(T, B, device=dev)
        st0 = {k: v.clone() for k, v in self.st.items()}
        hum = None
        if self.adversarial:
            hum = {"obs": torch.zeros(T, B, P, self.arena.hspec.size, device=dev), "act": torch.zeros(T, B, P, 2, device=dev),
                   "logp": torch.zeros(T, B, P, device=dev), "val": torch.zeros(T, B, P, device=dev), "rew": torch.zeros(T, B, P, device=dev),
                   "alive": torch.zeros(T, B, P, device=dev, dtype=torch.bool)}
        episodes = []
        self.brain.eval()
        W = self.brain.weights()                                   # the wiring does not change during a rollout
        for t in range(T):
            obs[t] = self.obs
            a, lp, self.st, _ = self.brain.act(self.obs, self.st, W=W)
            v = self.critic(self.obs)
            hdrive = None
            if self.hnet is not None:
                hobs = self.arena.observe_humans()
                hdrive, hlp, hv = self.hnet.act(hobs)
                if hum is not None:
                    hum["obs"][t], hum["act"][t], hum["logp"][t], hum["val"][t] = hobs, hdrive, hlp, hv
            r, d, info = self.arena.step(a, hdrive)
            act[t], logp[t], val[t], rew[t], done[t] = a, lp, v, r, d.float()
            if hum is not None:
                hum["rew"][t], hum["alive"][t] = info["human_reward"], info["human_active"]
            episodes.extend(info.get("episodes", []))
            if bool(d.any()):
                self.arena.reset(d.nonzero()[:, 0])
            self.obs = self.arena.observe()
        last_v = self.critic(self.obs)
        gamma, lam = float(self.p.get("gamma", 0.995)), float(self.p.get("lam", 0.95))
        adv = gae(rew, val, done, last_v, gamma, lam)
        self.total_steps += T * B
        out = {"obs": obs, "act": act, "logp": logp, "val": val, "rew": rew, "adv": adv, "ret": adv + val, "done": done, "st0": st0, "episodes": episodes}
        if hum is not None:
            _, hlast = self.hnet(self.arena.observe_humans())
            hum["adv"] = gae(hum["rew"], hum["val"], done[:, :, None].expand(-1, -1, P), hlast, float(self.hp.get("gamma", gamma)), float(self.hp.get("lam", lam)))
            hum["ret"] = hum["adv"] + hum["val"]
            out["humans"] = hum
        return out

    # ---- the update: the critic by regression, the brain by PPO one tick at a time ----
    def learn(self, ro: dict) -> dict:
        p = self.p
        T, B = ro["obs"].shape[:2]
        epochs, nmb = int(p.get("epochs", 2)), int(p.get("minibatches", 4))
        clip, ent_coef, max_gn = float(p.get("clip", 0.2)), float(p.get("entropy", 0.003)), float(p.get("max_grad_norm", 0.5))
        adv = ro["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        # critic
        obs_flat, ret_flat, val_flat = ro["obs"].reshape(-1, ro["obs"].shape[-1]), ro["ret"].reshape(-1), ro["val"].reshape(-1)
        vloss_sum, nv = 0.0, 0
        for _ in range(int(p.get("critic_epochs", 4))):
            for mb in torch.randperm(obs_flat.shape[0], device=self.device).chunk(8):
                v = self.critic(obs_flat[mb])
                vc = val_flat[mb] + (v - val_flat[mb]).clamp(-0.2, 0.2)
                vl = 0.5 * torch.maximum((v - ret_flat[mb]) ** 2, (vc - ret_flat[mb]) ** 2).mean()
                self.copt.zero_grad(set_to_none=True); vl.backward(); self.copt.step()
                vloss_sum += float(vl); nv += 1
        # the brain
        self.brain.train()
        stats = {"policy_loss": 0.0, "kl": 0.0, "clipfrac": 0.0}
        n = 0
        dense = self.device.type == "mps"                                   # see RateBrain.weights
        for _ in range(epochs):
            for mb in torch.randperm(B, device=self.device).chunk(nmb):
                st = RateBrain.index_state(ro["st0"], mb)
                self.opt.zero_grad(set_to_none=True)
                # the wiring is built once per minibatch with its graph to the gains; every tick uses a detached leaf
                # (the dense matrix on MPS, the CSR values elsewhere) and the leaf's gradient goes back to the gains once
                W_full = self.brain.weights(dense=True) if dense else self.brain.csr_values()
                W = W_full.detach().requires_grad_(True)
                pending = None
                for t in range(T):
                    mean, st, _ = self.brain.step(ro["obs"][t, mb], st, W=W if dense else self.brain.csr(W))
                    logp = self.brain.log_prob(mean, ro["act"][t, mb])
                    ratio = (logp - ro["logp"][t, mb]).exp()
                    a = adv[t, mb]
                    pl = -torch.minimum(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                    loss = (pl - ent_coef * self.brain.entropy() + self.reg * self.brain.regularizer()) / T
                    pending = loss if pending is None else pending + loss
                    if (t + 1) % self.bptt == 0 or t == T - 1:             # truncated BPTT: back through the last `bptt` ticks
                        pending.backward()
                        pending = None
                        st = RateBrain.detach_state(st)
                    with torch.no_grad():
                        stats["policy_loss"] += float(pl) / T; stats["kl"] += float((ro["logp"][t, mb] - logp).mean()) / T
                        stats["clipfrac"] += float(((ratio - 1).abs() > clip).float().mean()) / T
                if W.grad is not None:
                    W_full.backward(W.grad)                                  # the ticks' weight gradients, back to the synapse gains once
                nn.utils.clip_grad_norm_(self.brain.parameters(), max_gn)
                self.opt.step()
                n += 1
        del W_full, W, st, pending
        if self.device.type == "mps":                                  # keep the footprint flat: the dense weight matrices
            import gc                                                  # are big, and a swapping Mac punishes churn
            gc.collect(); torch.mps.empty_cache()
        out = {k: round(v / max(n, 1), 4) for k, v in stats.items()}
        out.update({"value_loss": round(vloss_sum / max(nv, 1), 4), "entropy": round(float(self.brain.entropy()), 4),
                    "gain_spread": round(float(self.brain.gain().log().std()), 4), "th_spread": round(float(self.brain.th_offset().std()), 4)})
        if "humans" in ro:
            out.update({"h_" + k: v for k, v in self.learn_humans(ro["humans"]).items()})
        return out

    def best_key(self, sm: dict) -> tuple:
        """The brain is kept for the contact it keeps first, then its tracking, then its touch rate."""
        return (sm.get("contact") or 0.0, sm.get("track") or 0.0, sm.get("touch_rate") or 0.0)

    def save(self, fly_name: str, humans_name: str, meta: dict) -> None:
        self.brain.save(self.out_dir / fly_name.replace("hunter_gpu", "hunter_brain").replace(".pt", ".npz"), meta=meta)
        if self.hnet is not None:
            self.hnet.save(self.out_dir / humans_name, meta=meta)


# ---- evaluation ---------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_brain(cfg, brain: RateBrain, episodes: int, device="auto", seed: int = 0, humans: EvaderNet | None = None, deterministic: bool = True) -> list[dict]:
    """The rate-model fly through `episodes` seeded rooms at once (like ppo.evaluate)."""
    dev = pick_device(device)
    seeds = [(seed * 7919 + k * 104729) & 0xFFFFFFFF for k in range(episodes)]
    g = dict(cfg.hunt_gpu)
    if dict(g.get("humans", {})).get("mode", "learn") == "learn" and humans is None:
        g = {**g, "humans": {**dict(g.get("humans", {})), "mode": "flee"}}
    torch.manual_seed(seed)                                    # the runners' waypoints and pauses: reproducible by the seed
    arena = BatchArena(cfg.hunt, g, len(seeds), dev, seed=seed)
    arena.reset(torch.arange(len(seeds), device=dev), seeds)
    brain = brain.to(dev).eval()
    brain.bind_arena(arena.spec, arena.bin_c, arena.fov, arena.vmax)
    st = brain.initial_state(arena.B)
    if humans is not None:
        humans = humans.to(dev).eval()
    results: dict[int, dict] = {}
    active = torch.ones(arena.B, device=dev, dtype=torch.bool)
    W = brain.weights()
    for _ in range(arena.max_steps + 1):
        a, _, st, _ = brain.act(arena.observe(noise=False), st, deterministic=deterministic, W=W)
        hdrive = humans.act(arena.observe_humans(), deterministic=deterministic)[0] if (humans is not None and arena.human_mode == "learn") else None
        _, done, info = arena.step(a, hdrive)
        for e in info.get("episodes", []):
            results.setdefault(e["seed"], e)
        active &= ~done
        if not bool(active.any()):
            break
        arena.steps[~active] = 0
    return [results[s] for s in seeds if s in results]


class SpikingHunter:
    """The hunter subcircuit as spiking LIF neurons with the learned brain applied, driven by the batch arena's
    senses and read out by the real decoder. One room at a time (a LIF network is one brain)."""

    def __init__(self, cfg, hc: HunterCircuit, brain_file: str | Path | None, verbose: bool = True):
        from ..config import apply_dotted
        from ..sim.runner import BrainRunner
        from ..body.decode import Decoder
        import copy
        self.cfg = copy.deepcopy(cfg)
        self.brain_meta = {}
        gains = {}
        if brain_file:
            b = load_brain(brain_file)
            gains, self.brain_meta = b["gains"], b["meta"]
            apply_dotted(self.cfg, gains)
        self.cfg.learning["enabled"] = False
        self.runner = BrainRunner(hc.circuit, self.cfg, seed=0)
        if brain_file:
            apply_to_runner(self.runner, b, verbose=verbose)
        base = self.runner.calibrate(float(self.cfg.decode.calibrate_s), [300.0, 500.0], warmup_s=float(self.cfg.decode.get("warmup_s", 2.0)), verbose=verbose)
        self.decoder = Decoder(self.cfg, base)
        self.rate = RateBrain(hc, self.cfg, gains)                     # only for its sensory mapping (drive_hz)
        self.tick_s = float(self.cfg.hunt.get("tick_s", 0.05))
        self.chunks = max(1, int(round(self.tick_s * 1000 / self.runner.chunk_ms)))
        self.spikes = np.zeros(hc.n, dtype=np.int32)
        self.hc = hc
        self.width = None

    def bind_arena(self, arena: BatchArena) -> None:
        self.rate.bind_arena(arena.spec, arena.bin_c.cpu(), arena.fov, arena.vmax)

    def sever_senses(self) -> int:
        """The lobotomy: cut every synapse leaving the sensory neurons the arena drives (the LC columns, the hot cells,
        the proximity cells). They still fire when stimulated, but nothing reaches the rest of the brain, so the
        descending neurons are left with their spontaneous activity and the vehicle twitches on baseline noise.
        Returns the number of synapses cut."""
        net, c = self.runner.net, self.hc.circuit
        cut = 0
        for feat, side, grp in self.rate.sense_rows:
            for p in c.groups.get(grp, {}).get(side, []):
                a, b = int(net.indptr[p]), int(net.indptr[p + 1])
                cut += int((net.data[a:b] != 0).sum())
                net.data[a:b] = 0.0
        return cut

    def reset(self) -> None:
        self.decoder.motor_state.clear()
        self.decoder.last_t = None
        self.width = None

    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """obs [1, D] -> drive [1, 2] (forward, turn), stepping the spiking brain one arena tick."""
        hz, self.width = self.rate.drive_hz(obs.cpu(), self.width)
        hz = hz[:, 0]
        r = self.runner
        r.clear_drive()
        for (feat, side, grp), v in zip(self.rate.sense_rows, hz.tolist()):
            if v > 0:
                r.drive(grp, v, side)
        self.spikes[:] = 0
        for _ in range(self.chunks):
            self.spikes += r.step_chunk()
        d = self.decoder.decode({w: r.rates(w) for w in self.decoder.windows}, now=r.brain_ms / 1000)
        m = d.motor
        turn = max(-1.0, min(1.0, float(m.get("turn", 0.0)) * float(self.cfg.hunt.get("turn_gain", 1.0))))
        forward = float(m.get("forward", 0.0)) - float(m.get("backward", 0.0))
        if self.cfg.hunt.get("freeze_brakes", False):                 # the freeze channel brakes the vehicle, as in the rate model and the CPU arena
            forward *= 1.0 - float(m.get("freeze", 0.0))
        return torch.tensor([[forward, turn]]), d


def evaluate_spiking(cfg, hunter: SpikingHunter, episodes: int, seed: int = 0, humans: EvaderNet | None = None, verbose: bool = True) -> list[dict]:
    """The spiking fly through `episodes` seeded rooms, one after another (CPU)."""
    g = dict(cfg.hunt_gpu)
    if dict(g.get("humans", {})).get("mode", "learn") == "learn" and humans is None:
        g = {**g, "humans": {**dict(g.get("humans", {})), "mode": "flee"}}
    arena = BatchArena(cfg.hunt, g, 1, "cpu", seed=seed)
    hunter.bind_arena(arena)
    if humans is not None:
        humans = humans.to("cpu").eval()                    # the spiking fly and its room live on the CPU
    out = []
    for k in range(episodes):
        s = (seed * 7919 + k * 104729) & 0xFFFFFFFF
        torch.manual_seed(s)                                   # the runners' waypoints and pauses too: an episode is reproducible by its seed
        arena.reset(torch.arange(1), [s])
        hunter.reset()
        t0 = time.time()
        while True:
            a, _ = hunter(arena.observe(noise=False))
            hdrive = humans.act(arena.observe_humans(), deterministic=True)[0] if (humans is not None and arena.human_mode == "learn") else None
            _, done, info = arena.step(a, hdrive)
            if bool(done[0]):
                break
        e = info["episodes"][0]
        out.append(e)
        if verbose:
            print(f"  seed {s:10d}: {('first touch at %5.1f s, tracked %3.0f%%, %d contacts' % (e['t_touch'], 100 * e['track'], e['touches'])) if e['touched'] else 'no touch'} "
                  f"| nobody in range {e['blind']:.0%} | {time.time() - t0:.0f} s wall", flush=True)
    return out
