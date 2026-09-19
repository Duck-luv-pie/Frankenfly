"""Recurrent PPO for the batched hunt arena: rollouts, advantages and updates all on one device.

One update = a rollout of `steps` ticks across all `envs` environments (the fly's hidden state carried
over), generalized advantage estimation, then `epochs` passes of clipped-surrogate updates on
minibatches of whole environment sequences (the GRU is replayed from the hidden state stored at the
start of the rollout, so gradients flow through time). With `humans.mode: learn` the runners are the
second player: their shared feed-forward policy (`EvaderNet`) collects its own transitions in the
same rollout (every living person is a sample) and is updated by the same PPO step with its own
optimizer, so fly and humans learn against each other. Episodes that finish during a rollout are
logged with the CPU arena's hunting score; the best fly by touch rate then score is kept as
`hunter_gpu.pt` (and its opponents as `evaders_gpu.pt`)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..sim.hunt import score
from .arena import BatchArena, pick_device, scripted_drive
from .brain import EvaderNet, HunterNet, build, build_evader


def summarize(episodes: list[dict], episode_s: float, side_reward: float) -> dict:
    """Per-batch statistics. The first-touch numbers (touch rate, time, angle, switches, score) come from the
    episodes that began as hunts; the tracking numbers from every episode that was locked on, including the
    ones that began at contact (`track.start_locked`)."""
    if not episodes:
        return {"n": 0}
    hunts = [e for e in episodes if not e.get("started_locked")]
    hit = [e for e in hunts if e["touched"]]
    locked = [e for e in episodes if e["touched"]]
    return {"n": len(episodes), "n_hunts": len(hunts),
            "touch_rate": round(len(hit) / len(hunts), 3) if hunts else None,
            "t_touch": round(float(np.mean([e["t_touch"] for e in hit])), 2) if hit else None,
            "frontal": round(float(np.mean([e["frontal"] for e in hit])), 3) if hit else None,
            "switches": round(float(np.mean([e["switches"] for e in hunts])), 2) if hunts else None,
            "touches": round(float(np.mean([e["touches"] for e in locked])), 2) if locked else None,
            "track": round(float(np.mean([e["track"] for e in locked])), 3) if locked else None,
            "contact": round(float(np.mean([e.get("contact") or 0.0 for e in locked])), 3) if locked else None,
            "contact_s": round(float(np.mean([e.get("contact_s") or 0.0 for e in locked])), 2) if locked else None,
            "spins": round(float(np.mean([e.get("spins", 0) for e in locked])), 2) if locked else None,
            "track_s": round(float(np.mean([e.get("track_s") or 0.0 for e in locked])), 2) if locked else None,
            "mean_dist": round(float(np.mean([e.get("mean_dist", 0.0) for e in episodes])), 2),
            "target_rest": round(float(np.mean([e.get("target_rest", 0.0) for e in episodes])), 3),
            "blind": round(float(np.mean([e.get("blind", 0.0) for e in episodes])), 3),
            "idle": round(float(np.mean([e.get("idle", 0.0) for e in episodes])), 3),
            "score": round(score(hunts, episode_s, side_reward), 3) if hunts else None}


def gae(rew: torch.Tensor, val: torch.Tensor, done: torch.Tensor, last_v: torch.Tensor, gamma: float, lam: float) -> torch.Tensor:
    """Generalized advantage estimation over the leading time axis; `done[t]` ends the episode at step t."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    g = torch.zeros_like(last_v)
    for t in reversed(range(T)):
        nv = last_v if t == T - 1 else val[t + 1]
        cont = 1.0 - done[t]
        delta = rew[t] + gamma * nv * cont - val[t]
        g = delta + gamma * lam * cont * g
        adv[t] = g
    return adv


def ppo_loss(net, mean, value, act, logp_old, adv, val_old, ret, p: dict):
    """The clipped surrogate + clipped value loss - entropy bonus, and its statistics."""
    clip, vclip = float(p.get("clip", 0.2)), float(p.get("value_clip", 0.2))
    ent_coef, v_coef = float(p.get("entropy", 0.003)), float(p.get("value_coef", 0.5))
    logp = net.log_prob(mean, act)
    ratio = (logp - logp_old).exp()
    pl = -torch.minimum(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
    v_clipped = val_old + (value - val_old).clamp(-vclip, vclip)
    vl = 0.5 * torch.maximum((value - ret) ** 2, (v_clipped - ret) ** 2).mean()
    ent = net.entropy()
    loss = pl + v_coef * vl - ent_coef * ent
    with torch.no_grad():
        st = {"policy_loss": float(pl), "value_loss": float(vl), "entropy": float(ent), "kl": float((logp_old - logp).mean()),
              "clipfrac": float(((ratio - 1).abs() > clip).float().mean())}
    return loss, st


class Trainer:
    def __init__(self, cfg, n_envs: int | None = None, device: str = "auto", seed: int = 0, out_dir: Path | None = None):
        g = dict(cfg.hunt_gpu)
        self.cfg, self.g = cfg, g
        self.p = dict(g.get("ppo", {}))
        self.device = pick_device(device)
        torch.manual_seed(seed)
        self.arena = BatchArena(cfg.hunt, g, n_envs or int(g.get("envs", 2048)), self.device, seed=seed)
        self.net = build(self.arena.spec, g.get("net", {}), float(self.p.get("log_std_init", -0.5))).to(self.device)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=float(self.p.get("lr", 3e-4)), eps=1e-5)
        hcfg = dict(g.get("humans", {}))
        self.adversarial = self.arena.human_mode == "learn" and not bool(hcfg.get("frozen", False))   # frozen: trained runners that do not learn further
        self.hp = {**self.p, **dict(hcfg.get("ppo", {}))}
        self.hnet = build_evader(self.arena.hspec, hcfg.get("net", {}), float(self.hp.get("log_std_init", -0.5))).to(self.device) if self.arena.human_mode == "learn" else None
        self.hopt = torch.optim.Adam(self.hnet.parameters(), lr=float(self.hp.get("lr", 3e-4)), eps=1e-5) if self.adversarial else None
        self.out_dir = Path(out_dir) if out_dir else None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        self.h = self.net.initial_state(self.arena.B, self.device)
        self.obs = self.arena.observe()
        self.update, self.total_steps, self.best = 0, 0, {"touch_rate": -1.0, "score": -1.0, "track": -1.0}
        self.recent: list[dict] = []

    # ---- one rollout ---------------------------------------------------------------------------
    @torch.no_grad()
    def rollout(self, T: int) -> dict:
        B, D, P, dev = self.arena.B, self.arena.spec.size, self.arena.px.shape[1], self.device
        obs = torch.zeros(T, B, D, device=dev)
        act = torch.zeros(T, B, 2, device=dev)
        logp, val, rew = torch.zeros(T, B, device=dev), torch.zeros(T, B, device=dev), torch.zeros(T, B, device=dev)
        done = torch.zeros(T, B, device=dev)               # done[t] = 1 if the step at t ended the episode
        h0 = self.h.clone()
        hum = None
        if self.adversarial:
            hum = {"obs": torch.zeros(T, B, P, self.arena.hspec.size, device=dev), "act": torch.zeros(T, B, P, 2, device=dev),
                   "logp": torch.zeros(T, B, P, device=dev), "val": torch.zeros(T, B, P, device=dev), "rew": torch.zeros(T, B, P, device=dev),
                   "alive": torch.zeros(T, B, P, device=dev, dtype=torch.bool)}
        episodes = []
        self.net.eval()
        for t in range(T):
            obs[t] = self.obs
            a, lp, v, self.h = self.net.act(self.obs, self.h)
            hdrive = None
            if self.hnet is not None:
                hobs = self.arena.observe_humans()
                hdrive, hlp, hv = self.hnet.act(hobs)
                if hum is not None:
                    hum["obs"][t], hum["act"][t], hum["logp"][t], hum["val"][t] = hobs, hdrive, hlp, hv
            r, d, info = self.arena.step(a, hdrive)
            act[t], logp[t], val[t], rew[t], done[t] = a, lp, v, r, d.float()
            if hum is not None:
                hum["rew"][t], hum["alive"][t] = info["human_reward"], info["human_active"]      # a runner learns from the ticks it minded the fly
            episodes.extend(info.get("episodes", []))
            if bool(d.any()):
                self.arena.reset(d.nonzero()[:, 0])          # the hidden state runs on: see HunterNet.sequence
            self.obs = self.arena.observe()
        _, last_v, _ = self.net(self.obs, self.h)
        gamma, lam = float(self.p.get("gamma", 0.995)), float(self.p.get("lam", 0.95))
        adv = gae(rew, val, done, last_v, gamma, lam)
        self.total_steps += T * B
        out = {"obs": obs, "act": act, "logp": logp, "val": val, "rew": rew, "adv": adv, "ret": adv + val, "done": done, "h0": h0, "episodes": episodes}
        if hum is not None:
            _, hlast = self.hnet(self.arena.observe_humans())
            hum["adv"] = gae(hum["rew"], hum["val"], done[:, :, None].expand(-1, -1, P), hlast, float(self.hp.get("gamma", gamma)), float(self.hp.get("lam", lam)))
            hum["ret"] = hum["adv"] + hum["val"]
            out["humans"] = hum
        return out

    # ---- the update ----------------------------------------------------------------------------
    def learn(self, ro: dict) -> dict:
        p = self.p
        T, B = ro["obs"].shape[:2]
        epochs, nmb, max_gn = int(p.get("epochs", 4)), int(p.get("minibatches", 4)), float(p.get("max_grad_norm", 0.5))
        adv = ro["adv"]
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        self.net.train()
        stats: dict[str, float] = {}
        n = 0
        for _ in range(epochs):
            perm = torch.randperm(B, device=self.device)
            for mb in perm.chunk(nmb):
                mean, value = self.net.sequence(ro["obs"][:, mb], ro["h0"][mb])      # [T, mb, 2], [T, mb]
                loss, st = ppo_loss(self.net, mean, value, ro["act"][:, mb], ro["logp"][:, mb], adv[:, mb], ro["val"][:, mb], ro["ret"][:, mb], p)
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_gn)
                self.opt.step()
                for k, v in st.items():
                    stats[k] = stats.get(k, 0.0) + v
                n += 1
        out = {k: round(v / max(n, 1), 4) for k, v in stats.items()}
        if "humans" in ro:
            out.update({"h_" + k: v for k, v in self.learn_humans(ro["humans"]).items()})
        return out

    def learn_humans(self, hum: dict) -> dict:
        """The runners' PPO step: every living person at every tick is one sample of the shared policy."""
        p = self.hp
        alive = hum["alive"].reshape(-1)
        obs, act = hum["obs"].reshape(-1, hum["obs"].shape[-1])[alive], hum["act"].reshape(-1, 2)[alive]
        logp, val, adv, ret = (hum[k].reshape(-1)[alive] for k in ("logp", "val", "adv", "ret"))
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        epochs, nmb, max_gn = int(p.get("epochs", 4)), int(p.get("minibatches", 4)), float(p.get("max_grad_norm", 0.5))
        self.hnet.train()
        stats: dict[str, float] = {}
        n = 0
        for _ in range(epochs):
            for mb in torch.randperm(N, device=self.device).chunk(nmb):
                mean, value = self.hnet(obs[mb])
                loss, st = ppo_loss(self.hnet, mean, value, act[mb], logp[mb], adv[mb], val[mb], ret[mb], p)
                self.hopt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.hnet.parameters(), max_gn)
                self.hopt.step()
                for k, v in st.items():
                    stats[k] = stats.get(k, 0.0) + v
                n += 1
        return {k: round(v / max(n, 1), 4) for k, v in stats.items()}

    # ---- the loop ------------------------------------------------------------------------------
    def train(self, updates: int, log_every: int = 1, save_every: int = 10, verbose: bool = True) -> dict:
        T = int(self.p.get("steps", 64))
        t0 = time.time()
        log_path = self.out_dir / "hunt_gpu_train.jsonl" if self.out_dir else None
        if verbose:
            who = {"patrol": "patrol walkers", "wander": "people walking about, minding their own business", "flee": "scripted runners",
                   "learn": "runners learning to evade (PPO, faster than the fly)" if self.adversarial else "trained runners (frozen)"}[self.arena.human_mode]
            print(f"[hunt-gpu] PPO on {self.device}: {self.arena.B} rooms x {T} ticks per update ({self.arena.B * T} ticks), {updates} updates; "
                  f"fly {sum(p.numel() for p in self.net.parameters())} parameters vs {who}"
                  f"{'' if self.hnet is None else ' (%d parameters)' % sum(p.numel() for p in self.hnet.parameters())}", flush=True)
        for _ in range(updates):
            tu = time.time()
            ro = self.rollout(T)
            st = self.learn(ro)
            self.update += 1
            self.recent = (self.recent + ro["episodes"])[-4 * self.arena.B:]
            sm = summarize(self.recent, self.arena.episode_s, self.arena.side_reward)
            line = {"update": self.update, "steps": self.total_steps, "wall_s": round(time.time() - t0), "ups": round(time.time() - tu, 2),
                    "reward_per_tick": round(float(ro["rew"].mean()), 4), **sm, **st, "log_std": [round(float(x), 3) for x in self.net.log_std.detach().cpu()]}
            if "humans" in ro:
                line["human_reward_per_tick"] = round(float(ro["humans"]["rew"].sum() / ro["humans"]["alive"].sum().clamp(min=1)), 4)
            if log_path:
                with open(log_path, "a") as f:
                    f.write(json.dumps(line) + "\n")
            if verbose and self.update % log_every == 0:
                tr = f"{sm['touch_rate']:6.1%}" if sm.get("touch_rate") is not None else "   -  "
                tt = f"{sm['t_touch']:5.1f} s" if sm.get("t_touch") is not None else "  -    "
                fr = f"{sm['frontal']:.0%}" if sm.get("frontal") is not None else "-"
                trk = (f" track {sm['track']:.0%} (longest trail {sm.get('track_s') or 0:.1f} s) touching {sm.get('contact') or 0:.0%} (longest {sm.get('contact_s') or 0:.1f} s, "
                       f"{sm['touches']:.1f} touches, {sm.get('spins') or 0:.2f} spins)") if sm.get("track") is not None else ""
                hum = f" | humans kept {sm['mean_dist']:.1f} m, rested {sm['target_rest']:.0%}, kl {st['h_kl']:.4f}" if "h_kl" in st and sm.get("n") else ""
                srch = f" | blind {sm['blind']:.0%} of the time, idle {sm['idle']:.0%} of it" if sm.get("n") else ""
                warm = " (warming up: no room has run a full episode yet)" if self.update * T < self.arena.max_steps else ""
                sc = f"{sm['score']:.2f}" if sm.get("score") is not None else "-"
                print(f"[hunt-gpu] upd {self.update:4d} | {sm.get('n', 0):5d} eps: touch {tr} in {tt} head-on {fr:>4}{trk} "
                      f"score {sc} | kl {st['kl']:.4f} ent {st['entropy']:.2f}{srch}{hum} | {self.total_steps / (time.time() - t0) / 1000:.0f}k ticks/s{warm}", flush=True)
            if self.adversarial and sm.get("n", 0) >= self.arena.B:              # two learners: the pair at the end is the result, not an early peak
                self.best = {"touch_rate": sm.get("touch_rate"), "score": sm.get("score"), "update": self.update}
            elif self.out_dir and sm.get("n", 0) >= self.arena.B and self.best_key(sm) > self.best_key(self.best):
                self.best = {"touch_rate": sm.get("touch_rate"), "score": sm.get("score"), "track": sm.get("track"), "contact": sm.get("contact"), "update": self.update}
                self.save("hunter_gpu.pt", "evaders_gpu.pt", meta={**self.best, "t_touch": sm.get("t_touch"), "frontal": sm.get("frontal"), "track": sm.get("track")})
            if self.out_dir and self.update % save_every == 0:
                self.save("hunter_gpu_last.pt", "evaders_gpu_last.pt", meta={"update": self.update, **sm})
        if self.out_dir:
            final = {"update": self.update, **summarize(self.recent, self.arena.episode_s, self.arena.side_reward)}
            self.save("hunter_gpu_last.pt", "evaders_gpu_last.pt", meta=final)
            if self.adversarial:
                self.save("hunter_gpu.pt", "evaders_gpu.pt", meta=final)
        return {"best": self.best, "updates": self.update, "steps": self.total_steps, "wall_s": round(time.time() - t0)}

    def best_key(self, sm: dict) -> tuple:
        """What makes one fly better than another when keeping the best checkpoint: touch rate, then score."""
        return (sm.get("touch_rate") or 0.0, sm.get("score") or 0.0)

    def save(self, fly_name: str, humans_name: str, meta: dict) -> None:
        self.net.save(self.out_dir / fly_name, meta={**meta, "humans": self.arena.human_mode})
        if self.hnet is not None:
            self.hnet.save(self.out_dir / humans_name, meta=meta)


@torch.no_grad()
def evaluate(cfg, net: HunterNet | None, episodes: int, device="auto", seed: int = 0, seeds: list[int] | None = None,
             deterministic: bool = True, scripted: bool = False, noise: bool = False, humans: EvaderNet | None = None) -> list[dict]:
    """Run `episodes` seeded hunts as one batch (seeds seed*7919 + k*104729 unless given) with the network
    (mean action) or the scripted hunter, against the humans of `hunt_gpu.humans.mode` (`learn` needs the
    trained `humans` network; without one the scripted runners stand in). Returns per-episode results."""
    dev = pick_device(device)
    seeds = list(seeds) if seeds is not None else [(seed * 7919 + k * 104729) & 0xFFFFFFFF for k in range(episodes)]
    g = dict(cfg.hunt_gpu)
    if dict(g.get("humans", {})).get("mode", "learn") == "learn" and humans is None:
        g = {**g, "humans": {**dict(g.get("humans", {})), "mode": "flee"}}
    arena = BatchArena(cfg.hunt, g, len(seeds), dev, seed=seed)
    arena.reset(torch.arange(len(seeds), device=dev), seeds)
    if net is not None:
        net = net.to(dev).eval()
        h = net.initial_state(arena.B, dev)
    if humans is not None:
        humans = humans.to(dev).eval()
    results: dict[int, dict] = {}
    active = torch.ones(arena.B, device=dev, dtype=torch.bool)
    for _ in range(arena.max_steps + 1):
        obs = arena.observe(noise=noise)
        if scripted or net is None:
            a = scripted_drive(obs, arena.spec)
        else:
            a, _, _, h = net.act(obs, h, deterministic=deterministic)
        hdrive = humans.act(arena.observe_humans(), deterministic=deterministic)[0] if (humans is not None and arena.human_mode == "learn") else None
        _, done, info = arena.step(a, hdrive)
        for e in info.get("episodes", []):
            results.setdefault(e["seed"], e)
        active &= ~done
        if not bool(active.any()):
            break
        arena.steps[~active] = 0            # finished rooms idle harmlessly (their records are already kept)
    return [results[s] for s in seeds if s in results]
