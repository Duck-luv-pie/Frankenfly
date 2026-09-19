"""The real fly's brain in the batch arena: the hunter subcircuit of the FlyWire connectome as a
batched, differentiable rate model, trained on the GPU, loaded back into the spiking brain.

The hunter subcircuit (`HunterCircuit`) is cut from the pruned circuit: every neuron within
`hops_forward` synapses of the hunter's sensory inputs (the LC visual projection neurons and the
arista's hot cells) that also lies within `hops_backward` synapses of its descending readouts
(DNa01 / DNa02 steering, the descending population, DNp09, MDN), plus the inputs and readouts
themselves and every connection among them. Nothing is invented: the wiring, its signs and its
synapse counts are the connectome's.

`RateBrain` runs that wiring as a mean-field version of the LIF neurons of `sim/lif.py` (Shiu et al.
2024): each neuron's mean synaptic input is the sum of its presynaptic rates times the signed synaptic
weights times tau_syn, its firing rate follows the LIF f-I curve 1 / (t_ref + tau_m ln(g / (g - 7 mV)))
with a soft threshold (the spiking brain's Poisson noise fires neurons a little below threshold), the
spike-frequency adaptation of the spiking model is its mean (adapt_mv x rate x tau_adapt), the 1 Hz
spontaneous floor of central neurons is added, and the stimulated sensory neurons fire at their drive
rate. Sensory drive and motor readout are the real ones: the hunt arena's heat and vision become
Poisson rates on the same neuron groups as in `sim/hunt.py`, and the forward / backward / turn drives
are the same z-scored descending rates as `body/decode.py`, against a resting baseline.

What learns, and can be written back into the spiking brain: a gain on every synapse (bounded, sign
kept: an excitatory synapse stays excitatory), a threshold offset on every neuron (mV), the sensory
gains and the decoder thresholds (`sim/hunt.py`'s PARAMS). `save` writes them as `hunter_brain.npz`,
keyed by FlyWire root ids; `apply_to_runner` puts them on a `BrainRunner`, so `companion hunt --load`
and `companion run --load-weights` run the trained fly as spiking neurons."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch import nn

from ..data.prune import Circuit, _bfs
from .arena import ObsSpec

HUNT_INPUTS = ["LC11", "LC10a", "LC12", "LC15", "TRN_hot"]
HUNT_READOUTS = ["DNa01", "DNa02", "DN_all", "DNp09", "MDN"]
SIDES = ("left", "right", "all")


# ---- the subcircuit ----------------------------------------------------------------------------
class HunterCircuit:
    def __init__(self, circuit: Circuit, cfg, hops: tuple[int, int] = (2, 1), inputs=None, readouts=None):
        self.cfg = cfg
        self.inputs, self.readouts = list(inputs or HUNT_INPUTS), list(readouts or HUNT_READOUTS)
        c = circuit
        adj = sp.csr_matrix((np.ones(c.m, np.int8), (c.pre, c.post)), shape=(c.n, c.n))
        src = np.zeros(c.n, bool)
        dst = np.zeros(c.n, bool)
        for g in self.inputs:
            src[c.idx(g)] = True
        for g in self.readouts:
            dst[c.idx(g)] = True
        fwd, bwd = _bfs(adj, src, int(hops[0])), _bfs(adj.T.tocsr(), dst, int(hops[1]))
        keep = (fwd & bwd) | src | dst
        self.keep = np.nonzero(keep)[0]
        new = np.full(c.n, -1, np.int64)
        new[self.keep] = np.arange(len(self.keep))
        em = keep[c.pre] & keep[c.post]
        pre, post, w = new[c.pre[em]], new[c.post[em]], c.weight_mv[em].astype(np.float32).copy()
        self.cell_type = c.cell_type[self.keep] if c.cell_type is not None else np.array(["unnamed"] * len(self.keep))
        # the runner's synapse-class gains (config lif.synapse_gains) are part of the spiking brain: apply them here too
        for spec in cfg.lif.get("synapse_gains", []):
            rpre, rpost = re.compile(spec["pre"]), re.compile(spec["post"])
            pre_ok = np.array([bool(rpre.search(t)) for t in self.cell_type])
            post_ok = np.array([bool(rpost.search(t)) for t in self.cell_type])
            w[pre_ok[pre] & post_ok[post]] *= np.float32(spec["gain"])
        groups = {}
        for name, d in c.groups.items():
            groups[name] = {s: np.sort(new[d[s]][new[d[s]] >= 0]).astype(np.int32) for s in SIDES}
        self.circuit = Circuit(root_ids=c.root_ids[self.keep], pre=pre.astype(np.int32), post=post.astype(np.int32), weight_mv=w, groups=groups,
                               meta={"full": False, "n": int(len(self.keep)), "m": int(len(pre)), "hunter": True, "hops": list(hops),
                                     "parent": {"n": int(c.n), "m": int(c.m)}},
                               pos=c.pos[self.keep] if c.pos is not None else None, cell_type=self.cell_type,
                               super_class=c.super_class[self.keep] if c.super_class is not None else None,
                               cell_class=c.cell_class[self.keep] if c.cell_class is not None else None)
        self.hops = tuple(int(h) for h in hops)

    @property
    def n(self) -> int:
        return self.circuit.n

    @property
    def m(self) -> int:
        return self.circuit.m

    def background_mask(self) -> np.ndarray:
        """The neurons that get the spontaneous floor, by the runner's rules."""
        c, lif = self.circuit, self.cfg.lif
        classes = set(lif.get("background_classes", ["central", "descending", "ascending", "visual_centrifugal", "endocrine"]))
        mask = np.isin(c.super_class, list(classes)) if c.super_class is not None else np.ones(c.n, bool)
        for pat in lif.get("background_exclude_types", []):
            rx = re.compile(pat)
            mask &= ~np.array([bool(rx.search(t)) for t in c.cell_type])
        return mask

    def neuron_classes(self) -> np.ndarray:
        """A coarse class per neuron: its hunter group if it is in one, else "DN" for other descending neurons, else its super class."""
        c = self.circuit
        cls = np.array(list(c.super_class), dtype=object) if c.super_class is not None else np.array(["central"] * c.n, dtype=object)
        cls[cls == "descending"] = "DN"
        for g in ("LC10a", "LC11", "LC12", "LC15", "TRN_hot", "DNa01", "DNa02", "DNp09", "MDN"):
            cls[c.groups[g]["all"]] = g
        return cls

    def summary(self) -> dict:
        c = self.circuit
        cls = {}
        if c.super_class is not None:
            u, cnt = np.unique(c.super_class, return_counts=True)
            cls = {str(k): int(v) for k, v in zip(u, cnt)}
        return {"neurons": c.n, "synapses": c.m, "hops": self.hops, "classes": cls,
                "groups": {g: {s: int(len(c.groups[g][s])) for s in SIDES} for g in self.inputs + self.readouts}}


# ---- sensory drive from the arena's observation ----------------------------------------------------
def hemifield_features(vis: torch.Tensor, bin_c: torch.Tensor, half: float) -> dict[str, torch.Tensor]:
    """The CPU arena's per-hemifield features from the retinal columns [B, K, 2]: the strongest object per
    hemifield, its eccentricity (0 at the midline, 1 at the edge, for the retinotopic drive) and the bar."""
    obj, bar = vis[..., 0], vis[..., 1]
    left = bin_c > 0                                        # columns on the fly's left (+ve bearing)
    out = {}
    for name, mask in (("left", left), ("right", ~left)):
        o = torch.where(mask[None], obj, torch.zeros_like(obj))
        strength = o.amax(dim=-1)
        # eccentricity of the object's centre of mass in the hemifield (a close person saturates every column it
        # covers; the first of the tied columns would put it at the edge on one side and the midline on the other)
        ecc = ((o * bin_c.abs()[None]).sum(-1) / o.sum(-1).clamp(min=1e-6) / half).clamp(0, 1)
        out[name] = {"small_object": strength, "ecc": ecc, "bar": torch.where(mask[None], bar, torch.zeros_like(bar)).amax(dim=-1)}
    return out


# ---- the rate model ------------------------------------------------------------------------------
class RateBrain(nn.Module):
    """The hunter subcircuit as a batch of rate neurons, with the real sensory mapping and decoder on either side."""

    CHANNELS = ("forward", "backward", "turn")

    def __init__(self, hc: HunterCircuit, cfg, gains: dict | None = None, brain_cfg: dict | None = None):
        super().__init__()
        c, lif, bc = hc.circuit, cfg.lif, dict(brain_cfg or {})
        self.hc, self.cfg = hc, cfg
        n, m = c.n, c.m
        self.n, self.m = n, m
        g = dict(gains or {})
        self.tau_syn, self.tau_mem = float(lif.tau_syn_ms) * 1e-3, float(lif.tau_mem_ms) * 1e-3
        self.t_ref, self.delta = float(lif.refractory_ms) * 1e-3, float(lif.v_thresh_mv) - float(lif.v_rest_mv)
        self.adapt_mv, self.tau_adapt = float(lif.get("adapt_mv", 0.0)), float(lif.get("tau_adapt_ms", 200.0)) * 1e-3
        self.r_max = 1.0 / self.t_ref
        self.substeps = int(bc.get("substeps", 4))
        self.dt = float(cfg.hunt.get("tick_s", 0.05)) / self.substeps
        self.kappa, self.syn_scale = float(bc.get("kappa_mv", 1.0)), float(bc.get("syn_scale", 1.0))
        self.gain_log_max, self.th_max = math.log(float(bc.get("gain_max", 10.0))), float(bc.get("th_max_mv", 4.0))
        # wiring
        self.register_buffer("pre", torch.as_tensor(c.pre, dtype=torch.long))
        self.register_buffer("post", torch.as_tensor(c.post, dtype=torch.long))
        self.register_buffer("w0", torch.as_tensor(c.weight_mv, dtype=torch.float32))
        # the same wiring as a CSR matrix [post, pre] (duplicate pre->post pairs summed, as in the dense form): on a
        # GPU the CSR matmul is several times faster than the COO one, and `csr_values` keeps the gradient to the gains
        key = torch.as_tensor(c.post, dtype=torch.long) * n + torch.as_tensor(c.pre, dtype=torch.long)
        order = torch.argsort(key)
        uniq, inverse = torch.unique_consecutive(key[order], return_inverse=True)
        crow = torch.zeros(n + 1, dtype=torch.long)
        crow[1:] = torch.cumsum(torch.bincount(uniq // n, minlength=n), 0)
        self.register_buffer("csr_order", order)
        self.register_buffer("csr_inverse", inverse)
        self.register_buffer("csr_crow", crow)
        self.register_buffer("csr_col", uniq % n)
        self.csr_nnz = int(uniq.shape[0])
        bg = torch.zeros(n)
        bg[torch.as_tensor(hc.background_mask())] = float(lif.get("background_hz", 0.0))
        self.register_buffer("bg", bg)
        # synapse classes (presynaptic class -> postsynaptic class, by hunter group or super class): a coarse, gradient-free
        # handle on the wiring for evolutionary search, multiplied on top of the per-synapse gains
        ncls = hc.neuron_classes()
        pairs = sorted(set(zip(ncls[c.pre], ncls[c.post])))
        self.class_names = [f"{a}->{b}" for a, b in pairs]
        pair_idx = {p: i for i, p in enumerate(pairs)}
        self.register_buffer("syn_class", torch.as_tensor([pair_idx[(a, b)] for a, b in zip(ncls[c.pre], ncls[c.post])], dtype=torch.long))
        self.register_buffer("class_log_gain", torch.zeros(len(pairs)))
        # sensory drive: (group, side) -> neurons; the hunt's sensory mapping (sim/hunt.py)
        f = cfg.senses.features
        self.max_hz = float(cfg.senses.max_rate_hz)
        self.sense_rows = []                                     # (feature, side, group)
        for feat in ("small_object", "bar"):
            for side in ("left", "right"):
                for grp in f[feat]["groups"]:
                    self.sense_rows.append((feat, side, grp))
        for side in ("left", "right"):
            for grp in cfg.senses.world.heat.groups:
                self.sense_rows.append(("heat", side, grp))
        # looming: the growth of a person's angular width drives the fly's own looming detectors (LC4, LPLC2). The hunt
        # zeroes this drive on the CPU to avoid the escape jump; here there is no jump, so the reflex is available.
        for side in ("left", "right"):
            for grp in cfg.senses.features.loom_fast.groups:
                if len(c.groups.get(grp, {}).get(side, [])):
                    self.sense_rows.append(("loom", side, grp))
        # proximity: a person filling more than `hunt.near_deg` of the field of view (within a few decimetres) drives
        # `brain.near_groups`, by default LC9: in the connectome LC9 is the visual projection type with the most synapses
        # onto DNp09, the freezing descending neuron, and driving it raises DNp09 without touching the forward
        # population or DNa02. With `hunt.freeze_brakes` that is the brain's brake: it can learn to stop at contact.
        # (LC4 / LPLC2 do not reach DNp09 or MDN within the cut; they only excite the forward population.)
        self.near_groups = [g_ for g_ in bc.get("near_groups", ["LC9"]) if any(len(c.groups.get(g_, {}).get(sd, [])) for sd in ("left", "right"))]
        for side in ("left", "right"):
            for grp in self.near_groups:
                if len(c.groups.get(grp, {}).get(side, [])):
                    self.sense_rows.append(("near", side, grp))
        self.fresh = set(bc.get("fresh_gains", []))          # gains that ignore a resumed file and start from the config
        M = torch.zeros(len(self.sense_rows), n)
        for i, (feat, side, grp) in enumerate(self.sense_rows):
            M[i, torch.as_tensor(c.groups[grp][side], dtype=torch.long)] = 1.0
        self.register_buffer("M", M)
        # readouts: mean rate per (group, side)
        self.read_keys = [(grp, s) for grp in hc.readouts for s in SIDES]
        R = torch.zeros(len(self.read_keys), n)
        for i, (grp, s) in enumerate(self.read_keys):
            idx = torch.as_tensor(c.groups[grp][s], dtype=torch.long)
            if len(idx):
                R[i, idx] = 1.0 / len(idx)
        self.register_buffer("R", R)
        self.register_buffer("base_mean", torch.zeros(len(self.read_keys)))
        self.register_buffer("base_std", torch.ones(len(self.read_keys)))
        d = cfg.decode
        self.window_ticks = max(1, int(round(float(d.motor.channels.forward.get("window_ms", d.window_ms)) / float(cfg.hunt.get("tick_s", 0.05)) / 1000)))
        self.min_std = {"forward": float(d.motor.channels.forward.get("min_std_hz", d.min_std_hz)), "backward": float(d.motor.channels.backward.get("min_std_hz", d.min_std_hz)),
                        "turn": float(d.motor.channels.turn.get("min_std_hz", d.min_std_hz))}
        # ---- learnable: the brain ----
        self.u_gain = nn.Parameter(torch.zeros(m))               # log gain = gain_log_max * tanh(u)
        self.u_th = nn.Parameter(torch.zeros(n))                 # threshold offset = th_max * tanh(u) mV
        # ---- learnable: senses and decoder (sim/hunt.py PARAMS, the ones the connectome fly carries) ----
        def logp(key, default):
            return nn.Parameter(torch.tensor(math.log(float(g.get(key, default)))))
        def rawp(key, default):
            return nn.Parameter(torch.tensor(float(g.get(key, default))))
        self.p_obj_gain = logp("senses.features.small_object.gain", float(f.small_object.get("gain", 1.0)))
        self.p_bar_gain = logp("senses.features.bar.gain", float(f.bar.get("gain", 0.4)))
        self.p_azimuth = rawp("senses.features.small_object.azimuth_weight", float(f.small_object.get("azimuth_weight", 0.0)))
        self.p_heat_gain = logp("hunt.heat_gain", float(cfg.hunt.get("heat_gain", 1.0)))
        self.p_loom_gain = rawp("hunt.loom_gain", float(cfg.hunt.get("loom_gain", 0.0)))     # 0 = off (brains trained without it stay faithful)
        self.p_loom_ref = logp("hunt.loom_ref", float(cfg.hunt.get("loom_ref", 2.0)))     # rad/s of width growth for full drive
        self.p_near_deg = rawp("hunt.near_deg", float(cfg.hunt.get("near_deg", 50.0)))    # a person wider than this (whole view) is "on top of me" ...
        self.p_near_gain = rawp("hunt.near_gain", float(cfg.hunt.get("near_gain", 0.0)))  # ... and drives brain.near_groups (LC9 -> DNp09, the brake); 0 = off
        self.p_efference = rawp("hunt.efference_copy_gain", float(cfg.hunt.get("efference_copy_gain", 0.3)))
        self.p_turn_gain = logp("hunt.turn_gain", float(cfg.hunt.get("turn_gain", 1.0)))
        self.p_fwd_zref = logp("decode.motor.channels.forward.z_ref", float(d.motor.channels.forward.z_ref))
        # the forward dead zone may go negative: a resting descending population then still drives the vehicle
        # forward (the fly walks when it sees nothing, instead of freezing); the spiking decoder reads the same z0
        self.p_fwd_z0 = rawp("decode.motor.channels.forward.z0", float(d.motor.channels.forward.get("z0", d.z0)))
        # the turn readout: right minus left z-score, or (`contrast`) that difference normalized by both sides' activity,
        # so a person straight ahead driving both eyes hard is a small correction and not a hard turn (the DNa02 pair is a
        # lopsided winner-take-all: symmetric input alone gives a large raw difference). z0 is then in contrast units.
        self.contrast = bool(g.get("decode.motor.channels.turn.contrast", d.motor.channels.turn.get("contrast", False)))   # a saved brain carries it
        self.p_turn_zref = logp("decode.motor.channels.turn.z_ref", float(d.motor.channels.turn.z_ref))
        self.p_turn_z0 = rawp("decode.motor.channels.turn.z0", float(d.motor.channels.turn.get("z0", 0.05 if self.contrast else d.z0)))
        self.p_smooth = logp("decode.motor.smoothing_ms", float(d.motor.smoothing_ms))
        self.bwd_zref, self.bwd_z0 = float(d.motor.channels.backward.z_ref), float(d.motor.channels.backward.get("z0", d.z0))
        # the freeze channel (DNp09) brakes the vehicle when hunt.freeze_brakes is on, as in the CPU arena: forward x (1 - freeze).
        # It gives the brain a way to slow down at contact instead of ramming through the person it just caught.
        self.brake = bool(cfg.hunt.get("freeze_brakes", False))
        self.p_freeze_zref = logp("decode.motor.channels.freeze.z_ref", float(d.motor.channels.freeze.z_ref))
        # the brake's dead zone is learnable too: DNp09's background activity must not leave a residual brake on while
        # the fly is following a person at a distance (it needs all its speed there) and still brake hard at contact
        self.p_freeze_z0 = rawp("decode.motor.channels.freeze.z0", float(d.motor.channels.freeze.get("z0", d.z0)))
        self.log_std = nn.Parameter(torch.full((2,), float(bc.get("log_std_init", -0.7))))
        self.log_std_min = float(bc.get("log_std_min", -10.0))   # a floor on the exploration noise: PPO shrinks it until nothing new is tried
        self.spec = None                                         # ObsSpec, set by bind_arena
        self.bin_c = None
        self.half = None

    # ---- geometry of the arena's retina ----
    def bind_arena(self, spec: ObsSpec, bin_c: torch.Tensor, fov: float, vmax: float) -> None:
        self.spec, self.bin_c, self.half, self.vmax = spec, bin_c.to(self.w0.device), fov / 2, float(vmax)
        self.bin_w = fov / spec.bins

    # ---- parameters as the spiking brain reads them ----
    def gain(self) -> torch.Tensor:
        return torch.exp(self.gain_log_max * torch.tanh(self.u_gain) + self.class_log_gain[self.syn_class])

    def th_offset(self) -> torch.Tensor:
        return self.th_max * torch.tanh(self.u_th)

    def gains_dict(self) -> dict:
        return {"senses.features.small_object.gain": float(self.p_obj_gain.exp()), "senses.features.bar.gain": float(self.p_bar_gain.exp()),
                "senses.features.small_object.azimuth_weight": float(self.p_azimuth.clamp(0, 1)), "hunt.heat_gain": float(self.p_heat_gain.exp()),
                "hunt.efference_copy_gain": float(self.p_efference.clamp(0, 0.9)), "hunt.turn_gain": float(self.p_turn_gain.exp()),
                "hunt.loom_gain": float(self.p_loom_gain.clamp(min=0)), "hunt.loom_ref": float(self.p_loom_ref.exp()),
                "hunt.near_deg": float(self.p_near_deg.clamp(10, 75)), "hunt.near_gain": float(self.p_near_gain.clamp(min=0)),
                "decode.motor.channels.forward.z_ref": float(self.p_fwd_zref.exp()), "decode.motor.channels.forward.z0": float(self.p_fwd_z0.clamp(-3.0, 1.0)),
                "decode.motor.channels.turn.z_ref": float(self.p_turn_zref.exp()),
                "decode.motor.channels.turn.z0": float(self.p_turn_z0.clamp(0, 0.9) if self.contrast else self.p_turn_z0.clamp(min=0)),
                "decode.motor.channels.turn.contrast": bool(self.contrast), "decode.motor.smoothing_ms": float(self.p_smooth.exp()),
                "decode.motor.channels.freeze.z_ref": float(self.p_freeze_zref.exp()), "decode.motor.channels.freeze.z0": float(self.p_freeze_z0.clamp(-2.0, 20.0)),
                "hunt.freeze_brakes": bool(self.brake),
                "decode.baseline_drift_s": 0.0,       # no slow re-centering of the resting baseline: the rate model has none, and with it the spiking fly stalls after a chase
                "learning.valence_steering": 0.0}

    # ---- state ----
    def initial_state(self, B: int) -> dict:
        dev = self.w0.device
        return {"r": torch.zeros(self.n, B, device=dev), "a": torch.zeros(self.n, B, device=dev),
                "ring": torch.zeros(self.window_ticks, len(self.read_keys), B, device=dev), "motor": torch.zeros(4, B, device=dev),
                "width": torch.zeros(2, B, device=dev)}

    @staticmethod
    def detach_state(st: dict) -> dict:
        return {k: v.detach() for k, v in st.items()}

    @staticmethod
    def index_state(st: dict, idx) -> dict:
        return {k: v[..., idx] for k, v in st.items()}

    # ---- senses -> Poisson drive per neuron ----
    def drive_hz(self, obs: torch.Tensor, width_prev: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """([G, B] Hz per (feature, side, group) row of M, [2, B] hemifield widths) from the arena's observation.
        `width_prev` is the previous tick's widths, for the looming drive."""
        heat, vis, body = self.spec.split(obs)
        hf = hemifield_features(vis, self.bin_c, self.half)
        left = self.bin_c > 0
        covered = (vis[..., 0] > 0.1).float()          # columns a person really covers (a person within see_m is > 0.19; the
                                                        # training's sensor noise, sigma 0.02, must not count as a wall of people)
        width = torch.stack([(covered * left[None]).sum(-1), (covered * (~left)[None]).sum(-1)], dim=0) * self.bin_w    # [2, B] rad
        growth = (width - (width_prev if width_prev is not None else width)) / (self.dt * self.substeps)
        # proximity from the person's whole width (both hemifields: at contact a person straight ahead is ~70 deg wide,
        # 35 per side): zero at near_deg, full 25 deg past it. With the defaults the brake starts ~20 cm before contact.
        near = ((width.sum(0) - math.radians(1.0) * self.p_near_deg.clamp(10, 75)) / math.radians(25.0)).clamp(0, 1) * self.p_near_gain.clamp(min=0)
        loom = (growth / self.p_loom_ref.exp()).clamp(0, 1) * self.p_loom_gain.clamp(min=0)
        sup = (1.0 - self.p_efference.clamp(0, 0.9) * body[:, 0].abs().clamp(max=1.0))      # efference copy: a moving fly sees less
        w = self.p_azimuth.clamp(0, 1)
        rows = []
        for feat, side, grp in self.sense_rows:
            if feat == "small_object":
                val = hf[side]["small_object"] * self.p_obj_gain.exp() * ((1 - w) + w * hf[side]["ecc"]) * sup
            elif feat == "bar":
                val = hf[side]["bar"] * self.p_bar_gain.exp() * sup
            elif feat == "loom":
                val = loom[0 if side == "left" else 1] * sup
            elif feat == "near":
                val = near                                                   # not suppressed by the efference copy: it is the brake
            else:
                val = heat[:, 0 if side == "left" else 1] * self.p_heat_gain.exp()
            rows.append(val.clamp(0, 1) * self.max_hz)
        return torch.stack(rows, dim=0), width

    # ---- one arena tick ----
    def csr_values(self) -> torch.Tensor:
        """The CSR matrix's values (mV), differentiable w.r.t. the synapse gains."""
        vals = self.w0 * self.gain() * self.syn_scale
        return torch.zeros(self.csr_nnz, device=vals.device, dtype=vals.dtype).index_add(0, self.csr_inverse, vals[self.csr_order])

    def csr(self, values: torch.Tensor) -> torch.Tensor:
        return torch.sparse_csr_tensor(self.csr_crow, self.csr_col, values, (self.n, self.n))

    def weights(self, dense: bool = False) -> torch.Tensor:
        """The effective weight matrix [post, pre] in mV: CSR (fast on CUDA and CPU), COO on Apple's MPS (which has no
        CSR kernels), or dense on request (on MPS the gradient of a sparse matmul never reaches the values, so
        training there uses the dense form)."""
        if dense:
            vals = self.w0 * self.gain() * self.syn_scale
            return torch.zeros(self.n, self.n, device=vals.device).index_put((self.post, self.pre), vals, accumulate=True)
        if self.w0.device.type == "mps":
            vals = self.w0 * self.gain() * self.syn_scale
            return torch.sparse_coo_tensor(torch.stack([self.post, self.pre]), vals, (self.n, self.n)).coalesce()
        return self.csr(self.csr_values())

    def neurons(self, r: torch.Tensor, a: torch.Tensor, drive: torch.Tensor, W: torch.Tensor, th: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`substeps` of the mean-field LIF dynamics. r, a, drive: [n, B]."""
        k_r, k_a = math.exp(-self.dt / self.tau_mem), math.exp(-self.dt / self.tau_adapt)
        for _ in range(self.substeps):
            I = (torch.sparse.mm(W, r) if W.layout == torch.sparse_coo else W @ r) * self.tau_syn     # mean synaptic input, mV
            x = I - a - th[:, None] - self.delta
            s = nn.functional.softplus(x / self.kappa) * self.kappa                    # how far above threshold, softly
            # the LIF f-I curve, gated by the probability of being above threshold at all (the spiking brain's
            # Poisson noise smears the threshold over ~kappa mV; the curve alone falls off only logarithmically)
            f = torch.sigmoid(x / self.kappa) / (self.t_ref + self.tau_mem * torch.log1p(self.delta / s.clamp(min=1e-6)))
            target = (f + self.bg[:, None] + drive).clamp(max=self.r_max)
            r = target + (r - target) * k_r
            a_target = self.adapt_mv * self.tau_adapt * r
            a = a_target + (a - a_target) * k_a
        return r, a

    readout_noise = False        # add the spiking brain's resting jitter to the window means (its std per group and side)

    def decode(self, ring: torch.Tensor, motor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The decoder of body/decode.py on the window's mean readout rates: z-scores -> smoothed channels."""
        rates = ring.mean(dim=0)                                                       # [K, B]
        if self.readout_noise:
            rates = rates + torch.randn_like(rates) * self.base_std[:, None]
        z = (rates - self.base_mean[:, None]) / self.base_std[:, None]
        zk = {k: z[i] for i, k in enumerate(self.read_keys)}
        def score(zv, z0, zref):
            return ((zv - z0) / (zref - z0).clamp(min=1e-3)).clamp(0, 1)
        fwd = score(zk[("DN_all", "all")], self.p_fwd_z0.clamp(-3.0, 1.0), self.p_fwd_zref.exp())
        bwd = score(zk[("MDN", "all")], self.bwd_z0, torch.tensor(self.bwd_zref, device=z.device))
        zR, zL = 0.5 * (zk[("DNa01", "right")] + zk[("DNa02", "right")]), 0.5 * (zk[("DNa01", "left")] + zk[("DNa02", "left")])
        if self.contrast:
            c = (zR - zL) / (zR.clamp(min=0) + zL.clamp(min=0) + self.p_turn_zref.exp())
            z0 = self.p_turn_z0.clamp(0, 0.9)
            turn = torch.sign(c) * ((c.abs() - z0).clamp(min=0) / (1 - z0)).clamp(max=1)
        else:
            dz = zR - zL
            z0 = self.p_turn_z0.clamp(min=0)
            mag = ((dz.abs() - z0).clamp(min=0) / (self.p_turn_zref.exp() - z0).clamp(min=1e-3)).clamp(max=1)
            turn = torch.sign(dz) * mag
        freeze = score(zk[("DNp09", "all")], self.p_freeze_z0.clamp(-2.0, 20.0), self.p_freeze_zref.exp())
        raw = torch.stack([fwd, bwd, turn, freeze], dim=0)
        alpha = 1.0 - torch.exp(-float(self.dt * self.substeps) / (self.p_smooth.exp() * 1e-3))
        motor = motor + alpha * (raw - motor)
        return motor, z

    def step(self, obs: torch.Tensor, st: dict, W: torch.Tensor | None = None) -> tuple[torch.Tensor, dict, dict]:
        """One arena tick for a batch: obs [B, D] -> (action mean [B, 2], new state, extras). `W` lets a training
        loop pass a weight matrix built once for many ticks."""
        W, th = self.weights() if W is None else W, self.th_offset()
        hz, width = self.drive_hz(obs, st.get("width"))
        drive = self.M.t() @ hz                                                          # [n, B]
        r, a = self.neurons(st["r"], st["a"], drive, W, th)
        rates = self.R @ r                                                               # [K, B]
        ring = torch.cat([st["ring"][1:], rates[None]], dim=0)
        motor, z = self.decode(ring, st["motor"])
        forward = (motor[0] - motor[1]) * ((1.0 - motor[3]) if self.brake else 1.0)
        mean = torch.stack([forward, (motor[2] * self.p_turn_gain.exp()).clamp(-1, 1)], dim=1)
        return mean, {"r": r, "a": a, "ring": ring, "motor": motor, "width": width}, {"rates": rates, "z": z}

    # ---- the Gaussian policy on the motor channels ----
    def sigma_log(self) -> torch.Tensor:
        return self.log_std.clamp(min=self.log_std_min)

    def log_prob(self, mean: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        ls = self.sigma_log()
        zz = (action - mean) / ls.exp()
        return (-0.5 * zz * zz - ls - 0.5 * math.log(2 * math.pi)).sum(-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.sigma_log()).sum()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, st: dict, deterministic: bool = False, W: torch.Tensor | None = None):
        mean, st, extra = self.step(obs, st, W=W)
        action = mean if deterministic else mean + self.sigma_log().exp() * torch.randn_like(mean)
        return action, self.log_prob(mean, action), st, extra

    def regularizer(self) -> torch.Tensor:
        """Stay close to the connectome: penalize log gains and threshold shifts."""
        return (self.gain_log_max * torch.tanh(self.u_gain)).pow(2).mean() + (self.th_offset() / self.th_max).pow(2).mean()

    # ---- resting baseline (what the decoder measures against) ----
    @torch.no_grad()
    def calibrate(self, seconds: float = 3.0, std: dict | None = None) -> dict:
        """Run the rate model with no senses to steady state; its resting readout rates are the decoder's means.
        The stds are the spiking brain's (from `std`, {group: {side: hz}}) since a rate model has none."""
        st = self.initial_state(1)
        obs = torch.zeros(1, self.spec.size, device=self.w0.device)
        for _ in range(int(seconds / (self.dt * self.substeps))):
            _, st, extra = self.step(obs, st)
        self.base_mean.copy_(extra["rates"][:, 0])
        for i, (grp, s) in enumerate(self.read_keys):
            spec_min = self.min_std["forward"] if grp == "DN_all" else self.min_std["turn"] if grp in ("DNa01", "DNa02") else self.min_std["backward"]
            self.base_std[i] = max(float((std or {}).get(grp, {}).get(s, 0.0)), spec_min)
        return {k: (round(float(self.base_mean[i]), 3), round(float(self.base_std[i]), 3)) for i, k in enumerate(self.read_keys)}

    @torch.no_grad()
    def probe(self, drives: dict[tuple[str, str], float], seconds: float = 1.0, settle_s: float = 1.0) -> dict:
        """Steady readout rates (Hz) with the given (group, side) -> Hz drives, like BrainRunner.probe."""
        st = self.initial_state(1)
        dev = self.w0.device
        drive = torch.zeros(self.n, 1, device=dev)
        for (grp, side), hz in drives.items():
            drive[torch.as_tensor(self.hc.circuit.groups[grp][side], dtype=torch.long, device=dev)] = float(hz)
        W, th = self.weights(), self.th_offset()
        zero = torch.zeros_like(drive)
        r, a = st["r"], st["a"]
        for _ in range(int(settle_s / (self.dt * self.substeps))):
            r, a = self.neurons(r, a, zero, W, th)
        for _ in range(int(seconds / (self.dt * self.substeps))):
            r, a = self.neurons(r, a, drive, W, th)
        rates = self.R @ r
        return {k: round(float(rates[i, 0]), 2) for i, k in enumerate(self.read_keys)}

    # ---- files ----
    def save(self, path: Path | str, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        c = self.hc.circuit
        np.savez_compressed(path, pre_root=c.root_ids[c.pre], post_root=c.root_ids[c.post], gain=self.gain().detach().cpu().numpy().astype(np.float32),
                            th_root=c.root_ids, th_mv=self.th_offset().detach().cpu().numpy().astype(np.float32),
                            gains=json.dumps(self.gains_dict()), meta=json.dumps({**(meta or {}), "hops": list(self.hc.hops), "n": c.n, "m": c.m,
                                                                                    "kappa_mv": self.kappa, "syn_scale": self.syn_scale}),
                            base_mean=self.base_mean.cpu().numpy(), base_std=self.base_std.cpu().numpy(), log_std=self.log_std.detach().cpu().numpy(),
                            u_gain=self.u_gain.detach().cpu().numpy().astype(np.float32), u_th=self.u_th.detach().cpu().numpy().astype(np.float32),
                            class_log_gain=self.class_log_gain.cpu().numpy().astype(np.float32), class_names=np.array(self.class_names),
                            brain="hunter")
        return path

    def load_params(self, path: Path | str) -> dict:
        """Resume from a saved brain: by position when it is the same cut, otherwise by FlyWire root id (a synapse or
        neuron the file does not know keeps the connectome's value), so a brain trained on a smaller cut seeds a larger one."""
        z = np.load(path)
        c = self.hc.circuit
        with torch.no_grad():
            if len(z["u_gain"]) == self.m and len(z["u_th"]) == self.n and "pre_root" in z.files and np.array_equal(z["th_root"], c.root_ids):
                self.u_gain.copy_(torch.as_tensor(z["u_gain"]))
                self.u_th.copy_(torch.as_tensor(z["u_th"]))
                if "class_log_gain" in z.files and len(z["class_log_gain"]) == len(self.class_log_gain):
                    self.class_log_gain.copy_(torch.as_tensor(z["class_log_gain"]))
            else:
                id2i = {int(r): i for i, r in enumerate(c.root_ids)}
                th_i = np.array([id2i.get(int(r), -1) for r in z["th_root"]])
                ok = th_i >= 0
                u_th = torch.zeros(self.n)
                u_th[torch.as_tensor(th_i[ok])] = torch.as_tensor(z["u_th"][ok], dtype=torch.float32)
                self.u_th.copy_(u_th)
                # synapses: (pre, post) root ids -> this cut's synapse index; the saved gain includes the class gain, so fold it in
                pair = {(int(c.root_ids[p]), int(c.root_ids[q])): i for i, (p, q) in enumerate(zip(c.pre, c.post))}
                idx = np.array([pair.get((int(a), int(b)), -1) for a, b in zip(z["pre_root"], z["post_root"])])
                oks = idx >= 0
                log_gain = np.log(np.clip(z["gain"][oks].astype(np.float64), 1e-6, None))
                u = torch.zeros(self.m)
                u[torch.as_tensor(idx[oks])] = torch.atanh(torch.as_tensor(log_gain / self.gain_log_max, dtype=torch.float32).clamp(-0.999, 0.999))
                self.u_gain.copy_(u)
                self.class_log_gain.zero_()
                print(f"[brain] resumed by root id: {int(oks.sum()):,}/{self.m:,} synapse gains and {int(ok.sum()):,}/{self.n:,} thresholds matched", flush=True)
            self.log_std.copy_(torch.as_tensor(z["log_std"]))
            self.base_mean.copy_(torch.as_tensor(z["base_mean"]))
            self.base_std.copy_(torch.as_tensor(z["base_std"]))
            g = {k: v for k, v in json.loads(str(z["gains"])).items() if k not in self.fresh}
            cur = self.gains_dict()
            g = {**{k: cur[k] for k in self.fresh if k in cur}, **g}                     # a fresh gain keeps its current (config) value
            self.p_obj_gain.fill_(math.log(g["senses.features.small_object.gain"])); self.p_bar_gain.fill_(math.log(g["senses.features.bar.gain"]))
            self.p_azimuth.fill_(g["senses.features.small_object.azimuth_weight"]); self.p_heat_gain.fill_(math.log(g["hunt.heat_gain"]))
            self.p_efference.fill_(g["hunt.efference_copy_gain"]); self.p_turn_gain.fill_(math.log(g["hunt.turn_gain"]))
            self.p_fwd_zref.fill_(math.log(g["decode.motor.channels.forward.z_ref"])); self.p_turn_zref.fill_(math.log(g["decode.motor.channels.turn.z_ref"]))
            self.p_fwd_z0.fill_(float(g.get("decode.motor.channels.forward.z0", 0.0)))
            self.p_loom_gain.fill_(float(g.get("hunt.loom_gain", 0.0))); self.p_loom_ref.fill_(math.log(g.get("hunt.loom_ref", 2.0)))
            self.p_near_deg.fill_(float(g.get("hunt.near_deg", 50.0))); self.p_near_gain.fill_(float(g.get("hunt.near_gain", 0.0)))
            if "decode.motor.channels.freeze.z_ref" in g:
                self.p_freeze_zref.fill_(math.log(g["decode.motor.channels.freeze.z_ref"]))
            if "decode.motor.channels.freeze.z0" in g:
                self.p_freeze_z0.fill_(float(g["decode.motor.channels.freeze.z0"]))
            self.p_turn_z0.fill_(g["decode.motor.channels.turn.z0"]); self.p_smooth.fill_(math.log(g["decode.motor.smoothing_ms"]))
        return json.loads(str(z["meta"]))


# ---- back into the spiking brain --------------------------------------------------------------------
def load_brain(path: Path | str) -> dict:
    z = np.load(path)
    return {"pre_root": z["pre_root"], "post_root": z["post_root"], "gain": z["gain"], "th_root": z["th_root"], "th_mv": z["th_mv"],
            "gains": json.loads(str(z["gains"])), "meta": json.loads(str(z["meta"]))}


def apply_to_runner(runner, brain: dict, verbose: bool = True) -> dict:
    """Write the learned synapse gains and threshold offsets into a BrainRunner's LIF network, matching
    synapses by (presynaptic, postsynaptic) FlyWire root id and neurons by root id."""
    c, net = runner.c, runner.net
    id2i = {int(r): i for i, r in enumerate(c.root_ids)}
    pre_i = np.array([id2i.get(int(r), -1) for r in brain["pre_root"]])
    post_i = np.array([id2i.get(int(r), -1) for r in brain["post_root"]])
    ok = (pre_i >= 0) & (post_i >= 0)
    hit = 0
    for p, q, gval in zip(pre_i[ok], post_i[ok], brain["gain"][ok]):
        a, b = net.indptr[p], net.indptr[p + 1]
        j = np.searchsorted(net.indices[a:b], q)
        if j < b - a and net.indices[a + j] == q:
            net.data[a + j] *= np.float32(gval)
            hit += 1
    th_i = np.array([id2i.get(int(r), -1) for r in brain["th_root"]])
    okt = th_i >= 0
    net.th_offset[th_i[okt]] += brain["th_mv"][okt].astype(np.float32)
    out = {"synapses": int(hit), "of": int(len(brain["gain"])), "neurons": int(okt.sum()), "of_neurons": int(len(brain["th_root"]))}
    if verbose:
        print(f"[brain] applied {out['synapses']:,}/{out['of']:,} learned synapse gains and {out['neurons']:,} threshold offsets", flush=True)
    return out
