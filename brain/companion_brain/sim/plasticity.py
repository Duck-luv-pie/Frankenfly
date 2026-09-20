"""Mushroom-body learning: dopamine-gated depression of Kenyon-cell -> MBON synapses.

This is the fly's own reinforcement-learning circuit (Aso et al. 2014, Hige et al. 2015,
Handler et al. 2019). Kenyon cells (KCs) encode the sensory situation sparsely; their synapses
onto mushroom-body output neurons (MBONs) set approach vs avoid; dopamine neurons (DANs) carry
the reinforcement. The rule is three-factor: a KC->MBON synapse is depressed when its KC was
recently active (eligibility trace) AND dopamine arrives in that MBON's compartment.

Compartments come from the connectome itself: for each MBON, its dopamine signal is the
activity of the PAM (reward) or PPL1 (punishment) neurons that synapse directly onto it.
Reward DANs innervate MBONs that drive avoidance, so reward pairing weakens avoidance of the
current odor; punishment DANs innervate approach-driving MBONs, so punishment weakens approach.

    e_k    <- e_k * exp(-dt/tau_elig) + spikes_k                     (per KC)
    D_m    <- max(0, rate(DANs of m) - rest_m) / dopamine_ref_hz     (per MBON, 0..~1)
    w_km   <- max(w_min, w_km - eta * e_k * D_m * dt)                (depression)
    w_km   <- w_km + (w0_km - w_km) * dt / tau_forget                 (slow forgetting)
"""
from __future__ import annotations

import numpy as np


class MushroomBodyPlasticity:
    def __init__(self, circuit, net, cfg, chunk_ms: float):
        self.cfg = cfg.learning
        self.net = net
        self.chunk_s = chunk_ms * 1e-3
        ct = circuit.cell_type
        kc = np.array([t.startswith("KC") for t in ct])
        mbon = np.array([t.startswith("MBON") for t in ct])
        dan_rew = np.array([t.startswith("PAM") for t in ct])
        dan_pun = np.array([t.startswith("PPL1") for t in ct])
        self.kc_idx = np.nonzero(kc)[0]
        self.mbon_idx = np.nonzero(mbon)[0]
        # KC->MBON synapses, located in the network's CSR arrays (rows = presynaptic neuron)
        indptr, indices = net.indptr, net.indices
        syn, pre, post = [], [], []
        for k in self.kc_idx:
            for j in range(indptr[k], indptr[k + 1]):
                if mbon[indices[j]]:
                    syn.append(j); pre.append(k); post.append(indices[j])
        self.syn = np.asarray(syn, dtype=np.int64)
        self.pre = np.asarray(pre, dtype=np.int64)
        self.post = np.asarray(post, dtype=np.int64)
        gain = float(self.cfg.get("kc_mbon_gain", 1.0))
        if gain != 1.0:
            net.data[self.syn] *= gain
        self.w0 = net.data[self.syn].copy()
        self.w_min = self.w0 * float(self.cfg.get("w_min_frac", 0.1))
        # compartments: for each MBON, the DANs that synapse onto it (>= min synapses)
        min_syn = float(self.cfg.get("compartment_min_synapses", 3)) * 0.275
        self.dans_of: dict[int, np.ndarray] = {}
        self.kind_of: dict[int, str] = {}
        by_post: dict[int, list] = {}
        for d in np.nonzero(dan_rew | dan_pun)[0]:
            for j in range(indptr[d], indptr[d + 1]):
                q = indices[j]
                if mbon[q] and abs(net.data[j]) >= min_syn:
                    by_post.setdefault(int(q), []).append(int(d))
        for m in self.mbon_idx:
            ds = np.asarray(by_post.get(int(m), []), dtype=np.int64)
            self.dans_of[int(m)] = ds
            if len(ds):
                self.kind_of[int(m)] = "reward" if dan_rew[ds].sum() >= dan_pun[ds].sum() else "punish"
        # per-synapse compartment index (into a compact list of MBONs with DAN input)
        self.mbons_with_dan = np.array([m for m in self.mbon_idx if len(self.dans_of[int(m)])], dtype=np.int64)
        comp_of = {int(m): i for i, m in enumerate(self.mbons_with_dan)}
        self.syn_comp = np.array([comp_of.get(int(q), -1) for q in self.post], dtype=np.int64)
        keep = self.syn_comp >= 0
        self.syn, self.pre, self.post, self.w0, self.w_min, self.syn_comp = (a[keep] for a in (self.syn, self.pre, self.post, self.w0, self.w_min, self.syn_comp))
        # state
        self.elig = np.zeros(circuit.n, dtype=np.float32)
        self.kc_rest = np.zeros(len(self.kc_idx), dtype=np.float32)   # slow resting rate per KC (Hz)
        self.kc_rate = np.zeros(len(self.kc_idx), dtype=np.float32)   # fast smoothed rate per KC (Hz)
        self.dop = np.zeros(len(self.mbons_with_dan), dtype=np.float32)      # current dopamine per compartment (0..)
        self.dan_rate = np.zeros(len(self.mbons_with_dan), dtype=np.float32)  # smoothed DAN rate per compartment
        self.dan_rest = np.zeros(len(self.mbons_with_dan), dtype=np.float32)  # slow resting estimate
        self.warm = 0.0
        self.enabled = bool(self.cfg.get("enabled", True))
        self.total_depression = 0.0
        self.tau_elig = float(self.cfg.get("tau_elig_s", 2.0))
        self.tau_kc_rest = float(self.cfg.get("tau_kc_rest_s", 30.0))
        self.tau_kc_rate = float(self.cfg.get("tau_kc_rate_s", 0.3))
        self.kc_excess = float(self.cfg.get("kc_excess_factor", 1.5))
        self.kc_margin = float(self.cfg.get("kc_margin_hz", 1.0))
        self.dan_margin = float(self.cfg.get("dopamine_margin_hz", 2.0))
        self.tau_rate = float(self.cfg.get("tau_dan_s", 0.3))
        self.tau_rest = float(self.cfg.get("tau_rest_s", 60.0))
        self.tau_forget = float(self.cfg.get("tau_forget_s", 900.0))
        self.eta = float(self.cfg.get("eta", 0.15))
        self.dref = float(self.cfg.get("dopamine_ref_hz", 5.0))
        self.n_kc, self.n_mbon, self.n_syn = len(self.kc_idx), len(self.mbons_with_dan), len(self.syn)

    def reset(self) -> None:
        self.net.data[self.syn] = self.w0
        self.elig[:] = 0
        self.total_depression = 0.0

    def update(self, counts: np.ndarray) -> None:
        """Call after every simulation chunk with that chunk's spike counts."""
        dt = self.chunk_s
        # eligibility of KCs: only firing above each cell's own resting rate counts as "this odor"
        # (in the model the antennal lobe is tonically active; a real KC is silent at rest)
        kc_hz = counts[self.kc_idx].astype(np.float32) / dt
        self.kc_rate += (1 - np.exp(-dt / self.tau_kc_rate)) * (kc_hz - self.kc_rate)
        self.kc_rest += (dt / self.tau_kc_rest) * (kc_hz - self.kc_rest)
        self.elig *= np.exp(-dt / self.tau_elig)
        self.elig[self.kc_idx] += np.maximum(0.0, self.kc_rate - self.kc_excess * self.kc_rest - self.kc_margin) * dt
        # dopamine per compartment: smoothed DAN rate above its slow resting level
        a = 1 - np.exp(-dt / self.tau_rate)
        for i, m in enumerate(self.mbons_with_dan):
            ds = self.dans_of[int(m)]
            r = counts[ds].mean() / dt
            self.dan_rate[i] += a * (r - self.dan_rate[i])
        self.warm += dt
        if self.warm < 5.0:                       # let the resting estimate settle first
            self.dan_rest += (dt / 2.0) * (self.dan_rate - self.dan_rest)
            return
        self.dan_rest += (dt / self.tau_rest) * (self.dan_rate - self.dan_rest)
        self.dop = np.minimum(1.0, np.maximum(0.0, self.dan_rate - self.dan_rest - self.dan_margin) / self.dref)
        if not self.enabled or not len(self.syn):
            return
        # three-factor depression + forgetting
        w = self.net.data[self.syn]
        dw = self.eta * self.elig[self.pre] * self.dop[self.syn_comp] * dt
        w_new = np.maximum(self.w_min, w - dw * np.abs(self.w0))
        w_new += (self.w0 - w_new) * (dt / self.tau_forget)
        self.total_depression += float(np.sum(w - w_new))
        self.net.data[self.syn] = w_new.astype(np.float32)

    # ---- reporting ---------------------------------------------------------------------
    def summary(self) -> dict:
        w = self.net.data[self.syn] / np.where(self.w0 == 0, 1, self.w0)
        rew = np.array([self.kind_of.get(int(m)) == "reward" for m in self.mbons_with_dan])
        comp_frac = np.array([w[self.syn_comp == i].mean() if (self.syn_comp == i).any() else 1.0 for i in range(self.n_mbon)])
        return {
            "enabled": self.enabled, "n_kc": self.n_kc, "n_mbon": self.n_mbon, "n_syn": self.n_syn,
            "dopamine_reward": float(self.dop[rew].mean()) if rew.any() else 0.0,
            "dopamine_punish": float(self.dop[~rew].mean()) if (~rew).any() else 0.0,
            "weight_reward_comp": float(comp_frac[rew].mean()) if rew.any() else 1.0,   # KC->MBON strength in reward compartments (1 = naive)
            "weight_punish_comp": float(comp_frac[~rew].mean()) if (~rew).any() else 1.0,
            "kc_eligible": float((self.elig[self.kc_idx] > 0.5).mean()),
        }

    def mbon_sets(self, cell_type: np.ndarray) -> dict[str, list[str]]:
        """Wiring-defined valence sets: MBONs under reward DANs drive avoidance, under punishment DANs drive approach."""
        out = {"avoid": [], "approach": []}
        for m in self.mbons_with_dan:
            out["avoid" if self.kind_of[int(m)] == "reward" else "approach"].append(str(cell_type[m]))
        return {k: sorted(set(v)) for k, v in out.items()}
