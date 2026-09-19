"""The hunter's brain: a compact recurrent network shaped like the fly's hunting circuit.

  thermal stream   the two hot cells -> VP2 projection neurons -> a Kenyon-cell expansion (a fixed
                   random projection with k-winners-take-all, a sparse code as in the mushroom body)
                   -> MBON readout (learned valence features)
  visual stream    the retinotopic LC columns (object + motion per column) -> two 1-D convolutions
                   over azimuth (the optic glomeruli / AOTU) -> a visual feature vector
  central complex  a GRU over the merged senses and body state (heading memory, target commitment)
  descending       a policy head (mean forward and turn drive, with a learned log-std) and a value
                   head; a direct sensory -> motor shortcut like the innate LC10a -> DNa02 pursuit,
                   so the network starts close to a reflex and learns the rest

Everything the trainer saves is in `save`: the weights, the network sizes and the observation spec,
so `load` rebuilds the same fly anywhere (GPU or CPU)."""
from __future__ import annotations

import math
from pathlib import Path

import torch
from torch import nn

from .arena import ObsSpec


class AzimuthConv(nn.Conv1d):
    """A 1-D convolution over the retinal columns. On a GPU it is `conv1d`; on the CPU, where conv1d on a
    2-channel, 24-column input takes 15x longer than the matmul it stands for, it is unfold + einsum."""

    def __init__(self, cin: int, cout: int, k: int, stride: int):
        super().__init__(cin, cout, kernel_size=k, stride=stride, padding=k // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cpu":
            return super().forward(x)
        cols = nn.functional.pad(x, (self.padding[0], self.padding[0])).unfold(2, self.kernel_size[0], self.stride[0])   # [B, cin, L, k]
        return torch.einsum("bclk,ock->bol", cols, self.weight) + self.bias[None, :, None]


class HunterNet(nn.Module):
    def __init__(self, spec: ObsSpec, hidden: int = 128, kc: int = 256, kc_active: int = 8, visual: int = 32, log_std_init: float = -0.5):
        super().__init__()
        self.spec, self.hidden, self.kc_active = spec, int(hidden), int(kc_active)
        self.sizes = {"hidden": int(hidden), "kc": int(kc), "kc_active": int(kc_active), "visual": int(visual)}
        # thermal: 2 hot cells -> 16 PNs -> KC (fixed, sparse) -> 8 MBON features
        self.pn = nn.Sequential(nn.Linear(2, 16), nn.Tanh())
        kc_w = torch.randn(16, kc) * (1.0 / 16 ** 0.5)
        self.register_buffer("kc_w", kc_w)
        self.mbon = nn.Linear(kc, 8)
        # visual: [bins, 2] columns -> conv over azimuth -> feature vector
        self.conv1 = AzimuthConv(2, 16, 5, 1)
        self.conv2 = AzimuthConv(16, 16, 5, 2)
        with torch.no_grad():
            n_vis = self.vis(torch.zeros(1, 2, spec.bins)).shape[1]
        self.vis_out = nn.Sequential(nn.Linear(n_vis, visual), nn.ReLU())
        n_merge = 8 + 2 + visual + 4
        self.merge = nn.Sequential(nn.Linear(n_merge, hidden), nn.ReLU())
        self.gru = nn.GRU(hidden, hidden)                     # one fused call over a whole sequence (cuDNN on CUDA)
        self.pi = nn.Linear(hidden, 2)
        self.reflex = nn.Linear(n_merge, 2)                   # sensory -> motor shortcut (innate pursuit)
        self.v = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.zeros_(self.pi.bias); nn.init.orthogonal_(self.pi.weight, gain=0.01)
        nn.init.zeros_(self.reflex.bias); nn.init.orthogonal_(self.reflex.weight, gain=0.01)
        nn.init.orthogonal_(self.v.weight, gain=1.0); nn.init.zeros_(self.v.bias)

    def vis(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.conv2(torch.relu(self.conv1(x)))).flatten(1)

    def senses(self, obs: torch.Tensor) -> torch.Tensor:
        heat, vis, body = self.spec.split(obs)
        pn = self.pn(heat)
        kc = torch.relu(pn @ self.kc_w)
        if 0 < self.kc_active < kc.shape[-1]:                # k winners take all: the sparse Kenyon-cell code
            thresh = kc.topk(self.kc_active, dim=-1).values[..., -1:]
            kc = torch.where(kc >= thresh, kc, torch.zeros_like(kc))
        mbon = self.mbon(kc)
        lead = vis.shape[:-2]                                 # [B] or [T, B]: the convolution wants one batch axis
        v = self.vis_out(self.vis(vis.reshape(-1, *vis.shape[-2:]).transpose(-1, -2).contiguous())).reshape(*lead, -1)
        return torch.cat([mbon, heat, v, body], dim=-1)

    def initial_state(self, n: int, device) -> torch.Tensor:
        return torch.zeros(n, self.hidden, device=device)

    def forward(self, obs: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One tick. obs [B, D], h [B, hidden] -> (action mean [B, 2], value [B], new hidden)."""
        s = self.senses(obs)
        out, h1 = self.gru(self.merge(s)[None], h[None])
        h = h1[0]
        mean = torch.tanh(self.pi(h) + self.reflex(s))
        return mean, self.v(h)[:, 0], h

    def sequence(self, obs: torch.Tensor, h0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """A whole rollout at once for training. obs [T, B, D], h0 [B, hidden] -> (means [T, B, 2], values [T, B]).
        The hidden state runs straight through episode boundaries, as it does during the rollout: a new episode
        announces itself in the body inputs (speed 0, time 0), so the fly learns to start afresh on its own."""
        s = self.senses(obs)
        out, _ = self.gru(self.merge(s), h0[None])
        return torch.tanh(self.pi(out) + self.reflex(s)), self.v(out)[..., 0]

    # the Gaussian policy, written out (torch.distributions costs more than the network on small batches)
    def log_prob(self, mean: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = (action - mean) / self.log_std.exp()
        return (-0.5 * z * z - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, h: torch.Tensor, deterministic: bool = False):
        mean, value, h = self(obs, h)
        action = mean if deterministic else mean + self.log_std.exp() * torch.randn_like(mean)
        return action, self.log_prob(mean, action), value, h

    # ---- files ----------------------------------------------------------------------------------
    def save(self, path: Path | str, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "sizes": self.sizes, "bins": self.spec.bins, "meta": meta or {}}, path)
        return path

    @classmethod
    def load(cls, path: Path | str, device="cpu") -> tuple["HunterNet", dict]:
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        net = cls(ObsSpec(int(ck["bins"])), **ck["sizes"])
        net.load_state_dict(ck["state_dict"])
        return net.to(device).eval(), ck.get("meta", {})


def build(spec: ObsSpec, net_cfg: dict, log_std_init: float = -0.5) -> HunterNet:
    n = dict(net_cfg or {})
    return HunterNet(spec, hidden=int(n.get("hidden", 128)), kc=int(n.get("kc", 256)), kc_active=int(n.get("kc_active", 8)),
                     visual=int(n.get("visual", 32)), log_std_init=log_std_init)


class EvaderNet(nn.Module):
    """The runners' brain, one network shared by every person: a small feed-forward policy over the
    runner's own-frame view of the fly, the walls and the nearest other person (`HumanObs`) -> a run
    drive and a turn drive (Gaussian with a learned spread) and a value. Runners have no memory: they
    react, but they are faster than the fly."""

    def __init__(self, obs_size: int, hidden: int = 64, log_std_init: float = -0.5):
        super().__init__()
        self.obs_size, self.sizes = int(obs_size), {"hidden": int(hidden)}
        self.body = nn.Sequential(nn.Linear(obs_size, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh())
        self.pi, self.v = nn.Linear(hidden, 2), nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.full((2,), float(log_std_init)))
        nn.init.orthogonal_(self.pi.weight, gain=0.01); nn.init.zeros_(self.pi.bias)
        nn.init.orthogonal_(self.v.weight, gain=1.0); nn.init.zeros_(self.v.bias)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """obs [..., D] -> (action mean [..., 2], value [...])."""
        z = self.body(obs)
        return torch.tanh(self.pi(z)), self.v(z)[..., 0]

    def log_prob(self, mean: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        z = (action - mean) / self.log_std.exp()
        return (-0.5 * z * z - self.log_std - 0.5 * math.log(2 * math.pi)).sum(-1)

    def entropy(self) -> torch.Tensor:
        return (0.5 + 0.5 * math.log(2 * math.pi) + self.log_std).sum()

    @torch.no_grad()
    def act(self, obs: torch.Tensor, deterministic: bool = False):
        mean, value = self(obs)
        action = mean if deterministic else mean + self.log_std.exp() * torch.randn_like(mean)
        return action, self.log_prob(mean, action), value

    def save(self, path: Path | str, meta: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "sizes": self.sizes, "obs_size": self.obs_size, "meta": meta or {}}, path)
        return path

    @classmethod
    def load(cls, path: Path | str, device="cpu") -> tuple["EvaderNet", dict]:
        ck = torch.load(Path(path), map_location="cpu", weights_only=False)
        net = cls(int(ck["obs_size"]), **ck["sizes"])
        net.load_state_dict(ck["state_dict"])
        return net.to(device).eval(), ck.get("meta", {})


def build_evader(spec, net_cfg: dict, log_std_init: float = -0.5) -> EvaderNet:
    n = dict(net_cfg or {})
    return EvaderNet(spec.size, hidden=int(n.get("hidden", 64)), log_std_init=log_std_init)
