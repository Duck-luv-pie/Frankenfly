"""
retina.py  --  24-column fly retina encoder for the RoboMaster camera.

Two front ends, ONE output format, so the brain never knows which fed it:
  Retina.from_state(...)  ground truth poses  -> columns   (batched GPU training)
  Retina.from_boxes(...)  detector boxes (px) -> columns   (his sim images, real robot)

Camera (from Ducks): 320x240, pinhole, no distortion, 82 deg vertical,
98.43 deg horizontal, turret-mounted 23.1 cm above ground. Keep turret at
neutral pan/tilt so the camera is body-fixed.

Output dict (all torch tensors, B = batch):
  pres   (B, n_cols)  fraction of the column's angular span covered by a human, 0..1
  size   (B, n_cols)  angular width of the covering box / hfov, 0..1
  mot    (B, n_cols)  d(pres)/dt, clipped to [-1, 1] per second-ish
  loom_L (B,)         positive rate of width expansion, left hemisphere
  loom_R (B,)         positive rate of width expansion, right hemisphere

Injection into neurons (do this in senses.py, not here):
  LC11 / LC12 / LC15 (small-object) <- pres * (1 - size)   prefers small, fades near contact
  LC10a (tracking)                  <- pres * size         takes over mid-range
  LC4 / LPLC2 (looming)             <- loom_L / loom_R
  columns 0..n/2-1 = left visual field -> left optic lobe; the rest -> right.
"""
from __future__ import annotations

import math
import torch


class Camera:
    def __init__(self, width: int = 320, height: int = 240, hfov_deg: float = 98.43, vfov_deg: float = 82.0,
                 height_m: float = 0.231) -> None:
        self.W, self.H = width, height
        self.hfov = math.radians(hfov_deg)
        self.vfov = math.radians(vfov_deg)
        self.fx = (width / 2) / math.tan(self.hfov / 2)    # ~138.1 px
        self.fy = (height / 2) / math.tan(self.vfov / 2)   # ~138.1 px
        self.cx, self.cy = width / 2, height / 2
        self.height_m = height_m

    def px_to_az(self, x: torch.Tensor) -> torch.Tensor:
        """Pixel column -> azimuth in radians. + is to the robot's right."""
        return torch.atan((x - self.cx) / self.fx)


class Retina:
    def __init__(self, cam: Camera, n_cols: int = 24, ema_tau: float = 0.05,
                 device: str | torch.device = "cpu") -> None:
        assert n_cols % 2 == 0, "need an even column count for left/right split"
        self.cam, self.n, self.tau = cam, n_cols, ema_tau
        # uniform in ANGLE, not pixels
        self.edges = torch.linspace(-cam.hfov / 2, cam.hfov / 2, n_cols + 1,
                                    device=device)
        self.col_w = (self.edges[1] - self.edges[0]).item()
        self.device = device
        self._prev_pres = None
        self._prev_width = None   # (B, 2) summed angular width per hemisphere

    def reset(self) -> None:
        self._prev_pres = None
        self._prev_width = None

    # ---------------------------------------------------------------- shared
    def _from_angles(self, az0, az1, valid, dt):
        """
        az0, az1: (B, K) left/right azimuth of each box in radians.
        valid:    (B, K) bool, False for padding / absent humans.
        """
        B, K = az0.shape
        half = self.cam.hfov / 2
        az0 = az0.clamp(-half, half)
        az1 = az1.clamp(-half, half)
        width = (az1 - az0).clamp(min=0.0) * valid            # (B, K)
        valid = valid & (width > 0)

        lo = self.edges[:-1].view(1, 1, self.n)               # (1,1,n)
        hi = self.edges[1:].view(1, 1, self.n)
        ov = (torch.minimum(az1.unsqueeze(-1), hi)
              - torch.maximum(az0.unsqueeze(-1), lo)).clamp(min=0.0) / self.col_w
        ov = ov * valid.unsqueeze(-1)                         # (B, K, n)

        pres, best = ov.max(dim=1)                            # (B, n), which box covers
        size = torch.gather(width, 1, best) / self.cam.hfov   # (B, n)
        size = size * (pres > 0)

        # motion: change in presence per column
        if self._prev_pres is None or self._prev_pres.shape != pres.shape:
            mot = torch.zeros_like(pres)
        else:
            mot = ((pres - self._prev_pres) / max(dt, 1e-3)).clamp(-1.0, 1.0)

        # looming: positive expansion of summed width per hemisphere
        centre = 0.5 * (az0 + az1)
        left = (centre < 0) & valid
        w_L = (width * left).sum(dim=1)
        w_R = (width * (~left & valid)).sum(dim=1)
        w = torch.stack([w_L, w_R], dim=1)                    # (B, 2)
        if self._prev_width is None or self._prev_width.shape != w.shape:
            loom = torch.zeros_like(w)
        else:
            loom = ((w - self._prev_width) / max(dt, 1e-3)).clamp(min=0.0)

        # light temporal smoothing so single-frame flicker does not drive spikes
        a = min(1.0, dt / self.tau)
        if self._prev_pres is not None and self._prev_pres.shape == pres.shape:
            pres = a * pres + (1 - a) * self._prev_pres
        self._prev_pres, self._prev_width = pres.detach(), w.detach()

        return {"pres": pres, "size": size, "mot": mot,
                "loom_L": loom[:, 0], "loom_R": loom[:, 1]}

    # ------------------------------------------------------ ground truth path
    def from_state(self, robot_xy: torch.Tensor, robot_yaw: torch.Tensor, human_xy: torch.Tensor,
                   human_r: torch.Tensor, valid: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
        """
        robot_xy  (B, 2)   robot_yaw (B,)  radians, 0 = +x, CCW positive
        human_xy  (B, K, 2)  human_r (B, K) cylinder radius  valid (B, K) bool
        Bearing is measured so + = to the robot's RIGHT (matches image x).
        """
        d = human_xy - robot_xy.unsqueeze(1)                  # (B, K, 2)
        dist = d.norm(dim=-1).clamp(min=1e-3)
        bearing_ccw = torch.atan2(d[..., 1], d[..., 0]) - robot_yaw.unsqueeze(1)
        bearing_ccw = (bearing_ccw + math.pi) % (2 * math.pi) - math.pi
        bearing = -bearing_ccw                                # + = right
        half_w = torch.asin((human_r / dist).clamp(max=0.999))
        az0, az1 = bearing - half_w, bearing + half_w
        # anything entirely outside the FOV is absent
        half = self.cam.hfov / 2
        inside = valid & (az1 > -half) & (az0 < half)
        return self._from_angles(az0, az1, inside, dt)

    # -------------------------------------------------------- detector path
    def from_boxes(self, boxes: torch.Tensor, valid: torch.Tensor, dt: float) -> dict[str, torch.Tensor]:
        """
        boxes (B, K, 4) pixel [x0, y0, x1, y1] from a person detector
        valid (B, K) bool. Only x edges matter (width, not height: at 23 cm
        camera height a person's head leaves the frame inside ~2 m, so height
        saturates exactly where the chase gets close).
        """
        az0 = self.cam.px_to_az(boxes[..., 0])
        az1 = self.cam.px_to_az(boxes[..., 2])
        return self._from_angles(az0, az1, valid, dt)


if __name__ == "__main__":
    # smoke test: one human dead ahead at 2 m, radius 0.25 m
    cam = Camera()
    ret = Retina(cam)
    out = ret.from_state(robot_xy=torch.zeros(1, 2), robot_yaw=torch.zeros(1),
                         human_xy=torch.tensor([[[2.0, 0.0]]]),
                         human_r=torch.tensor([[0.25]]),
                         valid=torch.tensor([[True]]), dt=1 / 60)
    print("focal px", round(cam.fx, 1), "col width deg", round(math.degrees(ret.col_w), 2))
    print("pres", out["pres"].round(decimals=2))
    print("size", out["size"].round(decimals=3))
