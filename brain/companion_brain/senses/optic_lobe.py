"""Synthetic optic lobe: turns camera frames into the features the fly's visual projection
neurons are known to encode, separately for the left and right visual hemifields.

    loom_fast     rapid expansion of a dark blob        -> LC4, LPLC2  (escape)
    loom_slow     slower, sustained expansion           -> LC16        (backward walking)
    small_object  small moving dark object (+ position)  -> LC11, LC10a (object tracking)
    bar           moving vertical edge energy           -> LC12, LC15
    motion_energy overall change, used for sleep/wake only

All features are in 0..1. This is deliberately simple image processing that a laptop can do at
15 fps; tune gains in configs/default.yaml."""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class HemifieldFeatures:
    loom_fast: float = 0.0
    loom_slow: float = 0.0
    small_object: float = 0.0
    object_x: float = 0.0     # -1 (left edge of hemifield) .. 1 (right edge)
    object_y: float = 0.0     # -1 (top) .. 1 (bottom)
    bar: float = 0.0


@dataclass
class Features:
    left: HemifieldFeatures = field(default_factory=HemifieldFeatures)
    right: HemifieldFeatures = field(default_factory=HemifieldFeatures)
    motion_energy: float = 0.0
    object_x: float = 0.0     # full-field object azimuth -1..1 (for gaze), 0 if none
    object_y: float = 0.0
    object_strength: float = 0.0

    def side(self, s: str) -> HemifieldFeatures:
        return self.left if s == "left" else self.right

    def as_dict(self) -> dict:
        return {"L": vars(self.left), "R": vars(self.right), "motion": self.motion_energy,
                "obj": [self.object_x, self.object_y, self.object_strength]}


class OpticLobe:
    def __init__(self, width: int = 80, height: int = 60, fps: float = 15.0,
                 diff_thresh: int = 25, small_max_frac: float = 0.05, loom_gain: float = 6.0,
                 loom_min_frac: float = 0.02, loom_min_rate: float = 0.25, slow_tau_s: float = 0.6):
        self.w, self.h, self.fps = width, height, fps
        self.diff_thresh = diff_thresh
        self.small_max_area = small_max_frac * (width // 2) * height
        self.loom_gain = loom_gain
        self.loom_min_frac = loom_min_frac
        self.loom_min_rate = loom_min_rate          # hemifield fraction / s below which growth is not an approach
        self.prev_cent = [(0.0, 0.0), (0.0, 0.0)]
        self.prev_size = [0.0, 0.0]
        self.slow_alpha = 1.0 / (slow_tau_s * fps)
        self.prev: np.ndarray | None = None
        self.prev_area = [0.0, 0.0]
        self.slow = [0.0, 0.0]

    def reset(self) -> None:
        self.prev = None
        self.prev_area = [0.0, 0.0]
        self.prev_size = [0.0, 0.0]
        self.slow = [0.0, 0.0]

    def process(self, frame: np.ndarray) -> Features:
        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        f = Features()
        if self.prev is None:
            self.prev = frame
            return f
        diff = cv2.absdiff(frame, self.prev)
        moving = (diff > self.diff_thresh).astype(np.uint8)
        f.motion_energy = float(np.clip(moving.mean() * 8.0, 0, 1))
        half = self.w // 2
        best_obj = (0.0, 0.0, 0.0)
        for si, (name, sl) in enumerate((("left", slice(0, half)), ("right", slice(half, self.w)))):
            hf = f.side(name)
            m = moving[:, sl]
            dark = ((frame[:, sl] < 100) & (m > 0)).astype(np.uint8)  # moving *dark* pixels
            n, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n > 1:
                areas = stats[1:, cv2.CC_STAT_AREA]
                k = int(np.argmax(areas)) + 1
                area = float(areas[k - 1])
                # --- looming: radial expansion of an already-visible region whose centre stays put.
                # Translation (an object crossing, or the world sliding past a moving fly) moves the
                # centroid about as fast as the edges; a real approach grows the box around a fixed centre.
                hemi_area = half * self.h
                cx, cy = cent[k]
                bw, bh = stats[k, cv2.CC_STAT_WIDTH], stats[k, cv2.CC_STAT_HEIGHT]
                pa, pc, ps = self.prev_area[si], self.prev_cent[si], self.prev_size[si]
                if pa >= self.loom_min_frac * hemi_area and area > pa:
                    growth = (area - pa) * self.fps / hemi_area                     # hemifield fraction per second
                    d_size = (bw + bh) - ps                                          # px of edge expansion
                    d_cent = abs(cx - pc[0]) + abs(cy - pc[1])                       # px of centre motion
                    if d_size > 0 and d_cent < 0.6 * d_size:                          # expansion, not translation
                        hf.loom_fast = float(np.clip((growth - self.loom_min_rate) * self.loom_gain, 0, 1))
                self.prev_area[si] = area
                self.prev_cent[si] = (float(cx), float(cy))
                self.prev_size[si] = float(bw + bh)
                # --- small object: smallest blob with a few pixels, prefer dark
                small = [(i + 1, a) for i, a in enumerate(areas) if 3 <= a <= self.small_max_area]
                if small:
                    i, a = max(small, key=lambda t: t[1])
                    cx, cy = cent[i]
                    darkness = dark[stats[i, cv2.CC_STAT_TOP]:stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT],
                                    stats[i, cv2.CC_STAT_LEFT]:stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH]].mean()
                    hf.small_object = float(np.clip(0.4 + 0.6 * darkness, 0, 1))
                    hf.object_x = float(cx / half * 2 - 1)
                    hf.object_y = float(cy / self.h * 2 - 1)
                    gx = float((cx + sl.start) / self.w * 2 - 1)
                    if hf.small_object > best_obj[2]:
                        best_obj = (gx, hf.object_y, hf.small_object)
            else:
                self.prev_area[si] = 0.0
                self.prev_size[si] = 0.0
            # --- slow looming: low-pass of positive growth
            self.slow[si] += self.slow_alpha * (hf.loom_fast - self.slow[si])
            hf.loom_slow = float(np.clip(self.slow[si] * 2.0, 0, 1))
            # --- bars: vertical edges that moved
            sob = cv2.Sobel(frame[:, sl], cv2.CV_32F, 1, 0, ksize=3)
            hf.bar = float(np.clip((np.abs(sob) * m).mean() / 200.0, 0, 1))
        f.object_x, f.object_y, f.object_strength = best_obj
        self.prev = frame
        return f


def features_to_rates(f: Features, cfg) -> dict[tuple[str, str], float]:
    """Map features to (group, side) -> Hz using cfg.senses.features."""
    max_hz = float(cfg.senses.max_rate_hz)
    out: dict[tuple[str, str], float] = {}
    for feat, spec in cfg.senses.features.items():
        for side in ("left", "right"):
            hf = f.side(side)
            val = getattr(hf, feat, 0.0) * float(spec.get("gain", 1.0))
            w = float(spec.get("azimuth_weight", 0.0))
            if w > 0 and feat == "small_object" and val > 0:
                # retinotopy: an object far out in the hemifield drives its side harder than one near the midline,
                # so the steering it evokes is proportional to the error, not bang-bang. object_x is +1 at the
                # hemifield's right edge, so the periphery is object_x=-1 on the left and +1 on the right.
                ecc = (1.0 - hf.object_x) / 2 if side == "left" else (1.0 + hf.object_x) / 2
                val *= (1.0 - w) + w * max(0.0, min(1.0, ecc))
            hz = float(np.clip(val, 0, 1)) * max_hz
            if hz > 0:
                for g in spec["groups"]:
                    out[(g, side)] = max(out.get((g, side), 0.0), hz)
    return out
