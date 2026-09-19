"""Behavior state -> eye parameters for the two GC9A01 displays.

Eye parameters (per eye): px, py pupil position (-1..1), pr pupil radius (0..1),
ut / lt upper / lower lid openness (0..1), tint iris RGB."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .decode import Decoded


@dataclass
class Eye:
    px: float = 0.0
    py: float = 0.0
    pr: float = 0.45
    ut: float = 0.85
    lt: float = 0.85
    tint: tuple = (90, 200, 255)


# Base pose per state: (pr, ut, lt)
POSES = {
    "idle":     (0.45, 0.85, 0.85),
    "sleep":    (0.40, 0.08, 0.10),
    "escape":   (0.25, 1.00, 1.00),
    "freeze":   (0.35, 0.95, 0.95),
    "backward": (0.40, 0.55, 0.75),
    "threat":   (0.30, 0.55, 0.35),
    "groom":    (0.50, 0.45, 0.65),
    "land":     (0.50, 0.75, 0.85),
    "social":   (0.60, 0.90, 0.80),
    "track":    (0.45, 0.90, 0.85),
    "feed":     (0.55, 0.60, 0.70),
}


def eyes_for(d: Decoded, gaze_x: float, gaze_y: float, blink: bool = False) -> dict:
    pr, ut, lt = POSES.get(d.state, POSES["idle"])
    pr = float(np.clip(pr + 0.25 * d.arousal, 0.15, 0.9))
    # iris tint: valence -1 (red) .. 0 (cyan) .. 1 (green)
    v = d.valence
    if v >= 0:
        tint = (int(90 * (1 - v)), int(200 + 40 * v), int(255 * (1 - v) + 80 * v))
    else:
        tint = (int(90 + 165 * -v), int(200 * (1 + v)), int(255 * (1 + v)))
    if d.state == "threat":
        tint = (255, 120, 40)
    px, py = float(np.clip(gaze_x, -1, 1)), float(np.clip(gaze_y, -1, 1))
    if d.state == "escape":
        px, py = -px * 0.8, py * 0.5          # look away from the threat
    elif d.state == "backward":
        px, py = px * 0.3, 0.3
    elif d.state == "groom":
        px, py = 0.0, 0.5
    elif d.state == "sleep":
        px, py = 0.0, 0.3
    steer = d.lateral.get("track", 0.0) * 0.5 if d.state in ("track", "idle") else 0.0
    px = float(np.clip(px + steer, -1, 1))
    l = Eye(px, py, pr, ut, lt, tint)
    r = Eye(px, py, pr, ut, lt, tint)
    if d.state == "groom":  # asymmetric squint while grooming
        r.ut, r.lt = 0.3, 0.5
    return {"l": _pack(l), "r": _pack(r)}


def _pack(e: Eye) -> dict:
    d = asdict(e)
    d["tint"] = list(e.tint)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}
