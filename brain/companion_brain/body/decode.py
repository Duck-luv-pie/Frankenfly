"""Turn readout-group firing rates into a behavior state and continuous modulators."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Decoded:
    state: str = "idle"
    scores: dict = field(default_factory=dict)      # behavior -> 0..1
    lateral: dict = field(default_factory=dict)     # behavior -> -1 (left) .. 1 (right)
    valence: float = 0.0
    arousal: float = 0.0
    reward: float = 0.0


class Decoder:
    def __init__(self, cfg):
        self.cfg = cfg.decode
        self.hold_until: dict[str, float] = {}
        self.current = "idle"
        self.current_priority = -1

    def _group_rate(self, rates: dict, groups: list[str], side: str = "all") -> float:
        vals = [rates[g][side] for g in groups if g in rates]
        return float(np.mean(vals)) if vals else 0.0

    def decode(self, rates: dict, asleep: bool = False, now: float | None = None) -> Decoded:
        now = time.time() if now is None else now
        d = Decoded()
        best, best_pri = None, -1
        for name, spec in self.cfg.behaviors.items():
            r = self._group_rate(rates, spec["groups"])
            score = float(np.clip(r / float(spec["ref_hz"]), 0, 1))
            d.scores[name] = score
            rl = self._group_rate(rates, spec["groups"], "left")
            rr = self._group_rate(rates, spec["groups"], "right")
            d.lateral[name] = float((rr - rl) / (rr + rl)) if (rr + rl) > 0 else 0.0
            if score >= float(spec["min_score"]):
                self.hold_until[name] = now + float(spec["hold_ms"]) * 1e-3
            if self.hold_until.get(name, 0.0) > now and int(spec["priority"]) > best_pri:
                best, best_pri = name, int(spec["priority"])
        d.state = best or ("sleep" if asleep else "idle")
        m = self.cfg.modulators
        d.valence = self._bipolar(rates, m["valence"])
        d.arousal = float(np.clip(self._group_rate(rates, [m["arousal"]["plus"]]) / float(m["arousal"]["ref_hz"]), 0, 1))
        d.reward = self._bipolar(rates, m["reward"])
        return d

    def _bipolar(self, rates: dict, spec: dict) -> float:
        plus = self._group_rate(rates, [spec["plus"]]) if "plus" in spec else 0.0
        minus = self._group_rate(rates, [spec["minus"]]) if "minus" in spec else 0.0
        return float(np.clip((plus - minus) / float(spec["ref_hz"]), -1, 1))
