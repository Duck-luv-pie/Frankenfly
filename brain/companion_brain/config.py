"""Configuration loading. The YAML is the single place where neuron groups, sensor mappings
and behavior readouts are defined; code only interprets it."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

BRAIN_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = BRAIN_DIR / "configs" / "default.yaml"


class Config(dict):
    """dict with attribute access and dotted `get`, so cfg.lif.dt_ms and cfg.get('body.port') work."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as e:  # pragma: no cover - trivial
            raise AttributeError(name) from e

    def path(self, key: str) -> Path:
        """Resolve a path from the config relative to the brain/ directory."""
        p = Path(self.get(key) if "." not in key else self.dotted(key))
        return p if p.is_absolute() else BRAIN_DIR / p

    def dotted(self, key: str, default: Any = None) -> Any:
        node: Any = self
        for part in key.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    with open(path or DEFAULT_CONFIG) as f:
        raw = yaml.safe_load(f)
    if overrides:
        raw = _deep_update(copy.deepcopy(raw), overrides)
    return _wrap(raw)


def apply_dotted(cfg: Config, values: dict) -> Config:
    """Set dotted keys (e.g. 'decode.motor.channels.turn.z_ref') on a loaded config, in place."""
    for key, v in values.items():
        node: Any = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if not isinstance(node.get(part), dict):
                node[part] = Config()
            node = node[part]
        node[parts[-1]] = _wrap(v)
    return cfg


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base
