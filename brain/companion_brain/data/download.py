"""Fetch the model-ready connectome (Shiu et al. 2024) and FlyWire annotations."""
from __future__ import annotations

import sys
from pathlib import Path

import requests

from ..config import Config


def download_all(cfg: Config, force: bool = False) -> list[Path]:
    raw_dir = cfg.path("data.raw_dir")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for key, spec in cfg.data.files.items():
        dest = raw_dir / spec["name"]
        expected = spec.get("size")
        if dest.exists() and not force and (expected is None or dest.stat().st_size == expected):
            print(f"[download] {key}: already present ({dest.stat().st_size:,} bytes)")
            out.append(dest)
            continue
        print(f"[download] {key}: {spec['url']}")
        try:
            _fetch(spec["url"], dest)
        except Exception as e:  # noqa: BLE001
            print(f"[download] FAILED ({e}). Download it by hand into {dest}:\n  {spec['url']}", file=sys.stderr)
            raise
        if expected is not None and dest.stat().st_size != expected:
            print(f"[download] warning: {dest.name} is {dest.stat().st_size:,} bytes, expected {expected:,}")
        out.append(dest)
    return out


def _fetch(url: str, dest: Path, chunk: int = 1 << 20) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for block in r.iter_content(chunk):
                f.write(block)
                done += len(block)
                if total:
                    print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        print()
    tmp.replace(dest)


def raw_paths(cfg: Config) -> dict[str, Path]:
    raw_dir = cfg.path("data.raw_dir")
    return {k: raw_dir / v["name"] for k, v in cfg.data.files.items()}


def have_data(cfg: Config) -> bool:
    return all(p.exists() for p in raw_paths(cfg).values())
