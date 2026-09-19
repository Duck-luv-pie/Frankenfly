#!/usr/bin/env python3
"""Synthesize the Companion's sounds for the DFPlayer SD card.

    uv run --project brain python tools/make_sounds.py [--out assets/sounds]

Writes WAV files; if `ffmpeg` is on PATH they are also converted to the MP3s the DFPlayer needs
(/mp3/0001.mp3 ...). Without ffmpeg, convert the WAVs with any tool and name them the same way.

Tracks (see brain/configs/default.yaml `decode.sounds`):
  0001 escape   - wing buzz: ~200 Hz wingbeat with harmonics, rising then fading
  0002 backward - short descending chirp
  0003 groom    - soft brushing noise bursts
  0004 threat   - harsh buzz with tremolo
  0005 social   - Drosophila courtship pulse song: pulses every ~35 ms (Bennet-Clark 1969)
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

SR = 22050


def env(n: int, attack: float, release: float) -> np.ndarray:
    e = np.ones(n)
    a, r = int(attack * SR), int(release * SR)
    e[:a] = np.linspace(0, 1, a)
    e[n - r:] = np.linspace(1, 0, r)
    return e


def wingbeat(seconds: float, f0: float = 200.0, wobble: float = 0.0) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    f = f0 * (1 + 0.15 * np.sin(2 * np.pi * 0.8 * t)) * (1 + wobble * np.sin(2 * np.pi * 27 * t))
    phase = 2 * np.pi * np.cumsum(f) / SR
    s = sum(np.sin(k * phase) / k for k in (1, 2, 3, 5))
    return s / np.abs(s).max()


def escape() -> np.ndarray:
    s = wingbeat(1.2, 210)
    t = np.arange(len(s)) / SR
    return s * env(len(s), 0.02, 0.5) * (0.6 + 0.4 * np.exp(-3 * t))


def backward() -> np.ndarray:
    t = np.arange(int(0.5 * SR)) / SR
    f = 900 * np.exp(-2.5 * t) + 180
    s = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return s * env(len(s), 0.01, 0.2)


def groom() -> np.ndarray:
    rng = np.random.default_rng(0)
    out = np.zeros(int(1.5 * SR))
    for k in range(6):
        start = int(k * 0.24 * SR)
        n = int(0.12 * SR)
        noise = rng.normal(size=n)
        noise = np.convolve(noise, np.ones(8) / 8, mode="same")  # low-pass
        out[start:start + n] += noise * env(n, 0.03, 0.06)
    return out / np.abs(out).max()


def threat() -> np.ndarray:
    s = wingbeat(1.0, 160, wobble=0.35)
    t = np.arange(len(s)) / SR
    trem = 0.6 + 0.4 * np.sign(np.sin(2 * np.pi * 14 * t))
    return s * trem * env(len(s), 0.01, 0.3)


def pulse_song(seconds: float = 2.0, ipi_ms: float = 35.0, pulse_hz: float = 170.0) -> np.ndarray:
    """Courtship pulse song: brief ~3-cycle pulses at the inter-pulse interval."""
    n = int(seconds * SR)
    out = np.zeros(n)
    pulse_len = int(3 / pulse_hz * SR)
    tp = np.arange(pulse_len) / SR
    pulse = np.sin(2 * np.pi * pulse_hz * tp) * np.hanning(pulse_len)
    t = 0.05
    while t + pulse_len / SR < seconds:
        i = int(t * SR)
        out[i:i + pulse_len] += pulse
        t += ipi_ms / 1000 * (1 + 0.05 * np.random.default_rng(int(t * 1000)).normal())
    return out * env(n, 0.05, 0.3)


TRACKS = {1: escape, 2: backward, 3: groom, 4: threat, 5: pulse_song}


def write_wav(path: Path, s: np.ndarray) -> None:
    pcm = (np.clip(s, -1, 1) * 32767 * 0.9).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=Path(__file__).resolve().parent.parent / "assets" / "sounds")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ffmpeg = shutil.which("ffmpeg")
    for n, fn in TRACKS.items():
        wav = out / f"{n:04d}.wav"
        write_wav(wav, fn())
        if ffmpeg:
            mp3 = out / f"{n:04d}.mp3"
            subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame", "-b:a", "96k", str(mp3)], check=True)
            print(f"wrote {mp3.name}")
        else:
            print(f"wrote {wav.name} (no ffmpeg on PATH: convert to {n:04d}.mp3 yourself)")
    print(f"copy the .mp3 files to the micro SD card as /mp3/0001.mp3 ... (FAT32)")


if __name__ == "__main__":
    main()
