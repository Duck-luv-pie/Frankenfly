"""
voice.py -- the fly says what just happened to it, out of cached audio.

    export ELEVENLABS_API_KEY=...
    python scripts/voice.py --generate            # synthesize once, into assets/voice/*.mp3
    python scripts/voice.py --list                # what it can say, and whether the audio exists
    python scripts/voice.py --say lesion_lc10a    # check a line

Then `scripts/demo.py --voice` speaks a line whenever the demo changes the brain: a lesion, a restore,
the wiring shuffle, a lobotomy. A demo table is loud and a judge is watching the robot, not the screen,
so the useful thing for the fly to say is what was just done to it, not a joke.

Audio is generated once and played from disk, so the demo needs no network and no API key at the venue,
and a failed or missing file is silently skipped rather than stalling the control loop.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

VOICE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "voice")

LINES = {
    "baseline":     "[calm] Real wiring. Fifteen thousand neurons. Nothing trained.",
    "lesion_lc10a": "[quietly] Tracking neurons removed... [pause] I cannot see you.",
    "restore":      "[relieved] Tracking restored.",
    "lobotomy":     "[flat] Learning wiped. [pause] The connectome remains.",
    "restore_plasticity": "[warm] Learned synapses back.",
    "shuffle_on":  "[slowly] Same neurons. Same connections. Random targets. [pause] Nothing reaches my legs.",
    "shuffle_off": "[relieved] Real wiring again.",
    "quarter":     "[matter-of-fact] A quarter of the tracking population, gone.",
    "searching":   "[uncertain] I have lost you. [pause] Searching.",
    "contact":     "[pleased] Found you.",
}

# ElevenLabs "George", a warm storyteller: with eleven_v3 the lines are delivered, not read.
# It must be a *premade* voice: free accounts are refused library voices over the API with HTTP 402.
# `python scripts/voice.py --voices` lists what this account can actually use. The id is not a secret.
DEFAULT_VOICE = "JBFqnCBsd6RMkjVDRZzb"
DEFAULT_SPEED = 1.0


def current_voice_id():
    """ELEVENLABS_VOICE_ID from .env, or George if Taka has not picked one yet. Read fresh on every
    call (not a module constant) so a test or a caller can monkeypatch os.environ after load_dotenv()."""
    return os.environ.get("ELEVENLABS_VOICE_ID", "").strip() or DEFAULT_VOICE


def current_voice_speed():
    """ELEVENLABS_VOICE_SPEED from .env, or 1.0. Falls back to 1.0 on anything that does not parse."""
    raw = os.environ.get("ELEVENLABS_VOICE_SPEED", "").strip()
    if not raw:
        return DEFAULT_SPEED
    try:
        return float(raw)
    except ValueError:
        return DEFAULT_SPEED


def load_dotenv():
    """Read the project's .env into os.environ without overwriting anything already set.

    The working directory is not a reliable place to look: a shell that wandered into a sibling repo
    silently wrote a live database password into the wrong file once, so this resolves .env relative to
    this source file instead."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


FALLBACK = {}


def path_for(key):
    return os.path.join(VOICE_DIR, f"{key}.mp3")


def list_voices():
    """What this account may use. Free plans see the premade voices only."""
    import requests
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY first")
    r = requests.get("https://api.elevenlabs.io/v1/voices", headers={"xi-api-key": key}, timeout=20)
    r.raise_for_status()
    return [v for v in r.json().get("voices", []) if v.get("category") == "premade"]


def generate(keys=None, voice_id=None, model="eleven_v3", speed=None):
    """voice_id defaults to ELEVENLABS_VOICE_ID (current_voice_id()); speed to ELEVENLABS_VOICE_SPEED
    (current_voice_speed()), clamped to the 0.7-1.2 range the API accepts for voice_settings.speed."""
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY first (the audio is generated once and then played from disk)")
    voice_id = voice_id or current_voice_id()
    speed = speed if speed is not None else current_voice_speed()
    speed = max(0.7, min(1.2, speed))
    import requests
    os.makedirs(VOICE_DIR, exist_ok=True)
    made = 0
    for name in (keys or LINES):
        text = LINES[name]
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                          headers={"xi-api-key": key, "accept": "audio/mpeg"},
                          json={"text": text, "model_id": model,
                                "voice_settings": {"stability": 0.35, "similarity_boost": 0.8,
                                                    "style": 0.4, "speed": speed}},
                          timeout=30)
        if r.status_code == 402 and voice_id != FALLBACK.get("id"):
            # a library voice on a free plan; drop to the first premade voice this account has
            prem = list_voices()
            if prem:
                FALLBACK["id"] = voice_id = prem[0]["voice_id"]
                print(f"  (voice refused on this plan, switching to {prem[0]['name']})")
                continue
        if r.status_code != 200:
            print(f"  {name}: HTTP {r.status_code} {r.text[:120]}")
            continue
        with open(path_for(name), "wb") as f:
            f.write(r.content)
        made += 1
        print(f"  {name}: {len(r.content) // 1024} KB  \"{text}\"")
    print(f"{made}/{len(keys or LINES)} lines written to {VOICE_DIR}")


def player_cmd():
    for exe, args in (("afplay", []), ("ffplay", ["-nodisp", "-autoexit", "-loglevel", "quiet"]),
                      ("mpg123", ["-q"]), ("aplay", [])):
        if shutil.which(exe):
            return [exe] + args
    return None


class Voice:
    """Fire-and-forget playback. Never blocks the control loop, never raises into it."""

    def __init__(self, enabled=True, min_gap_s=1.5):
        self.cmd = player_cmd() if enabled else None
        self.min_gap, self.last = min_gap_s, 0.0
        self.missing = set()
        if enabled and self.cmd is None:
            print("no audio player found (afplay, ffplay, mpg123 or aplay); voice disabled", flush=True)
        elif enabled:
            have = sum(os.path.exists(path_for(k)) for k in LINES)
            print(f"voice: {have}/{len(LINES)} lines cached in assets/voice, playing with {self.cmd[0]}", flush=True)

    def say(self, key):
        if self.cmd is None:
            return False
        p = path_for(key)
        if not os.path.exists(p):
            if key not in self.missing:
                self.missing.add(key)
                print(f"voice: no audio for '{key}' (run scripts/voice.py --generate)", flush=True)
            return False
        now = time.time()
        if now - self.last < self.min_gap:
            return False
        self.last = now
        threading.Thread(target=lambda: subprocess.run(self.cmd + [p], stdout=subprocess.DEVNULL,
                                                       stderr=subprocess.DEVNULL, check=False),
                         daemon=True).start()
        return True


if __name__ == "__main__":
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--voices", action="store_true", help="list the voices this account can use")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--say", default=None, choices=sorted(LINES))
    ap.add_argument("--voice-id", default=None,
                    help="override ELEVENLABS_VOICE_ID for this run (default: .env, then George)")
    a = ap.parse_args()
    if a.voices:
        for v in list_voices():
            lab = (v.get("labels") or {})
            print(f"  {v['voice_id']:24s} {v.get('name','')[:40]:42s} "
                  f"{lab.get('gender','')}/{lab.get('accent','')}")
    elif a.generate:
        generate(voice_id=a.voice_id)
    elif a.say:
        v = Voice()
        print(f"\"{LINES[a.say]}\"")
        if v.say(a.say):
            time.sleep(4)
    else:
        print(f"{'line':22s} {'cached':>7s}  text")
        for k, t in LINES.items():
            print(f"{k:22s} {'yes' if os.path.exists(path_for(k)) else 'no':>7s}  \"{t}\"")
        print(f"\naudio in {VOICE_DIR}; player: {(player_cmd() or ['none'])[0]}")
