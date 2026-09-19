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
    "baseline":     "Real wiring. Fifteen thousand neurons, nothing trained.",
    "lesion_lc10a": "Tracking neurons removed. I cannot see you.",
    "restore":      "Tracking restored.",
    "lobotomy":     "Learning wiped. The connectome remains.",
    "restore_plasticity": "Learned synapses back.",
    "shuffle_on":  "Same neurons, same connections, random targets. Nothing reaches my legs.",
    "shuffle_off": "Real wiring again.",
    "quarter":     "A quarter of the tracking population gone.",
    "searching":   "I have lost you. Searching.",
    "contact":     "Found you.",
}

# ElevenLabs "Rachel"; any voice id works, and the id is not a secret
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"


def path_for(key):
    return os.path.join(VOICE_DIR, f"{key}.mp3")


def generate(keys=None, voice_id=DEFAULT_VOICE, model="eleven_turbo_v2_5"):
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY first (the audio is generated once and then played from disk)")
    import requests
    os.makedirs(VOICE_DIR, exist_ok=True)
    made = 0
    for name in (keys or LINES):
        text = LINES[name]
        r = requests.post(f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
                          headers={"xi-api-key": key, "accept": "audio/mpeg"},
                          json={"text": text, "model_id": model,
                                "voice_settings": {"stability": 0.4, "similarity_boost": 0.8}},
                          timeout=30)
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--say", default=None, choices=sorted(LINES))
    ap.add_argument("--voice-id", default=DEFAULT_VOICE)
    a = ap.parse_args()
    if a.generate:
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
