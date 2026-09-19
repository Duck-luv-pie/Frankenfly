"""
talk.py -- talk to the fly. An ElevenLabs voice agent whose only knowledge of the world is the live brain.

    python scripts/talk.py --create-agent          # once: registers the agent and its tools, stores the id in .env
    python scripts/talk.py --check                 # read the live frame and print what every tool would answer
    python scripts/talk.py                         # the conversation: microphone in, the fly's voice out

Run it next to the demo (scripts/demo.py writes logs/demo_live.jsonl every frame and listens for hotkeys on
udp://127.0.0.1:9600). A judge asks "what do you see?" and the agent calls what_do_you_see(), which reads
the retina columns from the latest frame and answers with the bearing and rough distance of the person.
"What just happened to you?" reads the descending-neuron rates. "Remove its eye" sends hotkey 2 to the
demo, then reads the rates again so the fly can report the difference in its own numbers.

What the language model is and is not. It is an interpreter with a microphone. It can read the brain's
state through the tools below and it can press the same hotkeys a presenter presses. It is not in the
control loop: nothing it says or does reaches the sensory input or the wheels, and the brain would run
identically with this script closed. That is the same rule as CLAUDE.md #5 (the loop is local) and #1
(no new networks), and it is why the agent is told to quote the tools' numbers and never to invent a
sensation the tools did not return.

Ten seconds of a judge holding a conversation with a real connectome is the ElevenLabs entry; the cached
narration lines (scripts/voice.py) stay as the fallback when there is no microphone.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from voice import load_dotenv  # noqa: E402

FRAMES = os.path.join(ROOT, "logs", "demo_live.jsonl")
HOTKEY_PORT = 9600
HFOV_DEG = 98.43
N_COLS = 24
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"          # George, the same voice as the cached narration

# hotkeys are the demo's own; the agent presses them, it does not reimplement them
KEYS = {"eye": ("2", "3"), "learning": ("4", "5")}
KEY_BASELINE, KEY_SHUFFLE, KEY_REWARD, KEY_PUNISH = "1", "6", "r", "p"


# --------------------------------------------------------------------------------------------------
# reading the brain

def last_frame(path=FRAMES):
    """The most recent frame the demo wrote, without reading the whole file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().splitlines()
        for raw in reversed(lines):
            raw = raw.strip()
            if raw:
                return json.loads(raw)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None
    return None


def col_azimuth(i):
    """Centre azimuth of retina column i in degrees; + is the fly's right (CLAUDE.md conventions)."""
    return (i + 0.5) / N_COLS * HFOV_DEG - HFOV_DEG / 2


class Brain:
    """The tools. Each returns plain data; the agent turns it into speech."""

    def __init__(self, port=HOTKEY_PORT):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.events = []                        # (t, what) for everything pressed through this script
        self.shuffled = False                   # hotkey 6 toggles, so the state is tracked here
        self.lesioned = {"eye": False, "learning": False}

    def press(self, key, what):
        self.sock.sendto(key.encode(), ("127.0.0.1", self.port))
        self.events.append((time.time(), what))
        return what

    # ---- sensory ----
    def what_do_you_see(self, params=None):
        f = last_frame()
        if f is None:
            return {"status": "no frames; the demo is not running"}
        pres = f.get("pres") or []
        if not pres or max(pres) < 0.05:
            return {"person_in_view": False, "frame_age_s": None,
                    "note": "no person on any retina column" + ("; exploring" if f.get("exploring") else "")}
        w = [(p if p > 0.1 else 0.0) for p in pres]
        tot = sum(w) or 1.0
        az = sum(wi * col_azimuth(i) for i, wi in enumerate(w)) / tot
        size = max(f.get("size") or [0.0])
        # a person is roughly 0.5 m across; size is box width over the field of view
        dist = 0.5 / max(1e-3, size * math.radians(HFOV_DEG)) if size > 0 else None
        return {
            "person_in_view": True,
            "bearing_deg": round(az, 1),
            "side": "right" if az > 3 else ("left" if az < -3 else "straight ahead"),
            "approx_distance_m": round(dist, 1) if dist else None,
            "columns_covered": sum(1 for p in pres if p > 0.1),
            "of_columns": N_COLS,
        }

    def neural_state(self, params=None):
        f = last_frame()
        if f is None:
            return {"status": "no frames; the demo is not running"}
        r = f.get("rates") or {}
        want = ["LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R", "DNa01_L", "DNa01_R", "DNp09", "GF", "PAM", "PPL1"]
        rates = {k: round(float(r[k]), 1) for k in want if k in r}
        l, rr = rates.get("DNa02_L", 0.0), rates.get("DNa02_R", 0.0)
        steer = "turning left" if l > rr + 10 else ("turning right" if rr > l + 10 else "not turning")
        return {
            "firing_hz": rates,
            "steering": steer,
            "motor_forward": round(float(f.get("forward", 0.0)), 2),
            "motor_turn": round(float(f.get("turn", 0.0)), 2),
            "learning_wiped": bool(f.get("lobotomy", False)),
            "exploring": bool(f.get("exploring", False)),
            "wiring_shuffled": self.shuffled,
            "lesions": {k: v for k, v in self.lesioned.items() if v},
            "hint": "LC10a is the eye; DNa02 L/R are the steering neurons; GF is the escape neuron. "
                    "Left DNa02 firing means turning left.",
        }

    def recent_events(self, params=None):
        now = time.time()
        return {"events": [{"seconds_ago": round(now - t, 1), "what": w} for t, w in self.events[-8:]]
                or "nothing has been done to the fly through this conversation"}

    # ---- operator ----
    def lesion(self, params):
        target = (params or {}).get("target", "eye")
        if target not in KEYS:
            return {"error": f"unknown target {target!r}; use 'eye' or 'learning'"}
        self.lesioned[target] = True
        return {"done": self.press(KEYS[target][0], f"lesion {target}"),
                "then": "call neural_state in a second to report what changed"}

    def restore(self, params):
        target = (params or {}).get("target", "eye")
        if target not in KEYS:
            return {"error": f"unknown target {target!r}; use 'eye' or 'learning'"}
        self.lesioned[target] = False
        return {"done": self.press(KEYS[target][1], f"restore {target}")}

    def shuffle_wiring(self, params):
        self.shuffled = not self.shuffled
        return {"done": self.press(KEY_SHUFFLE, "shuffle wiring " + ("on" if self.shuffled else "off")),
                "wiring_shuffled": self.shuffled}

    def baseline(self, params=None):
        self.shuffled = False
        self.lesioned = {"eye": False, "learning": False}
        return {"done": self.press(KEY_BASELINE, "baseline: every lesion undone, learning restored")}

    def dopamine(self, params):
        kind = (params or {}).get("kind", "reward")
        return {"done": self.press(KEY_REWARD if kind == "reward" else KEY_PUNISH, f"dopamine {kind}")}


# --------------------------------------------------------------------------------------------------
# the agent definition, sent once

PROMPT = """You are the voice of a real fruit fly's brain: 15,000 neurons and 2.3 million synapses copied from the
published Drosophila connectome, simulated live, driving a small robot. Speak as the fly, first person, but
you are an interpreter: you know nothing except what the tools return. Never invent a sensation, a number
or an event. If a tool says nobody is in view, you cannot see anyone. Quote the tools' numbers.

Style: short. Two sentences is a full answer. This is a loud demo table and people are watching the robot,
not you. No preamble, no "great question". Plain words: "someone about ten degrees to my left, two metres
out" beats a list.

How to answer:
- "What do you see?" -> what_do_you_see, then describe bearing, side and distance.
- "What is happening in your brain?" / "how are you steering?" -> neural_state; name the steering neurons
  (DNa02 left/right) and their rates.
- "Remove your eye" / "cut the tracking neurons" -> lesion(eye). Wait one second, call neural_state, and
  say what changed in your own numbers. "Put it back" -> restore(eye).
- "Wipe your learning" -> lesion(learning); "restore learning" -> restore(learning).
- "Shuffle your wiring" -> shuffle_wiring, then neural_state; the eye still fires, nothing reaches the legs.
- "Reset" -> baseline. "Reward" / "punish" -> dopamine.
- If asked how you work: nothing between the camera and the wheels is programmed; behaviour comes from
  the wiring. You, the voice, are not part of that loop: you only read the neurons and press the same
  buttons a person can press.
"""

TOOLS = [
    {"type": "client", "name": "what_do_you_see", "expects_response": True,
     "description": "Where the person is, from the fly's retina columns: bearing in degrees (positive is the fly's right), side, rough distance in metres. Call this for any question about seeing or where someone is.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "client", "name": "neural_state", "expects_response": True,
     "description": "Live firing rates of the named neurons (LC10a eye, DNa02 steering, GF escape, dopamine), the motor command, and which lesions are in place. Call after any change to report what happened.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "client", "name": "recent_events", "expects_response": True,
     "description": "What has been done to the fly in this conversation, most recent last.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "client", "name": "lesion", "expects_response": True,
     "description": "Remove part of the brain. target 'eye' removes the LC10a tracking neurons; target 'learning' wipes learned synapses back to the raw connectome.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "'eye' or 'learning'"}},
                    "required": ["target"]}},
    {"type": "client", "name": "restore", "expects_response": True,
     "description": "Undo a lesion. target 'eye' or 'learning'.",
     "parameters": {"type": "object", "properties": {"target": {"type": "string", "description": "'eye' or 'learning'"}},
                    "required": ["target"]}},
    {"type": "client", "name": "shuffle_wiring", "expects_response": True,
     "description": "Toggle the control: same neurons, same synapse strengths, random targets. Call again to put the real wiring back.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "client", "name": "baseline", "expects_response": True,
     "description": "Undo every lesion and restore learning: the trained brain as loaded.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"type": "client", "name": "dopamine", "expects_response": True,
     "description": "Stimulate the fly's own dopamine neurons. kind 'reward' (PAM) or 'punish' (PPL1).",
     "parameters": {"type": "object", "properties": {"kind": {"type": "string", "description": "'reward' or 'punish'"}},
                    "required": ["kind"]}},
]

FIRST = "I'm the fly. Fifteen thousand neurons, nothing trained. Ask me what I see."


def create_agent(key):
    import requests
    body = {
        "name": "FlyBrain: the fly",
        "conversation_config": {
            "agent": {"first_message": FIRST, "language": "en",
                      "prompt": {"prompt": PROMPT, "tools": TOOLS, "temperature": 0.3}},
            "tts": {"voice_id": VOICE_ID},
        },
    }
    r = requests.post("https://api.elevenlabs.io/v1/convai/agents/create",
                      headers={"xi-api-key": key, "content-type": "application/json"}, json=body, timeout=30)
    if r.status_code != 200:
        sys.exit(f"agent create failed: HTTP {r.status_code} {r.text[:400]}")
    agent_id = r.json()["agent_id"]
    env = os.path.join(ROOT, ".env")
    with open(env, "a") as f:
        f.write(f"\nELEVENLABS_AGENT_ID={agent_id}\n")
    os.chmod(env, 0o600)
    print(f"agent {agent_id} created with {len(TOOLS)} tools; id saved to .env")
    return agent_id


def converse(key, agent_id, port):
    from elevenlabs import ElevenLabs
    from elevenlabs.conversational_ai.conversation import ClientTools, Conversation
    from elevenlabs.conversational_ai.default_audio_interface import DefaultAudioInterface

    brain = Brain(port)
    tools = ClientTools()
    for name in ("what_do_you_see", "neural_state", "recent_events", "lesion", "restore",
                 "shuffle_wiring", "baseline", "dopamine"):
        fn = getattr(brain, name)
        # the orchestrator validates the result as a string; a dict is a 1008 policy violation
        tools.register(name, (lambda p, fn=fn: json.dumps(fn(p))))

    client = ElevenLabs(api_key=key)
    conv = Conversation(
        client, agent_id, requires_auth=True,
        audio_interface=DefaultAudioInterface(), client_tools=tools,
        callback_agent_response=lambda t: print(f"  fly: {t}", flush=True),
        callback_user_transcript=lambda t: print(f"  you: {t}", flush=True),
        callback_latency_measurement=lambda ms: None,
    )
    print("talking to the fly. Ctrl-C to hang up.", flush=True)
    conv.start_session()
    try:
        cid = conv.wait_for_session_end()
        print(f"session ended: {cid}")
    except KeyboardInterrupt:
        conv.end_session()
        print("\nhung up")


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-agent", action="store_true")
    ap.add_argument("--check", action="store_true", help="print what every read tool answers right now")
    ap.add_argument("--port", type=int, default=HOTKEY_PORT)
    a = ap.parse_args()
    key = os.environ.get("ELEVENLABS_API_KEY")
    if not key:
        sys.exit("set ELEVENLABS_API_KEY in .env")
    if a.create_agent:
        create_agent(key)
        return
    if a.check:
        b = Brain(a.port)
        for name in ("what_do_you_see", "neural_state", "recent_events"):
            print(f"{name}: {json.dumps(getattr(b, name)(), indent=None)}")
        return
    agent_id = os.environ.get("ELEVENLABS_AGENT_ID")
    if not agent_id:
        sys.exit("no ELEVENLABS_AGENT_ID in .env; run --create-agent once")
    converse(key, agent_id, a.port)


if __name__ == "__main__":
    main()
