"""
SWAT, the badge game, driven end to end on a laptop.

badge/swat_app.lua runs on an ESP32-C3 we cannot test from here, but the game is pure Lua over a stubbed
`badge` table, so badge/test_swat.lua can run the whole state machine under desktop Lua with a fake clock
and print one CSV line per round. This file runs it and holds the numbers to badge/GAME_SPEC.md:

  * the fly's escape latency per difficulty matches the measured table (section 3) within 20 ms,
  * pressing A first is a HIT with the press time you actually pressed at,
  * pressing A while ARMED is a FALSE_START,
  * with LC4 and LPLC2 cut the fly never escapes, across 20 rounds at INSANE,
  * restoring them brings the latency back,
  * the built file fits the badge limits and starts with the manifest,
  * the app only calls widget methods badge/SDK_NOTES.md documents,
  * the brain (constants, reset_brain, step) is byte-identical to badge/app_template.lua,
  * and no `if` anywhere outside step() decides that the fly escapes (spec section 8 and criterion 7).

The harness itself also asserts, on every IDLE and ARMED tick, that the disc is the 10x10 dot at
(155,113), that difficulty wraps in both directions, and that no tick calls badge.led.show() twice.

Needs a desktop `lua` (5.3+ for integer division). Skipped when it is missing.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "badge", "swat_app.lua")
TEMPLATE = os.path.join(ROOT, "badge", "swat_template.lua")
BASE = os.path.join(ROOT, "badge", "app_template.lua")

pytestmark = pytest.mark.skipif(
    not (shutil.which("lua") and os.path.exists(APP)),
    reason="needs a desktop lua and a built badge/swat_app.lua "
           "(python badge/build_app.py --template badge/swat_template.lua --out badge/swat_app.lua)",
)

# GAME_SPEC.md section 3, measured: RAMP per 10 ms step -> giant fibre first spike
EXPECTED_GF_MS = {"SLOW": 420, "NORMAL": 250, "FAST": 170, "INSANE": 110}
RAMP_OF = {"SLOW": 45, "NORMAL": 80, "FAST": 120, "INSANE": 200}

# badge/SDK_NOTES.md: the only widget methods the badge Lua exposes
WIDGET_METHODS = {"align", "set_pos", "set_size", "set_text", "set_value", "style"}


@pytest.fixture(scope="module")
def rows():
    proc = subprocess.run(["lua", "badge/test_swat.lua"], cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"harness failed:\n{proc.stdout}\n{proc.stderr}"
    lines = proc.stdout.strip().splitlines()
    header = lines[0].split(",")
    assert header == ["scenario", "difficulty", "result", "press_ms", "gf_ms"]
    out = []
    for line in lines[1:]:
        r = dict(zip(header, line.split(",")))
        r["press_ms"] = int(r["press_ms"]) if r["press_ms"] else None
        r["gf_ms"] = int(r["gf_ms"]) if r["gf_ms"] else None
        out.append(r)
    return out


def scenario(rows, tag):
    return [r for r in rows if r["scenario"] == tag]


def test_fly_latency_matches_the_measured_table(rows):
    a = scenario(rows, "a")
    assert [r["difficulty"] for r in a] == ["SLOW", "NORMAL", "FAST", "INSANE"]
    for r in a:
        assert r["result"] == "ESCAPED", r
        assert r["press_ms"] is None, r
        want = EXPECTED_GF_MS[r["difficulty"]]
        assert abs(r["gf_ms"] - want) <= 20, f"{r['difficulty']}: fly at {r['gf_ms']} ms, spec says {want}"


def test_pressing_first_is_a_hit_with_your_press_time(rows):
    (b,) = scenario(rows, "b")
    assert b["difficulty"] == "NORMAL"
    assert b["result"] == "HIT", b
    assert b["press_ms"] == 100, b
    assert b["gf_ms"] is None, "the brain is frozen on a hit; the fly has no latency to report"


def test_pressing_while_armed_is_a_false_start(rows):
    (c,) = scenario(rows, "c")
    assert c["result"] == "FALSE_START", c
    assert c["press_ms"] is None and c["gf_ms"] is None, c


def test_lesioned_fly_never_escapes(rows):
    d = scenario(rows, "d")
    assert len(d) == 20
    escapes = [r for r in d if r["result"] == "ESCAPED" or r["gf_ms"] is not None]
    assert escapes == [], f"{len(escapes)} escapes with LC4/LPLC2 cut: a hidden path exists"
    assert all(r["result"] == "TIMEOUT" and r["difficulty"] == "INSANE" for r in d), d


def test_restoring_the_synapses_restores_the_latency(rows):
    (e,) = scenario(rows, "e")
    (a_insane,) = [r for r in scenario(rows, "a") if r["difficulty"] == "INSANE"]
    assert e["result"] == "ESCAPED", e
    assert abs(e["gf_ms"] - a_insane["gf_ms"]) <= 10, (e, a_insane)


def test_app_fits_the_documented_badge_limits():
    src = open(APP).read()
    assert src.startswith("--[==[badge-app\n"), "the IDE importer needs the manifest header first"
    manifest = src.split("]==]", 1)[0]
    assert "\nslug=swat\n" in manifest
    assert "\nheap_kb=96\n" in manifest and "\nwake_lock=1\n" in manifest and "\napi=2\n" in manifest
    assert len(src.encode()) < 65536, "over the 64 KiB main.lua limit"
    for handler in ("on_enter", "on_tick", "on_button", "on_exit"):
        assert f"function {handler}(" in src
    assert "--@CIRCUIT@" not in src and "local C = {}" in src, "the circuit was not substituted"
    for banned in ("pcall", "coroutine", "os", "io", "require", "setmetatable", "load", "dofile"):
        assert not re.search(rf"(?<![\w.]){banned}\s*[.(]", src), f"uses {banned}, absent on the badge"


def test_difficulty_table_is_the_measured_one():
    src = open(TEMPLATE).read()
    m = re.search(r"RAMP_OF = \{([^}]*)\}", src)
    assert m, "no RAMP_OF table"
    table = dict((k.strip(), int(v)) for k, v in re.findall(r"(\w+)\s*=\s*(\d+)", m.group(1)))
    assert table == RAMP_OF


# --- the brain must be the tested one -------------------------------------------------------------

def _block(src, start, end):
    m = re.search(start + r".*?" + end, src, re.S | re.M)
    assert m, f"no block {start!r} .. {end!r}"
    return m.group(0)


BRAIN_BLOCKS = {
    "constants": (r"^-- fixed point: 1 unit = 1/256 mV.*?$", r"^local CUR_MAX = [^\n]*$"),
    "reset_brain": (r"^local function reset_brain\(\)$", r"^end$"),
    "step": (r"^local function step\(cur, flirt\)$", r"^end$"),
}


@pytest.mark.parametrize("name", sorted(BRAIN_BLOCKS))
def test_brain_is_byte_identical_to_flybadge(name):
    start, end = BRAIN_BLOCKS[name]
    ours, theirs = _block(open(TEMPLATE).read(), start, end), _block(open(BASE).read(), start, end)
    assert ours == theirs, f"{name} differs from badge/app_template.lua"


def test_circuit_is_the_generated_one():
    circuit = open(os.path.join(ROOT, "badge", "flybadge_circuit.lua")).read()
    app = open(APP).read()
    for key in ("n", "eye_last", "gf_first", "gf_last", "n_edges"):
        m = re.search(rf"^M\.{key} = (-?\d+)", circuit, re.M)
        assert re.search(rf"^C\.{key} = {m.group(1)}\b", app, re.M), key


# --- no hidden reflex ------------------------------------------------------------------------------
#
# The whole entry rests on one property: the fly escapes only because the giant fibre spiked in step().
# Two checks, on the built file with step() removed. First, the literal rule from the task: no token
# matching /escap/i sits inside an `if` whose condition mentions accel, swat, press, loom or cur. Second,
# the structural rule: the identifier ESCAPED is used, outside its declaration, in exactly one function,
# and every call of that function is guarded by an `if` on the summed return value of step().

TOKEN = re.compile(
    r"--\[(=*)\[.*?\]\1\]"            # long comment (the manifest)
    r"|--[^\n]*"                       # line comment
    r'|"(?:[^"\\\n]|\\.)*"'            # strings, so a -- or an `if` inside one is not code
    r"|'(?:[^'\\\n]|\\.)*'"
    r"|[A-Za-z_]\w*"
    r"|\d+"
    r"|\S",
    re.S,
)
OPENERS = {"for", "while", "function", "repeat"}
SUSPECT = re.compile(r"accel|swat|press|loom|cur", re.I)


def _tokens(src):
    return [m.group(0) for m in TOKEN.finditer(src) if not m.group(0).startswith("--")]


def _without_step(src):
    start, end = BRAIN_BLOCKS["step"]
    block = _block(src, start, end)
    return src.replace(block, "", 1)


def test_only_documented_widget_methods_are_called():
    """A method the SDK does not list is a nil call on the badge, and on_enter dying there is the one
    failure no laptop harness can see. The harness stub defines exactly this set, so it would crash too;
    this check names the offender without running anything."""
    tokens = _tokens(open(APP).read())
    # Lua's own string methods are called with a colon too (A:sub(i, i) in the base64 decoder), and they
    # are not widget calls. Exclude the string library rather than widen WIDGET_METHODS, so a genuinely
    # undocumented widget method is still caught.
    STRING_METHODS = {"sub", "byte", "char", "find", "format", "gmatch", "gsub", "len", "lower",
                      "match", "rep", "reverse", "upper"}
    called = {tokens[i + 1] for i, tok in enumerate(tokens[:-2])
              if tok == ":" and re.fullmatch(r"[A-Za-z_]\w*", tokens[i + 1]) and tokens[i + 2] == "("}
    called -= STRING_METHODS
    assert called <= WIDGET_METHODS, f"undocumented widget methods: {sorted(called - WIDGET_METHODS)}"


def _walk(tokens):
    """Yield (index, token, [enclosing if conditions], [enclosing function names]) for every token in a
    block body. Condition tokens between `if`/`elseif` and `then` are folded into the frame instead."""
    stack = []   # frames: [kind, text]
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in ("if", "elseif"):
            j = i + 1
            while tokens[j] != "then":
                j += 1
            cond = " ".join(tokens[i + 1:j])
            if tok == "if":
                stack.append(["if", cond])
            else:
                assert stack and stack[-1][0] == "if", "elseif outside an if"
                stack[-1][1] += " " + cond
            i = j + 1
            continue
        if tok == "function":
            name = tokens[i + 1] if tokens[i + 1] != "(" else ""
            stack.append(["function", name])
        elif tok in OPENERS:
            stack.append([tok, ""])
        elif tok in ("end", "until"):
            stack.pop()
        yield i, tok, [t for k, t in stack if k == "if"], [t for k, t in stack if k == "function"]
        i += 1


def test_no_if_on_the_stimulus_or_the_button_decides_an_escape():
    tokens = _tokens(_without_step(open(APP).read()))
    offenders = []
    for _, tok, conds, _ in _walk(tokens):
        if re.search(r"escap", tok, re.I):
            bad = [c for c in conds if SUSPECT.search(c)]
            if bad:
                offenders.append((tok, bad))
    assert offenders == [], f"an if on the stimulus or the button leads to an escape: {offenders}"


def _in_local_declaration(tokens, idx):
    """True when tokens[idx] is a name in `local a, b, c = ...`."""
    j = idx - 1
    while j >= 0 and (tokens[j] == "," or re.fullmatch(r"[A-Za-z_]\w*", tokens[j])):
        if tokens[j] == "local":
            return True
        j -= 1
    return False


def test_every_transition_into_escaped_comes_from_a_giant_fibre_spike():
    tokens = _tokens(_without_step(open(APP).read()))
    users = set()       # functions whose body uses the ESCAPED identifier
    calls = []          # (enclosing if conditions) at every call of such a function
    walked = list(_walk(tokens))
    for idx, tok, conds, funcs in walked:
        if tok == "ESCAPED" and not _in_local_declaration(tokens, idx):
            users.add(funcs[-1] if funcs else "<top level>")
    assert users, "nothing ever enters ESCAPED"
    assert len(users) == 1 and "<top level>" not in users, f"ESCAPED is assigned in {sorted(users)}"
    (escaper,) = users
    for idx, tok, conds, funcs in walked:
        if tok == escaper and tokens[idx + 1] == "(" and tokens[idx - 1] != "function":
            calls.append(conds)
    assert calls, f"{escaper} is never called"
    for conds in calls:
        assert conds and re.search(r"\bgf_tick\b", conds[-1]), \
            f"{escaper} is called without a giant fibre spike guard: {conds}"
        assert not any(SUSPECT.search(c) for c in conds), conds
