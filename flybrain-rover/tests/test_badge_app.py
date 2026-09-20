"""
The badge app's integer LIF must agree with the Python reference step for step.

badge/main.lua runs on an ESP32-C3 that we cannot test from here, and a fixed-point port is exactly the
kind of code that silently disagrees with its reference: one floor() that rounds the other way on a
negative number and the giant fibre stops firing. So the Lua and the Python implement the same arithmetic
over the same generated circuit, and this test runs both and diffs the spike trace.

Needs a desktop `lua` (5.3+ for integer division). Skipped when it is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "badge"))

APP = os.path.join(ROOT, "badge", "flybadge_app.lua")
CIRCUIT = os.path.join(ROOT, "badge", "flybadge_circuit.lua")
pytestmark = pytest.mark.skipif(
    not (shutil.which("lua") and os.path.exists(APP) and os.path.exists(CIRCUIT)),
    reason="needs a desktop lua and a built badge/flybadge_app.lua (python badge/build_app.py)",
)

STEPS = 300


def lua_trace(lesion=False):
    args = ["lua", "badge/test_lua.lua"] + (["lesion", str(STEPS)] if lesion else ["intact", str(STEPS)])
    out = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True).stdout
    rows = [l.split(",") for l in out.strip().splitlines()[1:]]
    return [(int(c), int(f), int(g)) for _, c, f, g in rows]


def py_trace(lesion=False):
    from sim_fixed import load_circuit, run
    c = load_circuit(CIRCUIT)
    return run(c, STEPS, lesion=lesion, quiet=True)[3]


def test_lua_matches_python_intact():
    lua, py = lua_trace(), py_trace()
    assert len(lua) == len(py) == STEPS
    for i, (a, b) in enumerate(zip(lua, py)):
        assert a == b, f"step {i}: lua {a} != python {b} (cur, fired, gf)"


def test_lua_matches_python_lesioned():
    lua, py = lua_trace(lesion=True), py_trace(lesion=True)
    for i, (a, b) in enumerate(zip(lua, py)):
        assert a == b, f"step {i} lesioned: lua {a} != python {b}"


def test_giant_fibre_fires_on_looming_and_not_when_lesioned():
    intact = sum(g for _, _, g in lua_trace())
    lesioned = sum(g for _, _, g in lua_trace(lesion=True))
    assert intact > 0, "the escape circuit never fired on a looming ramp"
    assert lesioned == 0, f"lesioning LC4/LPLC2 left {lesioned} giant fibre spikes: a hidden path exists"


def test_silent_at_rest():
    """No stimulus, no spikes. A circuit that fires on its own would flash the LEDs at random."""
    from sim_fixed import FixedLIF, load_circuit
    lif = FixedLIF(load_circuit(CIRCUIT))
    assert sum(len(lif.step({})) for _ in range(200)) == 0


def test_app_fits_the_documented_badge_limits():
    src = open(APP).read()
    assert len(src.encode()) <= 65536, "over the 64 KiB main.lua limit"
    assert src.startswith("--[==[badge-app\n"), "the IDE importer needs the manifest header first"
    for handler in ("on_enter", "on_tick", "on_button", "on_exit"):
        assert f"function {handler}(" in src
