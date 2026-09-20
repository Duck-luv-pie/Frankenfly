"""
The badge app is an exhibit, so this tests what the exhibit does, not just what the arithmetic does.

tests/test_badge_app.py proves the fixed-point LIF still matches badge/sim_fixed.py spike for spike.
That is necessary and not sufficient: the whole point of the rewrite is that a stranger can pick the
badge up, swat it, and SEE the spike wave travel through the real neurons, then press B and SEE it
stop. So badge/test_visual.lua drives on_enter and a few hundred on_ticks against a stubbed `badge`
table that records every widget write and counts every native call, and this runs it and asserts on
the numbers it prints.

It also diffs the three pieces that are not allowed to move -- the fixed-point constants, reset_brain()
and step() -- against the previous committed app_template.lua, because a renderer rewrite that quietly
edits the membrane equation would still pass a spike-trace test written against the new equation.

Needs a desktop `lua` (5.3+ for integer division) and `git`. Skipped when either is missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "badge", "flybadge_app.lua")
TEMPLATE = os.path.join(ROOT, "badge", "app_template.lua")
HARNESS = os.path.join(ROOT, "badge", "test_visual.lua")

pytestmark = pytest.mark.skipif(
    not (shutil.which("lua") and shutil.which("git") and os.path.exists(APP)
         and os.path.exists(HARNESS)),
    reason="needs a desktop lua, git, and a built badge/flybadge_app.lua (python badge/build_app.py)",
)

# The app's own declared ceiling, repeated here so a silent bump to WORST cannot make the budget test
# pass by moving the goalposts. 44 is the counted phase 6 / phase 11 tick; see app_template.lua.
STATED_WORST = 44


def harness(*args):
    """Run badge/test_visual.lua and return its CSV as {scenario: {column: int}}."""
    p = subprocess.run(["lua", "badge/test_visual.lua", *args], cwd=ROOT,
                       capture_output=True, text=True)
    assert p.returncode == 0, f"badge/test_visual.lua failed:\n{p.stdout}\n{p.stderr}"
    lines = [l for l in p.stdout.strip().splitlines() if l and l != "OK"]
    head = lines[0].split(",")
    rows = {}
    for line in lines[1:]:
        cells = line.split(",")
        rows[cells[0]] = {k: int(v) for k, v in zip(head[1:], cells[1:])}
    return rows


@pytest.fixture(scope="module")
def full():
    return harness()


@pytest.fixture(scope="module")
def lowheap():
    return harness("lowheap")


def test_a_swat_fires_the_giant_fibre_and_lights_its_widgets(full):
    """The escape has to be visible, not just true: bolt, bar and LEDs all have to move."""
    hit = full["intact_swat"]
    assert hit["gf_spikes"] > 0, "a swat produced no giant-fibre spikes"
    assert hit["bolt_h"] == 70, f"the bolt stopped at {hit['bolt_h']} px instead of reaching the thorax"
    assert hit["gf_bar"] == 100, "the giant-fibre bar never latched"
    assert hit["led_gf_ticks"] > 0, "the giant-fibre LED pair never lit"
    assert hit["tract_ticks"] > 0, "the nerve bundle never showed spikes crossing"
    assert hit["eye_ticks"] > 0, "the eyes never lit"


def test_nothing_fires_at_rest(full):
    idle = full["idle"]
    assert idle["gf_spikes"] == 0 and idle["bolt_h"] == 0 and idle["led_gf_ticks"] == 0


def test_lesion_leaves_the_eye_lit_and_everything_downstream_dark(full):
    """The teaching moment. The fly can still see you; it just cannot act."""
    cut = full["lesioned"]
    assert cut["ticks"] == 200
    assert cut["gf_spikes"] == 0, f"{cut['gf_spikes']} giant-fibre spikes while lesioned: a path exists"
    assert cut["eye_ticks"] > 0, "the eye went dark when lesioned; only its OUTPUT should be cut"
    assert cut["bolt_h"] <= 22, f"the bolt ran past the slash to {cut['bolt_h']} px"
    assert cut["tract_ticks"] == 0, "the nerve bundle lit with the eye's output cut"
    assert cut["gf_bar"] == 0, "the giant-fibre bar moved while lesioned"
    assert cut["led_gf_ticks"] == 0, "the giant-fibre LED pair lit while lesioned"


def test_restore_brings_the_escape_back(full):
    assert full["restored"]["gf_spikes"] > 0
    assert full["restored"]["bolt_h"] == 70


def test_slow_motion_steps_the_real_brain_more_slowly(full):
    """LEFT is not a canned animation: the same stimulus over the same ticks produces fewer spikes."""
    assert 0 < full["slow_motion"]["gf_spikes"] < full["restored"]["gf_spikes"]


def test_the_swat_placeholder_costs_nothing(full):
    assert full["swat_screen"]["max_calls"] == 0


def test_native_calls_per_tick_stay_under_the_stated_worst_case(full, lowheap):
    for rows in (full, lowheap):
        budget = rows["budget"]
        assert budget["bolt_h"] == STATED_WORST, (
            f"the app now declares WORST={budget['bolt_h']}, not the reviewed {STATED_WORST}")
        assert budget["max_calls"] <= STATED_WORST, (
            f"a tick made {budget['max_calls']} native calls against a stated ceiling of {STATED_WORST}")
        for name, row in rows.items():
            if name not in ("budget", "on_enter"):
                assert row["max_calls"] <= STATED_WORST, f"{name} exceeded the budget"


def test_show_is_called_at_most_once_per_tick(full, lowheap):
    for rows in (full, lowheap):
        for name, row in rows.items():
            if name not in ("budget", "on_enter"):
                assert row["max_show"] <= 1, f"{name} called badge.led.show() {row['max_show']} times"


def test_a_tight_heap_drops_widgets_instead_of_the_app(full, lowheap):
    """on_enter reads free_heap before building anything; the optional bars go and nothing breaks."""
    assert lowheap["intact_swat"]["gf_spikes"] > 0
    assert lowheap["intact_swat"]["bolt_h"] == 70
    assert lowheap["intact_swat"]["gf_bar"] == 0, "the optional bar was built on a tight heap"
    assert lowheap["lesioned"]["gf_spikes"] == 0
    assert lowheap["on_enter"]["max_calls"] < full["on_enter"]["max_calls"], (
        "the tight-heap build made just as many widgets as the full one")


# ------------------------------------------------------------------ the parts that may not move ----

def _block(src: str, start: str, end: str) -> str:
    i = src.index(start)
    j = src.index(end, i) + len(end)
    return src[i:j]


CONST_START = "local V_REST, V_TH = -13312, -11520"
CONST_END = "local FLASH_MS, JUMP_MS, FLIRT_MS = 160, 260, 3000"
RESET_START = "local function reset_brain()"
RESET_END = "  spk, spk_n = {}, 0\nend"
STEP_START = "local function step(cur, flirt)"
STEP_END = "  return gf_now\nend"


# The commit whose circuit block and step() this app must still match, character for character.
# Comparing against HEAD compares the working file with the version of itself that was just committed,
# so the guard can never fail. Pin the last commit before the visual rebuild instead; if the arithmetic
# is ever deliberately changed, move this pin deliberately.
BASELINE_REV = "c9f3d36"


@pytest.fixture(scope="module")
def previous():
    out = subprocess.run(["git", "show", f"{BASELINE_REV}:badge/app_template.lua"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode != 0:
        pytest.skip(f"baseline {BASELINE_REV} not available (shallow clone?)")
    return out.stdout


@pytest.fixture(scope="module")
def current():
    return open(TEMPLATE).read()


@pytest.mark.parametrize("name,start,end", [
    ("the fixed-point constants", CONST_START, CONST_END),
    ("reset_brain()", RESET_START, RESET_END),
    ("step()", STEP_START, STEP_END),
])
def test_the_circuit_block_is_byte_identical_to_the_previous_app(previous, current, name, start, end):
    was, now = _block(previous, start, end), _block(current, start, end)
    assert now == was, f"{name} changed; the arithmetic is not allowed to move in a renderer rewrite"


def test_the_built_app_still_carries_the_manifest_and_the_lifecycle():
    src = open(APP).read()
    assert src.startswith("--[==[badge-app\n"), "the IDE importer needs the manifest header first"
    for key in ("slug=flybadge", "api=2", "heap_kb=96", "wake_lock=1"):
        assert key in src.split("]==]", 1)[0], f"the manifest lost {key}"
    for handler in ("on_enter", "on_tick", "on_button", "on_exit"):
        assert f"function {handler}(" in src


def test_the_pushed_app_stays_under_twenty_kilobytes():
    """The badge compiles main.lua in RAM and a serial push of a larger file is the flakiest step."""
    size = os.path.getsize(APP)
    assert size < 20 * 1024, f"{size} bytes: too big to push comfortably"


def test_no_banned_sandbox_builtins_in_the_pushed_app():
    import re
    src = open(APP).read()
    for banned in ("pcall", "coroutine", "setmetatable", "dofile", "require"):
        assert not re.search(rf"(?<![\w.]){banned}\s*[.(]", src), f"the sandbox has no {banned}"
