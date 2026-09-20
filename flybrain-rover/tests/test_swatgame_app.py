"""The badge game: Swat.

Two kinds of check. The Lua harness (`badge/test_swatgame.lua`) plays the game and asserts on what the
screen did. This file runs that, and then asserts the things a harness cannot: that the built file fits,
that the manifest is right, and above all that **nothing but a giant-fibre spike can make a mosquito
escape**. If a future edit adds `if elapsed > 470 then escape()` the game would still play and the
harness would still pass, and the entry would be dead, because the whole claim is that a measured
circuit decides it.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "badge" / "swatgame_template.lua"
APP = ROOT / "badge" / "swatgame_app.lua"
HARNESS = ROOT / "badge" / "test_swatgame.lua"

MAIN_LUA_LIMIT = 65536
STATED_WORST = 26          # must equal what the harness measures; see test_the_budget_is_not_a_guess


@pytest.fixture(scope="module")
def built():
    subprocess.run(["python", "badge/build_app.py", "--template", str(TEMPLATE), "--out", str(APP)],
                   cwd=ROOT, check=True, capture_output=True)
    return APP.read_text()


@pytest.fixture(scope="module")
def harness():
    r = subprocess.run(["lua", "badge/test_swatgame.lua"], cwd=ROOT, capture_output=True, text=True)
    return r


def test_the_harness_passes(built, harness):
    assert harness.returncode == 0, harness.stdout + harness.stderr
    assert harness.stdout.strip().endswith("OK")


@pytest.mark.parametrize("scenario,expect", [
    ("quiet_crossing", "crossed"),              # an untouched crossing never bolts
    ("centre_hit", "GOT IT"),
    ("early_shake", "IT GOT OUT"),              # the circuit punishes a panicky shake
    ("chicken_swatted", "THAT WAS A CHICKEN"),
    ("chicken_spared", "SPARED IT"),
    ("lesioned", "never bolted"),
])
def test_scenario(harness, scenario, expect):
    line = next((l for l in harness.stdout.splitlines() if l.startswith(scenario + ",")), None)
    assert line, f"{scenario} missing from:\n{harness.stdout}"
    assert expect in line, line


def test_the_budget_is_not_a_guess(harness):
    """The stated ceiling must be what was measured. A number with slack in it stops being a budget."""
    m = re.search(r"worst tick (\d+) native calls", harness.stderr)
    assert m, harness.stderr
    measured = int(m.group(1))
    assert measured == STATED_WORST, (
        f"the harness measured {measured} native calls, this file says {STATED_WORST}. "
        "If the tick got cheaper, lower the number rather than keeping the slack.")
    src = TEMPLATE.read_text()
    declared = int(re.search(r"local WORST = (\d+)", src).group(1))
    assert declared == measured, f"the template says WORST = {declared}, the harness measured {measured}"


# ---------------------------------------------------------------- THE ONE RULE
def _code_lines(src: str) -> list[str]:
    """The template's prose talks about the rule, so counting has to skip the prose."""
    return [l for l in src.splitlines() if not l.lstrip().startswith("--")]


def test_only_a_giant_fibre_spike_can_make_it_escape(built):
    src = TEMPLATE.read_text()
    code = "\n".join(_code_lines(src))

    # exactly one definition and exactly one call
    assert len(re.findall(r"^local function fly_escapes\(\)", code, re.M)) == 1
    calls = re.findall(r"^(.*)\bfly_escapes\(\)", code, re.M)
    calls = [c for c in calls if "local function" not in c]
    assert len(calls) == 1, f"fly_escapes() is called {len(calls)} times: {calls}"

    # and that one call is guarded by the spike count step() returned, not by a clock or a distance
    assert re.search(r"if gf > 0 and bbolt == 0 then fly_escapes\(\) end", code), \
        "the escape is no longer gated on the giant fibre's own spike count"

    # nothing anywhere may set the dodge flag except that function
    setters = [l.strip() for l in _code_lines(src)
               if re.search(r"^\s*bbolt\s*=", l) and "bbolt = 0" not in l]
    assert len(setters) == 1, f"bbolt is set outside fly_escapes(): {setters}"


def test_no_clock_decides_the_escape(built):
    """A timer comparison near the escape is the exact shortcut this entry must not take."""
    for line in _code_lines(TEMPLATE.read_text()):
        if "bbolt" in line and re.search(r"(now_ms|elapsed|MS)\s*[<>]", line):
            pytest.fail(f"a clock is involved in the escape decision: {line.strip()}")


# ---------------------------------------------------------------- what the badge will accept
def test_it_fits(built):
    n = len(built.encode())
    assert n < MAIN_LUA_LIMIT, f"{n} bytes exceeds the {MAIN_LUA_LIMIT} main.lua limit"
    assert n < 20000, f"{n} bytes: a serial push this big is the flakiest step in the loop"


def test_the_manifest_is_right(built):
    head = built[:400]
    for field in ("slug=swatgame", "api=2", "heap_kb=96", "wake_lock=1"):
        assert field in head, f"{field} missing from the manifest"
    assert built.startswith("--[==[badge-app"), "the manifest must be the first thing in the file"


def test_it_has_an_icon_and_a_name(built):
    assert re.search(r"^name=\S", built[:400], re.M), "no name: it shows as an untitled app"
    assert re.search(r"^icon=\S", built[:400], re.M), "no icon: it shows as text in the launcher"


def test_the_radio_is_never_touched(built):
    """badge.radio.enable() panics this device with the circuit loaded. See badge/SDK_NOTES.md."""
    assert "badge.radio" not in built


def test_only_proven_widget_factories(built):
    """badge.ui.line and non-root parenting are undocumented and unproven on this hardware; box, bar
    and label are what app_template.lua has actually run."""
    used = set(re.findall(r"badge\.ui\.(\w+)", built))
    assert used <= {"box", "bar", "label", "screen_width", "screen_height"}, used


def test_the_fixed_point_constants_match_the_reference(built):
    """These are shared with badge/app_template.lua, which is checked against badge/sim_fixed.py."""
    for const in ("local V_REST, V_TH = -13312, -11520", "local SYN_DECAY = 35",
                  "local GAIN_Q = 75", "local REFRAC = 1", "local CUR_MAX = 4000"):
        assert const in built, f"{const!r} was changed; the arithmetic no longer matches the reference"


def test_the_circuit_is_the_real_one(built):
    assert "M.n = 150" in built or "C.n = 150" in built or re.search(r"\bn\s*=\s*150\b", built)
    assert "male-cns" in built or "Drosophila" in built, "the provenance comment was stripped"


def test_style_tables_are_never_built_per_tick(built):
    """17.7 KB of peak heap for 900 bytes, once. Every style() argument is a preallocated table."""
    body = built.split("function on_tick()", 1)[-1]
    inline = re.findall(r":style\(\{", body)
    assert not inline, f"{len(inline)} fresh style tables inside on_tick"
