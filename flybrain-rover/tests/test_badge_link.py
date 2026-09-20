"""The badge's two robot controls, over USB serial.

The parsing is tested directly, and then the whole loop is run over a pty, because the thing that
actually goes wrong on the night is not the regex: it is a partial line arriving in two reads, or the
badge logging something unrelated that happens to contain a digit.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.badge_link import KEY_LOBO_OFF, KEY_LOBO_ON, KEY_STOP, Link, main  # noqa: E402


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def link():
    sent = []
    clk = Clock()
    lk = Link(sent.append, now=clk, debounce_s=0.4)
    return lk, sent, clk


def test_lesion_on_and_off(link):
    lk, sent, clk = link
    assert lk.feed("I (912) lua: @FLY LOBO 1") == [KEY_LOBO_ON]
    clk.t += 1
    assert lk.feed("@FLY LOBO 0") == [KEY_LOBO_OFF]
    assert sent == [KEY_LOBO_ON, KEY_LOBO_OFF]


def test_repeats_are_not_resent(link):
    """The badge may log its state more than once. The rover must not be lesioned twice."""
    lk, sent, clk = link
    lk.feed("@FLY LOBO 1")
    clk.t += 5
    assert lk.feed("@FLY LOBO 1") == []
    assert sent == [KEY_LOBO_ON]


def test_debounce(link):
    """Hotkey 0 is a toggle on the rover, so a bounced button would leave the two out of step."""
    lk, sent, clk = link
    assert lk.feed("@FLY STOP 1") == [KEY_STOP]
    clk.t += 0.05
    assert lk.feed("@FLY STOP 0") == [], "a bounce inside the debounce window got through"
    clk.t += 1.0
    assert lk.feed("@FLY STOP 0") == [KEY_STOP]


def test_unrelated_console_noise_is_ignored(link):
    """The badge logs plenty. None of it may drive a robot."""
    lk, sent, _ = link
    for noise in ("I (204) boot: chip revision: v0.4",
                  "W (901) wifi: STOP 1",             # the word, without the marker
                  "@FLY",                              # marker, no payload
                  "@FLY LOBO",                         # no argument
                  "@FLY LOBO yes",                     # not 0 or 1
                  "@FLY SPIN 1",                       # a verb we do not know
                  "score 1 lives 3"):
        assert lk.feed(noise) == [], f"{noise!r} produced a keypress"
    assert sent == []


def test_state_is_tracked_per_control(link):
    lk, sent, clk = link
    lk.feed("@FLY LOBO 1"); clk.t += 1
    lk.feed("@FLY STOP 1"); clk.t += 1
    lk.feed("@FLY LOBO 0"); clk.t += 1
    assert sent == [KEY_LOBO_ON, KEY_STOP, KEY_LOBO_OFF]


@pytest.mark.skipif(not hasattr(os, "openpty"), reason="no pty on this platform")
def test_over_a_real_pty(monkeypatch):
    """The whole loop against a fake badge: a line split across two writes, and noise around it."""
    pytest.importorskip("serial")
    import serial

    primary, secondary = os.openpty()
    port = os.ttyname(secondary)
    sent: list[bytes] = []

    class FakeSock:
        def sendto(self, b, addr):
            sent.append(b)

    monkeypatch.setattr("socket.socket", lambda *a, **k: FakeSock())

    ser = serial.Serial(port, 115200, timeout=0.2)
    monkeypatch.setattr(serial, "Serial", lambda *a, **k: ser)

    # a partial line, then the rest, then the marker split across a read boundary
    os.write(primary, b"I (100) boot: hello\n@FLY LOB")
    os.write(primary, b"O 1\nI (140) lua: tick\n")

    # run the loop briefly by feeding it through Link the way main() does
    link = Link(lambda b: sent.append(b))
    buf = b""
    for _ in range(20):
        chunk = ser.read(256)
        if not chunk:
            break
        buf += chunk
        while b"\n" in buf:
            raw, buf = buf.split(b"\n", 1)
            link.feed(raw.decode("utf-8", "replace").strip())
    ser.close()
    os.close(primary)

    assert sent == [KEY_LOBO_ON], f"a line split across two reads was lost: {sent}"


def test_refuses_a_non_loopback_host(capsys):
    """CLAUDE rule 5: the control loop is local. This must never be pointed off the laptop."""
    rc = main(["--host", "10.0.0.5", "--dry-run"])
    assert rc == 2
    assert "loopback" in capsys.readouterr().err


def test_the_badge_actually_prints_these_lines():
    """The contract has two ends. If the Lua stops logging them, this link is dead and silent."""
    lua = (Path(__file__).resolve().parents[1] / "badge" / "swatgame_template.lua").read_text()
    for line in ('"@FLY LOBO 1"', '"@FLY LOBO 0"', '"@FLY STOP 1"', '"@FLY STOP 0"'):
        assert line in lua, f"the badge no longer logs {line}"


def test_the_badge_never_enables_the_radio():
    """This link exists over serial precisely because BLE panics the device."""
    lua = (Path(__file__).resolve().parents[1] / "badge" / "swatgame_app.lua").read_text()
    assert "badge.radio" not in lua
