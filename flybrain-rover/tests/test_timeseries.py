"""export_timeseries.py against the SQLite backend only: no --dsn, no $TIMESERIES_DSN, no network.

Synthetic frames follow replay/FORMAT.md (the live demo writes the same shape, see
scripts/robot_bridge.py VizFeed.frame): t, forward, turn, reward, exploring, lobotomy, pres[24],
size[24], mot[24], rates{group: hz}, spikes[int, ...].
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import export_timeseries as ts  # noqa: E402


N_COLS = 24
RATES = {"LC10a_L": 40.0, "LC10a_R": 12.0, "DNa02_L": 80.0, "DNa02_R": 5.0, "GF": 0.0}


def synthetic_frame(t, forward=0.2, turn=-0.1, reward=0.0):
    return {
        "t": t, "forward": forward, "turn": turn, "reward": reward,
        "exploring": False, "lobotomy": False,
        "pres": [0.1 * (i % 3) for i in range(N_COLS)],
        "size": [0.05] * N_COLS,
        "mot": [0.0] * N_COLS,
        "rates": dict(RATES),
        "spikes": [3, 17, 128],
    }


def synthetic_replay(n_frames=3, dt=0.02):
    return [synthetic_frame(round(i * dt, 3)) for i in range(n_frames)]


# --------------------------------------------------------------------------------------------------
# frame_rows: one frame -> rows for each table

def test_frame_rows_shapes():
    f = synthetic_frame(1.5)
    motor, rates, retina, spikes = ts.frame_rows("run1", f)
    assert motor == [("run1", 1.5, 0.2, -0.1, 0.0, 0, 0)]
    assert len(rates) == len(RATES)
    assert all(row[0] == "run1" and row[1] == 1.5 for row in rates)
    assert {row[2] for row in rates} == set(RATES)
    assert {row[2]: row[3] for row in rates} == RATES
    assert len(retina) == N_COLS
    assert retina[0] == ("run1", 1.5, 0, f["pres"][0], f["size"][0], f["mot"][0])
    assert retina[-1][2] == N_COLS - 1
    assert len(spikes) == 3
    assert spikes == [("run1", 1.5, 3), ("run1", 1.5, 17), ("run1", 1.5, 128)]


def test_frame_rows_exploring_and_lobotomy_as_ints():
    f = synthetic_frame(0.0)
    f["exploring"], f["lobotomy"] = True, True
    motor, *_ = ts.frame_rows("run1", f)
    assert motor[0][5:] == (1, 1)


def test_frame_rows_missing_optional_fields_default_safely():
    """A frame with only t set (e.g. a malformed live frame) must not raise."""
    motor, rates, retina, spikes = ts.frame_rows("run1", {"t": 0.0})
    assert motor == [("run1", 0.0, 0.0, 0.0, 0.0, 0, 0)]
    assert rates == []
    assert retina == []
    assert spikes == []


def test_frame_rows_retina_columns_shorter_than_pres_default_to_zero():
    f = {"t": 0.0, "pres": [0.5, 0.5], "size": [0.2]}  # mot missing entirely, size short by one
    _, _, retina, _ = ts.frame_rows("run1", f)
    assert retina == [("run1", 0.0, 0, 0.5, 0.2, 0.0), ("run1", 0.0, 1, 0.5, 0.0, 0.0)]


# --------------------------------------------------------------------------------------------------
# Store + write_batch on SQLite (dsn=None -> sqlite, psycopg never imported)

@pytest.fixture
def store(tmp_path):
    s = ts.Store(dsn=None, path=str(tmp_path / "timeseries.db"))
    assert s.kind == "sqlite"          # no network backend snuck in
    return s


def test_write_batch_row_counts(store):
    frames = synthetic_replay(3)
    n = ts.write_batch(store, "run1", frames, base=0.0)
    per_frame = 1 + len(RATES) + N_COLS + 3          # motor + rates + retina + spikes
    assert n == 3 * per_frame
    assert store.query("SELECT COUNT(*) FROM motor")[0][0] == 3
    assert store.query("SELECT COUNT(*) FROM neuron_rates")[0][0] == 3 * len(RATES)
    assert store.query("SELECT COUNT(*) FROM retina")[0][0] == 3 * N_COLS
    assert store.query("SELECT COUNT(*) FROM spikes")[0][0] == 9


def test_write_batch_empty_frames_returns_zero(store):
    assert ts.write_batch(store, "run1", [], base=0.0) == 0


def test_write_frames_creates_episode_row(store):
    ts.write_frames(store, "run1", synthetic_replay(3), {"brain": "test.npz", "stage": "A", "seed": 7})
    rows = store.query("SELECT run_id, n_frames, stage, seed FROM episodes WHERE run_id = ?", ("run1",))
    assert rows == [("run1", 3, "A", 7)]


def test_write_frames_two_episodes_stay_independent(store):
    ts.write_frames(store, "run1", synthetic_replay(3), {"stage": "A"})
    ts.write_frames(store, "run2", synthetic_replay(2), {"stage": "B"})
    assert store.query("SELECT COUNT(*) FROM motor WHERE run_id='run1'")[0][0] == 3
    assert store.query("SELECT COUNT(*) FROM motor WHERE run_id='run2'")[0][0] == 2
    assert store.query("SELECT COUNT(*) FROM episodes")[0][0] == 2


# --------------------------------------------------------------------------------------------------
# report(): must run end to end on SQLite with no network and no crash, empty or populated

def test_report_on_empty_store_prints_nothing_stored(store, capsys):
    ts.report(store)
    assert "nothing stored yet" in capsys.readouterr().out


def test_report_after_writing_prints_episode_and_asymmetry(store, capsys):
    ts.write_frames(store, "run1", synthetic_replay(3), {"brain": "test.npz", "stage": "A", "seed": 7})
    ts.report(store)
    out = capsys.readouterr().out
    assert "run1" in out
    assert "steering asymmetry" in out
    assert "busiest neurons" in out
    # DNa02_L is 80.0 Hz on every frame -> the per-episode average must show up verbatim
    assert "80.0" in out
