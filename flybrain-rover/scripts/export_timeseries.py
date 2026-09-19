"""
export_timeseries.py -- put the brain's own recordings into a time-series database.

    python scripts/export_timeseries.py --replay replay/episode_7_trained.json      # SQLite, no setup
    python scripts/export_timeseries.py --replay replay/episode_7_trained.json --dsn "$TIMESERIES_DSN"
    python scripts/export_timeseries.py --live ws://localhost:8765 --run-id demo1   # while the robot runs
    python scripts/export_timeseries.py --report                                    # query it back

A 15-second episode is 750 frames at 50 Hz carrying 25 neuron-group firing rates, 24 retina columns,
the motor command, the reward, and the identity of every neuron that spiked in a fixed 512-cell sample:
roughly 60,000 rows per episode, and the robot produces them continuously. That is a time series, so it
belongs in a time-series database rather than a pile of JSON.

Backends, chosen automatically: a PostgreSQL DSN (TimescaleDB, TigerData Cloud, or plain Postgres) if
`--dsn` or $TIMESERIES_DSN is set and `psycopg` is installed, otherwise a local SQLite file, so the
pipeline runs with no account and the same queries work either way. On Timescale/Tiger the two large
tables are turned into hypertables partitioned by time, which is the point of using one.

Schema
  episodes(run_id pk, started_at, brain, checkpoint, stage, seed, n_frames, note)
  motor(run_id, t, forward, turn, reward, exploring, lobotomy)
  neuron_rates(run_id, t, grp, hz)        -- 25 named populations per frame
  retina(run_id, t, col, pres, size, mot) -- 24 angular columns per frame
  spikes(run_id, t, neuron)               -- one row per spiking neuron per frame
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time

DDL = [
    """CREATE TABLE IF NOT EXISTS episodes (run_id TEXT PRIMARY KEY, started_at DOUBLE PRECISION,
        brain TEXT, checkpoint TEXT, stage TEXT, seed INTEGER, n_frames INTEGER, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS motor (run_id TEXT, t DOUBLE PRECISION, forward REAL, turn REAL,
        reward REAL, exploring INTEGER, lobotomy INTEGER)""",
    """CREATE TABLE IF NOT EXISTS neuron_rates (run_id TEXT, t DOUBLE PRECISION, grp TEXT, hz REAL)""",
    """CREATE TABLE IF NOT EXISTS retina (run_id TEXT, t DOUBLE PRECISION, col INTEGER,
        pres REAL, size REAL, mot REAL)""",
    """CREATE TABLE IF NOT EXISTS spikes (run_id TEXT, t DOUBLE PRECISION, neuron INTEGER)""",
]
INDEXES = [
    "CREATE INDEX IF NOT EXISTS neuron_rates_grp_t ON neuron_rates (grp, t)",
    "CREATE INDEX IF NOT EXISTS motor_run_t ON motor (run_id, t)",
    "CREATE INDEX IF NOT EXISTS spikes_run_t ON spikes (run_id, t)",
]


class Store:
    """One interface over Postgres/TimescaleDB/TigerData and SQLite, so nothing here needs an account."""

    def __init__(self, dsn=None, path="logs/timeseries.db"):
        self.dsn, self.kind = dsn, "sqlite"
        if dsn:
            try:
                import psycopg
                self.conn = psycopg.connect(dsn)
                self.kind = "postgres"
            except ImportError:
                print("psycopg is not installed (pip install 'psycopg[binary]'); using SQLite", flush=True)
                dsn = None
            except Exception as e:  # noqa: BLE001
                print(f"cannot reach the database ({type(e).__name__}: {str(e)[:70]}); using SQLite", flush=True)
                dsn = None
        if not dsn:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.conn = sqlite3.connect(path)
            self.path = path
        self._setup()

    @property
    def ph(self):
        return "%s" if self.kind == "postgres" else "?"

    def _setup(self):
        cur = self.conn.cursor()
        for stmt in DDL:
            cur.execute(stmt if self.kind == "postgres" else stmt.replace("DOUBLE PRECISION", "REAL"))
        if self.kind == "postgres":
            for tbl in ("neuron_rates", "spikes"):                 # the two that actually get large
                try:
                    cur.execute(f"SELECT create_hypertable('{tbl}', 't', if_not_exists => TRUE, "
                                f"chunk_time_interval => 60)")
                except Exception:                                   # plain Postgres has no Timescale
                    self.conn.rollback()
        for stmt in INDEXES:
            try:
                cur.execute(stmt)
            except Exception:  # noqa: BLE001
                pass
        self.conn.commit()

    def insert(self, table, cols, rows):
        if not rows:
            return 0
        cur = self.conn.cursor()
        q = f"INSERT INTO {table} ({','.join(cols)}) VALUES ({','.join([self.ph] * len(cols))})"
        cur.executemany(q, rows)
        self.conn.commit()
        return len(rows)

    def query(self, sql, args=()):
        cur = self.conn.cursor()
        cur.execute(sql if self.kind == "postgres" else sql.replace("%s", "?"), args)
        return cur.fetchall()

    def where(self):
        return f"{self.kind} ({self.dsn.split('@')[-1] if self.dsn else self.path})"


def frame_rows(run_id, f):
    """One replay/FORMAT.md frame -> rows for each table."""
    t = float(f.get("t", 0.0))
    motor = [(run_id, t, float(f.get("forward", 0)), float(f.get("turn", 0)), float(f.get("reward", 0)),
              int(bool(f.get("exploring"))), int(bool(f.get("lobotomy"))))]
    rates = [(run_id, t, g, float(v)) for g, v in (f.get("rates") or {}).items()]
    pres, size, mot = f.get("pres") or [], f.get("size") or [], f.get("mot") or []
    ret = [(run_id, t, i, float(pres[i]), float(size[i]) if i < len(size) else 0.0,
            float(mot[i]) if i < len(mot) else 0.0) for i in range(len(pres))]
    spk = [(run_id, t, int(n)) for n in (f.get("spikes") or [])]
    return motor, rates, ret, spk


def write_frames(store, run_id, frames, meta, note=""):
    t0 = time.time()
    store.insert("episodes", ["run_id", "started_at", "brain", "checkpoint", "stage", "seed", "n_frames", "note"],
                 [(run_id, time.time(), str(meta.get("brain", "")), str(meta.get("checkpoint") or ""),
                   str(meta.get("stage", "")), int(meta.get("seed", 0) or 0), len(frames), note)])
    M, R, T, S = [], [], [], []
    for f in frames:
        m, r, t, s = frame_rows(run_id, f)
        M += m; R += r; T += t; S += s
    n = (store.insert("motor", ["run_id", "t", "forward", "turn", "reward", "exploring", "lobotomy"], M)
         + store.insert("neuron_rates", ["run_id", "t", "grp", "hz"], R)
         + store.insert("retina", ["run_id", "t", "col", "pres", "size", "mot"], T)
         + store.insert("spikes", ["run_id", "t", "neuron"], S))
    print(f"{run_id}: {len(frames)} frames -> {n:,} rows into {store.where()} in {time.time() - t0:.1f} s")
    return n


def report(store):
    """What a time-series database is for: ask the recording questions instead of replaying it."""
    eps = store.query("SELECT run_id, n_frames, stage FROM episodes ORDER BY started_at DESC LIMIT 10")
    if not eps:
        print("nothing stored yet"); return
    print(f"stored in {store.where()}\n")
    print(f"{'episode':28s} {'frames':>7s} {'stage':>6s}")
    for r in eps:
        print(f"{str(r[0]):28s} {r[1]:7d} {str(r[2]):>6s}")
    print("\nsteering asymmetry per episode, the number the whole demo turns on:")
    rows = store.query("""SELECT run_id,
                                 AVG(CASE WHEN grp='DNa02_L' THEN hz END) AS l,
                                 AVG(CASE WHEN grp='DNa02_R' THEN hz END) AS r,
                                 MAX(CASE WHEN grp='GF' THEN hz END) AS gf
                          FROM neuron_rates WHERE grp IN ('DNa02_L','DNa02_R','GF')
                          GROUP BY run_id ORDER BY run_id""")
    print(f"{'episode':28s} {'DNa02_L':>9s} {'DNa02_R':>9s} {'peak GF':>9s}")
    for run, l, r, gf in rows:
        print(f"{str(run):28s} {(l or 0):9.1f} {(r or 0):9.1f} {(gf or 0):9.1f}")
    print("\nbusiest neurons in the sampled population, across everything stored:")
    for neuron, c in store.query("SELECT neuron, COUNT(*) c FROM spikes GROUP BY neuron ORDER BY c DESC LIMIT 5"):
        print(f"  sample neuron {neuron:4d}: {c:6d} spiking frames")
    print("\nwhere the eye was looking when the steering neurons were loudest:")
    for col, p in store.query("""SELECT col, AVG(pres) p FROM retina
                                 GROUP BY col ORDER BY p DESC LIMIT 5"""):
        print(f"  column {col:2d} (azimuth {(col - 11.5) * 4.1:+5.1f} deg): mean presence {p:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None, help="a replay JSON from scripts/dump_replay.py")
    ap.add_argument("--live", default=None, metavar="WS", help="stream from the demo feed, e.g. ws://localhost:8765")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dsn", default=os.environ.get("TIMESERIES_DSN"),
                    help="postgres://... (TimescaleDB / TigerData Cloud); omit to use a local SQLite file")
    ap.add_argument("--sqlite", default="logs/timeseries.db")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seconds", type=float, default=30.0, help="how long to record in --live mode")
    a = ap.parse_args()
    store = Store(a.dsn, a.sqlite)

    if a.replay:
        d = json.load(open(a.replay))
        run = a.run_id or os.path.splitext(os.path.basename(a.replay))[0]
        write_frames(store, run, d["frames"], d.get("meta", {}), note=a.replay)
    if a.live:
        import asyncio
        import websockets
        run = a.run_id or f"live_{int(time.time())}"
        frames = []

        async def pump():
            async with websockets.connect(a.live) as ws:
                end = time.time() + a.seconds
                while time.time() < end:
                    frames.append(json.loads(await ws.recv()))
        print(f"recording {a.seconds:.0f} s from {a.live} ...", flush=True)
        asyncio.run(pump())
        write_frames(store, run, frames, {"brain": "live", "stage": "robot"}, note=a.live)
    if a.report or not (a.replay or a.live):
        report(store)


if __name__ == "__main__":
    main()
