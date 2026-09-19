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
from datetime import datetime, timezone
import sqlite3
import sys
import time

DDL = [
    """CREATE TABLE IF NOT EXISTS episodes (run_id TEXT PRIMARY KEY, started_at DOUBLE PRECISION,
        brain TEXT, checkpoint TEXT, stage TEXT, seed INTEGER, n_frames INTEGER, note TEXT)""",
    """CREATE TABLE IF NOT EXISTS motor (run_id TEXT, t {T}, forward REAL, turn REAL,
        reward REAL, exploring INTEGER, lobotomy INTEGER)""",
    """CREATE TABLE IF NOT EXISTS neuron_rates (run_id TEXT, t {T}, grp TEXT, hz REAL)""",
    """CREATE TABLE IF NOT EXISTS retina (run_id TEXT, t {T}, col INTEGER,
        pres REAL, size REAL, mot REAL)""",
    """CREATE TABLE IF NOT EXISTS spikes (run_id TEXT, t {T}, neuron INTEGER)""",
]
INDEXES = [
    "CREATE INDEX IF NOT EXISTS neuron_rates_grp_t ON neuron_rates (grp, t)",
    "CREATE INDEX IF NOT EXISTS motor_run_t ON motor (run_id, t)",
    "CREATE INDEX IF NOT EXISTS spikes_run_t ON spikes (run_id, t)",
]


class Store:
    """One interface over Postgres/TimescaleDB/TigerData and SQLite, so nothing here needs an account."""

    def __init__(self, dsn=None, path="logs/timeseries.db"):
        self.dsn, self.kind, self.hypertables, self.cagg = dsn, "sqlite", [], False
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
        ts = "TIMESTAMPTZ" if self.kind == "postgres" else "REAL"
        for stmt in DDL:
            stmt = stmt.format(T=ts)
            cur.execute(stmt if self.kind == "postgres" else stmt.replace("DOUBLE PRECISION", "REAL"))
        self.conn.commit()          # the tables are safe before anything that might roll back
        if self.kind == "postgres":
            for tbl in ("neuron_rates", "spikes"):                 # the two that actually get large
                try:
                    cur.execute(f"SELECT create_hypertable('{tbl}', 't', if_not_exists => TRUE, "
                                f"chunk_time_interval => INTERVAL '1 minute')")
                    self.conn.commit()
                    self.hypertables.append(tbl)
                except Exception as e:                              # plain Postgres has no Timescale
                    self.conn.rollback()
                    print(f"  no hypertable on {tbl}: {type(e).__name__}: {str(e).splitlines()[0][:90]}",
                          flush=True)
        for stmt in INDEXES:
            try:
                cur.execute(stmt)
            except Exception:  # noqa: BLE001
                pass
        self.conn.commit()
        if self.hypertables:
            self._timescale_extras()

    def _timescale_extras(self):
        """A continuous aggregate of per-second population rates, refreshed every 5 s and read in real
        time (the not-yet-materialized tail is computed on query), so a chart can poll it with no lag
        and no scan of the raw table. Plus compression on the two big tables after two minutes, which
        is what lets a free-tier instance hold a whole day of the robot's recordings.
        CREATE MATERIALIZED VIEW ... WITH (timescaledb.continuous) refuses to run inside a transaction,
        hence autocommit for this block."""
        self.conn.autocommit = True
        cur = self.conn.cursor()
        try:
            cur.execute("""CREATE MATERIALIZED VIEW IF NOT EXISTS rates_1s
                           WITH (timescaledb.continuous) AS
                           SELECT time_bucket(INTERVAL '1 second', t) AS bucket, run_id, grp,
                                  avg(hz) AS hz, max(hz) AS peak
                           FROM neuron_rates GROUP BY 1, 2, 3 WITH NO DATA""")
            cur.execute("""SELECT add_continuous_aggregate_policy('rates_1s',
                               start_offset => INTERVAL '10 minutes', end_offset => INTERVAL '1 second',
                               schedule_interval => INTERVAL '5 seconds', if_not_exists => TRUE)""")
            # TimescaleDB 2.13+ creates continuous aggregates with real-time reads OFF; turn them on, so a
            # query sees the newest second before the 5 s refresh has materialized it
            cur.execute("ALTER MATERIALIZED VIEW rates_1s SET (timescaledb.materialized_only = false)")
            self.cagg = True
        except Exception as e:  # noqa: BLE001
            print(f"  no continuous aggregate: {type(e).__name__}: {str(e).splitlines()[0][:90]}", flush=True)
        for tbl, order in (("spikes", "t, neuron"), ("neuron_rates", "t, grp")):
            try:
                cur.execute(f"""ALTER TABLE {tbl} SET (timescaledb.compress,
                                timescaledb.compress_segmentby = 'run_id',
                                timescaledb.compress_orderby = '{order}')""")
                cur.execute(f"SELECT add_compression_policy('{tbl}', INTERVAL '2 minutes', if_not_exists => TRUE)")
            except Exception as e:  # noqa: BLE001
                print(f"  no compression on {tbl}: {type(e).__name__}: {str(e).splitlines()[0][:90]}", flush=True)
        self.conn.autocommit = False

    def insert(self, table, cols, rows):
        if not rows:
            return 0
        cur = self.conn.cursor()
        if self.kind == "postgres" and len(rows) > 64:
            with cur.copy(f"COPY {table} ({','.join(cols)}) FROM STDIN") as cp:   # the fast path
                for r in rows:
                    cp.write_row(r)
        else:
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


def open_episode(store, run_id, meta, n_frames, note=""):
    store.insert("episodes", ["run_id", "started_at", "brain", "checkpoint", "stage", "seed", "n_frames", "note"],
                 [(run_id, time.time(), str(meta.get("brain", "")), str(meta.get("checkpoint") or ""),
                   str(meta.get("stage", "")), int(meta.get("seed", 0) or 0), n_frames, note)])


def write_batch(store, run_id, frames, base):
    """Rows for a batch of frames. On Postgres t is an absolute instant: base + the frame's offset."""
    M, R, T, S = [], [], [], []
    for f in frames:
        m, r, t, s = frame_rows(run_id, f)
        M += m; R += r; T += t; S += s
    if store.kind == "postgres":
        def at(rows):
            return [(r[0], datetime.fromtimestamp(base + r[1], timezone.utc)) + tuple(r[2:]) for r in rows]
        M, R, T, S = at(M), at(R), at(T), at(S)
    return (store.insert("motor", ["run_id", "t", "forward", "turn", "reward", "exploring", "lobotomy"], M)
            + store.insert("neuron_rates", ["run_id", "t", "grp", "hz"], R)
            + store.insert("retina", ["run_id", "t", "col", "pres", "size", "mot"], T)
            + store.insert("spikes", ["run_id", "t", "neuron"], S))


def write_frames(store, run_id, frames, meta, note=""):
    t0 = time.time()
    open_episode(store, run_id, meta, len(frames), note)
    n = write_batch(store, run_id, frames, t0)
    print(f"{run_id}: {len(frames)} frames -> {n:,} rows into {store.where()} in {time.time() - t0:.1f} s")
    return n


def stream_live(store, ws_url, run_id, seconds):
    """Real-time ingest: a batch every second, for `seconds` seconds or until Ctrl-C when 0. The frames
    carry their own t relative to the demo's start; base is when this recording began, so the database
    holds wall-clock instants and the continuous aggregate can be read for 'the last 30 seconds'."""
    import asyncio
    import websockets
    base = time.time()
    open_episode(store, run_id, {"brain": "live", "stage": "robot"}, 0, ws_url)
    total = frames_total = 0
    t_off = None

    async def pump():
        nonlocal total, frames_total, t_off
        async with websockets.connect(ws_url) as ws:
            end = base + seconds if seconds > 0 else float("inf")
            batch, last_flush = [], time.time()
            while time.time() < end:
                f = json.loads(await asyncio.wait_for(ws.recv(), timeout=10))
                if t_off is None:
                    t_off = float(f.get("t", 0.0))         # so frame 0 lands at base
                f["t"] = float(f.get("t", 0.0)) - t_off
                batch.append(f)
                if time.time() - last_flush >= 1.0:
                    total += write_batch(store, run_id, batch, base)
                    frames_total += len(batch)
                    batch, last_flush = [], time.time()
                    if frames_total % 150 < 30:
                        print(f"  {run_id}: {frames_total} frames, {total:,} rows, "
                              f"{time.time() - base:.0f} s", flush=True)
            if batch:
                total += write_batch(store, run_id, batch, base)
                frames_total += len(batch)
    print(f"streaming {ws_url} -> {store.where()} as {run_id}"
          + (f" for {seconds:.0f} s" if seconds > 0 else " until Ctrl-C"), flush=True)
    try:
        asyncio.run(pump())
    except KeyboardInterrupt:
        pass
    cur = store.conn.cursor()
    cur.execute(f"UPDATE episodes SET n_frames = {store.ph} WHERE run_id = {store.ph}", (frames_total, run_id))
    store.conn.commit()
    print(f"{run_id}: {frames_total} frames, {total:,} rows, {time.time() - base:.0f} s", flush=True)
    return total


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
    if store.kind == "postgres":
        try:
            n_b, latest = store.query("SELECT count(*), max(bucket) FROM rates_1s")[0]
            lag = store.query("SELECT extract(epoch FROM now() - max(bucket)) FROM rates_1s")[0][0]
            print(f"\ncontinuous aggregate rates_1s: {n_b:,} one-second buckets, newest {lag:.0f} s ago")
        except Exception as e:  # noqa: BLE001
            store.conn.rollback(); print(f"\ncontinuous aggregate: not available ({type(e).__name__})")
        try:
            rows = store.query("""SELECT hypertable_name, before_compression_total_bytes, after_compression_total_bytes
                                  FROM hypertable_compression_stats('spikes')
                                  UNION ALL SELECT hypertable_name, before_compression_total_bytes, after_compression_total_bytes
                                  FROM hypertable_compression_stats('neuron_rates')""")
            for name, before, after in rows:
                if before and after:
                    print(f"compression {name}: {before/1e6:.1f} MB -> {after/1e6:.1f} MB, "
                          f"{100 * (1 - after / before):.0f}% smaller")
                else:
                    print(f"compression {name}: policy set, no chunk old enough yet (2 minutes)")
        except Exception as e:  # noqa: BLE001
            store.conn.rollback(); print(f"compression stats: not available ({type(e).__name__})")
    print("\nwhere the eye was looking when the steering neurons were loudest:")
    for col, p in store.query("""SELECT col, AVG(pres) p FROM retina
                                 GROUP BY col ORDER BY p DESC LIMIT 5"""):
        print(f"  column {col:2d} (azimuth {(col - 11.5) * 4.1:+5.1f} deg): mean presence {p:.3f}")


def main() -> None:
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--replay", default=None, help="a replay JSON from scripts/dump_replay.py")
    ap.add_argument("--live", default=None, metavar="WS", help="stream from the demo feed, e.g. ws://localhost:8765")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--dsn", default=os.environ.get("TIMESERIES_DSN"),
                    help="postgres://... (TimescaleDB / TigerData Cloud); omit to use a local SQLite file")
    ap.add_argument("--sqlite", default="logs/timeseries.db")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--seconds", type=float, default=0.0, help="--live: seconds to record; 0 = until Ctrl-C")
    a = ap.parse_args()
    store = Store(a.dsn, a.sqlite)

    if a.replay:
        d = json.load(open(a.replay))
        run = a.run_id or os.path.splitext(os.path.basename(a.replay))[0]
        write_frames(store, run, d["frames"], d.get("meta", {}), note=a.replay)
    if a.live:
        run = a.run_id or f"live_{int(time.time())}"
        stream_live(store, a.live, run, a.seconds)
    if a.report or not (a.replay or a.live):
        report(store)


if __name__ == "__main__":
    main()
