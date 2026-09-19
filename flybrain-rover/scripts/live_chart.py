"""
live_chart.py -- the fly's neurons on the wall, straight out of Tiger Data, while the robot runs.

    python scripts/live_chart.py                     # http://localhost:8790, reads $TIMESERIES_DSN
    python scripts/live_chart.py --run live_1758... # pin a run instead of following the newest

Reads the continuous aggregate `rates_1s` (one-second population firing rates, kept fresh by
TimescaleDB and computed for the newest second on the fly), never the raw neuron_rates table, so each
poll is a few dozen rows however long the robot has been running. The page polls once a second and draws
the last 30 seconds of the eye (LC10a), the steering neurons (DNa02) and the escape neuron (GF).

This is the demo's fourth terminal: robot daemon, brain, voice, and this. The robot's brain does not know
the database exists; export_timeseries.py --live copies frames out of the viz websocket into Tiger, and this
reads them back. Cloud is downstream of the loop, never in it.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from voice import load_dotenv  # noqa: E402

GROUPS = ["LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R", "GF"]

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><title>FlyBrain live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>
 :root{--bg:#0a0a0a;--panel:#151312;--rule:#2b2724;--text:#e8e6e4;--dim:#8a8580;--coral:#FF715B;--blue:#5aa9ff}
 body{margin:0;background:var(--bg);color:var(--text);font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;padding:22px}
 h1{font-size:18px;font-weight:600;margin:0 0 4px;letter-spacing:-.01em}
 .sub{color:var(--dim);font-size:12px;margin-bottom:16px}
 .sub b{color:var(--text);font-weight:600}
 .grid{display:grid;grid-template-columns:1fr;gap:14px;max-width:1100px}
 .panel{background:var(--panel);border:1px solid var(--rule);padding:12px 14px}
 .panel h2{font-size:12px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--dim);margin:0 0 8px;display:flex;justify-content:space-between}
 .panel h2 span{color:var(--text);font-variant-numeric:tabular-nums}
 canvas{width:100%;height:150px;display:block}
 .foot{color:var(--dim);font-size:11px;margin-top:14px}
</style></head><body>
<h1>FlyBrain, live from Tiger Data</h1>
<div class="sub">run <b id="run">…</b> · continuous aggregate <b>rates_1s</b> · last 30 s · newest bucket <b id="lag">…</b> ago · <b id="rows">…</b> rows in the raw table</div>
<div class="grid">
 <div class="panel"><h2>eye · LC10a left <i style="color:#FF715B">■</i> right <i style="color:#5aa9ff">■</i><span id="v0"></span></h2><canvas id="c0"></canvas></div>
 <div class="panel"><h2>steering · DNa02 left <i style="color:#FF715B">■</i> right <i style="color:#5aa9ff">■</i><span id="v1"></span></h2><canvas id="c1"></canvas></div>
 <div class="panel"><h2>escape · giant fibre<span id="v2"></span></h2><canvas id="c2"></canvas></div>
</div>
<div class="foot">The brain does not know this page exists. Frames leave the demo over a local websocket, land in TimescaleDB hypertables, are rolled up by a continuous aggregate every 5 s (the newest second computed on read), and this page polls that aggregate once a second.</div>
<script>
const PAIRS=[["LC10a_L","LC10a_R"],["DNa02_L","DNa02_R"],["GF",null]];
function draw(id,series,vmax){
  const c=document.getElementById(id),dpr=window.devicePixelRatio||1;
  c.width=c.clientWidth*dpr;c.height=c.clientHeight*dpr;const x=c.getContext('2d');x.scale(dpr,dpr);
  const W=c.clientWidth,H=c.clientHeight;x.fillStyle='#0d0b0a';x.fillRect(0,0,W,H);
  x.strokeStyle='#221e1b';x.lineWidth=1;for(let g=1;g<4;g++){x.beginPath();x.moveTo(0,H*g/4);x.lineTo(W,H*g/4);x.stroke();}
  x.fillStyle='#5c5854';x.font='10px IBM Plex Mono';x.fillText(vmax.toFixed(0)+' Hz',4,11);x.fillText('0',4,H-4);
  series.forEach(([pts,color])=>{if(!pts||!pts.length)return;x.strokeStyle=color;x.lineWidth=2;x.beginPath();
    pts.forEach(([t,v],i)=>{const px=W*(t/30),py=H-(H-6)*Math.min(1,v/vmax)-3;i?x.lineTo(px,py):x.moveTo(px,py);});x.stroke();
    const [t,v]=pts[pts.length-1];x.fillStyle=color;x.beginPath();x.arc(W*(t/30),H-(H-6)*Math.min(1,v/vmax)-3,3,0,7);x.fill();});
}
async function tick(){
  try{
    const r=await fetch('/api/rates?seconds=30');const d=await r.json();
    document.getElementById('run').textContent=d.run||'none';
    document.getElementById('lag').textContent=d.lag_s==null?'–':d.lag_s.toFixed(1)+' s';
    document.getElementById('rows').textContent=(d.raw_rows||0).toLocaleString();
    PAIRS.forEach(([a,b],i)=>{
      const sa=d.series[a]||[],sb=b?(d.series[b]||[]):[];
      const vmax=Math.max(20,...sa.map(p=>p[1]),...sb.map(p=>p[1]))*1.15;
      draw('c'+i,[[sa,'#FF715B'],[sb,'#5aa9ff']],vmax);
      const la=sa.length?sa[sa.length-1][1].toFixed(0):'–',lb=sb.length?sb[sb.length-1][1].toFixed(0):'';
      document.getElementById('v'+i).textContent=b?(la+' / '+lb+' Hz'):(la+' Hz');
    });
  }catch(e){document.getElementById('lag').textContent='no data';}
  setTimeout(tick,1000);
}
tick();
</script></body></html>"""


class Source:
    def __init__(self, dsn, run=None):
        import psycopg
        self.conn = psycopg.connect(dsn, autocommit=True)
        self.run = run
        self.lock = threading.Lock()

    def newest_run(self):
        with self.conn.cursor() as cur:
            cur.execute("SELECT run_id FROM episodes ORDER BY started_at DESC LIMIT 1")
            r = cur.fetchone()
        return r[0] if r else None

    def rates(self, seconds=30):
        run = self.run or self.newest_run()
        if not run:
            return {"run": None, "series": {}, "lag_s": None, "raw_rows": 0}
        with self.lock, self.conn.cursor() as cur:
            cur.execute("SELECT max(bucket) FROM rates_1s WHERE run_id = %s", (run,))
            newest = cur.fetchone()[0]
            if newest is None:
                return {"run": run, "series": {}, "lag_s": None, "raw_rows": 0}
            cur.execute("""SELECT grp, extract(epoch FROM bucket - (%s::timestamptz - make_interval(secs => %s))), hz
                           FROM rates_1s
                           WHERE run_id = %s AND grp = ANY(%s) AND bucket > %s::timestamptz - make_interval(secs => %s)
                           ORDER BY grp, bucket""", (newest, seconds, run, GROUPS, newest, seconds))
            series = {}
            for grp, t, hz in cur.fetchall():
                series.setdefault(grp, []).append([float(t), float(hz)])
            cur.execute("SELECT extract(epoch FROM now() - %s::timestamptz)", (newest,))
            lag = float(cur.fetchone()[0])
            cur.execute("SELECT count(*) FROM neuron_rates WHERE run_id = %s", (run,))
            raw = int(cur.fetchone()[0])
        return {"run": run, "series": series, "lag_s": lag, "raw_rows": raw}


def main():
    load_dotenv()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8790)
    ap.add_argument("--run", default=None, help="pin a run_id; default follows the newest episode")
    ap.add_argument("--dsn", default=os.environ.get("TIMESERIES_DSN"))
    a = ap.parse_args()
    if not a.dsn:
        sys.exit("set TIMESERIES_DSN (Tiger Cloud / TimescaleDB); the live chart reads the continuous aggregate")
    src = Source(a.dsn, a.run)

    class H(BaseHTTPRequestHandler):
        def log_message(self, *args):  # quiet
            pass

        def do_GET(self):
            if self.path.startswith("/api/rates"):
                try:
                    secs = int(self.path.split("seconds=")[1].split("&")[0]) if "seconds=" in self.path else 30
                    body = json.dumps(src.rates(max(5, min(300, secs)))).encode()
                    self.send_response(200); self.send_header("content-type", "application/json")
                except Exception as e:  # noqa: BLE001
                    body = json.dumps({"error": f"{type(e).__name__}: {e}"}).encode()
                    self.send_response(500); self.send_header("content-type", "application/json")
            else:
                body = PAGE.encode()
                self.send_response(200); self.send_header("content-type", "text/html; charset=utf-8")
            self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)

    print(f"live chart on http://localhost:{a.port}  (reads rates_1s from {a.dsn.split('@')[-1].split('?')[0]})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()


if __name__ == "__main__":
    main()
