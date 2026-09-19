"""Companion brain command line.

    companion download            fetch connectome + annotations
    companion prune [--full]      build (and cache) the simulated circuit
    companion bench               measure brain-time / wall-time
    companion test-gf             looming neurons -> Giant Fiber escape sanity check
    companion run [...]           live: camera + PIR -> brain -> body
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .config import load_config


def cmd_download(args, cfg):
    from .data.download import download_all
    download_all(cfg, force=args.force)


def cmd_prune(args, cfg):
    from .data.prune import load_or_build
    c = load_or_build(cfg, full=args.full, rebuild=args.rebuild)
    print(json.dumps({"neurons": c.n, "connections": c.m, "full": args.full}))


def cmd_bench(args, cfg):
    from .data.prune import load_or_build
    from .sim.runner import BrainRunner
    from .sim.lif import HAVE_NUMBA
    c = load_or_build(cfg, full=args.full)
    r = BrainRunner(c, cfg, seed=0, use_numba=not args.numpy)
    print(f"[bench] engine: {'numba' if r.net.use_numba else 'numpy'} (numba available: {HAVE_NUMBA}), "
          f"dt={cfg.lif.dt_ms} ms, {c.n:,} neurons, {c.m:,} connections")
    # warm-up (also JIT compile)
    r.step_chunk()
    # drive a realistic load: looming on both sides
    for g in ("LC4", "LPLC2"):
        r.drive(g, 100.0)
    t0 = time.perf_counter()
    spikes = 0
    for _ in range(int(args.seconds * 1000 / r.chunk_ms)):
        spikes += int(r.step_chunk().sum())
    wall = time.perf_counter() - t0
    ratio = args.seconds / wall
    print(f"[bench] simulated {args.seconds:.1f} s of brain in {wall:.2f} s wall -> ratio {ratio:.2f}x "
          f"({'real-time OK' if ratio >= 1 else 'TOO SLOW'}), {spikes:,} spikes")
    rates = r.rates()
    top = sorted(rates.items(), key=lambda kv: -kv[1]["all"])[:6]
    print("[bench] most active readouts (Hz/neuron, last window):", {k: round(v["all"], 1) for k, v in top})


def cmd_test_gf(args, cfg):
    """Reference-model check (Shiu et al.): no background, no adaptation -> silent brain, looming fires GF."""
    from .data.prune import load_or_build
    from .sim.runner import BrainRunner
    cfg.lif["background_hz"] = 0.0
    cfg.lif["adapt_mv"] = 0.0
    c = load_or_build(cfg, full=args.full)
    r = BrainRunner(c, cfg, seed=1)
    # 1) silence: nothing should fire without input
    quiet = sum(int(r.step_chunk().sum()) for _ in range(20))
    print(f"[test-gf] 200 ms with no input: {quiet} spikes (expect 0)")
    # 2) looming on the right: LC4 + LPLC2 at 100 Hz for 300 ms
    gf = c.idx("GF", "all")
    r.drive("LC4", 100.0, "right")
    r.drive("LPLC2", 100.0, "right")
    first_ms = None
    gf_spikes = 0
    for i in range(30):
        counts = r.step_chunk()
        n = int(counts[gf].sum())
        gf_spikes += n
        if n and first_ms is None:
            first_ms = (i + 1) * r.chunk_ms
    rates = r.rates()
    print(f"[test-gf] Giant Fiber spikes in 300 ms of looming: {gf_spikes} (first at ~{first_ms} ms)")
    print(f"[test-gf] GF rate {rates['GF']['all']:.1f} Hz/neuron; other readouts: "
          + ", ".join(f"{g}={v['all']:.0f}" for g, v in rates.items() if v["all"] > 1))
    ok = quiet == 0 and gf_spikes > 0
    print("[test-gf]", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


def cmd_run(args, cfg):
    from .data.prune import load_or_build
    from .sim.runner import BrainRunner
    from .senses.camera import CameraStream, SyntheticLooming
    from .senses.optic_lobe import OpticLobe, features_to_rates
    from .body.decode import Decoder, load_or_calibrate
    from .body.eyes import eyes_for
    from .body.link import BodyLink, DryBody
    from .data.prune import circuit_key

    c = load_or_build(cfg, full=args.full)
    brain = BrainRunner(c, cfg, seed=None)
    baseline = load_or_calibrate(brain, cfg, circuit_key(cfg, args.full), rebuild=args.recalibrate)
    cam_cfg = cfg.senses.camera
    dash = None
    if not args.no_ui:
        from .ui.server import Dashboard
        dash = Dashboard(c, cfg, port=args.ui_port, open_browser=args.open)
    if args.sim_camera == "synthetic":
        source = SyntheticLooming(cam_cfg.width, cam_cfg.height, cam_cfg.fps, loop=dash is not None)
    else:
        source = CameraStream(args.sim_camera or cam_cfg.url, cam_cfg.width, cam_cfg.height)
    lobe = OpticLobe(cam_cfg.width, cam_cfg.height, cam_cfg.fps)
    decoder = Decoder(cfg, baseline)
    body = DryBody() if args.dry_body else BodyLink(cfg.body.host, cfg.body.port, cfg.body.listen_port)
    period = 1.0 / float(cfg.body.rate_hz)
    pir_until = 0.0
    last_pir = 0
    last_motion = time.time()
    last_sound = (0, 0.0)
    t_start = time.time()
    print(f"[run] {c.n:,} neurons; camera={'synthetic' if args.sim_camera == 'synthetic' else (args.sim_camera or cam_cfg.url)}; "
          f"body={'dry' if args.dry_body else cfg.body.host}. Ctrl-C to stop.", flush=True)
    feats = None
    try:
        while True:
            tick = time.time()
            # --- senses
            frame = source.read()
            if frame is not None:
                feats = lobe.process(frame)
                if feats.motion_energy > float(cfg.senses.wake_motion):
                    last_motion = tick
            elif args.sim_camera:
                if isinstance(source, SyntheticLooming) or source.is_file:
                    print("[run] video finished"); break
            body.poll()
            if body.pir and not last_pir:
                pir_until = tick + float(cfg.senses.pir.burst_ms) * 1e-3
                last_motion = tick
            last_pir = body.pir
            # --- drive the input neurons
            brain.clear_drive()
            if feats is not None:
                for (g, side), hz in features_to_rates(feats, cfg).items():
                    brain.drive(g, hz, side)
            if tick < pir_until:
                for g in cfg.senses.pir.groups:
                    brain.drive(g, float(cfg.senses.pir.rate_hz))
            # --- think
            brain.advance_to_wall()
            asleep = (tick - last_motion) > float(cfg.senses.sleep_after_s)
            d = decoder.decode({w: brain.rates(w) for w in decoder.windows}, asleep=asleep, now=tick)
            # --- act
            gx, gy = (feats.object_x, feats.object_y) if feats and feats.object_strength > 0 else (0.0, 0.0)
            track = cfg.decode.sounds.get(d.state, 0)
            if track and (last_sound[0] != track or tick - last_sound[1] > 3.0):
                last_sound = (track, tick)
            elif not track or tick - last_sound[1] > 0.5:
                track = 0
            packet = {
                "t": int((tick - t_start) * 1000), "state": d.state,
                "eyes": eyes_for(d, gx, gy),
                "blink": d.state == "escape",
                "sound": {"track": int(track), "vol": int(cfg.decode.volume)},
                "arms": {"l": round(d.lateral.get("track", 0.0), 2), "r": 0.0},
                "motor": d.motor,
            }
            body.send(packet)
            if dash is not None:
                recent = brain.take_recent()
                spiking = np.nonzero(recent)[0]
                dash.publish({
                    **packet,
                    "scores": {k: round(v, 3) for k, v in d.scores.items()},
                    "lateral": {k: round(v, 3) for k, v in d.lateral.items()},
                    "z": d.z,
                    "valence": round(d.valence, 3), "arousal": round(d.arousal, 3), "reward": round(d.reward, 3),
                    "asleep": asleep, "pir": int(body.pir),
                    "sees": feats.as_dict() if feats else None,
                    "rates": {g: {s: round(v, 1) for s, v in r.items()} for g, r in brain.rates().items()},
                    "spikes": spiking.tolist(),
                    "n_spikes": int(recent.sum()),
                    "top_types": dash.top_types(brain.window_counts()),
                    "brain": {"behind_ms": round(brain.behind_ms), "chunk_ms": round(brain.last_chunk_wall_ms, 1),
                              "brain_s": round(brain.brain_ms / 1000, 1), "n": c.n},
                }, frame=source.preview)
            if args.verbose:
                act = " ".join(f"{k}={v:.2f}" for k, v in d.motor.items() if abs(v) > 0.05)
                fl, fr = (feats.left, feats.right) if feats else (None, None)
                see = (f"loom L/R={fl.loom_fast:.2f}/{fr.loom_fast:.2f} obj L/R={fl.small_object:.2f}/{fr.small_object:.2f} "
                       f"motion={feats.motion_energy:.2f}") if feats else "no frame"
                print(f"[brain] t={packet['t'] / 1000:6.1f}s  {d.state:<8} {act:<40} | sees: {see}", flush=True)
            # --- pace
            dt = time.time() - tick
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        print("\n[run] stopped")
    finally:
        source.close()
        body.close()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="companion", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. --set prune.hops_forward=2 (repeatable)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("download"); p.add_argument("--force", action="store_true"); p.set_defaults(fn=cmd_download)
    p = sub.add_parser("prune"); p.add_argument("--full", action="store_true"); p.add_argument("--rebuild", action="store_true"); p.set_defaults(fn=cmd_prune)
    p = sub.add_parser("bench"); p.add_argument("--full", action="store_true"); p.add_argument("--numpy", action="store_true", help="force the NumPy engine")
    p.add_argument("--seconds", type=float, default=3.0); p.set_defaults(fn=cmd_bench)
    p = sub.add_parser("test-gf"); p.add_argument("--full", action="store_true"); p.set_defaults(fn=cmd_test_gf)
    p = sub.add_parser("run"); p.add_argument("--full", action="store_true")
    p.add_argument("--sim-camera", default=None, metavar="SRC", help="'synthetic' or a video file / URL instead of the ESP32-CAM")
    p.add_argument("--dry-body", action="store_true", help="print body packets instead of sending UDP")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--recalibrate", action="store_true", help="re-measure the resting baseline")
    p.add_argument("--no-ui", action="store_true", help="do not start the live dashboard")
    p.add_argument("--ui-port", type=int, default=8600)
    p.add_argument("--open", action="store_true", help="open the dashboard in the default browser")
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    cfg = load_config(args.config, overrides=_parse_overrides(args.set))
    args.fn(args, cfg)


def _parse_overrides(items):
    import yaml
    out = {}
    for item in items:
        key, _, val = item.partition("=")
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(val)
    return out


if __name__ == "__main__":
    main()
