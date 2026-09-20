"""Companion brain command line.

    companion download            fetch connectome + annotations
    companion prune [--full]      build (and cache) the simulated circuit
    companion bench               measure brain-time / wall-time
    companion test-gf             looming neurons -> Giant Fiber escape sanity check
    companion run [...]           live: camera + PIR -> brain -> body
    companion hunt [...]          the hunt arena: train / evaluate the fly as a human-hunting vehicle
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
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


def cmd_experiment(args, cfg):
    """Two-odor differential conditioning, headless: odor A (grape) is paired with sugar, odor B
    (lemon) is presented alone. Before and after each training block the brain is probed with
    each odor and the MBON valence (approach minus avoid, z-scored) is reported."""
    from .data.prune import load_or_build, circuit_key
    from .sim.runner import BrainRunner
    from .body.decode import Decoder, load_or_calibrate
    c = load_or_build(cfg, full=False)
    r = BrainRunner(c, cfg, seed=args.seed)
    base = load_or_calibrate(r, cfg, circuit_key(cfg, False))
    dec = Decoder(cfg, base)
    pl = r.plasticity
    if pl is None:
        print("[experiment] learning is disabled in the config"); sys.exit(1)
    pl.enabled = not args.no_learn
    print(f"[experiment] mushroom body: {pl.n_kc} Kenyon cells, {pl.n_mbon} MBONs with dopamine compartments, {pl.n_syn:,} plastic KC->MBON synapses; learning {'ON' if pl.enabled else 'OFF (control)'}")
    odors = {"A (grape)": {"ORN_fruit": args.odor_hz}, "B (lemon)": {"ORN_lemon": args.odor_hz}}
    kc = pl.kc_idx

    mb_av, mb_ap = c.idx("MBON_avoid"), c.idx("MBON_approach")

    def probe(reps=3):
        """Odor-evoked MBON responses (Hz above rest) and decoded valence, averaged over probes."""
        out = {}
        for name, drive in odors.items():
            vals, av, ap = [], [], []
            for _ in range(reps):
                cap = {}
                rest, odor = r.probe(drive, seconds=1.0, capture=lambda rr: cap.update(w=rr.window_counts().copy()))
                vals.append(dec.decode({w: odor for w in dec.windows}, now=0).valence
                            - dec.decode({w: rest for w in dec.windows}, now=0).valence)
                av.append(cap["w"][mb_av].mean() - rest["MBON_avoid"]["all"])
                ap.append(cap["w"][mb_ap].mean() - rest["MBON_approach"]["all"])
            out[name] = {"valence": float(np.mean(vals)), "avoid": float(np.mean(av)), "approach": float(np.mean(ap))}
        return out

    def odor_kcs(drive, hz):
        """Kenyon cells excited by an odor (rate above rest during a probe)."""
        found = {}
        r.probe(drive, seconds=1.0, capture=lambda rr: found.update(e=pl.elig[kc].copy()))
        return set(kc[np.nonzero(found["e"] > 0.05)[0]].tolist())

    def specific_weights():
        wa = [i for i, k in enumerate(pl.pre) if k in kcA and pl.kind_of[int(pl.post[i])] == "reward"]
        wb = [i for i, k in enumerate(pl.pre) if k in kcB and pl.kind_of[int(pl.post[i])] == "reward"]
        w = pl.net.data[pl.syn] / np.where(pl.w0 == 0, 1, pl.w0)
        return (float(w[wa].mean()) if wa else 1.0, float(w[wb].mean()) if wb else 1.0, len(wa), len(wb))

    # warm-up so DAN resting levels settle
    for _ in range(int(6 * 100)):
        r.step_chunk()
    kcA = odor_kcs({"ORN_fruit": args.odor_hz}, args.odor_hz); kcB = odor_kcs({"ORN_lemon": args.odor_hz}, args.odor_hz)
    print(f"[experiment] Kenyon cells excited by A: {len(kcA)}, by B: {len(kcB)}, shared: {len(kcA & kcB)}")
    v0 = probe()
    fmt = lambda v: f"avoid-MBONs {v['avoid']:+.1f} Hz, approach-MBONs {v['approach']:+.1f} Hz, valence {v['valence']:+.2f}"
    print(f"[experiment] naive: A -> {fmt(v0['A (grape)'])} | B -> {fmt(v0['B (lemon)'])}")
    for block in range(args.blocks):
        # training: A + sugar for train_s, then rest, then B alone for train_s, then rest
        for name, drive, sugar in (("A+sugar", {"ORN_fruit": args.odor_hz}, True), ("B alone", {"ORN_lemon": args.odor_hz}, False)):
            dmax = 0.0
            for _ in range(int(args.train_s * 100)):
                r.clear_drive()
                for g, hz in drive.items():
                    r.drive(g, hz)
                if sugar:
                    r.drive("GRN_sugar", 150.0)
                    for g in cfg.learning.reward.groups:
                        r.drive(g, float(cfg.learning.reward.hz))
                r.step_chunk()
                dmax = max(dmax, float(pl.dop.max()) if len(pl.dop) else 0.0)
            print(f"  block {block + 1} {name:8s}: peak dopamine {dmax:.2f}, eligible KCs {pl.summary()['kc_eligible']:.0%}")
            r.clear_drive()
            for _ in range(int(3 * 100)):
                r.step_chunk()
            sm = pl.summary(); wa, wb, na, nb = specific_weights()
            print(f"  block {block + 1} {name:8s}: KC->MBON weight in reward compartments from A-cells {wa:.2f} (n={na}) vs B-cells {wb:.2f} (n={nb}); "
                  f"all reward comps {sm['weight_reward_comp']:.2f}, punishment comps {sm['weight_punish_comp']:.2f}")
        v = probe()
        print(f"[experiment] after block {block + 1}: A -> {fmt(v['A (grape)'])} | B -> {fmt(v['B (lemon)'])}")
    a0, a1 = v0["A (grape)"]["avoid"], v["A (grape)"]["avoid"]
    b0, b1 = v0["B (lemon)"]["avoid"], v["B (lemon)"]["avoid"]
    wa, wb, _, _ = specific_weights()
    drop_a = (a0 - a1) / max(a0, 0.5); drop_b = (b0 - b1) / max(b0, 0.5)
    ok = a0 > 1.0 and drop_a > 0.5 and drop_a > drop_b + 0.25 and wa < 0.7 * wb
    print(f"[experiment] avoidance drive of the rewarded odor fell {100 * drop_a:.0f}% (unrewarded: {100 * drop_b:.0f}%); "
          f"KC->MBON synapses from A-cells at {wa:.2f} of naive vs B-cells {wb:.2f}")
    print("[experiment]", "LEARNED: the rewarded smell no longer drives avoidance; the unrewarded one is unchanged" if ok else "no differential learning measured")
    return ok


def cmd_batch(args, cfg):
    """Many headless arena assays in parallel: train and test the fly quickly, with controls."""
    from .batch import run_batch
    from .config import BRAIN_DIR
    save = (BRAIN_DIR / "data" / "cache" / "learned_weights.npz") if args.save else None
    run_batch(args.config, _parse_overrides(args.set), runs=args.runs, workers=args.workers, control=args.control, save=save)


def cmd_hunt(args, cfg):
    """The hunt arena, headless. One fly through N episodes (default), CMA-ES training of many flies
    (--train), or evaluation of a saved hunter (--load, usually with --no-learn)."""
    from .sim.hunt import run_hunt, train, save_hunter, load_hunter, nested, default_gains
    from .config import BRAIN_DIR, apply_dotted
    out_dir = BRAIN_DIR / "data" / "cache"
    workers = args.workers or (os.cpu_count() or 4)
    if args.train:
        start = json.loads(Path(args.gains).read_text())["gains"] if args.gains else None
        train(args.config, _parse_overrides(args.set), generations=args.generations, popsize=args.pop, episodes=args.episodes,
              workers=workers, out_dir=out_dir, seed=args.seed, final_episodes=args.final_episodes, start_gains=start)
        return
    weights = None
    gains = {}                                   # no hunter loaded: the config's own defaults
    brain = None
    if args.load:
        weights, gains, meta = load_hunter(args.load)
        if meta.get("brain"):
            from .hunt_gpu.connectome import load_brain
            brain = load_brain(args.load)
            print(f"[hunt] loaded the GPU-trained connectome brain {args.load}: {brain['meta']}", flush=True)
        else:
            print(f"[hunt] loaded {args.load}: trained touch rate {float(meta.get('touch_rate', 0)):.0%}, score {float(meta.get('score', 0)):.2f}", flush=True)
    elif args.gains:
        gains = json.loads(Path(args.gains).read_text())["gains"]
    apply_dotted(cfg, gains)
    learn = not args.no_learn
    print(f"[hunt] one fly, {args.episodes} episodes of {cfg.hunt.episode_s:.0f} s, {int(cfg.hunt.people)} people, learning {'ON' if learn else 'OFF'}"
          f"{', scripted controller (not the brain)' if args.scripted else ''}", flush=True)
    res = run_hunt(cfg, seed=args.seed, episodes=args.episodes, learn=learn, weights=weights, scripted=args.scripted, verbose=True, brain=brain)
    print(f"[hunt] touch rate {res['touch_rate']:.0%}, score {res['score']:.2f} (later half {res['score_late']:.2f}), {res['wall_s']:.0f} s wall", flush=True)
    if args.save and "weights" in res:
        path = save_hunter(out_dir / "hunter.npz", res, gains)
        print(f"[hunt] saved the fly's synapses + gains to {path}", flush=True)


def cmd_hunt_gpu(args, cfg):
    """The hunt arena as a batch of tensors on a GPU: PPO training (--train), evaluation of a saved
    hunter (--load) or of the scripted hunter (default), and the CPU-arena cross-check (--cpu-check)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("hunt-gpu needs PyTorch: run `uv sync --extra gpu` in brain/")
    from .config import BRAIN_DIR
    from .sim.hunt import score
    from .hunt_gpu.ppo import Trainer, evaluate, summarize
    from .hunt_gpu.brain import HunterNet, EvaderNet
    from .hunt_gpu.arena import pick_device
    out_dir = Path(args.out) if args.out else BRAIN_DIR / "data" / "cache"
    dev = pick_device(args.device)
    humans = None                                     # the other player: patrol | flee | a trained EvaderNet
    choice = args.humans
    if choice is None and args.load and not args.train:
        sibling = Path(args.load).with_name(Path(args.load).name.replace("hunter", "evaders"))
        choice = str(sibling) if sibling.exists() and sibling != Path(args.load) else None
    if choice in ("patrol", "flee", "learn", "wander"):
        cfg.hunt_gpu.humans["mode"] = choice
    elif choice:
        humans, hmeta = EvaderNet.load(choice, dev)
        cfg.hunt_gpu.humans["mode"] = "learn"
        print(f"[hunt-gpu] loaded the runners {choice}: {hmeta}", flush=True)
    if args.train:
        tr = Trainer(cfg, n_envs=args.envs, device=args.device, seed=args.seed, out_dir=out_dir)
        if args.load:
            net, meta = HunterNet.load(args.load, dev)
            tr.net.load_state_dict(net.state_dict())
            print(f"[hunt-gpu] resumed the fly from {args.load} ({meta})", flush=True)
        if humans is not None and tr.hnet is not None:
            tr.hnet.load_state_dict(humans.state_dict())
            print("[hunt-gpu] resumed the runners too", flush=True)
        res = tr.train(updates=args.updates or int(cfg.hunt_gpu.get("updates", 300)), save_every=args.save_every)
        print(f"[hunt-gpu] done: {res['steps'] / 1e6:.1f} M ticks in {res['wall_s'] / 60:.1f} min; best {res['best']} -> {out_dir / 'hunter_gpu.pt'} "
              f"(evaluate it: companion hunt-gpu --load {out_dir / 'hunter_gpu.pt'})", flush=True)
        return
    net = None
    if args.load:
        net, meta = HunterNet.load(args.load, dev)
        print(f"[hunt-gpu] loaded {args.load}: {meta}", flush=True)
    if args.watch:
        from .hunt_gpu.viewer import serve
        serve(cfg, net, port=args.port, open_browser=args.open, seed=args.seed, speed=args.speed, net_label=Path(args.load).name if args.load else "", humans=humans, body=_s1_body(args, cfg))
        return
    what = "scripted hunter" if net is None else "trained fly"
    mode = cfg.hunt_gpu.humans.get("mode", "learn")
    opp = {"patrol": "patrol walkers", "wander": "people walking about", "flee": "scripted runners", "learn": "trained runners" if humans is not None else "scripted runners (no trained runners given: --humans PT)"}[mode]
    print(f"[hunt-gpu] {what} vs {opp}, {args.episodes} seeded episodes of {cfg.hunt.episode_s:.0f} s as one batch on {dev}", flush=True)
    t0 = time.time()
    res = evaluate(cfg, net, args.episodes, device=args.device, seed=args.seed, deterministic=not args.sample, humans=humans)
    sm = summarize(res, float(cfg.hunt.episode_s), float(cfg.hunt.side_reward))
    trk = f", then tracked them {sm['track']:.0%} of the time, {sm['touches']:.1f} touches per episode" if sm.get("track") is not None else ""
    print(f"[hunt-gpu] touch rate {sm['touch_rate']:.0%}, mean time to first touch {sm['t_touch']} s, head-on {sm['frontal']}, score {sm['score']:.2f}, "
          f"target switches {sm['switches']}{trk}; the humans kept {sm['mean_dist']:.1f} m away on average; nobody in range {sm['blind']:.0%} of the time, of which standing still {sm['idle']:.0%} | {time.time() - t0:.1f} s wall", flush=True)
    if args.cpu_check:
        if net is None:
            raise SystemExit("--cpu-check needs --load")
        from .hunt_gpu.bridge import run_cpu
        seeds = [e["seed"] for e in res]
        print(f"[hunt-gpu] the same fly through the CPU arena (sim/hunt_arena.py, patrol walkers), same {len(seeds)} seeds ...", flush=True)
        cpu = run_cpu(cfg, net, seeds, verbose=args.verbose)
        print(f"[hunt-gpu] CPU arena: touch rate {np.mean([e['touched'] for e in cpu]):.0%}, score {score(cpu, float(cfg.hunt.episode_s), float(cfg.hunt.side_reward)):.2f} "
              f"(GPU arena on the same seeds: {sm['touch_rate']:.0%}, {sm['score']:.2f})", flush=True)


def cmd_hunt_brain(args, cfg):
    """The real fly's brain in the hunt arena: the hunter subcircuit of the connectome as a batched rate model
    trained by PPO (--train), evaluated as a rate model (default) or as spiking neurons (--spiking), checked
    against the spiking brain's responses (--check), or watched live with its neurons (--watch)."""
    try:
        import torch  # noqa: F401
    except ImportError:
        raise SystemExit("hunt-brain needs PyTorch: run `uv sync --extra gpu` in brain/")
    from .config import BRAIN_DIR, apply_dotted
    from .hunt_gpu.brain import EvaderNet
    from .hunt_gpu.arena import pick_device
    from .hunt_gpu.ppo import summarize
    from .hunt_gpu.brain_train import BrainTrainer, build_brain, evaluate_brain, evaluate_spiking, SpikingHunter
    from .hunt_gpu.connectome import load_brain
    out_dir = Path(args.out) if args.out else BRAIN_DIR / "data" / "cache"
    dev = pick_device(args.device)
    apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))          # the hunter's fixed settings (no looming -> no escape)
    cfg.learning["enabled"] = False
    if not args.no_brake:
        cfg.hunt["freeze_brakes"] = True                             # DNp09 may brake the vehicle: the brain can learn to slow at contact
    if bool(cfg.hunt_gpu.brain.get("turn_contrast", True)):            # the normalized turn readout (see RateBrain.decode)
        cfg.decode.motor.channels.turn["contrast"] = True
        if "z0" not in cfg.decode.motor.channels.turn:
            cfg.decode.motor.channels.turn["z0"] = 0.05
    if args.fresh_brake:                                             # a brain trained before the brake reached DNp09 had switched it off
        cfg.hunt_gpu.brain["fresh_gains"] = ["hunt.near_gain", "hunt.near_deg", "decode.motor.channels.freeze.z_ref", "decode.motor.channels.turn.z0"]
    hops = tuple(int(h) for h in (args.hops.split(",") if args.hops else cfg.hunt_gpu.brain.get("hops", [2, 1])))
    gains = {}
    if args.gains:
        from .sim.hunt import load_hunter
        _, gains, _ = load_hunter(args.gains)
        print(f"[brain] starting from the connectome fly's tuned gains in {args.gains}", flush=True)
    if args.load:
        gains = {k: v for k, v in load_brain(args.load)["gains"].items() if k not in set(cfg.hunt_gpu.brain.get("fresh_gains", []))}
    humans = None
    choice = args.humans
    if choice is None and args.load:
        sibling = Path(args.load).with_name("evaders_gpu.pt")
        choice = str(sibling) if sibling.exists() else None
    if choice in ("patrol", "flee", "learn", "wander"):
        cfg.hunt_gpu.humans["mode"] = choice
    elif choice:
        humans, hmeta = EvaderNet.load(choice, dev)
        cfg.hunt_gpu.humans["mode"] = "learn"
        cfg.hunt_gpu.humans["frozen"] = not args.adversarial            # the benchmark's runners, as they are, unless asked to co-train
        print(f"[brain] loaded the runners {choice} ({'co-training' if args.adversarial else 'frozen'}): {hmeta}", flush=True)
    if args.evolve_spiking:
        from .hunt_gpu.brain_es import evolve_spiking
        best = evolve_spiking(cfg, generations=args.generations, popsize=args.pop, episodes=args.episodes, out_dir=out_dir, seed=args.seed, start=args.load,
                              humans=humans, hops=hops)
        print(f"[brain-es] best (spiking): {best}", flush=True)
        return
    if args.evolve:
        from .hunt_gpu.brain_es import evolve
        best = evolve(cfg, generations=args.generations, popsize=args.pop, episodes=args.episodes, out_dir=out_dir, seed=args.seed, start=args.load,
                      humans=humans, hops=hops, device="cpu" if args.device == "auto" else args.device)
        print(f"[brain-es] best: {best}", flush=True)
        return
    if args.train:
        tr = BrainTrainer(cfg, n_envs=args.envs, device=args.device, seed=args.seed, out_dir=out_dir, hops=hops, gains=gains, resume=args.load)
        if humans is not None and tr.hnet is not None:
            tr.hnet.load_state_dict(humans.state_dict())
        res = tr.train(updates=args.updates or int(cfg.hunt_gpu.brain.get("updates", 150)), save_every=args.save_every)
        print(f"[brain] done: {res['steps'] / 1e6:.1f} M ticks in {res['wall_s'] / 60:.1f} min -> {out_dir / 'hunter_brain.npz'}", flush=True)
        return
    hc, brain = build_brain(cfg, hops=hops, gains=gains, device=dev)
    if args.load:
        meta = brain.load_params(args.load)
        print(f"[brain] loaded {args.load}: {meta}", flush=True)
    if args.check:
        from .sim.runner import BrainRunner
        r = BrainRunner(hc.circuit, cfg, seed=0)
        if args.load:
            from .hunt_gpu.connectome import apply_to_runner
            apply_to_runner(r, load_brain(args.load))
        probes = {"rest": {}, "object left 150 Hz": {("LC10a", "left"): 150, ("LC11", "left"): 150}, "object left 75 Hz": {("LC10a", "left"): 75, ("LC11", "left"): 75},
                  "object right 150 Hz": {("LC10a", "right"): 150, ("LC11", "right"): 150}, "bar left": {("LC12", "left"): 150, ("LC15", "left"): 150}, "heat left": {("TRN_hot", "left"): 150}}
        print(f"[brain] spiking vs rate model, readout rates in Hz (left / right):")
        for name, drives in probes.items():
            r.net.clear_rates()
            for _ in range(60):
                r.step_chunk()
            for _ in range(100):
                r.net.clear_rates()
                for (g, s), hz in drives.items():
                    r.drive(g, hz, s)
                r.step_chunk()
            lif = r.rates(300)
            rate = brain.probe(drives)
            print(f"  {name:20s} " + " | ".join(f"{g}: spiking {lif[g]['left']:6.1f}/{lif[g]['right']:6.1f}  rate {rate[(g, 'left')]:6.1f}/{rate[(g, 'right')]:6.1f}" for g in ("DNa02", "DNa01", "DN_all")), flush=True)
        return
    if args.real:
        from .hunt_gpu.real import serve as serve_real
        hunter = SpikingHunter(cfg, hc, args.load)
        pir = None
        if args.pir:
            from .body.link import BodyLink
            pir = BodyLink(cfg.body.host, cfg.body.port, cfg.body.listen_port)
        serve_real(cfg, hunter, camera=args.camera or str(cfg.senses.camera.url), cam_fov_deg=float(args.cam_fov), body=_s1_body(args, cfg), pir=pir,
                   port=args.port, open_browser=args.open, rover_on=not args.rover_off)
        return
    if args.watch:
        from .hunt_gpu.viewer import serve
        hunter = SpikingHunter(cfg, hc, args.load)
        serve(cfg, None, port=args.port, open_browser=args.open, seed=args.seed, speed=args.speed, net_label=Path(args.load).name if args.load else "the untrained connectome", humans=humans, spiking=hunter, body=_s1_body(args, cfg))
        return
    opp = {"patrol": "patrol walkers", "wander": "people walking about", "flee": "scripted runners", "learn": "trained runners" if humans is not None else "scripted runners (no trained runners given)"}[cfg.hunt_gpu.humans.get("mode", "learn")]
    t0 = time.time()
    if args.spiking:
        print(f"[brain] the spiking fly ({hc.n:,} LIF neurons, {'trained' if args.load else 'untrained'}) vs {opp}, {args.episodes} seeded episodes one after another on the CPU", flush=True)
        hunter = SpikingHunter(cfg, hc, args.load)
        res = evaluate_spiking(cfg, hunter, args.episodes, seed=args.seed, humans=humans, verbose=True)
    else:
        print(f"[brain] the rate-model fly ({'trained' if args.load else 'untrained'}) vs {opp}, {args.episodes} seeded episodes as one batch on {dev}", flush=True)
        res = evaluate_brain(cfg, brain, args.episodes, device=args.device, seed=args.seed, humans=humans)
    sm = summarize(res, float(cfg.hunt.episode_s), float(cfg.hunt.side_reward))
    trk = f", then tracked them {sm['track']:.0%} of the time, {sm['touches']:.1f} touches per episode" if sm.get("track") is not None else ""
    print(f"[brain] touch rate {sm['touch_rate']:.0%}, mean time to first touch {sm['t_touch']} s, head-on {sm['frontal']}, score {sm['score']:.2f}{trk}; "
          f"nobody in range {sm['blind']:.0%} of the time, of which standing still {sm['idle']:.0%} | {time.time() - t0:.0f} s wall", flush=True)


def _s1_body(args, cfg, with_link: bool = True):
    """`--s1 [PORT]`: a RoboMaster S1 on S-Bus mirroring the fly (docs/wiring.md); None when not asked for.
    `--s1 udp`: the body ESP32 streams the S-Bus (firmware/body); the channels ride in the brain packet, so this
    returns a MultiBody(S1Body, BodyLink) unless `with_link` is False (the caller has its own BodyLink)."""
    s1_cfg = dict(cfg.body.get("s1", {}))
    mode = getattr(args, "s1", None)
    if mode is None and not s1_cfg.get("enabled"):
        return None
    from .body.s1 import S1Body, MultiBody
    if mode == "udp" or s1_cfg.get("port") == "udp":
        s1 = S1Body(s1_cfg, udp=True)
        print(f"[body] RoboMaster S1 via the body ESP32 ({cfg.body.host}: S-Bus channels in the packet, free_mode={s1.cfg['free_mode']}, speed {s1.cfg['speed']})", flush=True)
        if not with_link:
            return s1
        from .body.link import BodyLink
        return MultiBody(s1, BodyLink(cfg.body.host, cfg.body.port, cfg.body.listen_port))
    s1 = S1Body(s1_cfg, port=(mode or None))
    print(f"[body] RoboMaster S1 on {s1.cfg['port']} (S-Bus, free_mode={s1.cfg['free_mode']}, speed {s1.cfg['speed']}, sticks fwd {s1.cfg['stick_forward']} / yaw {s1.cfg['stick_yaw']})", flush=True)
    return s1


def cmd_run(args, cfg):
    from .data.prune import load_or_build
    from .sim.runner import BrainRunner
    from .senses.camera import CameraStream, SyntheticLooming
    from .senses.optic_lobe import OpticLobe, features_to_rates
    from .body.decode import Decoder, load_or_calibrate
    from .body.eyes import eyes_for
    from .body.link import BodyLink, DryBody
    from .data.prune import circuit_key

    hunter_gains = {}
    brain_file = None
    if args.load_weights:
        from .sim.hunt import load_hunter, is_brain_file
        brain_file = args.load_weights if is_brain_file(args.load_weights) else None
        from .config import apply_dotted
        _, hunter_gains, _ = load_hunter(args.load_weights)
        if hunter_gains:
            apply_dotted(cfg, hunter_gains)              # a trained hunter carries its tuned gains
            apply_dotted(cfg, dict(cfg.hunt.get("overrides", {})))   # ... and the hunter's fixed settings (no escape)
            print(f"[hunt] applied {len(hunter_gains)} tuned gains from {args.load_weights}", flush=True)
    c = load_or_build(cfg, full=args.full)
    brain = BrainRunner(c, cfg, seed=None)
    if brain.plasticity is not None:
        brain.plasticity.enabled = not args.no_learn
        pl = brain.plasticity
        print(f"[learning] mushroom body: {pl.n_kc} Kenyon cells, {pl.n_mbon} MBON compartments, {pl.n_syn:,} plastic synapses; "
              f"{'ON' if pl.enabled else 'OFF'} (toggle from the dashboard)", flush=True)
    if brain_file:                                          # a GPU-trained connectome brain: synapse gains + thresholds, then a fresh baseline
        from .hunt_gpu.connectome import load_brain, apply_to_runner
        apply_to_runner(brain, load_brain(brain_file))
        args.recalibrate = True
    baseline = load_or_calibrate(brain, cfg, circuit_key(cfg, args.full) + ("-brain" if brain_file else ""), rebuild=args.recalibrate)
    if args.load_weights and not brain_file and brain.plasticity is not None:
        z = np.load(args.load_weights)
        if len(z["weights"]) == brain.plasticity.n_syn:
            brain.plasticity.net.data[brain.plasticity.syn] = z["weights"].astype(np.float32)
            what = f"hunt touch rate {float(z['touch_rate']):.0%}" if "touch_rate" in z.files else f"PI {float(z['naive_pi']):+.2f} -> {float(z['trained_pi']):+.2f}"
            print(f"[learning] loaded learned synapses from {args.load_weights} (seed {int(z['seed'])}, {what})", flush=True)
        else:
            print(f"[learning] {args.load_weights} does not match this circuit; ignored", flush=True)
    cam_cfg = cfg.senses.camera
    probe_result = {}
    odor_probes = {"A": {"ORN_fruit": 120.0}, "B": {"ORN_lemon": 120.0}}

    def run_probe():
        if brain.plasticity is None:
            return
        out = {}
        for name, drive in odor_probes.items():
            vals = []
            for _ in range(2):
                rest, odor = brain.probe(drive, seconds=1.0)
                vals.append(decoder.decode({w: odor for w in decoder.windows}, now=0).valence
                            - decoder.decode({w: rest for w in decoder.windows}, now=0).valence)
            out[name] = round(float(np.mean(vals)), 3)
        out["t"] = round(time.time() - t_start, 1)
        probe_result.clear(); probe_result.update(out)
        print(f"[learning] probe: odor A (grape) valence {out['A']:+.2f}, odor B (lemon) {out['B']:+.2f}", flush=True)
    dash = None
    if not args.no_ui:
        from .ui.server import Dashboard
        dash = Dashboard(c, cfg, port=args.ui_port, open_browser=args.open)
    def open_source():
        if args.sim_camera == "synthetic":
            return SyntheticLooming(cam_cfg.width, cam_cfg.height, cam_cfg.fps, loop=dash is not None)
        return CameraStream(args.sim_camera or cam_cfg.url, cam_cfg.width, cam_cfg.height)
    webcam_on = not args.no_webcam
    source = open_source() if webcam_on else None
    if dash is not None:
        dash.webcam_on = webcam_on
    cam_lobe = OpticLobe(cam_cfg.width, cam_cfg.height, cam_cfg.fps)      # the webcam, as fallback vision
    lobe = OpticLobe(cam_cfg.width, cam_cfg.height, 10.0)                  # the fly's own eyes (retina from its world)
    decoder = Decoder(cfg, baseline)
    body = DryBody() if args.dry_body else BodyLink(cfg.body.host, cfg.body.port, cfg.body.listen_port)
    s1 = _s1_body(args, cfg, with_link=False)
    if s1 is not None:
        from .body.s1 import MultiBody
        body = MultiBody(s1, body)                       # the S1 first: in udp mode it annotates the packet the body link then sends
    period = 1.0 / float(cfg.body.rate_hz)
    pir_until = 0.0
    last_pir = 0
    last_motion = time.time()
    last_sound = (0, 0.0)
    t_start = time.time()
    want_retina = cfg.senses.get("vision", "retina") == "retina" and dash is not None
    retina_seq = -1
    vision = "camera"
    print(f"[run] {c.n:,} neurons; camera={'synthetic' if args.sim_camera == 'synthetic' else (args.sim_camera or cam_cfg.url)}; "
          f"body={'dry' if args.dry_body else cfg.body.host}; vision={'retina (world) with camera fallback' if want_retina else 'camera'}. Ctrl-C to stop.", flush=True)
    feats = None
    tick_hz, last_tick = 0.0, time.time()
    try:
        while True:
            tick = time.time()
            tick_hz += 0.1 * (1.0 / max(tick - last_tick, 1e-3) - tick_hz); last_tick = tick
            # --- controls: webcam on/off (closing releases the camera; the window in the world goes dark)
            if dash is not None:
                for ctl in [c_ for c_ in dash.controls if "webcam" in c_]:
                    want = bool(ctl["webcam"])
                    if want and source is None:
                        source = open_source(); cam_lobe.reset(); webcam_on = True
                        print("[run] webcam opened", flush=True)
                    elif not want and source is not None:
                        source.close(); source = None; webcam_on = False; dash.jpeg = None
                        print("[run] webcam closed", flush=True)
                    dash.webcam_on = webcam_on
                dash.controls[:] = [c_ for c_ in dash.controls if "webcam" not in c_]
            # --- senses: the webcam (the human world) ...
            frame = source.read() if source is not None else None
            cam_feats = None
            if frame is not None:
                cam_feats = cam_lobe.process(frame)
                if cam_feats.motion_energy > float(cfg.senses.wake_motion):
                    last_motion = tick
            elif source is not None and args.sim_camera and (isinstance(source, SyntheticLooming) or source.is_file):
                print("[run] video finished"); break
            # ... and the fly's own eyes in its world, whenever the dashboard is rendering them
            use_retina = want_retina and dash.retina is not None and (tick - dash.retina_at) < 1.0
            if use_retina:
                if dash.retina_seq != retina_seq:
                    retina_seq = dash.retina_seq
                    rf = dash.retina
                    if rf.shape != (lobe.h, lobe.w):          # the world chose another retina size (the hunt arena's wide eye is 160x120)
                        lobe = OpticLobe(rf.shape[1], rf.shape[0], 10.0)
                    feats = lobe.process(rf)
                vision = "retina"
            else:
                feats = cam_feats            # None when the webcam is closed: the fly is blind, not stuck on its last view
                vision = "camera" if cam_feats is not None else "none"
            world = dash.world if (dash is not None and tick - dash.world_at < 1.0) else {}
            if any(_world_value(world.get(k)) > 0.3 for k in cfg.senses.world):
                last_motion = tick
            # --- controls from the dashboard
            if dash is not None and dash.controls:
                for ctl in dash.controls:
                    if brain.plasticity is not None and "learning" in ctl:
                        brain.plasticity.enabled = bool(ctl["learning"])
                        print(f"[learning] {'ON' if brain.plasticity.enabled else 'OFF'}", flush=True)
                    if brain.plasticity is not None and ctl.get("reset"):
                        brain.plasticity.reset(); probe_result.clear()
                        print("[learning] synapses reset to naive", flush=True)
                    if ctl.get("probe"):
                        run_probe()
                dash.controls.clear()
            body.poll()
            if body.pir and not last_pir:
                pir_until = tick + float(cfg.senses.pir.burst_ms) * 1e-3
                last_motion = tick
            last_pir = body.pir
            # --- drive the input neurons
            brain.clear_drive()
            if feats is not None:
                # efference copy: the fly's own motion moves the whole retina, which is not a threat
                suppress = 1.0 - float(cfg.senses.get("efference_copy_gain", 0.8)) * float(world.get("self_motion", 0.0)) if vision == "retina" else 1.0
                for (g, side), hz in features_to_rates(feats, cfg).items():
                    brain.drive(g, hz * max(0.0, suppress), side)
            if tick < pir_until:
                for g in cfg.senses.pir.groups:
                    brain.drive(g, float(cfg.senses.pir.rate_hz))
            if brain.plasticity is not None:
                for key in ("reward", "punish"):
                    spec = cfg.learning.get(key)
                    if spec and _world_value(world.get(spec["world"])) > 0:
                        for g in spec["groups"]:
                            brain.drive(g, float(np.clip(_world_value(world.get(spec["world"])), 0, 1)) * float(spec["hz"]))
            for name, spec in cfg.senses.world.items():
                v = world.get(name)
                if v is None:
                    continue
                if spec.get("lateral") and isinstance(v, (list, tuple)) and len(v) == 2:
                    for side, vv in zip(("left", "right"), v):
                        for g in spec["groups"]:
                            brain.drive(g, float(np.clip(vv, 0, 1)) * float(spec["max_hz"]), side)
                else:
                    for g in spec["groups"]:
                        brain.drive(g, float(np.clip(_world_value(v), 0, 1)) * float(spec["max_hz"]))
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
                    "vision": vision, "world": world, "webcam": webcam_on,
                    "learning": ({**brain.plasticity.summary(), "probe": probe_result} if brain.plasticity is not None else None),
                    "rates": {g: {s: round(v, 1) for s, v in r.items()} for g, r in brain.rates().items()},
                    "spikes": spiking.tolist(),
                    "n_spikes": int(recent.sum()),
                    "top_types": dash.top_types(brain.window_counts()),
                    "brain": {"behind_ms": round(brain.behind_ms), "chunk_ms": round(brain.last_chunk_wall_ms, 1),
                              "brain_s": round(brain.brain_ms / 1000, 1), "n": c.n,
                              "tick_hz": round(tick_hz, 1), "camera_fps": round(getattr(source, "fps", 0.0), 1) if source else 0.0},
                }, frame=source.preview if source is not None else None)
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
        if source is not None:
            source.close()
        body.close()


def _world_value(v) -> float:
    if v is None:
        return 0.0
    if isinstance(v, (list, tuple)):
        return max((float(x) for x in v), default=0.0)
    return float(v)


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
    p = sub.add_parser("experiment", help="two-odor conditioning, headless")
    p.add_argument("--blocks", type=int, default=3); p.add_argument("--train-s", type=float, default=15.0)
    p.add_argument("--odor-hz", type=float, default=80.0); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-learn", action="store_true", help="control run with plasticity switched off"); p.set_defaults(fn=cmd_experiment)
    p = sub.add_parser("batch", help="many headless two-odor arena assays in parallel")
    p.add_argument("--runs", type=int, default=4, help="flies with learning on"); p.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    p.add_argument("--control", action="store_true", help="also run the same number with learning off")
    p.add_argument("--save", action="store_true", help="save the best fly's learned synapses for `run --load-weights`"); p.set_defaults(fn=cmd_batch)
    p = sub.add_parser("hunt", help="the hunt arena: one fly, CMA-ES training (--train) or evaluation (--load)")
    p.add_argument("--episodes", type=int, default=6, help="episodes per fly"); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-learn", action="store_true", help="plasticity off (evaluate a saved hunter, or a control)")
    p.add_argument("--scripted", action="store_true", help="drive with the hand-written hunter instead of the brain (arena check)")
    p.add_argument("--load", default=None, metavar="NPZ", help="a saved hunter (synapses + gains) from --train or --save")
    p.add_argument("--gains", default=None, metavar="JSON", help="tuned gains (hunter_gains.json) without synapses")
    p.add_argument("--save", action="store_true", help="save this fly's synapses + gains as data/cache/hunter.npz")
    p.add_argument("--train", action="store_true", help="CMA-ES over the gains with many flies in parallel")
    p.add_argument("--generations", type=int, default=20); p.add_argument("--pop", type=int, default=None, help="population (default 4+3 ln n = 12)")
    p.add_argument("--final-episodes", type=int, default=24, help="episodes for the final fly trained with the best gains")
    p.add_argument("--workers", type=int, default=None); p.set_defaults(fn=cmd_hunt)
    p = sub.add_parser("hunt-gpu", help="the hunt arena on a GPU: PPO training (--train) or evaluation of a saved hunter (--load)")
    p.add_argument("--train", action="store_true", help="train with PPO; saves data/cache/hunter_gpu.pt (best) and hunter_gpu_last.pt")
    p.add_argument("--updates", type=int, default=None, help="PPO updates (default hunt_gpu.updates)")
    p.add_argument("--envs", type=int, default=None, help="rooms in parallel (default hunt_gpu.envs)")
    p.add_argument("--device", default="auto", help="cuda, mps, cpu or auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--load", default=None, metavar="PT", help="a saved hunter (evaluate it, or start training from it)")
    p.add_argument("--episodes", type=int, default=256, help="seeded episodes to evaluate, all in one batch")
    p.add_argument("--sample", action="store_true", help="evaluate with the stochastic policy instead of the mean action")
    p.add_argument("--cpu-check", action="store_true", help="also run the loaded fly through the CPU arena on the same seeds")
    p.add_argument("--humans", default=None, metavar="MODE|PT", help="the other player: patrol, flee, learn, or a trained runners file (default: the evaders_*.pt next to --load, else the config)")
    p.add_argument("--watch", action="store_true", help="live 3-D view of the loaded fly (or the scripted hunter) hunting, at http://localhost:8601")
    p.add_argument("--s1", nargs="?", const="", default=None, metavar="PORT", help="with --watch: a RoboMaster S1 on S-Bus mirrors the fly (default body.s1.port)")
    p.add_argument("--port", type=int, default=8601); p.add_argument("--open", action="store_true", help="open the view in the browser")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed of the live view (1 = real time)")
    p.add_argument("--save-every", type=int, default=10, help="updates between hunter_gpu_last.pt checkpoints")
    p.add_argument("--out", default=None, help="output directory (default data/cache)")
    p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(fn=cmd_hunt_gpu)
    p = sub.add_parser("hunt-brain", help="the real fly's brain (hunter subcircuit of the connectome) in the GPU hunt arena: PPO on its synapse gains")
    p.add_argument("--train", action="store_true", help="train; saves data/cache/hunter_brain.npz (synapse gains + thresholds + sensory / decoder gains)")
    p.add_argument("--updates", type=int, default=None); p.add_argument("--envs", type=int, default=None)
    p.add_argument("--hops", default=None, metavar="F,B", help="subcircuit depth, e.g. 2,1 (default hunt_gpu.brain.hops)")
    p.add_argument("--gains", default=None, metavar="NPZ", help="start from a hunter.npz's tuned sensory / decoder gains")
    p.add_argument("--load", default=None, metavar="NPZ", help="a saved hunter_brain.npz (evaluate, resume, watch)")
    p.add_argument("--humans", default=None, metavar="MODE|PT", help="patrol, flee, learn, or a trained runners file")
    p.add_argument("--episodes", type=int, default=256); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--spiking", action="store_true", help="evaluate as spiking LIF neurons (one room at a time, CPU) instead of the rate model")
    p.add_argument("--adversarial", action="store_true", help="with --humans PT: keep training the runners too (default: frozen)")
    p.add_argument("--no-brake", action="store_true", help="do not let the freeze channel (DNp09) brake the vehicle")
    p.add_argument("--fresh-brake", action="store_true", help="with --load: the brake gains (near_gain, near_deg, freeze z_ref) restart from the config")
    p.add_argument("--evolve", action="store_true", help="gradient-free: CMA-ES over the tunables and per-class synapse gains (forward passes only, fits a small machine)")
    p.add_argument("--evolve-spiking", action="store_true", help="the polish: the same search scored on the spiking neurons themselves (slow; start from --load)")
    p.add_argument("--generations", type=int, default=20); p.add_argument("--pop", type=int, default=None)
    p.add_argument("--check", action="store_true", help="compare the rate model's responses with the spiking subcircuit's")
    p.add_argument("--watch", action="store_true", help="live view of the spiking fly hunting, with its neurons, at http://localhost:8601")
    p.add_argument("--s1", nargs="?", const="", default=None, metavar="PORT", help="with --watch / --real: a RoboMaster S1 on S-Bus is the fly's legs (default body.s1.port)")
    p.add_argument("--real", action="store_true", help="the real world: a camera (and --pir) instead of the arena, the spiking fly hunting people, page at :8601")
    p.add_argument("--camera", default=None, metavar="SRC", help="with --real: the ESP32-CAM URL (default senses.camera.url), a device index like 0, or a file")
    p.add_argument("--cam-fov", type=float, default=62.0, help="with --real: the camera's horizontal field of view in degrees (ESP32-CAM OV2640 ~62)")
    p.add_argument("--pir", action="store_true", help="with --real: read the body ESP32's PIR over UDP as the hot cells")
    p.add_argument("--rover-off", action="store_true", help="with --real --s1: attach the S1 but do not drive it until the page's rover button is pressed (safe autostart)")
    p.add_argument("--port", type=int, default=8601); p.add_argument("--open", action="store_true"); p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--device", default="auto"); p.add_argument("--save-every", type=int, default=5); p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_hunt_brain)
    p = sub.add_parser("run"); p.add_argument("--full", action="store_true")
    p.add_argument("--sim-camera", default=None, metavar="SRC", help="'synthetic' or a video file / URL instead of the ESP32-CAM")
    p.add_argument("--dry-body", action="store_true", help="print body packets instead of sending UDP")
    p.add_argument("--s1", nargs="?", const="", default=None, metavar="PORT",
                   help="also drive a RoboMaster S1 over S-Bus from this serial port, or 'udp' = via the body ESP32 (default body.s1.port)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--recalibrate", action="store_true", help="re-measure the resting baseline")
    p.add_argument("--no-learn", action="store_true", help="start with mushroom-body plasticity switched off")
    p.add_argument("--no-webcam", action="store_true", help="start with the webcam closed (toggle from the dashboard)")
    p.add_argument("--load-weights", default=None, metavar="NPZ", help="start with learned KC->MBON synapses saved by `batch --save`")
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
