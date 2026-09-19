"""
sim_fixed.py -- the badge's integer LIF, run on a laptop, so the Lua is not written blind.

    python badge/sim_fixed.py                 # looming ramp, then the same ramp with LC4/LPLC2 lesioned
    python badge/sim_fixed.py --steps 400 --plot

The badge has no floating point unit and 48 KiB of Lua heap, so `badge/main.lua` runs the membrane
equation in integers: one unit is 1/256 mV, the decay factors are 8-bit fractions, and a step is 10 ms.
Integer rounding is not a detail you can hand-wave at that word size, so this file is the reference
implementation of exactly the same arithmetic, in Python, against the same generated circuit file.
Every constant here appears verbatim in main.lua; if you change one, change both and re-run this.

What it checks:
  * at rest the giant fibre is silent (no spontaneous escape),
  * a looming ramp drives LC4 and LPLC2 and the giant fibre fires,
  * with LC4/LPLC2 outgoing synapses zeroed (button B) the same ramp produces nothing,
  * the per-step cost, in synaptic events, which is what decides whether 10 ms is achievable.
"""
from __future__ import annotations

import argparse
import re

# --- fixed point: 1 unit = 1/256 mV, dt = 10 ms ------------------------------------------------
Q = 256
V_REST = -52 * Q            # -13312
V_TH = -45 * Q              # -11520
V_RESET = -52 * Q
MEM_NUM, MEM_DEN = 1, 2     # dt / tau_m = 10 ms / 20 ms
SYN_DECAY = 35              # exp(-dt/tau_syn) = exp(-10/5) = 0.135 -> 35/256
REFRAC = 1                  # 10 ms, so a neuron tops out at 100 Hz
GAIN_Q = 75                 # one int8 weight unit in membrane units: 5.82 synapses x 0.05 mV x 256


def load_circuit(path="badge/flybadge_circuit.lua"):
    """Parse the generated Lua back into Python. Same bytes the badge will see."""
    src = open(path).read()
    m = {}
    for k in ("n", "eye_last", "lc10a_first", "lc10a_last", "gf_first", "gf_last", "n_edges",
              "n_lc4", "n_lplc2", "n_lc10a"):
        m[k] = int(re.search(rf"^M\.{k} = (-?\d+)", src, re.M).group(1))
    m["colptr"] = [int(x) for x in re.search(r"M\.colptr = \{([^}]*)\}", src).group(1).split(",")]
    def unesc(field):
        raw = re.search(rf'M\.{field} = "((?:[^"\\]|\\\d+)*)"', src).group(1)
        return [int(x) for x in re.findall(r"\\(\d+)", raw)]
    m["post"] = unesc("post")
    m["w"] = [v - 256 if v > 127 else v for v in unesc("w")]
    assert len(m["post"]) == len(m["w"]) == m["n_edges"], "circuit file is inconsistent"
    return m


class FixedLIF:
    """Integer leaky integrate-and-fire. Identical arithmetic to badge/main.lua."""

    def __init__(self, c):
        self.c = c
        self.n = c["n"]
        self.V = [V_REST] * (self.n + 1)
        self.g = [0] * (self.n + 1)
        self.ref = [0] * (self.n + 1)
        self.spk = []
        self.lesion = False
        self.events = 0

    def step(self, inject):
        """inject: dict neuron -> current in membrane units. Returns the list of neurons that fired."""
        c = self.c
        syn = [0] * (self.n + 1)
        for i in self.spk:                                    # event driven: only what fired last step
            if self.lesion and i <= c["eye_last"]:
                continue                                      # button B: eye cells keep spiking, send nothing
            a, b = c["colptr"][i - 1], c["colptr"][i]
            self.events += b - a
            for e in range(a, b):
                syn[c["post"][e]] += c["w"][e] * GAIN_Q
        fired = []
        for i in range(1, self.n + 1):
            self.g[i] = (self.g[i] * SYN_DECAY) // Q + syn[i]
            if self.ref[i] > 0:
                self.ref[i] -= 1
                self.V[i] = V_RESET
                continue
            drive = V_REST - self.V[i] + self.g[i] + inject.get(i, 0)
            self.V[i] += (drive * MEM_NUM) // MEM_DEN
            if self.V[i] >= V_TH:
                self.V[i] = V_RESET
                self.ref[i] = REFRAC
                fired.append(i)
        self.spk = fired
        return fired


def looming(step, steps):
    """A hand sweeping in: drive ramps, as an expanding edge would. Units are membrane units."""
    if step < steps // 4:
        return 0
    t = (step - steps // 4) / max(1, steps - steps // 4)
    return int(2600 * t)


def run(c, steps, lesion=False, quiet=False):
    lif = FixedLIF(c)
    lif.lesion = lesion
    gf = range(c["gf_first"], c["gf_last"] + 1)
    gf_spikes = eye_spikes = 0
    trace = []
    for s in range(steps):
        cur = looming(s, steps)
        inj = {i: cur for i in range(1, c["eye_last"] + 1)}
        fired = lif.step(inj)
        n_gf = sum(1 for i in fired if i in gf)
        gf_spikes += n_gf
        eye_spikes += sum(1 for i in fired if i <= c["eye_last"])
        trace.append((cur, len(fired), n_gf))
    secs = steps * 0.01
    if not quiet:
        tag = "LESIONED (B)" if lesion else "intact"
        print(f"  {tag:14s} eye {eye_spikes / (c['eye_last'] * secs):6.1f} Hz/cell   "
              f"GF {gf_spikes / (len(list(gf)) * secs):6.1f} Hz   "
              f"{lif.events / steps:6.0f} synaptic events/step")
    return gf_spikes, eye_spikes, lif.events / steps, trace


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuit", default="badge/flybadge_circuit.lua")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args()
    c = load_circuit(a.circuit)
    print(f"{a.circuit}: {c['n']} neurons ({c['n_lc4']} LC4, {c['n_lplc2']} LPLC2, {c['n_lc10a']} LC10a, "
          f"{c['gf_last'] - c['gf_first'] + 1} GF), {c['n_edges']} synapses")

    print("\nrest (no stimulus, 100 steps):")
    lif = FixedLIF(c)
    rest_spikes = sum(len(lif.step({})) for _ in range(100))
    print(f"  spikes anywhere in the circuit with no input: {rest_spikes}")

    print(f"\nlooming ramp ({a.steps} steps = {a.steps / 100:.1f} s):")
    gf_i, eye_i, ev_i, _ = run(c, a.steps)
    gf_l, eye_l, _, _ = run(c, a.steps, lesion=True)

    print("\nverdict")
    ok_rest = rest_spikes == 0
    ok_fire = gf_i > 0
    ok_lesion = gf_l == 0
    print(f"  silent at rest                {'PASS' if ok_rest else 'FAIL'}")
    print(f"  giant fibre fires on looming  {'PASS' if ok_fire else 'FAIL'}  ({gf_i} spikes)")
    print(f"  lesion (B) abolishes escape   {'PASS' if ok_lesion else 'FAIL'}  ({gf_l} spikes)")
    print(f"  budget: {ev_i:.0f} synaptic events per 10 ms step")


if __name__ == "__main__":
    main()
