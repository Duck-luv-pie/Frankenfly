"""
make_figures.py -- the three figures for the Devpost gallery and the poster, from measured numbers.

    python scripts/make_figures.py [--out devpost]

1. control.png   the wiring-shuffle control: the eye is untouched, the steering neurons go silent.
2. graded.png    deleting LC10a a quarter at a time: a population degrades, it does not switch off.
3. lesions.png   every lesion against the intact circuit, as turn-toward fraction.
Every panel carries its own protocol line, so a figure cannot be quoted without its provenance.
"""
from __future__ import annotations

import argparse
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BG, FG, DIM, CORAL, BLUE, GREY = "#0a0a0a", "#e8e8e8", "#8a8a8a", "#FF715B", "#5aa9ff", "#3a3a3a"


def frame(title, sub, figsize=(8.2, 5.4), left=0.13):
    fig, ax = plt.subplots(figsize=figsize, facecolor=BG)
    ax.set_facecolor(BG)
    wrapped = "\n".join(textwrap.wrap(sub, 96))
    fig.subplots_adjust(top=0.84 - 0.03 * (wrapped.count("\n") + 1), left=left, right=0.96, bottom=0.14)
    fig.text(0.035, 0.945, title, color=FG, fontsize=15, fontweight="bold", va="top")
    fig.text(0.035, 0.885, wrapped, color=DIM, fontsize=8.5, va="top", linespacing=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GREY)
    ax.tick_params(colors=DIM, labelsize=9.5)
    ax.yaxis.label.set_color(DIM)
    ax.xaxis.label.set_color(DIM)
    return fig, ax


def fig_control(out):
    fig, ax = frame("Same neurons. Same degrees. Same weights. Different map.",
                    "A person at the left of frame, 45 frames at 30 Hz. The shuffle permutes the target of every one of "
                    "2,334,959 synapses, so each neuron keeps its exact number of inputs, outputs and their strengths.")
    real, shuf = [66.6, 158.4], [69.9, 0.0]
    for i, (r, s) in enumerate(zip(real, shuf)):
        ax.bar(i - 0.19, r, 0.36, color=CORAL, label="real connectome" if i == 0 else None)
        ax.bar(i + 0.19, s, 0.36, color=BLUE, label="wiring shuffled" if i == 0 else None)
        ax.text(i - 0.19, r + 5, f"{r:.0f} Hz", ha="center", color=FG, fontsize=11)
        ax.text(i + 0.19, s + 5, f"{s:.0f} Hz", ha="center", color=FG, fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["LC10a\nthe tracking neurons (the eye)", "DNa02\nthe steering neurons"], color=FG, fontsize=11)
    ax.set_ylabel("firing rate (Hz)"); ax.set_ylim(0, 200); ax.set_xlim(-0.55, 1.55)
    ax.text(0, 92, "the eye still fires", ha="center", color=DIM, fontsize=10)
    ax.text(1.19, 26, "nothing arrives", ha="center", color=DIM, fontsize=10)
    leg = ax.legend(frameon=False, loc="upper left", fontsize=10.5, bbox_to_anchor=(0.02, 0.99))
    for t in leg.get_texts():
        t.set_color(FG)
    fig.savefig(f"{out}/control.png", dpi=200, facecolor=BG); plt.close(fig)


def fig_graded(out):
    fig, ax = frame("A population, not a switch.",
                    "LC10a deleted a quarter at a time in the running robot, 275 cells in total. Restoring them returns "
                    "158.4 Hz and a full turn, exactly.")
    pct, rate, turn = [0, 25, 50, 75, 100], [158.4, 137.5, 83.4, 40.3, 0.0], [1.00, 1.00, 1.00, 0.81, 0.00]
    l1, = ax.plot(pct, rate, "-o", color=CORAL, lw=2.5, ms=8, label="DNa02 steering drive (Hz)")
    for i, (px, r) in enumerate(zip(pct, rate)):
        below = i < 2                                    # keep the first labels clear of the dashed line above
        ax.text(px, r + (-13 if below else 9), f"{r:.0f}", ha="center",
                va="top" if below else "bottom", color=FG, fontsize=10.5)
    ax.set_xlabel("LC10a neurons deleted (%)"); ax.set_ylabel("DNa02 firing rate (Hz)", color=CORAL)
    ax.set_xticks(pct); ax.set_ylim(0, 195); ax.set_xlim(-6, 106)
    ax2 = ax.twinx()
    l2, = ax2.plot(pct, turn, "--s", color=BLUE, lw=2, ms=7, label="turn command sent to the wheels")
    ax2.set_ylim(0, 1.22); ax2.set_ylabel("turn command", color=BLUE)
    ax2.tick_params(colors=DIM, labelsize=9.5)
    for sp in ("top", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.spines["right"].set_color(GREY)
    ax.text(3, 30, "half the population still turns\nthe robot at full command", color=DIM, fontsize=10, linespacing=1.6)
    leg = ax.legend(handles=[l1, l2], frameon=False, fontsize=10.5, loc="lower left", bbox_to_anchor=(0.01, 0.32))
    for t in leg.get_texts():
        t.set_color(FG)
    fig.savefig(f"{out}/graded.png", dpi=200, facecolor=BG); plt.close(fig)


def fig_lesions(out):
    fig, ax = frame("Break it in named, predictable ways.",
                    "256 arenas, seed 2000, 2 s each. Every lesion is a cell type a neuroscientist can point to. Removing "
                    "the looming cells takes the giant fibre from 130 Hz to 0.1 Hz while steering survives.", left=0.30)
    rows = [("tracking cells removed\nLC10a, 275 of 15,000", 0.00, BLUE),
            ("wiring shuffled\nsame neurons and degrees", 0.11, BLUE),
            ("one steering neuron removed\nDNa02 left", 0.25, BLUE),
            ("looming cells removed\nLC4 and LPLC2", 0.75, CORAL),
            ("intact circuit", 0.78, CORAL)]
    ax.barh([r[0] for r in rows], [r[1] for r in rows], color=[r[2] for r in rows], height=0.6)
    for i, r in enumerate(rows):
        ax.text(r[1] + 0.018, i, f"{100 * r[1]:.0f}%", va="center", color=FG, fontsize=11.5)
    ax.set_xlim(0, 0.95); ax.set_xlabel("episodes where the robot turned toward the person")
    ax.set_xticks([0, 0.25, 0.5, 0.75]); ax.set_xticklabels(["0%", "25%", "50%", "75%"])
    ax.tick_params(axis="y", labelsize=10); ax.set_yticklabels([r[0] for r in rows], color=FG)
    fig.savefig(f"{out}/lesions.png", dpi=200, facecolor=BG); plt.close(fig)


def fig_silence(out):
    fig, ax = frame("Two different ways for a neuron to be silent.",
                    "The eye is driven; the inhibitory inputs onto each descending neuron are then removed, strongest "
                    "first. DNa10 receives 801 synapses straight from the tracking cells and still never fires, until "
                    "its inhibition is lifted. DNp09 and DNa01 never fire at all.")
    dna10 = [0.0, 0.0, 32.5, 91.0, 106.8, 117.3, 112.4]
    ax.plot(range(len(dna10)), dna10, "-o", color=CORAL, lw=2.5, ms=8, label="DNa10 (801 direct synapses from the eye)")
    ax.plot(range(9), [0.0] * 9, "--s", color=BLUE, lw=2, ms=7, label="DNp09 and DNa01 (forward walking)")
    ax.annotate("held down by feedforward inhibition\nfrom AOTU041 and AOTU063", xy=(3, 91), xytext=(3.25, 45),
                color=DIM, fontsize=10, linespacing=1.6,
                arrowprops=dict(arrowstyle="->", color=DIM, lw=1.2))
    ax.text(4.1, 6, "no excitation ever arrives", color=BLUE, fontsize=10)
    ax.set_xlabel("inhibitory sources removed (cumulative, strongest first)")
    ax.set_ylabel("firing rate (Hz)"); ax.set_ylim(-6, 155); ax.set_xlim(-0.3, 8.3)
    leg = ax.legend(frameon=False, fontsize=10, loc="upper left", bbox_to_anchor=(0.005, 1.0))
    for t in leg.get_texts():
        t.set_color(FG)
    fig.savefig(f"{out}/silence.png", dpi=200, facecolor=BG); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="devpost"); a = ap.parse_args()
    fig_control(a.out); fig_graded(a.out); fig_lesions(a.out); fig_silence(a.out)
    print(f"wrote {a.out}/control.png, {a.out}/graded.png, {a.out}/lesions.png, {a.out}/silence.png")
