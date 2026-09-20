"""The fly's-eye view: the retina tensors drawn as the fly sees them, next to the camera.

Nothing here computes anything new. It takes what robot_bridge already has in `brain.last_r`
(pres, size, mot, loom_L, loom_R) and the four group rates VizFeed already reads, and draws them.

    left   the camera frame with the detector boxes
    right  24 angular columns: brightness = pres x size, warm tint = |mot|,
           red flare over the hemisphere that is looming
    below  LC10a_L/R and DNa02_L/R as bars, 0 to RATE_MAX Hz, tick at the 150 Hz target

Preview it against a recorded episode (no camera, no brain, no checkpoint):

    .venv/bin/python scripts/flyeye.py --replay replay/episode_7_trained.json

and tune the five constants live in that window: b/B bright, g/G gamma, m/M motion,
k/K loom, y/Y rate scale, space pause, [ ] step, s save a PNG, q quit (prints the block).
"""
from __future__ import annotations

import argparse
import json

import cv2
import numpy as np

# --- the five constants. Defaults derived from episode_7_trained.json, see --replay to retune.
BRIGHT_REF = 0.90   # pres*size that renders as full brightness (episode p50 0.64, p90 0.79, max 1.00)
GAMMA      = 0.80   # <1 lifts the midtones
MOT_REF    = 1.00   # |mot| that renders as fully warm (this channel pins at 1.0 over half the time)
LOOM_REF   = 4.40   # loom rate that renders as a full flare (p50 0.30, p90 1.22, p99 4.41, max 42.2)
RATE_MAX   = 350    # bar full scale in Hz (these four hit p90 315, max 334; 200 would clip constantly)

BARS = ("LC10a_L", "LC10a_R", "DNa02_L", "DNa02_R")
HOMEOSTATIC_HZ = 150.0
SATURATED_HZ = 200.0

# BGR, from the project palette: #0a0a0a ground, #FF715B coral
GROUND = (10, 10, 10)
PANEL = (5, 5, 5)
LINE = (32, 34, 36)
TEXT = (226, 228, 232)
DIM = (103, 106, 112)
COOL = (160, 144, 125)      # #7d90a0, the still end of the motion ramp
CORAL = (91, 113, 255)      # #FF715B, the moving end
LOOMC = (26, 45, 255)       # #ff2d1a, semantic, not the accent
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _lerp(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _clip01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else float(x))


def fly_panel(pres, size, mot, loom_l, loom_r, w, h):
    """The 24 columns. `pres`/`size`/`mot` are equal-length sequences, column 0 = the left of the image."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = PANEL
    n = len(pres)
    for c in range(n):
        v = _clip01(pres[c] * size[c] / BRIGHT_REF) ** GAMMA
        col = _lerp(COOL, CORAL, _clip01(abs(mot[c]) / MOT_REF))
        x0, x1 = int(round(c * w / n)), int(round((c + 1) * w / n))
        img[:, x0:x1] = [int(ch * v) for ch in col]
    for i, loom in enumerate((loom_l, loom_r)):        # left hemisphere = columns 0..n/2-1
        a = _clip01(loom / LOOM_REF)
        if a <= 0.001:
            continue
        x0, x1 = i * (w // 2), i * (w // 2) + (w // 2)
        wash = np.empty_like(img[:, x0:x1])
        wash[:] = LOOMC
        img[:, x0:x1] = cv2.addWeighted(img[:, x0:x1], 1.0 - 0.55 * a, wash, 0.55 * a, 0.0)
        img[0:max(6, h // 60), x0:x1] = LOOMC          # a hard rule so the hemisphere reads at 3 m
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), LINE, 1)
    cv2.putText(img, "-49", (8, h - 12), FONT, 0.5, DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "+49", (w - 54, h - 12), FONT, 0.5, DIM, 1, cv2.LINE_AA)
    return img


def bars_panel(rates, w, h, names=BARS):
    """`rates` maps group name to Hz. One row per name, value printed at the right."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = GROUND
    row = h / max(1, len(names))
    x_bar, x_val = 250, w - 130
    for i, name in enumerate(names):
        hz = float(rates.get(name, 0.0))
        y0 = int(i * row + row * 0.22)
        y1 = int(i * row + row * 0.78)
        base = int(i * row + row * 0.72)
        cv2.putText(img, name, (6, base), FONT, 1.0, TEXT, 2, cv2.LINE_AA)
        cv2.rectangle(img, (x_bar, y0), (x_val - 24, y1), (21, 19, 18), -1)
        span = (x_val - 24) - x_bar
        fill = int(span * _clip01(hz / RATE_MAX))
        if fill > 0:
            cv2.rectangle(img, (x_bar, y0), (x_bar + fill, y1), LOOMC if hz > SATURATED_HZ else CORAL, -1)
        tick = x_bar + int(span * _clip01(HOMEOSTATIC_HZ / RATE_MAX))
        cv2.rectangle(img, (tick, y0 - 5), (tick + 2, y1 + 5), (74, 70, 67), -1)
        cv2.putText(img, f"{hz:.0f}", (x_val, base), FONT, 1.0, TEXT, 2, cv2.LINE_AA)
    return img


def _label(canvas, x, y, text):
    cv2.putText(canvas, text, (x + 2, y - 8), FONT, 0.45, DIM, 1, cv2.LINE_AA)


def compose(camera, pres, size, mot, loom, rates, status="", w=1600, h=900):
    """camera: a BGR frame, or None to leave that half empty. Returns the full canvas."""
    canvas = np.zeros((h, w, 3), np.uint8)
    canvas[:] = GROUND
    pad, gap = 26, 20
    top_h = int(h * 0.58)
    pw = (w - pad * 2 - gap) // 2
    y0 = pad + 22

    left = np.zeros((top_h, pw, 3), np.uint8)
    left[:] = PANEL
    if camera is not None:
        ch, cw = camera.shape[:2]
        s = min(pw / cw, top_h / ch)
        r = cv2.resize(camera, (int(cw * s), int(ch * s)), interpolation=cv2.INTER_AREA)
        oy, ox = (top_h - r.shape[0]) // 2, (pw - r.shape[1]) // 2
        left[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    cv2.rectangle(left, (0, 0), (pw - 1, top_h - 1), LINE, 1)
    canvas[y0:y0 + top_h, pad:pad + pw] = left
    canvas[y0:y0 + top_h, pad + pw + gap:pad + pw + gap + pw] = fly_panel(
        pres, size, mot, loom[0], loom[1], pw, top_h)
    _label(canvas, pad, y0, "CAMERA")
    _label(canvas, pad + pw + gap, y0, "FLY'S EYE   24 COLUMNS")

    by = y0 + top_h + 46
    bh = h - by - 54
    canvas[by:by + bh, pad:w - pad] = bars_panel(rates, w - pad * 2, bh)
    _label(canvas, pad, by, f"GROUP RATES   0 - {RATE_MAX} Hz   (tick = {HOMEOSTATIC_HZ:.0f})")
    if status:
        cv2.putText(canvas, status, (pad, h - 18), FONT, 0.7, CORAL, 2, cv2.LINE_AA)
    return canvas


# --- preview against a recorded episode -------------------------------------------------------

def _fake_camera(pres, size, w, h):
    """The replay has no images, so draw the detector's view back from the columns: one box per
    run of covered columns. It is here to prove column 0 lands on the left of the image."""
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = (7, 7, 7)
    floor = h - 34
    cv2.line(img, (0, floor), (w, floor), (25, 23, 22), 1)
    n, c = len(pres), 0
    while c < n:
        if pres[c] <= 0:
            c += 1
            continue
        s, mx = c, 0.0
        while c < n and pres[c] > 0:
            mx = max(mx, size[c])
            c += 1
        x0, x1 = int(s * w / n), int(c * w / n)
        bh = max(26, int(_clip01(mx) * (h - 60)))
        cv2.rectangle(img, (x0 + 2, floor - bh), (max(x0 + 8, x1 - 2), floor), CORAL, 3)
    return img


def _adjust(key, step):
    g = globals()
    for ch, name, d in (("b", "BRIGHT_REF", 0.05), ("g", "GAMMA", 0.05), ("m", "MOT_REF", 0.05),
                        ("k", "LOOM_REF", 0.25), ("y", "RATE_MAX", 10)):
        if key == ord(ch):
            g[name] = max(d, round(g[name] - d, 3))
            return True
        if key == ord(ch.upper()):
            g[name] = round(g[name] + d, 3)
            return True
    return False


def _constants_block():
    return (f"BRIGHT_REF = {BRIGHT_REF:.2f}\nGAMMA      = {GAMMA:.2f}\nMOT_REF    = {MOT_REF:.2f}\n"
            f"LOOM_REF   = {LOOM_REF:.2f}\nRATE_MAX   = {RATE_MAX}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--replay", default="replay/episode_7_trained.json")
    ap.add_argument("--fullscreen", action="store_true", help="as the demo window will open")
    ap.add_argument("--save", default=None, metavar="PNG", help="write one frame and exit (no window)")
    ap.add_argument("--frame", type=int, default=120, help="which frame --save writes")
    a = ap.parse_args(argv)

    frames = json.load(open(a.replay))["frames"]
    print(f"{len(frames)} frames from {a.replay}")

    def canvas_for(i):
        f = frames[i % len(frames)]
        cam = _fake_camera(f["pres"], f["size"], 640, 480)
        return compose(cam, f["pres"], f["size"], f["mot"], f["loom"], f["rates"],
                       status=f"t {f['t']:.2f}s   fwd {f['forward']:+.2f}   turn {f['turn']:+.2f}   "
                              f"loom {f['loom'][0]:.2f}/{f['loom'][1]:.2f}   "
                              f"[bright {BRIGHT_REF:.2f} gamma {GAMMA:.2f} mot {MOT_REF:.2f} "
                              f"loomref {LOOM_REF:.2f} scale {RATE_MAX}]")

    if a.save:
        cv2.imwrite(a.save, canvas_for(a.frame))
        print(f"wrote {a.save} (frame {a.frame})")
        return

    win = "fly's eye"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    if a.fullscreen:
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(__doc__.split("and tune")[-1])
    i, playing = 0, True
    while True:
        cv2.imshow(win, canvas_for(i))
        k = cv2.waitKey(50) & 0xFF
        if k == ord("q"):
            break
        elif k == ord(" "):
            playing = not playing
        elif k == ord("["):
            playing, i = False, i - 1
        elif k == ord("]"):
            playing, i = False, i + 1
        elif k == ord("s"):
            cv2.imwrite("flyeye.png", canvas_for(i))
            print("wrote flyeye.png")
        elif k == ord("f"):
            a.fullscreen = not a.fullscreen
            cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if a.fullscreen else cv2.WINDOW_NORMAL)
        elif _adjust(k, 1):
            pass
        if playing:
            i += 1
    cv2.destroyAllWindows()
    print("\n# scripts/flyeye.py\n" + _constants_block())


if __name__ == "__main__":
    main()
