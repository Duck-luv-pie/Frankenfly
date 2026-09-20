"""Measure a camera's real horizontal field of view by clicking two marks on a wall.

Why not the spec sheet: DroidCam (and every phone app) crops, and the crop changes with the
resolution and aspect you stream at, so the lens's nominal FOV is not the stream's FOV. The
retina's `Camera.px_to_az` needs the stream's.

Method. Put two marks on a flat wall a measured distance L apart, horizontally. Stand the camera
a measured perpendicular distance D from that wall, facing it square on. For a pinhole camera with
its image plane parallel to the wall, both marks project as

    xA = cx + fx * (0 - p)/D        xB = cx + fx * (L - p)/D

for whatever lateral offset p the camera happens to sit at, so

    xB - xA = fx * L / D           ->   fx = (xB - xA) * D / L

The camera's position and aim cancel. Only D, L and the two pixel columns matter.

    .venv/bin/python scripts/measure_fov.py --source http://192.168.1.42:4747/video -D 2.0 -L 1.0

Click the left mark, then the right mark. SPACE records the pair and averages it in, r clears the
current pair, u undoes the last recorded pair, q quits and prints the numbers.
"""
from __future__ import annotations

import argparse
import math
import statistics

import cv2


def open_source(src):
    cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {src!r}. Open the URL in a browser first to check it streams.")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="webcam index, or the DroidCam URL (http://IP:4747/video)")
    ap.add_argument("-D", "--distance", type=float, required=True, metavar="M",
                    help="perpendicular distance from the camera to the wall, metres")
    ap.add_argument("-L", "--separation", type=float, required=True, metavar="M",
                    help="horizontal distance between the two marks on the wall, metres")
    a = ap.parse_args(argv)

    cap = open_source(a.source)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise SystemExit("opened the source but got no frame")
    H, W = frame.shape[:2]
    print(f"stream is {W}x{H}. D = {a.distance} m, L = {a.separation} m")
    print("click the left mark, then the right mark, then SPACE. q when you have 3 or 4 pairs.")

    clicks, samples = [], []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 2:
            clicks.append((x, y))

    win = "measure fov"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)

    while True:
        ok, frame = cap.read()
        if ok and frame is not None:
            last = frame
        vis = last.copy()
        cv2.line(vis, (W // 2, 0), (W // 2, H), (60, 60, 60), 1)
        for i, (x, y) in enumerate(clicks):
            cv2.drawMarker(vis, (x, y), (91, 113, 255), cv2.MARKER_CROSS, 22, 2)
            cv2.putText(vis, "AB"[i], (x + 9, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (91, 113, 255), 2)
        if len(clicks) == 2:
            cv2.line(vis, clicks[0], clicks[1], (91, 113, 255), 1)
            dx = abs(clicks[1][0] - clicks[0][0])
            fx = dx * a.distance / a.separation
            hfov = 2 * math.degrees(math.atan((W / 2) / fx)) if fx > 0 else 0.0
            msg = f"dx {dx}px  fx {fx:.1f}  hfov {hfov:.1f}deg   SPACE to keep"
        else:
            msg = f"click {'the left mark' if not clicks else 'the right mark'}"
        cv2.putText(vis, msg, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (226, 228, 232), 2)
        cv2.putText(vis, f"pairs kept: {len(samples)}", (8, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.imshow(win, vis)

        k = cv2.waitKey(30) & 0xFF
        if k == ord("q"):
            break
        elif k == ord("r"):
            clicks.clear()
        elif k == ord("u") and samples:
            samples.pop()
            print(f"dropped one, {len(samples)} left")
        elif k == ord(" ") and len(clicks) == 2:
            dx = abs(clicks[1][0] - clicks[0][0])
            if dx < 20:
                print("those two clicks are too close together to be accurate. Move further apart.")
            else:
                samples.append(dx * a.distance / a.separation)
                print(f"kept pair {len(samples)}: dx {dx}px -> fx {samples[-1]:.1f}")
            clicks.clear()

    cap.release()
    cv2.destroyAllWindows()

    if not samples:
        print("\nno pairs recorded, nothing to report")
        return
    fx = statistics.mean(samples)
    spread = (max(samples) - min(samples)) / fx * 100 if len(samples) > 1 else 0.0
    hfov = 2 * math.degrees(math.atan((W / 2) / fx))
    vfov = 2 * math.degrees(math.atan((H / 2) / fx))      # square pixels, so fy = fx
    print(f"\n{len(samples)} pairs, fx {fx:.1f} px at {W}x{H}, spread {spread:.1f}%")
    if spread > 5:
        print("  spread above 5%: check the camera is square to the wall and D is measured to the lens")
    print(f"\n  hfov {hfov:.2f} deg      vfov {vfov:.2f} deg")
    print(f"\n  --hfov {hfov:.2f} --vfov {vfov:.2f}")
    dx20 = fx * math.tan(math.radians(20.0)) * (320.0 / W)      # where 20 deg lands, in the retina's 320-wide frame
    fx_hardcoded = (320 / 2) / math.tan(math.radians(98.43) / 2)
    read = math.degrees(math.atan(dx20 / fx_hardcoded))
    print(f"\nWithout it, a person truly at 20 deg reads as {read:.1f} deg: the hardcoded 98.43 "
          f"assumes the RoboMaster lens.")


if __name__ == "__main__":
    main()
