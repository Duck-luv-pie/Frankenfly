"""
robot_probe.py -- READ-ONLY check of a RoboMaster over the SDK. Never moves anything.

    .venv38/bin/python scripts/robot_probe.py [--conn ap] [--frames 5] [--out logs/robot_probe]

Prints firmware version, which modules answer (chassis, gimbal, camera, battery), battery level,
chassis attitude and position readings for ~2 s (subscriptions only), then grabs a few camera frames and
saves them as JPEG (for detector tests without the robot). No drive_speed, no gimbal moves, no LEDs.
"""
import argparse
import os
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conn", default="ap", choices=["ap", "sta", "rndis"])
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--out", default="logs/robot_probe")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from robomaster import robot, version as rm_version
    print("robomaster sdk", rm_version.__version__, flush=True)
    ep = robot.Robot()
    t0 = time.time()
    ep.initialize(conn_type=sys.intern(a.conn))   # SDK uses `is` on its string constants
    print(f"connected in {time.time() - t0:.1f} s; firmware {ep.get_version()}; sn {ep.get_sn()}", flush=True)
    mods = {}
    for name in ("chassis", "gimbal", "camera", "battery", "led", "blaster", "vision", "sensor", "armor"):
        try:
            mods[name] = getattr(ep, name) is not None
        except Exception as e:  # noqa: BLE001
            mods[name] = f"err {e}"
    print("modules:", mods, flush=True)
    # subscriptions (read-only)
    readings = {"attitude": [], "position": [], "battery": []}
    try:
        ep.chassis.sub_attitude(freq=10, callback=lambda d: readings["attitude"].append(d))
        ep.chassis.sub_position(freq=10, callback=lambda d: readings["position"].append(d))
    except Exception as e:  # noqa: BLE001
        print("chassis subscription failed:", e)
    try:
        ep.battery.sub_battery_info(freq=1, callback=lambda d: readings["battery"].append(d))
    except Exception as e:  # noqa: BLE001
        print("battery subscription failed:", e)
    time.sleep(2.5)
    for k in ("chassis.unsub_attitude", "chassis.unsub_position", "battery.unsub_battery_info"):
        try:
            obj, fn = k.split("."); getattr(getattr(ep, obj), fn)()
        except Exception:  # noqa: BLE001
            pass
    att = readings["attitude"][-1] if readings["attitude"] else None
    pos = readings["position"][-1] if readings["position"] else None
    bat = readings["battery"][-1] if readings["battery"] else None
    print(f"attitude (yaw, pitch, roll) {att}; position (x, y, z) {pos}; battery {bat}", flush=True)
    # camera frames (read-only)
    try:
        import cv2
        ep.camera.start_video_stream(display=False, resolution="360p")
        got = 0; t1 = time.time()
        for i in range(a.frames * 3):
            img = ep.camera.read_cv2_image(strategy="newest", timeout=2.0)
            if img is None:
                continue
            cv2.imwrite(f"{a.out}/frame_{got:02d}.jpg", img); got += 1
            if got == 1:
                print(f"camera: first frame {img.shape[1]}x{img.shape[0]} after {time.time() - t1:.1f} s", flush=True)
            if got >= a.frames:
                break
        ep.camera.stop_video_stream()
        print(f"camera: saved {got} frames to {a.out}/ ({(time.time() - t1):.1f} s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print("camera failed:", e, flush=True)
    ep.close()
    print("closed; nothing was moved.", flush=True)


if __name__ == "__main__":
    sys.exit(main())
