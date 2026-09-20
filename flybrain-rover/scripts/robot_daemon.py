"""
robot_daemon.py -- the only process that touches the DJI RoboMaster SDK. Runs under Python 3.8 x86_64
(the SDK ships cp36-cp38 wheels only; on Apple Silicon that is Rosetta): `.venv38/bin/python`.

    .venv38/bin/python scripts/robot_daemon.py --conn ap            # real robot (Wi-Fi AP mode)
    .venv38/bin/python scripts/robot_daemon.py --fake               # synthetic frames, prints commands

Protocol (localhost TCP, default port 9500, one client = scripts/robot_bridge.py --robot-daemon):
    daemon -> client :  b"F" + uint32 length + JPEG bytes            (newest camera frame, ~30 fps)
    client -> daemon :  text line "V x y z\\n"  (m/s, m/s, deg/s)     chassis.drive_speed(x, y, z)
                        "S\\n" stop, "Q\\n" quit
Safety: a watchdog zeroes the wheels if no "V" line arrives for 300 ms; Ctrl-C stops the wheels first.
The gimbal is recentred at start and never moved (camera body-fixed, like a fly's head).
"""
import argparse
import socket
import struct
import sys
import threading
import time

import numpy as np


class FakeRobot:
    def __init__(self):
        self.t0 = time.time()

    def frame(self):
        import cv2
        img = np.full((240, 320, 3), 90, np.uint8)
        img[160:] = 60                                            # floor
        ph = ((time.time() - self.t0) % 6.0) / 6.0
        cx = int(320 * (0.15 + 0.7 * (0.5 - 0.5 * np.cos(2 * np.pi * ph))))
        cv2.rectangle(img, (cx - 20, 30), (cx + 20, 239), (60, 60, 200), -1)   # person-ish blob
        return img

    def drive(self, x, y, z):
        print(f"[fake] drive_speed(x={x:+.2f}, y={y:+.2f}, z={z:+.1f})", flush=True)

    def stop(self):
        print("[fake] stop", flush=True)

    def close(self):
        pass


class RealRobot:
    def __init__(self, conn="ap", camera=True, local_ip=None):
        from robomaster import robot, config
        if local_ip:
            config.LOCAL_IP_STR = local_ip          # macOS with two interfaces needs the Wi-Fi address pinned
        self.camera_on = camera
        self.ep = robot.Robot()
        # the SDK compares conn_type with `is` against its interned literals; argv strings are not interned
        self.ep.initialize(conn_type=sys.intern(conn))
        self.ep.gimbal.recenter().wait_for_completed()
        # CHASSIS_LEAD = the gimbal follows the chassis, so the camera is body-fixed like a fly's head.
        # (FREE mode would keep the camera pointed at the wall while the chassis turns and break the loop.)
        try:
            self.ep.set_robot_mode(mode=robot.CHASSIS_LEAD)
        except Exception as e:  # noqa: BLE001
            print("warning: could not set CHASSIS_LEAD gimbal mode:", e, flush=True)
        if self.camera_on:
            self.ep.camera.start_video_stream(display=False, resolution="360p")
        print(f"robot connected, gimbal recentred + locked to chassis, video {'on' if self.camera_on else 'OFF (laptop webcam is the eye)'}", flush=True)

    def frame(self):
        if not self.camera_on:
            return None
        return self.ep.camera.read_cv2_image(strategy="newest", timeout=1.0)

    def drive(self, x, y, z):
        self.ep.chassis.drive_speed(x=x, y=y, z=z, timeout=0.5)

    def stop(self):
        self.ep.chassis.drive_speed(x=0, y=0, z=0)

    def close(self):
        self.stop()
        if self.camera_on:
            try:
                self.ep.camera.stop_video_stream()
            except Exception:  # noqa: BLE001
                pass
        self.ep.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--conn", default="ap", choices=["ap", "sta", "rndis"])
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--watchdog-ms", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--no-camera", action="store_true", help="wheels only; the bridge uses the laptop webcam (--source 0)")
    ap.add_argument("--local-ip", default=None, help="this machine's address on the robot network, e.g. 192.168.2.30")
    a = ap.parse_args()
    import cv2
    bot = FakeRobot() if a.fake else RealRobot(a.conn, camera=not a.no_camera, local_ip=a.local_ip)
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", a.port)); srv.listen(1)
    print(f"daemon listening on 127.0.0.1:{a.port} ({'FAKE' if a.fake else 'REAL'} robot)", flush=True)
    last_cmd = [time.time()]
    stop_flag = threading.Event()

    def watchdog():
        while not stop_flag.is_set():
            if (time.time() - last_cmd[0]) * 1000 > a.watchdog_ms:
                bot.stop(); last_cmd[0] = time.time()
            time.sleep(0.05)

    threading.Thread(target=watchdog, daemon=True).start()
    try:
        while not stop_flag.is_set():
            conn, _ = srv.accept()
            conn.settimeout(0.0)
            print("client connected", flush=True)
            buf = b""
            try:
                while not stop_flag.is_set():
                    t = time.time()
                    img = bot.frame()
                    if img is not None:
                        ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
                        if ok:
                            conn.sendall(b"F" + struct.pack("<I", len(jpg)) + jpg.tobytes())
                    elif getattr(bot, "camera_on", True) is False:
                        conn.sendall(b"F" + struct.pack("<I", 0))          # heartbeat, no frame
                    try:
                        buf += conn.recv(4096)
                    except BlockingIOError:
                        pass
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        parts = line.decode(errors="ignore").split()
                        if not parts:
                            continue
                        if parts[0] == "V" and len(parts) == 4:
                            x, y, z = map(float, parts[1:])
                            bot.drive(x, y, z); last_cmd[0] = time.time()
                        elif parts[0] == "S":
                            bot.stop(); last_cmd[0] = time.time()
                        elif parts[0] == "Q":
                            stop_flag.set()
                    time.sleep(max(0.0, 1 / a.fps - (time.time() - t)))
            except (BrokenPipeError, ConnectionResetError):
                print("client gone; wheels stopped", flush=True)
            finally:
                bot.stop(); conn.close()
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set(); bot.close(); srv.close()


if __name__ == "__main__":
    sys.exit(main())
