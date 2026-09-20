"""Frame sources: the ESP32-CAM MJPEG stream, a video file, or a synthetic looming stimulus."""
from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


class FrameSource:
    preview: np.ndarray | None = None     # last frame at display size (BGR or gray), for the dashboard

    def read(self) -> np.ndarray | None:  # grayscale uint8 (H, W) or None when exhausted
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraStream(FrameSource):
    """Reads the ESP32-CAM MJPEG stream (or any OpenCV-readable URL / file / device index).

    Network streams are read in a background thread that always keeps only the newest frame, so
    the brain loop never waits on the camera and never falls behind it (latency stays at one frame)."""

    def __init__(self, url: str, width: int = 80, height: int = 60, reconnect_s: float = 2.0):
        self.url, self.w, self.h = url, width, height
        self.reconnect_s = reconnect_s
        self.cap: cv2.VideoCapture | None = None
        self.last_attempt = 0.0
        self.is_device = url.isdigit()          # "0" = the computer's own webcam
        self.is_file = not self.is_device and not url.startswith(("http://", "https://", "rtsp://"))
        self.threaded = not self.is_file
        self._latest: np.ndarray | None = None
        self._seq = 0
        self._taken = 0
        self._lock = threading.Lock()
        self._stop = False
        self.fps = 0.0
        self._open()
        if self.threaded:
            threading.Thread(target=self._pump, daemon=True).start()

    def _open(self) -> None:
        self.last_attempt = time.time()
        url = self.url
        if not self.is_device and not self.is_file:
            # OpenCV's stream reader cannot resolve mDNS names (companion-cam.local): resolve here
            from urllib.parse import urlsplit, urlunsplit
            import socket
            u = urlsplit(url)
            if u.hostname and u.hostname.endswith(".local"):
                try:
                    ip = socket.gethostbyname(u.hostname)
                    netloc = ip + (f":{u.port}" if u.port else "")
                    url = urlunsplit((u.scheme, netloc, u.path, u.query, u.fragment))
                    print(f"[camera] {u.hostname} -> {ip}", flush=True)
                except OSError:
                    print(f"[camera] cannot resolve {u.hostname}; is the camera on the network?", flush=True)
        cap = cv2.VideoCapture(int(self.url) if self.is_device else url)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
        else:
            self.cap = None

    def _pump(self) -> None:
        """Background reader: keep only the newest frame."""
        n, t0 = 0, time.time()
        while not self._stop:
            cap = self.cap
            if cap is None:
                if time.time() - self.last_attempt > self.reconnect_s:
                    self._open()
                time.sleep(0.1)
                continue
            ok, frame = cap.read()
            if not ok:
                fails = getattr(self, "_fails", 0) + 1
                self._fails = fails
                if fails >= 5:
                    print("[camera] stream stalled, reconnecting", flush=True)
                    cap.release(); self.cap = None; self._fails = 0
                else:
                    time.sleep(0.05)
                continue
            self._fails = 0
            with self._lock:
                self._latest = frame
                self._seq += 1
            n += 1
            if time.time() - t0 >= 2.0:
                self.fps = n / (time.time() - t0); n, t0 = 0, time.time()

    def read(self) -> np.ndarray | None:
        if self.threaded:
            with self._lock:
                if self._seq == self._taken or self._latest is None:
                    return None                   # nothing new since the last call
                self._taken = self._seq
                frame = self._latest
        else:
            if self.cap is None:
                if time.time() - self.last_attempt > self.reconnect_s:
                    self._open()
                return None
            ok, frame = self.cap.read()
            if not ok:
                return None
        self.preview = cv2.resize(frame, (320, 240), interpolation=cv2.INTER_AREA) if frame.shape[1] != 320 else frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        self._stop = True
        time.sleep(0.05)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class SyntheticLooming(FrameSource):
    """A dark disc that expands toward the camera (a looming stimulus), used for tests and demos.
    Timeline (seconds): 0-1 static, 1-2 small object drifts left->right (LC11/LC10a), 2-3 static,
    3-4 disc expands rapidly (LC4/LPLC2), 4-5 static."""

    def __init__(self, width: int = 80, height: int = 60, fps: float = 15.0, duration_s: float = 5.0, loop: bool = False):
        self.w, self.h, self.fps = width, height, fps
        self.n_frames = int(duration_s * fps)
        self.loop = loop
        self.i = 0

    def read(self) -> np.ndarray | None:
        if self.i >= self.n_frames:
            if not self.loop:
                return None
            self.i = 0
        t = self.i / self.fps
        self.i += 1
        img = np.full((self.h, self.w), 200, dtype=np.uint8)
        if 1.0 <= t < 2.0:
            x = int(self.w * (0.15 + 0.7 * (t - 1.0)))
            cv2.circle(img, (x, self.h // 3), 2, 40, -1)
        elif 3.0 <= t < 4.0:
            r = int(2 + (self.h * 0.6) * ((t - 3.0) ** 2))  # accelerating expansion
            cv2.circle(img, (int(self.w * 0.6), self.h // 2), max(1, r), 30, -1)
        noise = np.random.default_rng(self.i).integers(-3, 4, img.shape)
        out = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        self.preview = cv2.resize(out, (320, 240), interpolation=cv2.INTER_NEAREST)
        return out
