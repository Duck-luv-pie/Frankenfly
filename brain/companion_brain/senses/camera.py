"""Frame sources: the ESP32-CAM MJPEG stream, a video file, or a synthetic looming stimulus."""
from __future__ import annotations

import math
import time

import cv2
import numpy as np


class FrameSource:
    def read(self) -> np.ndarray | None:  # grayscale uint8 (H, W) or None when exhausted
        raise NotImplementedError

    def close(self) -> None:
        pass


class CameraStream(FrameSource):
    """Reads the ESP32-CAM MJPEG stream (or any OpenCV-readable URL / file)."""

    def __init__(self, url: str, width: int = 80, height: int = 60, reconnect_s: float = 2.0):
        self.url, self.w, self.h = url, width, height
        self.reconnect_s = reconnect_s
        self.cap: cv2.VideoCapture | None = None
        self.last_attempt = 0.0
        self.is_file = not url.startswith(("http://", "https://", "rtsp://"))
        self._open()

    def _open(self) -> None:
        self.last_attempt = time.time()
        self.cap = cv2.VideoCapture(self.url)
        if not self.cap.isOpened():
            self.cap = None

    def read(self) -> np.ndarray | None:
        if self.cap is None:
            if time.time() - self.last_attempt > self.reconnect_s:
                self._open()
            return None
        ok, frame = self.cap.read()
        if not ok:
            if self.is_file:
                return None
            self.cap.release()
            self.cap = None
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        return cv2.resize(gray, (self.w, self.h), interpolation=cv2.INTER_AREA)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class SyntheticLooming(FrameSource):
    """A dark disc that expands toward the camera (a looming stimulus), used for tests and demos.
    Timeline (seconds): 0-1 static, 1-2 small object drifts left->right (LC11/LC10a), 2-3 static,
    3-4 disc expands rapidly (LC4/LPLC2), 4-5 static."""

    def __init__(self, width: int = 80, height: int = 60, fps: float = 15.0, duration_s: float = 5.0):
        self.w, self.h, self.fps = width, height, fps
        self.n_frames = int(duration_s * fps)
        self.i = 0

    def read(self) -> np.ndarray | None:
        if self.i >= self.n_frames:
            return None
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
        return np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
