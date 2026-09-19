"""Real-world senses for the hunter: a camera frame -> the retinotopic columns the hunt brain was trained on.

The arena (hunt_gpu/arena.py, `sense`) gives the fly, per column across its field of view, an "object"
value = the person's angular width / 25 deg (capped at 1) and a "motion" value = 0.8 x object when
the person or the fly moves, 0.4 x object when both stand still. A person detector on the frame gives
the same numbers: a box's horizontal position is its bearing, its width its angular width, and frame
differencing inside the box says whether it moves. Columns outside the camera's view stay dark (the
fly's eye is 150 deg, a camera ~60: the fly is blind at the sides until the gimbal or the chassis turns).
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


MODEL = Path(__file__).resolve().parents[2] / "data" / "models" / "nanodet_2022nov.onnx"
MODEL_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/object_detection_nanodet/object_detection_nanodet_2022nov.onnx"


class NanoDetPeople:
    """NanoDet-Plus-m 416 from the OpenCV Zoo (COCO, 3.8 MB) through OpenCV's DNN module: person boxes
    (x, y, w, h) in the frame's pixels. Pre/post-processing ported from opencv_zoo's nanodet.py.
    ~40-80 ms a frame on a Pi 5 CPU; a missing model is downloaded once (`MODEL_URL`)."""

    def __init__(self, model: str | Path = MODEL, prob: float = 0.35, iou: float = 0.6):
        model = Path(model)
        if not model.exists():
            import urllib.request
            model.parent.mkdir(parents=True, exist_ok=True)
            print(f"[people] downloading the person detector to {model} ...", flush=True)
            urllib.request.urlretrieve(MODEL_URL, model)
        self.net = cv2.dnn.readNet(str(model))
        self.prob, self.iou = prob, iou
        self.strides, self.size, self.reg_max = (8, 16, 32, 64), 416, 7
        self.project = np.arange(self.reg_max + 1, dtype=np.float32)
        self.mean = np.array([103.53, 116.28, 123.675], dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array([57.375, 57.12, 58.395], dtype=np.float32).reshape(1, 1, 3)
        self.anchors = []
        for st in self.strides:
            n = self.size // st
            xv, yv = np.meshgrid(np.arange(n) * st, np.arange(n) * st)
            self.anchors.append(np.column_stack((xv.ravel() + 0.5 * (st - 1), yv.ravel() + 0.5 * (st - 1))).astype(np.float32))

    def __call__(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        h, w = frame.shape[:2]
        # letterbox into 416x416 (as the zoo demo does)
        top = left = 0
        if h != w:
            if h > w:
                newh, neww = self.size, int(self.size * w / h)
                img = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_AREA)
                left = (self.size - neww) // 2
                img = cv2.copyMakeBorder(img, 0, 0, left, self.size - neww - left, cv2.BORDER_CONSTANT, value=0)
            else:
                newh, neww = int(self.size * h / w), self.size
                img = cv2.resize(frame, (neww, newh), interpolation=cv2.INTER_AREA)
                top = (self.size - newh) // 2
                img = cv2.copyMakeBorder(img, top, self.size - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            newh = neww = self.size
            img = cv2.resize(frame, (self.size, self.size), interpolation=cv2.INTER_AREA)
        blob = cv2.dnn.blobFromImage(((img.astype(np.float32) - self.mean) / self.std))
        self.net.setInput(blob)
        outs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # pair the heads by shape (the output order is not the stride order): rows = (416/stride)^2, 80 channels = classes, 32 = box
        heads: dict[tuple[int, str], np.ndarray] = {}
        for o in outs:
            o = o.reshape(-1, o.shape[-1])
            st = int(round(self.size / math.sqrt(o.shape[0])))
            heads[(st, "cls" if o.shape[1] == 80 else "reg")] = o
        boxes, scores = [], []
        for st, anc in zip(self.strides, self.anchors):
            if (st, "cls") not in heads or (st, "reg") not in heads:
                continue
            cls, reg = heads[(st, "cls")], heads[(st, "reg")]
            x = np.exp(reg.reshape(-1, self.reg_max + 1)); x /= x.sum(axis=1, keepdims=True)
            d = (x @ self.project).reshape(-1, 4) * st
            person = cls[:, 0]                                        # COCO class 0
            keep = person > self.prob
            if not keep.any():
                continue
            a, d, sc = anc[keep], d[keep], person[keep]
            x1, y1 = np.clip(a[:, 0] - d[:, 0], 0, self.size), np.clip(a[:, 1] - d[:, 1], 0, self.size)
            x2, y2 = np.clip(a[:, 0] + d[:, 2], 0, self.size), np.clip(a[:, 1] + d[:, 3], 0, self.size)
            boxes += np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist(); scores += sc.tolist()
        if not boxes:
            return []
        idx = cv2.dnn.NMSBoxes(boxes, scores, self.prob, self.iou)
        out = []
        rh, rw = h / newh, w / neww
        for i in np.ravel(idx):
            bx, by, bw, bh = boxes[i]
            x1, y1 = max((bx - left) * rw, 0), max((by - top) * rh, 0)
            x2, y2 = min((bx + bw - left) * rw, w), min((by + bh - top) * rh, h)
            if x2 > x1 and y2 > y1:
                out.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return out


class PeopleSense:
    """Frames in, the hunter's vision map out: `observe(gray, fly_moving) -> (vis [bins, 2], boxes)`."""

    def __init__(self, bins: int = 24, fly_fov_deg: float = 150.0, cam_fov_deg: float = 62.0, detector=None,
                 motion_frac: float = 0.02, hold_s: float = 0.4, tick_s: float = 0.05):
        self.bins, self.fov, self.cam_fov = int(bins), float(fly_fov_deg), float(cam_fov_deg)
        self.bin_w = self.fov / self.bins
        self.bin_c = self.fov / 2 - self.bin_w * (np.arange(self.bins) + 0.5)   # column 0 = left edge, bearing +ve = left
        self.detector = detector if detector is not None else NanoDetPeople()
        self.motion_frac = float(motion_frac)
        self.hold_ticks = max(1, int(round(hold_s / tick_s)))   # a detector misses frames: keep the last boxes briefly
        self.prev: np.ndarray | None = None
        self.boxes: list[tuple[int, int, int, int]] = []
        self.moving: list[bool] = []
        self.age = 10 ** 6

    def boxes_to_vis(self, boxes, moving_flags, frame_w: int, fly_moving: bool) -> np.ndarray:
        vis = np.zeros((self.bins, 2), dtype=np.float32)
        for (x, y, w, h), mv in zip(boxes, moving_flags):
            cx = x + w / 2
            bearing = (frame_w / 2 - cx) / frame_w * self.cam_fov                 # image left = +bearing (left)
            width = w / frame_w * self.cam_fov
            strength = min(1.0, width / 25.0)
            moving = 1.0 if (mv or fly_moving) else 0.4
            cover = np.abs(bearing - self.bin_c) <= (width / 2 + self.bin_w / 2)
            vis[cover, 0] = np.maximum(vis[cover, 0], strength)
            vis[cover, 1] = np.maximum(vis[cover, 1], 0.8 * strength * moving)
        return vis

    def observe(self, gray: np.ndarray | None, fly_moving: bool = False, color: np.ndarray | None = None) -> tuple[np.ndarray, list]:
        """`gray` (H, W) drives the motion channel; the detector sees `color` (H, W, 3, same size) when given."""
        if gray is not None:
            boxes = self.detector(color if color is not None and color.shape[:2] == gray.shape[:2] else gray)
            diff = cv2.absdiff(gray, self.prev) if self.prev is not None and self.prev.shape == gray.shape else None
            self.prev = gray
            if boxes:
                moving = []
                for (x, y, w, h) in boxes:
                    if diff is None or w <= 0 or h <= 0:
                        moving.append(False)
                    else:
                        patch = diff[max(0, y):y + h, max(0, x):x + w]
                        moving.append(bool(patch.size and (patch > 25).mean() > self.motion_frac))
                self.boxes, self.moving, self.age = boxes, moving, 0
            else:
                self.age += 1
        else:
            self.age += 1
        if self.age > self.hold_ticks:
            self.boxes, self.moving = [], []
        w = gray.shape[1] if gray is not None else (self.prev.shape[1] if self.prev is not None else 320)
        return self.boxes_to_vis(self.boxes, self.moving, w, fly_moving), self.boxes
