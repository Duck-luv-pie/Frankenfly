"""
test_retina_agreement.py -- Retina.from_state and Retina.from_boxes must
land on the same 24-column output for the same physical scene.

We synthesize 320x240 RGB frames of a person-like cylinder (radius r,
height 1.7 m, feet on the floor, camera 0.231 m up) using the *same*
pinhole model brain/retina.py::Camera implements, run a trivial colour
threshold "detector" on the frame to get a pixel bounding box (this
stands in for the YOLO/ultralytics wrapper in scripts/robot_bridge.py --
opencv-python and ultralytics are not installed in this venv, so we
cannot exercise the real detector here), and feed that box into
Retina.from_boxes. In parallel, we feed the exact ground-truth
robot/human pose into Retina.from_state for the same scene. The two
outputs should agree.
"""
import math

import numpy as np
import pytest
import torch

from brain.retina import Camera, Retina


# --------------------------------------------------------------------------
# synthetic scene rendering
# --------------------------------------------------------------------------

PERSON_HEIGHT_M = 1.7
RED = (230, 20, 20)


def _project_box(cam: Camera, bearing_rad: float, d: float, r: float):
    """Analytic pinhole projection of a standing cylinder (person) blob.

    bearing_rad is the azimuth of the cylinder's centre axis, + = the
    robot's right = image +x (same convention as Camera.px_to_az).
    Horizontal edges come from the two tangent lines to the cylinder,
    at azimuth bearing +/- asin(r/d); the vertical projection uses
    d_forward = d*cos(bearing) as the depth Z, per x = fx*X/Z + cx with
    X = d*sin(bearing), Z = d*cos(bearing).

    Returns (x0f, y_floor, x1f, y_head, az0, az1): continuous pixel
    coordinates (unclipped to the frame) and the true left/right
    azimuths in radians. x0f/az0 is the left (smaller-x) edge.
    """
    half_w = math.asin(min(r / d, 0.999))
    az0 = bearing_rad - half_w
    az1 = bearing_rad + half_w
    x0f = cam.cx + cam.fx * math.tan(az0)
    x1f = cam.cx + cam.fx * math.tan(az1)

    d_forward = d * math.cos(bearing_rad)
    y_floor = cam.cy + cam.fy * (cam.height_m / d_forward)
    y_head = cam.cy - cam.fy * ((PERSON_HEIGHT_M - cam.height_m) / d_forward)
    return x0f, y_floor, x1f, y_head, az0, az1


def _make_frame(cam: Camera, bearing_rad: float, d: float, r: float):
    """Render a flat grey wall / floor-gradient background with a solid
    red rectangle standing in for the projected person-cylinder blob."""
    W, H = cam.W, cam.H
    frame = np.zeros((H, W, 3), dtype=np.uint8)

    horizon = int(round(cam.cy))
    frame[:horizon, :] = (110, 110, 115)
    if H > horizon:
        ramp = np.linspace(70, 170, H - horizon).astype(np.uint8)
        frame[horizon:, :, 0] = ramp[:, None]
        frame[horizon:, :, 1] = ramp[:, None]
        frame[horizon:, :, 2] = ramp[:, None]

    x0f, y_floor, x1f, y_head, az0, az1 = _project_box(cam, bearing_rad, d, r)

    ix0, ix1 = int(math.floor(x0f)), int(math.ceil(x1f))
    iy_top, iy_bottom = int(math.floor(y_head)), int(math.ceil(y_floor))

    cix0, cix1 = max(0, ix0), min(W, ix1)
    ciy0, ciy1 = max(0, iy_top), min(H, iy_bottom)
    if cix1 > cix0 and ciy1 > ciy0:
        frame[ciy0:ciy1, cix0:cix1] = RED

    return frame, (x0f, y_floor, x1f, y_head), (az0, az1)


def _detect_red_bbox(frame: np.ndarray):
    """Tiny stand-in for the YOLO person-detector wrapper used by
    scripts/robot_bridge.py (ultralytics/opencv-python are not installed
    in this environment). Thresholds the solid-red blob and returns the
    pixel bounding box of the mask as (x0, y0, x1, y1), or None if
    nothing matched -- i.e. the blob is entirely outside the frame."""
    r = frame[..., 0].astype(np.int16)
    g = frame[..., 1].astype(np.int16)
    b = frame[..., 2].astype(np.int16)
    mask = (r > 150) & ((r - g) > 60) & ((r - b) > 60)
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _build_scenes():
    cam = Camera()
    rng = np.random.default_rng(42)

    n_random = 40
    bearings = rng.uniform(-40.0, 40.0, n_random)
    dists = rng.uniform(0.8, 4.0, n_random)
    radii = rng.uniform(0.22, 0.30, n_random)
    raw = [dict(tag=f"rand{i}", bearing_deg=float(bearings[i]),
                d=float(dists[i]), r=float(radii[i]))
           for i in range(n_random)]

    # forced partly-out-of-frame cases: extreme bearing + close range so
    # the cylinder's tangent lines fall past the horizontal FOV edge.
    edge_bearing = [-40.0, -39.0, -38.0, -37.0, 37.0, 38.0, 39.0, 40.0, -40.0, 40.0]
    edge_d = [0.80, 0.85, 0.90, 0.95, 0.80, 0.85, 0.90, 0.95, 1.00, 1.00]
    edge_r = [0.30, 0.29, 0.30, 0.28, 0.30, 0.29, 0.30, 0.28, 0.30, 0.30]
    raw += [dict(tag=f"edge{i}", bearing_deg=bd, d=d, r=r)
            for i, (bd, d, r) in enumerate(zip(edge_bearing, edge_d, edge_r))]

    assert len(raw) == 50

    built = []
    for s in raw:
        bearing_rad = math.radians(s["bearing_deg"])
        frame, (x0f, y_floor, x1f, y_head), (az0, az1) = _make_frame(
            cam, bearing_rad, s["d"], s["r"])
        box = _detect_red_bbox(frame)
        fully_inside_h = (x0f >= 0.0) and (x1f <= cam.W)
        built.append(dict(
            id=s["tag"], bearing_deg=s["bearing_deg"], bearing_rad=bearing_rad,
            d=s["d"], r=s["r"], frame=frame, box=box,
            x0f=x0f, x1f=x1f, az0=az0, az1=az1, fully_inside_h=fully_inside_h,
        ))
    return built


SCENES = _build_scenes()
VISIBLE_SCENES = [s for s in SCENES if s["box"] is not None]
FULLY_INSIDE_SCENES = [s for s in VISIBLE_SCENES if s["fully_inside_h"]]


# --------------------------------------------------------------------------
# retina-output helpers
# --------------------------------------------------------------------------

def _covered_indices(pres, thresh=0.05):
    return (pres > thresh).nonzero(as_tuple=True)[0]


def _centroid_col(pres):
    total = pres.sum()
    cols = torch.arange(pres.numel(), dtype=pres.dtype)
    return (cols * pres).sum() / total


CAM = Camera()


# --------------------------------------------------------------------------
# fixture sanity checks (make sure the scene set actually stresses both
# the "partly clipped" and "fully inside" regimes the assignment asked for)
# --------------------------------------------------------------------------

def test_have_partly_out_of_frame_cases():
    clipped = [s for s in SCENES if s["box"] is not None and not s["fully_inside_h"]]
    assert len(clipped) >= 3, "expected several partly-out-of-frame scenes in the fixture"


def test_have_fully_inside_cases():
    assert len(FULLY_INSIDE_SCENES) >= 5, "expected several fully-inside scenes for the round-trip test"


# --------------------------------------------------------------------------
# main agreement test
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scene", SCENES, ids=[s["id"] for s in SCENES])
def test_state_and_boxes_agree(scene):
    if scene["box"] is None:
        pytest.skip(
            f"blob entirely outside frame: bearing={scene['bearing_deg']:.1f} deg, "
            f"d={scene['d']:.2f} m, r={scene['r']:.2f} m"
        )

    cam = CAM
    ret_state = Retina(cam)   # fresh per scene: no temporal smoothing carry-over
    ret_boxes = Retina(cam)

    bearing_ccw = -scene["bearing_rad"]
    human_xy = torch.tensor(
        [[[scene["d"] * math.cos(bearing_ccw), scene["d"] * math.sin(bearing_ccw)]]],
        dtype=torch.float32,
    )
    robot_xy = torch.zeros(1, 2, dtype=torch.float32)
    robot_yaw = torch.zeros(1, dtype=torch.float32)
    human_r = torch.tensor([[scene["r"]]], dtype=torch.float32)
    valid = torch.tensor([[True]])

    out_state = ret_state.from_state(robot_xy, robot_yaw, human_xy, human_r, valid, dt=1 / 30)

    x0, y0, x1, y1 = scene["box"]
    boxes = torch.tensor([[[x0, y0, x1, y1]]], dtype=torch.float32)
    valid_b = torch.tensor([[True]])
    out_boxes = ret_boxes.from_boxes(boxes, valid_b, dt=1 / 30)

    pres_s, pres_b = out_state["pres"][0], out_boxes["pres"][0]
    size_s, size_b = out_state["size"][0], out_boxes["size"][0]

    idx_s = _covered_indices(pres_s)
    idx_b = _covered_indices(pres_b)
    assert idx_s.numel() > 0, "ground-truth path found no covered columns"
    assert idx_b.numel() > 0, "detector path found no covered columns"

    left_diff = abs(idx_s.min().item() - idx_b.min().item())
    right_diff = abs(idx_s.max().item() - idx_b.max().item())
    assert left_diff <= 1, (
        f"leftmost covered column differs by {left_diff}: "
        f"state={idx_s.min().item()} boxes={idx_b.min().item()}"
    )
    assert right_diff <= 1, (
        f"rightmost covered column differs by {right_diff}: "
        f"state={idx_s.max().item()} boxes={idx_b.max().item()}"
    )

    c_s, c_b = _centroid_col(pres_s).item(), _centroid_col(pres_b).item()
    assert abs(c_s - c_b) < 1.0, (
        f"pres-weighted centroid column differs: state={c_s:.3f} boxes={c_b:.3f}"
    )

    common = ((pres_s > 0.05) & (pres_b > 0.05)).nonzero(as_tuple=True)[0]
    assert common.numel() > 0, "no columns covered by both paths"
    size_diff = (size_s[common] - size_b[common]).abs().max().item()
    assert size_diff < 0.02, (
        f"size disagreement {size_diff:.4f} on common columns {common.tolist()}: "
        f"state={size_s[common].tolist()} boxes={size_b[common].tolist()}"
    )


# --------------------------------------------------------------------------
# pixel-box <-> azimuth round trip
# --------------------------------------------------------------------------

@pytest.mark.parametrize("scene", FULLY_INSIDE_SCENES, ids=[s["id"] for s in FULLY_INSIDE_SCENES])
def test_pixel_box_roundtrips_to_azimuth(scene):
    """px_to_az of the drawn-then-detected box edges must recover
    bearing +/- asin(r/d) within 0.5 deg when the box is fully inside
    the frame (no clipping to distort the edges)."""
    cam = CAM
    x0, _, x1, _ = scene["box"]
    az0_rec = cam.px_to_az(torch.tensor(x0, dtype=torch.float32)).item()
    az1_rec = cam.px_to_az(torch.tensor(x1, dtype=torch.float32)).item()

    err0_deg = math.degrees(abs(az0_rec - scene["az0"]))
    err1_deg = math.degrees(abs(az1_rec - scene["az1"]))
    assert err0_deg < 0.5, f"left edge azimuth off by {err0_deg:.3f} deg"
    assert err1_deg < 0.5, f"right edge azimuth off by {err1_deg:.3f} deg"
