"""Vehicles as boxes in the camera image, from video tracking or from a simulation. This is what the
image-space score (image_score.py) compares: nothing here is lifted onto the ground.

Video: the tracker's boxes, as exported. Only tracks that follow a drawn route are kept (the simulation
is not asked to reproduce the others); broken tracks keep the id they were stitched to.

Simulation: each vehicle is a box of its kind's length, width and height on the road, projected through
the layout's camera (`camera.world_to_clip`, the same matrix Unity renders with); its image box is the
box around the eight projected corners (from_result). The simulation is not rendered and passed through
the detector, which would take minutes per clip on a GPU; so the projected boxes have none of the
detector's misses, merges and jitter that the video's boxes have (docs/twin.md section 4).

A vehicle's position is the bottom centre of its box, in pixels (x right, y down). Its `scale` is the
picture's pixels per metre of road at that point: a fixed map of the image, worked out once from the
camera, the same for video and simulation. Dividing image distances by it takes out the perspective
without using the box's own size, which differs between a detector's box and a projected one.

Both sides drop boxes cut by the frame edge and boxes where the road is under MIN_SCALE pixels per
metre (too far away to be detected reliably).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..vehicles import DIMENSIONS
from .tracks import EDGE, KIND_MAP

# Body height (m) by kind; a motorcycle or bicycle includes its rider.
HEIGHTS = {"motorcycle": 1.5, "bicycle": 1.5, "car": 1.5, "van": 2.0, "truck": 2.8, "bus": 3.0}
TWO_WHEEL = ("motorcycle", "bicycle")
MIN_SCALE = 2.5       # px per metre; vehicles further away than this are dropped
SCALE_STEP_PX = 4     # resolution of the scale map


@dataclass
class ImageTracks:
    """One row per observation, sorted by (track, time)."""
    width: int
    height: int
    duration: float
    step: float           # time between frames (s)
    t: np.ndarray
    tid: np.ndarray       # track id
    group: np.ndarray     # 0 two-wheelers, 1 four-wheelers and larger
    x: np.ndarray         # bottom centre (px)
    y: np.ndarray
    size: np.ndarray      # sqrt(box width * box height) (px)
    scale: np.ndarray     # pixels per metre of road at (x, y)

    @property
    def frames(self):
        return max(int(round(self.duration / self.step)), 1)


def _build(camera, duration, step, t, tid, group, box):
    """box: (n, 4) normalised x1, y1, x2, y2."""
    width, height = camera.width, camera.height
    box = np.asarray(box, float).reshape(-1, 4)
    t, tid, group = np.asarray(t, float), np.asarray(tid, np.int64), np.asarray(group, np.int64)
    w, h = (box[:, 2] - box[:, 0]) * width, (box[:, 3] - box[:, 1]) * height
    size = np.sqrt(np.maximum(w * h, 0))
    x, y = (box[:, 0] + box[:, 2]) / 2 * width, box[:, 3] * height
    scale = camera.scale(x, y)
    cut = (box[:, 0] <= EDGE) | (box[:, 2] >= 1 - EDGE) | (box[:, 3] >= 1 - EDGE)
    keep = ~cut & (scale >= MIN_SCALE) & (t >= 0) & (t <= duration)
    order = np.flatnonzero(keep)
    order = order[np.lexsort((t[order], tid[order]))]
    return ImageTracks(width, height, float(duration), float(step), t[order], tid[order], group[order],
                       x[order], y[order], size[order], scale[order])


# ------------------------------------------------------------------ video

def from_video(path, camera, matched_tracks=None, *, time_shift=0.0, duration=None):
    """Boxes from a video trajectories JSON. matched_tracks: the stitched, route-matched tracks of the same
    file (tracks.load_matched); when given, only tracks on a route are kept, under their stitched id.
    time_shift and duration move the boxes onto another clock (as from_result)."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    stitched_id = None
    if matched_tracks is not None:
        stitched_id = {raw: tr.id for tr in matched_tracks if tr.route >= 0 for raw in tr.source_ids}
    t, tid, group, box = [], [], [], []
    for v in data["vehicles"]:
        kind = KIND_MAP.get(v.get("kind"))
        if kind is None:
            continue
        track = int(v["track_id"])
        if stitched_id is not None:
            if track not in stitched_id:
                continue
            track = stitched_id[track]
        g = 0 if kind in TWO_WHEEL else 1
        for o in v["observations"]:
            if o.get("box"):
                t.append(o["time_s"] - time_shift); tid.append(track); group.append(g); box.append(o["box"])
    fps = float(data.get("fps") or 25.0)
    duration = float(duration if duration is not None else data["duration_s"] - time_shift)
    return _build(camera, duration, 1 / fps, t, tid, group, box)


# ------------------------------------------------------------------ simulation

class Camera:
    """The layout's camera: world points (Unity x, y up, z) to normalised image points."""

    def __init__(self, layout):
        if isinstance(layout, (str, Path)):
            layout = json.loads(Path(layout).read_text(encoding="utf-8-sig"))
        cam = layout["camera"]
        if "world_to_clip" not in cam:
            raise ValueError("The layout has no camera.world_to_clip; export it again from the Lane Editor")
        self.matrix = np.array(cam["world_to_clip"], float).reshape(4, 4)
        self.near = float(cam.get("near_m", 0.1))
        self.width, self.height = int(layout["image"]["width"]), int(layout["image"]["height"])
        self.ground_y = float(layout.get("ground_y", 0.0))
        self.layout = layout
        self._scale_map = self._build_scale_map()

    def _build_scale_map(self):
        """Pixels per metre of road at each image point (0 above the horizon): the square root of the
        area, in pixels, that one square metre of ground covers there."""
        m, g = self.matrix, self.ground_y
        # Ground (x, z, 1) -> clip (x, y, w): a homography.
        h = np.stack([m[[0, 1, 3], 0], m[[0, 1, 3], 2], m[[0, 1, 3], 1] * g + m[[0, 1, 3], 3]], axis=1)
        inverse = np.linalg.inv(h)
        xs = np.arange(0, self.width + SCALE_STEP_PX, SCALE_STEP_PX, dtype=float)
        ys = np.arange(0, self.height + SCALE_STEP_PX, SCALE_STEP_PX, dtype=float)
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        ndc = np.stack([gx / self.width * 2 - 1, (1 - gy / self.height) * 2 - 1, np.ones_like(gx)], axis=-1)
        ground = ndc @ inverse.T
        valid = np.abs(ground[..., 2]) > 1e-12
        ground = ground[..., :2] / np.where(valid, ground[..., 2], 1.0)[..., None]

        def image(dx, dz):
            world = np.stack([ground[..., 0] + dx, np.full(gx.shape, g), ground[..., 1] + dz], axis=-1)
            pts, front = self.project(world)
            return pts * np.array([self.width, self.height]), front

        p0, f0 = image(0, 0)
        px, f1 = image(0.5, 0)
        pz, f2 = image(0, 0.5)
        a, b = (px - p0) / 0.5, (pz - p0) / 0.5
        area = np.abs(a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0])
        # A ray above the horizon meets the ground plane behind the camera: no scale there.
        return np.where(valid & f0 & f1 & f2, np.sqrt(area), 0.0)

    def scale(self, x, y):
        """Pixels per metre of road at image points (px)."""
        i = np.clip(np.rint(np.asarray(x) / SCALE_STEP_PX).astype(np.int64), 0, self._scale_map.shape[0] - 1)
        j = np.clip(np.rint(np.asarray(y) / SCALE_STEP_PX).astype(np.int64), 0, self._scale_map.shape[1] - 1)
        return self._scale_map[i, j]

    def project(self, xyz):
        """(..., 3) world points -> (..., 2) normalised image points (y down) and whether in front."""
        p = np.concatenate([xyz, np.ones(xyz.shape[:-1] + (1,))], axis=-1) @ self.matrix.T
        w = p[..., 3]
        front = w > self.near
        w = np.where(front, w, 1.0)
        return np.stack([(p[..., 0] / w + 1) / 2, 1 - (p[..., 1] / w + 1) / 2], axis=-1), front

    def boxes(self, xy, heading, length, width, height):
        """Image boxes (n, 4) of upright boxes standing on the ground at xy (n, 2: Unity x, z), and
        whether each is wholly in front of the camera."""
        side = np.stack([-heading[:, 1], heading[:, 0]], axis=1)
        corners = []
        for a in (-0.5, 0.5):
            for b in (-0.5, 0.5):
                base = xy + heading * (a * length)[:, None] + side * (b * width)[:, None]
                for y in (0.0, 1.0):
                    corners.append(np.stack([base[:, 0], self.ground_y + y * height, base[:, 1]], axis=1))
        pts, front = self.project(np.stack(corners, axis=1))          # (n, 8, 2)
        box = np.concatenate([pts.min(axis=1), pts.max(axis=1)], axis=1)
        return box, front.all(axis=1)


def from_result(result, camera, *, time_shift=0.0, duration=None):
    """Boxes of a simulation Result as the camera would see them. time_shift: seconds the simulation
    started before the video (demand report's time_shift_s); duration: the video's length."""
    from ..output import C
    rows = result.samples
    kinds = {v["uid"]: v["kind"] for v in result.vehicles}
    uid = rows[:, C["uid"]].astype(np.int64)
    uniq, inverse = np.unique(uid, return_inverse=True)
    dims = np.array([DIMENSIONS[kinds[int(u)]] + (HEIGHTS[kinds[int(u)]],) for u in uniq])[inverse]
    group = np.array([0 if kinds[int(u)] in TWO_WHEEL else 1 for u in uniq])[inverse]
    box, front = camera.boxes(rows[:, [C["x"], C["y"]]], rows[:, [C["hx"], C["hy"]]], dims[:, 0], dims[:, 1], dims[:, 2])
    inside = front & (box[:, 2] > 0) & (box[:, 0] < 1) & (box[:, 3] > 0) & (box[:, 1] < 1)
    t = rows[:, C["t"]] - time_shift
    duration = float(duration if duration is not None else result.end_t - time_shift)
    return _build(camera, duration, result.dt, t[inside], uid[inside], group[inside], np.clip(box[inside], 0, 1))


def from_simulation_file(path, camera, *, time_shift=0.0, duration=None):
    """The same from a saved simulation trajectories JSON (Result.save)."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    t, tid, group, xy, heading, dims = [], [], [], [], [], []
    for v in data["vehicles"]:
        size = DIMENSIONS[v["kind"]] + (HEIGHTS[v["kind"]],)
        for o in v["observations"]:
            t.append(o["time_s"]); tid.append(int(v["track_id"])); group.append(0 if v["kind"] in TWO_WHEEL else 1)
            xy.append((o["world_position"][0], o["world_position"][2])); heading.append(o["heading"]); dims.append(size)
    t, xy, heading, dims = np.array(t) - time_shift, np.array(xy), np.array(heading), np.array(dims)
    box, front = camera.boxes(xy, heading, dims[:, 0], dims[:, 1], dims[:, 2])
    inside = front & (box[:, 2] > 0) & (box[:, 0] < 1) & (box[:, 3] > 0) & (box[:, 1] < 1)
    duration = float(duration if duration is not None else data["duration_s"] - time_shift)
    return _build(camera, duration, 1 / float(data.get("fps") or 10.0), t[inside],
                  np.array(tid)[inside], np.array(group)[inside], np.clip(box[inside], 0, 1))


# ------------------------------------------------------------------ gates

@dataclass
class Gate:
    name: str
    a: np.ndarray        # end points (px)
    b: np.ndarray
    forward: np.ndarray  # direction of travel in the image (unit)


def gates(camera):
    """One counting line per drawn lane, across its middle, in the image. Lanes whose line is not wholly
    in the picture, or is too short there to count on, have none."""
    out = []
    for lane in camera.layout["lanes"]:
        rows = lane.get("samples") or lane["sections"]
        k = len(rows) // 2
        if lane.get("direction", "Forward") == "Backward":
            rows = rows[::-1]
            k = len(rows) - 1 - k
        k0, k1 = max(k - 1, 0), min(k + 1, len(rows) - 1)
        world = np.array([rows[k]["left"], rows[k]["right"], rows[k0]["center"], rows[k1]["center"]], float)
        pts, front = camera.project(world)
        px = pts * np.array([camera.width, camera.height])
        inside = (pts > 0.02).all(axis=1) & (pts < 0.98).all(axis=1)
        if not front.all() or not inside[:2].all() or np.linalg.norm(px[1] - px[0]) < 12:
            continue
        forward = px[3] - px[2]
        if np.linalg.norm(forward) < 1e-6:
            continue
        out.append(Gate(f"{lane.get('name') or 'lane'} [{lane['id'][:4]}]", px[0], px[1], forward / np.linalg.norm(forward)))
    return out
