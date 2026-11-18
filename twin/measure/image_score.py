"""How far a simulation is from a video, measured only in the camera image (boxes from image.py).

Nothing follows a vehicle for longer than MOTION_WINDOW_S, so broken tracks do not matter. Six parts,
each a distance from 0 (the same) to 1:

  occupancy_map    where vehicles are: mean vehicles per frame in CELL_PX cells over the whole clip,
                   lightly blurred; sum|a - b| / sum(a + b)
  occupancy_time   when they are there: the same per coarse zone and TIME_BIN_S seconds
  motion           how they move: per zone and group (two-wheelers, larger), the image velocity over
                   MOTION_WINDOW_S divided by the picture's scale there (pixels per metre of road, so
                   roughly m/s); Wasserstein distance of its x and y parts / the video's mean speed
  crossings        throughput and delay: per counting line (one per drawn lane), area between the
                   cumulative crossing curves / area under the upper one
  headways         time between successive crossings of each line; Wasserstein / the video's mean
  spacing          image distance to the nearest other vehicle divided by the picture's scale (roughly
                   metres), per group; Wasserstein / the video's mean

The video side is the tracker's boxes; the simulation side is boxes projected straight from the simulation
(image.from_result). Rendering the simulation and tracking it with the detector would put the detector's
misses, merges and jitter on both sides, but takes minutes per clip on a GPU against 0.1 s here, so it is
not done; the gap that leaves is in docs/twin.md section 4.

occupancy_time and crossings line the two sides up in time, so they need the simulation to replay the
video's own arrivals. total is the weighted mean (DEFAULT_WEIGHTS) of the parts both sides have data for.

Score a saved simulation of a video's demand (twin.run --video ...) against that video:

    .venv\\Scripts\\python.exe -m twin.measure.image_score --layout L.camera-lanes.json --video V.trajectories.json `
        --simulation S.trajectories.json
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

CELL_PX = 16.0
ZONES = (6, 4)             # coarse zones across and down the image
TIME_BIN_S = 10.0
MOTION_WINDOW_S = 0.6
SPACING_EVERY_S = 0.5
SPACING_CAP = 15.0         # nearest-neighbour distances are capped at this (about metres)
MIN_SAMPLES = 30
MIN_CROSSINGS = 8
PARTS = ("occupancy_map", "occupancy_time", "motion", "crossings", "headways", "spacing")
# crossings and headways need a track to hold across a line, which a tracker does less reliably than it
# finds vehicles frame by frame, so they count half.
DEFAULT_WEIGHTS = {"occupancy_map": 1.0, "occupancy_time": 1.0, "motion": 1.0, "spacing": 1.0,
                   "crossings": 0.5, "headways": 0.5}


def wasserstein(a, b):
    a, b = np.sort(np.asarray(a, float)), np.sort(np.asarray(b, float))
    grid = np.sort(np.concatenate([a, b]))
    cdf_a = np.searchsorted(a, grid[:-1], "right") / len(a)
    cdf_b = np.searchsorted(b, grid[:-1], "right") / len(b)
    return float(np.sum(np.abs(cdf_a - cdf_b) * np.diff(grid)))


def _bray_curtis(a, b):
    total = float(np.sum(a + b))
    return float(np.sum(np.abs(a - b)) / total) if total > 0 else None


def _zone(tr):
    zx = np.clip((tr.x / tr.width * ZONES[0]).astype(np.int64), 0, ZONES[0] - 1)
    zy = np.clip((tr.y / tr.height * ZONES[1]).astype(np.int64), 0, ZONES[1] - 1)
    return zx * ZONES[1] + zy


# ------------------------------------------------------------------ features of one side

def velocities(tr, window=MOTION_WINDOW_S):
    """Per observation that has an earlier one of the same track about `window` ago: index, and velocity
    in the image divided by the picture's scale there (about m/s; x, y)."""
    if len(tr.t) == 0:
        return np.zeros(0, np.int64), np.zeros((0, 2))
    span = 1e6
    key = tr.tid * span + tr.t                  # sorted, since rows are sorted by (track, time)
    j = np.searchsorted(key, key - window)
    j = np.clip(np.stack([j - 1, j]), 0, len(key) - 1)
    err = np.abs(tr.t[j] - (tr.t - window)[None, :])
    err[tr.tid[j] != tr.tid[None, :]] = np.inf
    pick = np.argmin(err, axis=0)
    cols = np.arange(len(key))
    j, err = j[pick, cols], err[pick, cols]
    ok = err <= 0.15 * window
    i, j = cols[ok], j[ok]
    dt = tr.t[i] - tr.t[j]
    scale = (tr.scale[i] + tr.scale[j]) / 2
    u = np.stack([tr.x[i] - tr.x[j], tr.y[i] - tr.y[j]], axis=1) / (dt * scale)[:, None]
    return i, u


def crossings(tr, gate_list):
    """{gate name: sorted times at which a track first crossed the line going forward}."""
    out = {g.name: [] for g in gate_list}
    if len(tr.t) < 2:
        return {k: np.zeros(0) for k in out}
    same = tr.tid[1:] == tr.tid[:-1]
    p0 = np.stack([tr.x[:-1], tr.y[:-1]], axis=1)[same]
    p1 = np.stack([tr.x[1:], tr.y[1:]], axis=1)[same]
    t1, tid = tr.t[1:][same], tr.tid[1:][same]
    step = p1 - p0
    for g in gate_list:
        line = g.b - g.a
        normal = np.array([-line[1], line[0]])
        if normal @ g.forward < 0:
            normal = -normal
        s0, s1 = (p0 - g.a) @ normal, (p1 - g.a) @ normal
        hit = (s0 < 0) & (s1 >= 0)
        if not hit.any():
            continue
        k = np.flatnonzero(hit)
        f = s0[k] / (s0[k] - s1[k])
        along = ((p0[k] + step[k] * f[:, None]) - g.a) @ line / (line @ line)
        k = k[(along >= 0) & (along <= 1)]
        _, first = np.unique(tid[k], return_index=True)        # rows are time-ordered within a track
        out[g.name] = np.sort(t1[k][first])
    return {k: np.asarray(v, float) for k, v in out.items()}


def spacing(tr):
    """Per group: image distance from each vehicle to the nearest other one over the picture's scale, sampled
    every SPACING_EVERY_S. A vehicle alone in the picture counts as SPACING_CAP."""
    out = {0: [], 1: []}
    if len(tr.t) == 0:
        return {g: np.zeros(0) for g in out}
    every = max(int(round(SPACING_EVERY_S / tr.step)), 1)
    frame = np.rint(tr.t / tr.step).astype(np.int64)
    sel = np.flatnonzero(frame % every == 0)
    order = sel[np.argsort(frame[sel], kind="stable")]
    f = frame[order]
    bounds = np.flatnonzero(np.diff(f)) + 1
    for rows in np.split(order, bounds):
        if len(rows) == 1:
            nearest = np.array([SPACING_CAP])
        else:
            pts = np.stack([tr.x[rows], tr.y[rows]], axis=1)
            dist = cKDTree(pts).query(pts, k=2)[0][:, 1]
            nearest = np.minimum(dist / tr.scale[rows], SPACING_CAP)
        for g in (0, 1):
            out[g].append(nearest[tr.group[rows] == g])
    return {g: np.concatenate(v) if v else np.zeros(0) for g, v in out.items()}


def features(tr, gate_list):
    """Everything the score needs from one side. Computing it once per video saves time in a search."""
    nx, ny = int(np.ceil(tr.width / CELL_PX)), int(np.ceil(tr.height / CELL_PX))
    cell = np.histogram2d(tr.x, tr.y, bins=[nx, ny], range=[[0, nx * CELL_PX], [0, ny * CELL_PX]])[0] / tr.frames
    zone = _zone(tr)
    bins = max(int(np.ceil(tr.duration / TIME_BIN_S)), 1)
    tb = np.clip((tr.t / TIME_BIN_S).astype(np.int64), 0, bins - 1)
    frames_per_bin = TIME_BIN_S / tr.step
    zone_time = np.zeros((ZONES[0] * ZONES[1], bins))
    np.add.at(zone_time, (zone, tb), 1.0 / frames_per_bin)
    i, u = velocities(tr)
    motion = {}
    key = zone[i] * 2 + tr.group[i]
    for k in np.unique(key):
        motion[int(k)] = u[key == k]
    return {"map": gaussian_filter(cell, 1.0, mode="constant"), "zone_time": zone_time, "motion": motion,
            "crossings": crossings(tr, gate_list), "spacing": spacing(tr), "duration": tr.duration,
            "observations": len(tr.t)}


# ------------------------------------------------------------------ the score

def _w1_part(real, sim, scale_of, min_samples=MIN_SAMPLES):
    """Weighted mean over keys of a clipped, scaled Wasserstein distance. A key the video has and the
    simulation does not counts as 1."""
    rows, weights = {}, {}
    for key, r in real.items():
        if len(r) < min_samples:
            continue
        s = sim.get(key, ())
        rows[key] = 1.0 if len(s) < max(min_samples // 3, 3) else min(1.0, scale_of(r, s))
        weights[key] = len(r)
    if not rows:
        return None, {}
    total = sum(weights.values())
    return sum(rows[k] * weights[k] for k in rows) / total, rows


def _motion_distance(r, s):
    scale = max(float(np.mean(np.linalg.norm(r, axis=1))), 0.5)
    return (wasserstein(r[:, 0], s[:, 0]) + wasserstein(r[:, 1], s[:, 1])) / (2 * scale)


def _cumulative_distance(r, s, duration):
    grid = np.arange(0.0, duration + 1.0, 1.0)
    nr, ns = np.searchsorted(r, grid, "right"), np.searchsorted(s, grid, "right")
    upper = float(np.sum(np.maximum(nr, ns)))
    return float(np.sum(np.abs(nr - ns))) / upper if upper > 0 else 0.0


def score(real, sim, weights=None):
    """Distance between two feature sets (features()). Returns {"total", "parts", "detail"}."""
    parts, detail = {}, {}
    parts["occupancy_map"] = _bray_curtis(real["map"], sim["map"])
    bins = min(real["zone_time"].shape[1], sim["zone_time"].shape[1])
    parts["occupancy_time"] = _bray_curtis(real["zone_time"][:, :bins], sim["zone_time"][:, :bins])
    parts["motion"], detail["motion"] = _w1_part(real["motion"], sim["motion"], _motion_distance)

    duration = min(real["duration"], sim["duration"])
    rows, weight = {}, {}
    for name, r in real["crossings"].items():
        s = sim["crossings"].get(name, np.zeros(0))
        if max(len(r), len(s)) < MIN_CROSSINGS:
            continue
        rows[name] = _cumulative_distance(r, s, duration)
        weight[name] = max(len(r), len(s))
    detail["crossings"] = rows
    parts["crossings"] = sum(rows[k] * weight[k] for k in rows) / sum(weight.values()) if rows else None

    gaps = lambda side: {k: np.diff(v) for k, v in side["crossings"].items()}
    parts["headways"], detail["headways"] = _w1_part(
        gaps(real), gaps(sim), lambda r, s: wasserstein(r, s) / max(float(np.mean(r)), 0.2), MIN_CROSSINGS)
    parts["spacing"], detail["spacing"] = _w1_part(
        real["spacing"], sim["spacing"], lambda r, s: wasserstein(r, s) / max(float(np.mean(r)), 0.2))

    parts = {k: v for k, v in parts.items() if v is not None}
    w = {k: float((DEFAULT_WEIGHTS if weights is None else weights).get(k, 1.0)) for k in parts}
    total = sum(parts[k] * w[k] for k in parts) / sum(w.values()) if parts and sum(w.values()) > 0 else float("inf")
    return {"total": round(total, 4), "parts": {k: round(v, 4) for k, v in parts.items()},
            "detail": {name: {str(k): round(float(v), 3) for k, v in rows.items()} for name, rows in detail.items()}}


def main(argv=None):
    import argparse
    import json
    from pathlib import Path

    from ..network import Network
    from . import image
    from .demand import from_tracks
    from .tracks import load_matched

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--simulation", type=Path, required=True, help="Simulation trajectories of this video's demand")
    args = parser.parse_args(argv)
    network, camera = Network.from_layout(args.layout), image.Camera(args.layout)
    gate_list = image.gates(camera)
    tracks, meta = load_matched(args.video, network)
    shift = from_tracks(tracks, network)[1]["time_shift_s"]
    video = features(image.from_video(args.video, camera, tracks), gate_list)
    projected = features(image.from_simulation_file(args.simulation, camera, time_shift=shift, duration=meta["duration_s"]), gate_list)
    s = score(video, projected)
    print(f"video_vs_simulation      {s['total']:.4f}  {json.dumps(s['parts'])}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
