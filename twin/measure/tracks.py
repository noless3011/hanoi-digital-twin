"""Tracks in metres on the road surface, smoothed, matched to routes. Video and simulation go through the
same code, so every metric is measured the same way on both sides.

Loading: a vehicle-trajectories JSON (video export or simulation output). World positions are Unity x, z
metres on the ground plane.

Smoothing: a constant-velocity Kalman filter with a Rauch-Tung-Striebel backward pass, per track, then
resampled to a common 10 Hz grid. Speed and acceleration come from the smoothed state, not from
differencing noisy positions.

Stitching: video tracks break when the tracker loses a vehicle. A track that starts shortly after another
ends, close to where that one was heading, with the same kind, is joined to it.

Route matching: each track is projected onto every route; the routes it stays on (inside the drivable
width, moving forward) are its candidates. Routes that share a start are identical there, so a short track
may keep several candidates; anything measured on the shared part is the same for all of them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..vehicles import DIMENSIONS

STEP = 0.1                      # resampling grid (s)
KIND_MAP = {"motorcycle": "motorcycle", "motorbike": "motorcycle", "car": "car", "van": "van", "bus": "bus",
            "truck": "truck", "bicycle": "bicycle"}
PROCESS_NOISE = 3.0             # white-acceleration spectral density (m2/s3)
MEASUREMENT_NOISE = 0.5         # position noise, standard deviation (m)


@dataclass
class Track:
    id: int
    kind: str
    t: np.ndarray               # (n,) on the STEP grid
    xy: np.ndarray              # (n, 2) smoothed
    vel: np.ndarray             # (n, 2)
    raw_t: np.ndarray = None    # original observations, for tracking-error estimates
    raw_xy: np.ndarray = None
    source_ids: list = field(default_factory=list)
    route: int = -1             # chosen route (-1: none)
    candidates: tuple = ()
    s: np.ndarray = None
    d: np.ndarray = None

    @property
    def speed(self):
        return np.linalg.norm(self.vel, axis=1)

    @property
    def length(self):
        return DIMENSIONS[self.kind][0]

    @property
    def width(self):
        return DIMENSIONS[self.kind][1]

    @property
    def duration(self):
        return float(self.t[-1] - self.t[0]) if len(self.t) else 0.0

    def accel(self):
        """Acceleration along the direction of travel (m/s2), from the smoothed speed."""
        v = self.speed
        if len(v) < 3:
            return np.zeros_like(v)
        # Averaged over 1 s: differencing even a smoothed speed amplifies tracking noise.
        window = int(round(1.0 / STEP))
        acc = np.gradient(v, STEP)
        if len(acc) >= window:
            acc = np.convolve(acc, np.ones(window) / window, mode="same")
        return acc


# ------------------------------------------------------------------ smoothing

def kalman_smooth(t, z, q=PROCESS_NOISE, r=MEASUREMENT_NOISE):
    """Constant-velocity RTS smoother. t (n,), z (n, 2). Returns positions and velocities (n, 2)."""
    n = len(t)
    x = np.zeros((n, 4))
    P = np.zeros((n, 4, 4))
    xp, Pp = np.zeros((n, 4)), np.zeros((n, 4, 4))
    H = np.array([[1., 0, 0, 0], [0, 1., 0, 0]])
    R = np.eye(2) * r * r
    v0 = (z[min(n - 1, 5)] - z[0]) / max(t[min(n - 1, 5)] - t[0], 1e-3) if n > 1 else np.zeros(2)
    x[0] = [z[0, 0], z[0, 1], v0[0], v0[1]]
    P[0] = np.diag([r * r, r * r, 9.0, 9.0])
    xp[0], Pp[0] = x[0], P[0]
    for k in range(1, n):
        dt = max(t[k] - t[k - 1], 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.]])
        Q = q * np.array([[dt ** 3 / 3, 0, dt ** 2 / 2, 0], [0, dt ** 3 / 3, 0, dt ** 2 / 2],
                          [dt ** 2 / 2, 0, dt, 0], [0, dt ** 2 / 2, 0, dt]])
        xp[k] = F @ x[k - 1]
        Pp[k] = F @ P[k - 1] @ F.T + Q
        S = H @ Pp[k] @ H.T + R
        K = Pp[k] @ H.T @ np.linalg.inv(S)
        x[k] = xp[k] + K @ (z[k] - H @ xp[k])
        P[k] = (np.eye(4) - K @ H) @ Pp[k]
    xs = x.copy()
    for k in range(n - 2, -1, -1):
        dt = max(t[k + 1] - t[k], 1e-3)
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1.]])
        C = P[k] @ F.T @ np.linalg.inv(Pp[k + 1])
        xs[k] = x[k] + C @ (xs[k + 1] - xp[k + 1])
    return xs[:, :2], xs[:, 2:]


def _resample(t, xy, vel, start=None):
    start = np.ceil(t[0] / STEP) * STEP if start is None else start
    grid = np.arange(start, t[-1] + 1e-9, STEP)
    if len(grid) == 0:
        grid = np.array([t[0]])
    interp = lambda a: np.stack([np.interp(grid, t, a[:, 0]), np.interp(grid, t, a[:, 1])], axis=1)
    return np.round(grid, 3), interp(xy), interp(vel)


# ------------------------------------------------------------------ loading

EDGE = 0.01   # a box this close to the frame edge is cut off, so its ground point is wrong


def _cut(observation):
    box = observation.get("box")
    if not box:
        return False
    x1, y1, x2, y2 = box
    return x1 <= EDGE or x2 >= 1 - EDGE or y2 >= 1 - EDGE

def load(path, *, smooth=True, min_observations=3):
    """Tracks from a vehicle-trajectories JSON. Video tracks are smoothed; simulated ones are only resampled."""
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    simulated = data.get("source") == "simulation"
    tracks = []
    for v in data["vehicles"]:
        kind = KIND_MAP.get(v.get("kind"))
        obs = [o for o in v["observations"] if o.get("world_position") is not None and not _cut(o)]
        if kind is None or len(obs) < min_observations:
            continue
        t = np.array([o["time_s"] for o in obs], float)
        z = np.array([[o["world_position"][0], o["world_position"][2]] for o in obs], float)
        keep = np.concatenate([[True], np.diff(t) > 1e-6])
        t, z = t[keep], z[keep]
        if len(t) < min_observations:
            continue
        if simulated or not smooth:
            if simulated and "speed_mps" in obs[0]:
                h = np.array([o["heading"] for o in obs], float)[keep]
                vel = h * np.array([o["speed_mps"] for o in obs], float)[keep][:, None]
            else:
                vel = np.gradient(z, t, axis=0)
            xy = z
        else:
            xy, vel = kalman_smooth(t, z)
        gt, gxy, gvel = _resample(t, xy, vel)
        tracks.append(Track(int(v["track_id"]), kind, gt, gxy, gvel, raw_t=t, raw_xy=z, source_ids=[int(v["track_id"])]))
    meta = {k: data.get(k) for k in ("feed", "duration_s", "source", "endpoints_file")}
    meta["duration_s"] = float(meta["duration_s"] or max((tr.t[-1] for tr in tracks), default=0.0))
    meta["simulated"] = simulated
    meta["path"] = str(path)
    return tracks, meta


# ------------------------------------------------------------------ stitching

def stitch(tracks, max_gap_s=2.0, max_distance_m=4.0):
    """Join track fragments: a track starting within max_gap_s after another ended, within max_distance_m
    of where that one would be by then, of the same kind. Greedy, closest first. Returns new tracks."""
    order = sorted(range(len(tracks)), key=lambda k: tracks[k].t[0])
    end_t = np.array([tr.t[-1] for tr in tracks])
    end_xy = np.array([tr.xy[-1] for tr in tracks])
    end_v = np.array([tr.vel[-1] for tr in tracks])
    kinds = np.array([tr.kind for tr in tracks])
    next_of, candidates = {}, []
    for b, tb in enumerate(tracks):
        gap = tb.t[0] - end_t
        ok = (gap >= 0) & (gap <= max_gap_s) & (kinds == tb.kind)
        ok[b] = False
        for a in np.flatnonzero(ok):
            dist = float(np.linalg.norm(tb.xy[0] - (end_xy[a] + end_v[a] * gap[a])))
            if dist <= max_distance_m:
                candidates.append((dist + gap[a], int(a), b))
    candidates.sort()
    has_prev = set()
    for _, a, b in candidates:
        if a in next_of or b in has_prev:
            continue
        next_of[a] = b
        has_prev.add(b)
    out = []
    for k in order:
        if k in has_prev:
            continue
        chain = [k]
        while chain[-1] in next_of:
            chain.append(next_of[chain[-1]])
        if len(chain) == 1:
            out.append(tracks[k])
            continue
        parts = [tracks[c] for c in chain]
        t = np.concatenate([p.raw_t for p in parts])
        z = np.concatenate([p.raw_xy for p in parts])
        xy, vel = kalman_smooth(t, z)
        gt, gxy, gvel = _resample(t, xy, vel)
        out.append(Track(parts[0].id, parts[0].kind, gt, gxy, gvel, raw_t=t, raw_xy=z,
                         source_ids=[i for p in parts for i in p.source_ids]))
    return out


# ------------------------------------------------------------------ routes

def match_routes(tracks, network, *, margin=1.0, min_share=0.85):
    """Set each track's candidate routes and its (s, d) on the chosen one."""
    R = len(network.routes)
    for tr in tracks:
        best, scores = [], []
        moving = tr.speed > 1.0
        for r in range(R):
            s, d, valid, idx = network.project(np.full(len(tr.t), r), tr.xy)
            dr, dl = network.DR[r, idx], network.DL[r, idx]
            on = valid & (d >= dr - margin) & (d <= dl + margin) & (s >= -5) & (s <= network.length[r] + 5)
            forward = np.ones(len(tr.t), bool)
            if moving.any():
                tangent = network.T[r, idx]
                forward = ~moving | (np.sum(tr.vel * tangent, axis=1) > 0)
            share = float(np.mean(on & forward))
            scores.append(share)
            if share >= min_share:
                best.append(r)
        if not best:
            tr.candidates, tr.route = (), -1
            continue
        tr.candidates = tuple(best)
        # With several candidates, prefer the one followed for longest (most forward coverage).
        tr.route = max(best, key=lambda r: scores[r])
        tr.s, tr.d, _, _ = network.project(np.full(len(tr.t), tr.route), tr.xy)
    return tracks


def load_matched(path, network, *, stitch_fragments=True):
    tracks, meta = load(path)
    if stitch_fragments and not meta["simulated"]:
        before = len(tracks)
        tracks = stitch(tracks)
        meta["stitched"] = before - len(tracks)
    match_routes(tracks, network)
    meta["unmatched"] = sum(tr.route < 0 for tr in tracks)
    return tracks, meta


def from_result(result):
    """Tracks straight from a simulation Result (no file round trip)."""
    from ..output import C
    tracks = []
    meta_by_uid = {v["uid"]: v for v in result.vehicles}
    for uid, rows in result.tracks().items():
        meta = meta_by_uid[uid]
        heading = rows[:, [C["hx"], C["hy"]]]
        vel = heading * rows[:, C["v"]][:, None]
        tr = Track(uid, meta["kind"], np.round(rows[:, C["t"]], 3), rows[:, [C["x"], C["y"]]], vel, source_ids=[uid])
        tr.route, tr.candidates = meta["route"], (meta["route"],)
        tr.s, tr.d = rows[:, C["s"]], rows[:, C["d"]]
        tracks.append(tr)
    return tracks, {"duration_s": result.end_t, "simulated": True}

