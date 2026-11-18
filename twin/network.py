"""Road network: routes, projection onto routes, and conflict points between routes.

A route is a vehicle's path from an entry to an exit: a centreline sampled every DS metres, with the
drivable width on each side. Positions are (s, d): s the distance along the centreline, d the signed
sideways offset along the left-pointing normal. There are no lanes inside a route; the width is one
continuous space.

The network is built from lanes (directed polylines between nodes, each with a left and a right edge),
either from a Unity camera-lane layout (from_layout) or written in code for test roads (scenarios.py).
Routes are all lane sequences from an entry node to an exit node.

Two routes meet in two ways:
  crossing  their centrelines cross at an angle (after any shared start, before any shared end);
  merge     they come from different lanes into the same lane.
Routes that run along the same lanes in the same or the opposite direction do not have a conflict
point: vehicles there see each other directly, as something in their path.

World coordinates are 2D (x, y) metres. For Unity layouts x is Unity x and y is Unity z.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

DS = 0.5               # route sample spacing (m)
GRID_CELL = 0.5        # projection grid cell (m)
GRID_BAND = 8.0        # points further than this beyond a route's edge do not project onto it (m)
MAX_ZONES = 3          # conflict points kept per route pair
CROSS_MIN_SIN = 0.34   # crossings flatter than ~20 degrees are treated as running alongside
CROSS, MERGE = 0, 1
CLUSTER_GAP = 3.0      # overlap stretches closer than this along a route share one junction area (m)


@dataclass
class Lane:
    id: str
    start: str
    end: str
    center: np.ndarray   # (n, 2)
    left: np.ndarray     # (n, 2) left edge points
    right: np.ndarray    # (n, 2) right edge points


@dataclass
class Route:
    index: int
    id: str
    lanes: tuple
    xy: np.ndarray
    s: np.ndarray
    tangent: np.ndarray
    normal: np.ndarray
    d_left: np.ndarray       # signed offset of the left edge
    d_right: np.ndarray      # signed offset of the right edge
    lane_of: np.ndarray      # lane position (index into lanes) of each sample
    lane_start: dict = field(default_factory=dict)  # lane id -> station where it starts

    @property
    def length(self):
        return float(self.s[-1])

    @property
    def origin(self):
        return self.lanes[0]

    def describe(self):
        return f"{self.id} ({self.length:.0f} m)"


def _resample(center, left, right):
    """Centreline resampled every DS metres, with the signed edge offsets carried along."""
    center = np.asarray(center, float)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(center, axis=0), axis=1) > 1e-6])
    center, left, right = center[keep], np.asarray(left, float)[keep], np.asarray(right, float)[keep]
    seg = np.linalg.norm(np.diff(center, axis=0), axis=1)
    arc = np.concatenate([[0.], np.cumsum(seg)])
    t = np.gradient(center, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    dl = np.sum((left - center) * n, axis=1)
    dr = np.sum((right - center) * n, axis=1)
    count = max(int(math.ceil(arc[-1] / DS)) + 1, 2)
    s = np.linspace(0., arc[-1], count)
    xy = np.stack([np.interp(s, arc, center[:, 0]), np.interp(s, arc, center[:, 1])], axis=1)
    return s, xy, np.interp(s, arc, dl), np.interp(s, arc, dr)


def _round_corners(xy, half_window_m=2.0):
    """Moving average of the centreline (ends kept), so lane joints drawn as sharp corners become short
    curves. Keeps the sample spacing close to DS."""
    k = int(round(half_window_m / DS))
    if len(xy) <= 2 * k + 1:
        return xy
    pad = np.concatenate([np.repeat(xy[:1], k, 0), xy, np.repeat(xy[-1:], k, 0)])
    kernel = np.ones(2 * k + 1) / (2 * k + 1)
    out = np.stack([np.convolve(pad[:, 0], kernel, "valid"), np.convolve(pad[:, 1], kernel, "valid")], axis=1)
    out[0], out[-1] = xy[0], xy[-1]
    return out


def _smooth_tangent(xy, window=2):
    t = np.zeros_like(xy)
    n = len(xy)
    for k in range(n):
        a, b = max(0, k - window), min(n - 1, k + window)
        t[k] = xy[b] - xy[a]
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-9)
    return t


class Network:
    """Routes plus the lookups the simulation needs every step, all as dense arrays."""

    def __init__(self, lanes, routes=None, name="network", ground_y=0.0, view=None, entries=None):
        self.name = name
        self.ground_y = ground_y
        self.view = None if view is None else np.asarray(view, float)  # camera ground footprint polygon
        self.lanes = {lane.id: lane for lane in lanes}
        sequences = routes if routes is not None else self._enumerate(entries)
        if not sequences:
            raise ValueError("Network has no routes")
        self.routes = [self._build_route(i, seq) for i, seq in enumerate(sequences)]
        self.route_ids = {r.id: r.index for r in self.routes}
        self._stack()
        self._build_grid()
        self._build_conflicts()

    # ------------------------------------------------------------ construction

    def _enumerate(self, entries=None):
        outgoing, incoming = {}, {}
        for lane in self.lanes.values():
            outgoing.setdefault(lane.start, []).append(lane.id)
            incoming.setdefault(lane.end, []).append(lane.id)
        starts = entries or sorted({l.start for l in self.lanes.values()} - set(incoming))
        sequences = []

        def walk(node, path, seen):
            nexts = [l for l in outgoing.get(node, []) if self.lanes[l].end not in seen]
            if not nexts:
                if path:
                    sequences.append(tuple(path))
                return
            for lane_id in nexts:
                walk(self.lanes[lane_id].end, path + [lane_id], seen | {self.lanes[lane_id].end})

        for start in starts:
            walk(start, [], {start})
        return sequences

    def _build_route(self, index, seq):
        centers, lefts, rights, owner = [], [], [], []
        for k, lane_id in enumerate(seq):
            lane = self.lanes[lane_id]
            c, l, r = lane.center, lane.left, lane.right
            if centers:
                c, l, r = c[1:], l[1:], r[1:]   # the joint is shared with the previous lane
            centers.append(c)
            lefts.append(l)
            rights.append(r)
            owner.append(np.full(len(c), k))
        center, left, right = np.concatenate(centers), np.concatenate(lefts), np.concatenate(rights)
        owner = np.concatenate(owner)
        keep = np.concatenate([[True], np.linalg.norm(np.diff(center, axis=0), axis=1) > 1e-6])
        raw_arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(center[keep], axis=0), axis=1))])
        s, xy, dl, dr = _resample(center, left, right)
        # Round sharp joints, then re-sample so stations are true distances along the rounded line.
        xy = _round_corners(xy)
        arc = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        s_new = np.linspace(0., arc[-1], max(int(math.ceil(arc[-1] / DS)) + 1, 2))
        xy = np.stack([np.interp(s_new, arc, xy[:, 0]), np.interp(s_new, arc, xy[:, 1])], axis=1)
        dl, dr = np.interp(s_new, arc, dl), np.interp(s_new, arc, dr)
        to_new = lambda station: float(np.interp(station, s, arc))
        raw_arc = np.array([to_new(x) for x in raw_arc])
        s = s_new
        # A lane starts at the joint shared with the previous lane, exactly where it was drawn.
        lane_start = {seq[0]: 0.}
        kept = owner[keep]
        for joint in np.flatnonzero(np.diff(kept) != 0) + 1:
            lane_start[seq[kept[joint]]] = float(raw_arc[joint - 1])
        lane_of = np.zeros(len(s), dtype=int)
        for k, lane_id in enumerate(seq):
            lane_of[s >= lane_start[lane_id] - 1e-9] = k
        t = _smooth_tangent(xy)
        n = np.stack([-t[:, 1], t[:, 0]], axis=1)
        if np.mean(dl) < np.mean(dr):  # mirrored coordinates (Unity x, z): point the normal to the left edge
            n, dl, dr = -n, -dl, -dr
        return Route(index, "|".join(seq), tuple(seq), xy, s, t, n, dl, dr, lane_of, lane_start)

    def _stack(self):
        """Routes padded into (R, M) arrays so lookups for many vehicles are single gathers."""
        R, M = len(self.routes), max(len(r.s) for r in self.routes)
        self.count = np.array([len(r.s) for r in self.routes])
        self.length = np.array([r.length for r in self.routes])
        self.spacing = np.array([r.s[1] - r.s[0] for r in self.routes])   # just under DS
        self.XY, self.T, self.N = (np.zeros((R, M, 2)) for _ in range(3))
        self.S, self.DL, self.DR = (np.zeros((R, M)) for _ in range(3))
        for r in self.routes:
            k = len(r.s)
            pad = M - k
            self.XY[r.index] = np.pad(r.xy, ((0, pad), (0, 0)), mode="edge")
            self.T[r.index] = np.pad(r.tangent, ((0, pad), (0, 0)), mode="edge")
            self.N[r.index] = np.pad(r.normal, ((0, pad), (0, 0)), mode="edge")
            self.S[r.index] = np.concatenate([r.s, r.s[-1] + (r.s[1] - r.s[0]) * np.arange(1, pad + 1)])
            self.DL[r.index] = np.pad(r.d_left, (0, pad), mode="edge")
            self.DR[r.index] = np.pad(r.d_right, (0, pad), mode="edge")

    def _build_grid(self):
        pts = np.concatenate([r.xy for r in self.routes])
        reach = max(float(np.max(np.abs(np.concatenate([r.d_left, r.d_right])))) for r in self.routes) + GRID_BAND
        self.x0, self.y0 = pts.min(axis=0) - reach
        x1, y1 = pts.max(axis=0) + reach
        self.nx = int(math.ceil((x1 - self.x0) / GRID_CELL))
        self.ny = int(math.ceil((y1 - self.y0) / GRID_CELL))
        gx = self.x0 + (np.arange(self.nx) + .5) * GRID_CELL
        gy = self.y0 + (np.arange(self.ny) + .5) * GRID_CELL
        cells = np.stack(np.meshgrid(gx, gy, indexing="ij"), axis=-1).reshape(-1, 2)
        self.grid = np.full((len(self.routes), self.nx, self.ny), -1, dtype=np.int32)
        for r in self.routes:
            dist, idx = cKDTree(r.xy).query(cells)
            # Signed offset at the nearest sample, so the band follows each side's own edge.
            off = np.sum((cells - r.xy[idx]) * r.normal[idx], axis=1)
            edge = np.where(off >= 0, np.maximum(r.d_left[idx], 0), -np.minimum(r.d_right[idx], 0))
            ok = (np.abs(off) <= edge + GRID_BAND) & (dist <= edge + GRID_BAND + DS)
            self.grid[r.index] = np.where(ok, idx, -1).reshape(self.nx, self.ny)

    # ------------------------------------------------------------ lookups

    def project(self, route, p):
        """Project world points onto routes. route and p broadcast: route (...), p (..., 2).
        Returns s, d, valid, sample index."""
        route = np.asarray(route)
        shape = np.broadcast_shapes(route.shape, p.shape[:-1])
        p = np.broadcast_to(p, shape + (2,))
        cx = np.floor((p[..., 0] - self.x0) / GRID_CELL).astype(np.int64)
        cy = np.floor((p[..., 1] - self.y0) / GRID_CELL).astype(np.int64)
        inside = (cx >= 0) & (cx < self.nx) & (cy >= 0) & (cy < self.ny)
        cx, cy = np.where(inside, cx, 0), np.where(inside, cy, 0)
        route_b = np.broadcast_to(route, shape)
        idx = self.grid[route_b, cx, cy]
        valid = inside & (idx >= 0)
        idx = np.where(valid, idx, 0)
        dp = p - self.XY[route_b, idx]
        s = self.S[route_b, idx] + np.sum(dp * self.T[route_b, idx], axis=-1)
        d = np.sum(dp * self.N[route_b, idx], axis=-1)
        return s, d, valid, idx

    def sample_index(self, route, s):
        return np.clip(np.rint(s / self.spacing[route]).astype(np.int64), 0, self.count[route] - 1)

    def frame(self, route, s):
        """Centre point, tangent and normal at stations (interpolated)."""
        k = np.clip(s / self.spacing[route], 0, self.count[route] - 1.000001)
        k0 = np.floor(k).astype(np.int64)
        f = (k - k0)[..., None]
        xy = self.XY[route, k0] * (1 - f) + self.XY[route, k0 + 1] * f
        t = self.T[route, k0] * (1 - f) + self.T[route, k0 + 1] * f
        t /= np.maximum(np.linalg.norm(t, axis=-1, keepdims=True), 1e-9)
        n = np.stack([-t[..., 1], t[..., 0]], axis=-1)
        # Keep the stored normal's orientation (routes may have flipped normals).
        flip = np.sum(n * self.N[route, k0], axis=-1, keepdims=True) < 0
        n = np.where(flip, -n, n)
        # Extend past the end along the last tangent (vehicles leaving).
        over = np.maximum(s - self.length[route], 0)[..., None]
        under = np.minimum(s, 0)[..., None]
        xy = xy + t * (over + under)
        return xy, t, n

    def edges(self, route, s):
        """Signed (right, left) drivable offsets at stations."""
        k = self.sample_index(route, s)
        return self.DR[route, k], self.DL[route, k]

    def world(self, route, s, d):
        xy, t, n = self.frame(route, s)
        return xy + n * d[..., None], t, n

    # ------------------------------------------------------------ conflicts

    def _build_conflicts(self):
        R = len(self.routes)
        self.cf_valid = np.zeros((R, R, MAX_ZONES), bool)
        self.cf_kind = np.zeros((R, R, MAX_ZONES), np.int8)
        self.cf_s = np.zeros((R, R, MAX_ZONES))            # station of the conflict point on the row route
        self.cf_t = np.zeros((R, R, MAX_ZONES, 2))         # row route tangent there
        self.cf_n = np.zeros((R, R, MAX_ZONES, 2))
        self.cf_sin = np.ones((R, R, MAX_ZONES))
        self.cf_lo = np.zeros((R, R, MAX_ZONES))           # the row route's stretch whose width overlaps
        self.cf_hi = np.zeros((R, R, MAX_ZONES))           # the other route's carriageway, around this point
        for a in self.routes:
            for b in self.routes:
                if b.index <= a.index:
                    continue
                # Routes from the same entry lane are one stream that splits: vehicles in it are ordered by
                # their place in the queue and see each other directly, so no crossing points between them.
                if a.lanes[0] == b.lanes[0]:
                    continue
                runs_a, runs_b = self._overlap_runs(a, b), self._overlap_runs(b, a)
                zones = self._merge(a, b) + self._crossings(a, b, runs_a)
                zones.sort(key=lambda z: z[0])
                for z, (sa, sb, kind, sin) in enumerate(zones[:MAX_ZONES]):
                    for row, col, s_row, s_col, runs in ((a, b, sa, sb, runs_a), (b, a, sb, sa, runs_b)):
                        _, t, n = self.frame(np.array(row.index), np.array(s_row))
                        lo, hi = self._stretch(row, col, s_row, kind, sin, runs)
                        self.cf_valid[row.index, col.index, z] = True
                        self.cf_kind[row.index, col.index, z] = kind
                        self.cf_s[row.index, col.index, z] = s_row
                        self.cf_t[row.index, col.index, z] = t
                        self.cf_n[row.index, col.index, z] = n
                        self.cf_sin[row.index, col.index, z] = sin
                        self.cf_lo[row.index, col.index, z], self.cf_hi[row.index, col.index, z] = lo, hi
        self._cluster_conflicts()

    def _overlap_runs(self, a, b):
        """Stretches of route a (station intervals) where any part of a's width lies on b's carriageway,
        crossing it at an angle. Shared lanes and the few metres where routes split or join are left out."""
        shared = set(a.lanes) & set(b.lanes)
        mine = np.array([a.lanes[k] not in shared for k in a.lane_of])
        if not mine.any():
            return []
        theirs_ok = np.array([b.lanes[k] not in shared for k in range(len(b.lanes))])
        probes = np.linspace(0, 1, 7)
        offsets = a.d_right[:, None] + (a.d_left - a.d_right)[:, None] * probes[None, :]
        pts = a.xy[:, None, :] + a.normal[:, None, :] * offsets[..., None]
        sb, d, valid, idx = self.project(np.full(pts.shape[:2], b.index), pts)
        valid &= mine[:, None] & theirs_ok[b.lane_of[idx]]
        on_b = valid & (d >= b.d_right[idx] - 0.2) & (d <= b.d_left[idx] + 0.2)
        cos = np.abs(np.sum(a.tangent[:, None, :] * b.tangent[idx], axis=-1))
        hit = (on_b & (cos < 1 - 0.5 * CROSS_MIN_SIN ** 2)).any(axis=1)
        for route in (a, b):
            for lane in route.lanes:
                if lane in shared:
                    lo, hi = route.lane_start[lane] - 3., route.lane_start[lane] + self._lane_length(lane) + 3.
                    if route is a:
                        hit &= ~((a.s >= lo) & (a.s <= hi))
        runs, start = [], None
        for k, h in enumerate(hit):
            if h and start is None:
                start = k
            if (not h or k == len(hit) - 1) and start is not None:
                end = k if h else k - 1
                runs.append((float(a.s[start]), float(a.s[end])))
                start = None
        return runs

    def _stretch(self, row, col, s_row, kind, sin, runs):
        for lo, hi in runs:
            if lo - 1.0 <= s_row <= hi + 1.0:
                return lo, hi
        half = (np.mean(col.d_left - col.d_right) / 2) / max(sin, CROSS_MIN_SIN)
        half = min(half, 8.0)
        return (s_row - half, s_row) if kind == MERGE else (s_row - half, s_row + half)

    def _cluster_conflicts(self):
        """Overlapping or nearly touching stretches along a route form one junction area. cf_entry and
        cf_exit are where the area holding each conflict point starts and ends on the row route."""
        self.cf_entry, self.cf_exit = self.cf_lo.copy(), self.cf_hi.copy()
        for a in range(len(self.routes)):
            mask = self.cf_valid[a]
            spans = sorted(zip(self.cf_lo[a][mask], self.cf_hi[a][mask]))
            areas = []
            for lo, hi in spans:
                if areas and lo <= areas[-1][1] + CLUSTER_GAP:
                    areas[-1][1] = max(areas[-1][1], hi)
                else:
                    areas.append([lo, hi])
            for lo, hi in areas:
                inside = mask & (self.cf_lo[a] >= lo - 1e-9) & (self.cf_hi[a] <= hi + 1e-9)
                self.cf_entry[a][inside], self.cf_exit[a][inside] = lo, hi

    def _merge(self, a, b):
        shared = [l for l in a.lanes if l in b.lanes]
        for lane in shared:
            ia, ib = a.lanes.index(lane), b.lanes.index(lane)
            if ia > 0 and ib > 0 and a.lanes[ia - 1] != b.lanes[ib - 1]:
                sa, sb = a.lane_start[lane], b.lane_start[lane]
                _, ta, _ = self.frame(np.array(a.index), np.array(max(sa - 3, 0)))
                _, tb, _ = self.frame(np.array(b.index), np.array(max(sb - 3, 0)))
                sin = abs(ta[0] * tb[1] - ta[1] * tb[0])
                return [(sa, sb, MERGE, max(sin, CROSS_MIN_SIN))]
            if ia == 0 or ib == 0:
                break
        return []

    def _crossings(self, a, b, runs):
        shared = set(a.lanes) & set(b.lanes)
        mine = np.array([a.lanes[k] not in shared for k in a.lane_of])
        theirs_ok = np.array([b.lanes[k] not in shared for k in range(len(b.lanes))])
        if not mine.any():
            return []
        # Where routes split or join, their centrelines touch; skip a few metres around those nodes.
        skip = []
        for route, other in ((a, b), (b, a)):
            for k, lane in enumerate(route.lanes):
                if lane in shared:
                    end = route.lane_start[lane] + self._lane_length(lane)
                    skip.append((route.index, route.lane_start[lane] - 3., end + 3.))
        sb, d, valid, idx = self.project(np.full(len(a.s), b.index), a.xy)
        valid &= mine & theirs_ok[b.lane_of[idx]]
        for ri, lo, hi in skip:
            if ri == a.index:
                valid &= ~((a.s >= lo) & (a.s <= hi))
            else:
                valid &= ~((sb >= lo) & (sb <= hi))
        zones = []
        for k in range(len(a.s) - 1):
            if not (valid[k] and valid[k + 1]) or abs(sb[k + 1] - sb[k]) > 3 * DS:
                continue
            if d[k] == 0 or np.sign(d[k]) != np.sign(d[k + 1]):
                f = d[k] / (d[k] - d[k + 1]) if d[k] != d[k + 1] else 0.
                sa_x = a.s[k] + f * (a.s[k + 1] - a.s[k])
                sb_x = sb[k] + f * (sb[k + 1] - sb[k])
                _, ta, _ = self.frame(np.array(a.index), np.array(sa_x))
                _, tb, _ = self.frame(np.array(b.index), np.array(sb_x))
                sin = abs(ta[0] * tb[1] - ta[1] * tb[0])
                if sin < CROSS_MIN_SIN:
                    continue
                if zones and abs(zones[-1][0] - sa_x) < 5.:
                    continue
                zones.append((float(sa_x), float(sb_x), CROSS, float(sin)))
        # Carriageways that overlap at an angle without their centrelines crossing still conflict: use
        # the middle of the overlap.
        for lo, hi in runs:
            if any(lo - 1.0 <= z[0] <= hi + 1.0 for z in zones):
                continue
            mid = (lo + hi) / 2
            k = int(np.clip(np.searchsorted(a.s, mid), 0, len(a.s) - 1))
            s_b, _, ok, _ = self.project(np.array(b.index), a.xy[k])
            if not ok:
                continue
            _, ta, _ = self.frame(np.array(a.index), np.array(mid))
            _, tb, _ = self.frame(np.array(b.index), np.array(float(s_b)))
            sin = abs(ta[0] * tb[1] - ta[1] * tb[0])
            zones.append((float(mid), float(s_b), CROSS, float(max(sin, CROSS_MIN_SIN))))
        return zones

    def _lane_length(self, lane_id):
        c = self.lanes[lane_id].center
        return float(np.sum(np.linalg.norm(np.diff(c, axis=0), axis=1)))

    def conflicts(self):
        """Readable list of conflict points: (route a, route b, station on a, station on b, kind)."""
        out = []
        for a in range(len(self.routes)):
            for b in range(len(self.routes)):
                for z in range(MAX_ZONES):
                    if a < b and self.cf_valid[a, b, z]:
                        out.append((self.routes[a].id, self.routes[b].id, round(float(self.cf_s[a, b, z]), 2),
                                    round(float(self.cf_s[b, a, z]), 2), "merge" if self.cf_kind[a, b, z] else "cross"))
        return out

    # ------------------------------------------------------------ view

    def in_view(self, xy):
        """True for points inside the camera footprint (always True without one)."""
        xy = np.asarray(xy, float)
        if self.view is None:
            return np.ones(xy.shape[:-1], bool)
        from matplotlib.path import Path as MplPath
        return MplPath(self.view).contains_points(xy.reshape(-1, 2)).reshape(xy.shape[:-1])

    # ------------------------------------------------------------ loading

    @classmethod
    def from_layout(cls, path):
        """A network from a Unity camera-lane layout (simulation/Assets/StreamingAssets/Traffic/*.camera-lanes.json)."""
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        nodes = {n["id"]: np.array([n["world"][0], n["world"][2]]) for n in data["nodes"]}

        def nearest(p):
            return min(nodes, key=lambda k: float(np.sum((nodes[k] - p) ** 2)))

        lanes = []
        for lane in data["lanes"]:
            rows = lane.get("samples") or lane["sections"]
            center = np.array([[r["center"][0], r["center"][2]] for r in rows])
            left = np.array([[r["left"][0], r["left"][2]] for r in rows])
            right = np.array([[r["right"][0], r["right"][2]] for r in rows])
            if lane.get("direction", "Forward") == "Backward":
                center, left, right = center[::-1], right[::-1], left[::-1]
            start, end = nearest(center[0]), nearest(center[-1])
            lanes.append(Lane(lane["id"], start, end, center, left, right))
        entries = [n["id"] for n in data["nodes"] if n.get("spawn_point")] or None
        view = None
        if data.get("world_points"):
            from scipy.spatial import ConvexHull
            pts = np.array([[p[0], p[2]] for p in data["world_points"]])
            view = pts[ConvexHull(pts).vertices]
        return cls(lanes, name=data.get("scene", Path(path).stem), ground_y=float(data.get("ground_y", 0.)),
                   view=view, entries=entries)


def lane_from_polyline(lane_id, start, end, points, half_left, half_right=None):
    """A lane from a centreline polyline with a constant half width on each side (left of travel = +normal)."""
    half_right = half_left if half_right is None else half_right
    points = np.asarray(points, float)
    t = _smooth_tangent(points, window=1)
    n = np.stack([-t[:, 1], t[:, 0]], axis=1)
    return Lane(lane_id, start, end, points, points + n * half_left, points - n * half_right)
