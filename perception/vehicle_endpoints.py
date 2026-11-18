"""Collect the first and last real observations of each vehicle identity.

With keep_paths=True every real observation is also kept as a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Integral


VEHICLE_KINDS = {
    "car": "car", "bus": "bus", "truck": "truck", "van": "van",
    "motorcycle": "motorcycle", "motorbike": "motorcycle", "motor": "motorcycle",
}


class Calibration:
    """Optional normalized-image to Unity X/Z planar calibration.

    image_points and world_points contain >=4 corresponding ground landmarks.
    End nodes are explicit authored terminals, never inferred lane junctions.
    """

    def __init__(self, data):
        import numpy as np
        image = np.asarray(data["image_points"], dtype=float)
        world = np.asarray(data["world_points"], dtype=float)
        if (image.ndim != 2 or image.shape[1] != 2 or len(image) < 4
                or world.shape != (len(image), 3) or not np.isfinite(image).all()
                or not np.isfinite(world).all() or ((image < 0) | (image > 1)).any()):
            raise ValueError("Calibration needs >=4 normalized image_points and corresponding Unity [x,y,z] world_points")
        self.ground_y = float(data.get("ground_y", world[:, 1].mean()))
        if not math.isfinite(self.ground_y):
            raise ValueError("ground_y must be finite")
        rows, values = [], []
        for (u, v), (x, _, z) in zip(image, world):
            rows.extend(([u, v, 1, 0, 0, 0, -x*u, -x*v],
                         [0, 0, 0, u, v, 1, -z*u, -z*v]))
            values.extend((x, z))
        coefficients, _, rank, _ = np.linalg.lstsq(rows, values, rcond=None)
        if rank != 8:
            raise ValueError("Calibration landmarks must span a ground plane (not a line)")
        self.matrix = np.append(coefficients, 1).reshape(3, 3)
        self.end_nodes = data.get("end_nodes", [])
        for node in self.end_nodes:
            position = node.get("position", [])
            radius = node.get("radius_m", 3)
            if (not isinstance(node.get("id"), str) or not node["id"].strip()
                    or len(position) != 3 or not all(math.isfinite(x) for x in position)
                    or not math.isfinite(radius) or radius <= 0):
                raise ValueError("end_nodes need id, finite [x,y,z] position, and positive radius_m")

    def project(self, point):
        x, z, divisor = self.matrix @ [*point, 1]
        if not math.isfinite(divisor) or abs(divisor) < 1e-9:
            raise ValueError("Endpoint projects onto the calibration horizon")
        world = [float(x / divisor), self.ground_y, float(z / divisor)]
        if not all(math.isfinite(value) for value in world):
            raise ValueError("Endpoint calibration produced a non-finite point")
        return world

    def end_node(self, position):
        candidates = [(math.hypot(position[0] - node["position"][0], position[2] - node["position"][2]), node)
                      for node in self.end_nodes]
        candidates = [(distance, node) for distance, node in candidates
                      if distance <= node.get("radius_m", 3)]
        return min(candidates, key=lambda pair: pair[0])[1]["id"] if candidates else None


@dataclass
class _Endpoints:
    kind: str
    spawn: dict
    last: dict
    border: bool
    hits: int = 1
    path: list = field(default_factory=list)


class EndpointCollector:
    """Memory is O(vehicles) with two points per identity, or O(observations) with keep_paths.

    LOST predictions never replace an observation. Retirement after the tracker
    grace period confirms disappearance at the last *observed* video timestamp.
    Active/pending tracks at EOF are censored as clip_end, not fabricated exits.
    """

    def __init__(self, width, height, *, min_hits=3, edge_margin=.04, calibration=None, keep_paths=False):
        if (any(isinstance(value, bool) or not isinstance(value, Integral)
                for value in (width, height, min_hits))
                or width < 2 or height < 2 or min_hits < 1):
            raise ValueError("Frame dimensions must be integers >=2 and min_hits a positive integer")
        if not math.isfinite(edge_margin) or not 0 <= edge_margin < .5:
            raise ValueError("edge_margin must be finite in [0, .5)")
        self.width, self.height = width, height
        self.min_hits, self.edge_margin = min_hits, edge_margin
        self.calibration = calibration
        self.keep_paths = keep_paths
        self.active = {}
        self.vehicles = []
        self.trajectories = []
        self.rejected_tracks = 0
        self._last_time = -1
        self._last_frame = -1

    def _point(self, track, frame, time_s):
        x1, y1, x2, y2 = track.box
        point = [min(1., max(0., (x1 + x2) / 2 / self.width)),
                 min(1., max(0., y2 / self.height))]
        result = {"time_s": round(time_s, 6), "frame": frame, "point": point}
        if self.calibration:
            result["world_position"] = self.calibration.project(point)
        margin = self.edge_margin
        border = (x1 / self.width <= margin or y1 / self.height <= margin
                  or x2 / self.width >= 1-margin or y2 / self.height >= 1-margin)
        return result, border

    def update(self, snapshot, names, *, frame, time_s):
        if (isinstance(frame, bool) or not isinstance(frame, Integral)
                or not math.isfinite(time_s) or time_s < 0 or time_s < self._last_time or frame <= self._last_frame):
            raise ValueError("Observations need increasing frames and nonnegative monotonic video times")
        self._last_time, self._last_frame = time_s, frame
        present = {track.track_id for track in snapshot.tracks}
        for track_id in list(self.active):
            if track_id not in present:
                self._finish(track_id, censored=False)
        for track in snapshot.tracks:
            if track.missed or track.state == "LOST":
                continue
            kind = VEHICLE_KINDS.get(str(names.get(track.class_id, "")).lower())
            if kind is None:
                continue
            endpoint, border = self._point(track, frame, time_s)
            history = self.active.get(track.track_id)
            if history is None:
                history = self.active[track.track_id] = _Endpoints(kind, endpoint, endpoint, border)
            else:
                history.last, history.border = endpoint, border
                history.hits += 1
            if self.keep_paths:
                history.path.append(self._observation(track, endpoint))

    def _observation(self, track, endpoint):
        x1, y1, x2, y2 = track.box
        observation = dict(endpoint)
        observation["box"] = [round(x1 / self.width, 6), round(y1 / self.height, 6),
                              round(x2 / self.width, 6), round(y2 / self.height, 6)]
        confidence = getattr(track, "confidence", None)
        if confidence is not None:
            observation["confidence"] = round(float(confidence), 4)
        return observation

    def _finish(self, track_id, *, censored):
        track = self.active.pop(track_id)
        if track.hits < self.min_hits:
            self.rejected_tracks += 1
            return
        last = dict(track.last)
        last["reason"] = "clip_end" if censored else "out_of_view" if track.border else "tracking_lost"
        if not censored and self.calibration:
            node_id = self.calibration.end_node(last["world_position"])
            if node_id:
                last.update(reason="end_node", node_id=node_id)
        self.vehicles.append({"track_id": track_id, "kind": track.kind,
                              "spawn": track.spawn, "disappearance": last})
        if self.keep_paths:
            self.trajectories.append({"track_id": track_id, "kind": track.kind,
                                      "termination": last["reason"], "observations": track.path})

    def finish(self):
        for track_id in list(self.active):
            self._finish(track_id, censored=True)
        self.trajectories.sort(key=lambda path: (path["observations"][0]["time_s"], path["track_id"]))
        return sorted(self.vehicles, key=lambda vehicle: (vehicle["spawn"]["time_s"], vehicle["track_id"]))
