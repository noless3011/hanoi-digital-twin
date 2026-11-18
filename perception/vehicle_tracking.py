"""Detector-independent object IDs and bounded trails from actual TrackTrack.

The tracker advances once per processed image. Its frame counts and trails are
diagnostics, not source timestamps, calibrated speeds, or tracking accuracy.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from numbers import Integral
import os
from pathlib import Path

import numpy as np


CONFIG = Path(__file__).resolve().with_name("tracktrack.yaml")
os.environ.setdefault("YOLO_CONFIG_DIR", str(CONFIG.parent / ".ultralytics"))


@dataclass(frozen=True)
class TrackView:
    track_id: int
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]
    trail: tuple[tuple[float, float], ...]
    hits: int
    missed: int
    state: str


@dataclass(frozen=True)
class TrackingSnapshot:
    tracks: list[TrackView]
    detection_ids: tuple[int | None, ...]
    visible: int
    lost: int
    new: int
    recovered: int
    total_ids: int


@dataclass
class _History:
    display_id: int
    trail: deque
    hits: int = 0
    box: tuple = field(default_factory=tuple)
    was_lost: bool = False


class _Detections:
    """Small NumPy Results adapter expected by Ultralytics tracker.update()."""

    def __init__(self, boxes, classes, confidence):
        self.xywh = np.empty_like(boxes, dtype=np.float32)
        self.xywh[:, :2] = (boxes[:, :2] + boxes[:, 2:]) / 2
        self.xywh[:, 2:] = boxes[:, 2:] - boxes[:, :2]
        self.cls = classes
        self.conf = confidence


class VehicleTracker:
    """Maintain TrackTrack state independently of either detector's API.

    NEW includes tentative tracks while TrackTrack confirms repeated matches.
    RECOVERED denotes one update after a lost track regains its observation.
    LOST retains only its last observed box/trail, without drawing new motion.
    Session IDs remain increasing when image dimensions change; that change
    retires existing histories and restarts association in the new coordinates.
    """

    def __init__(self, trail_length=40, max_missed=30):
        if isinstance(trail_length, bool) or not isinstance(trail_length, Integral) or trail_length < 1:
            raise ValueError("trail_length must be a positive integer")
        if isinstance(max_missed, bool) or not isinstance(max_missed, Integral) or max_missed < 0:
            raise ValueError("max_missed must be a nonnegative integer")
        Path(os.environ["YOLO_CONFIG_DIR"]).mkdir(parents=True, exist_ok=True)
        # Ultralytics otherwise tries installing optional tracker requirements
        # during import. Dependencies belong in requirements/setup instead.
        os.environ["YOLO_AUTOINSTALL"] = "false"
        from ultralytics.trackers.basetrack import TrackState
        from ultralytics.trackers.track_tracker import TRACKTRACK
        from ultralytics.utils import IterableSimpleNamespace, YAML

        self._states = TrackState
        self._backend_class = TRACKTRACK
        settings = YAML.load(str(CONFIG))
        settings.update(track_buffer=int(max_missed), with_reid=False, gmc_method="none")
        self._settings = IterableSimpleNamespace(**settings)
        self.low_confidence_threshold = float(self._settings.track_low_thresh)
        self.trail_length = int(trail_length)
        self.max_missed = int(max_missed)
        self._backend = self._backend_class(self._settings)
        self._histories: dict[int, _History] = {}
        self._dimensions = None
        self._total_ids = 0
        self._recovered_events = 0
        self._retired_tracks = 0
        self._resets = 0
        self._frames = 0
        self._max_visible = 0
        self._snapshot = TrackingSnapshot([], (), 0, 0, 0, 0, 0)

    @staticmethod
    def _validate(xyxy, class_id, confidence, frame):
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3 or min(frame.shape[:2]) < 1:
            raise ValueError("frame must be a nonempty H x W x 3 image")
        boxes = np.asarray(xyxy, dtype=np.float64)
        classes = np.asarray(class_id, dtype=np.float64)
        scores = np.asarray(confidence, dtype=np.float64)
        # Empty Python lists are convenient for advancing missing observations.
        if boxes.shape == (0,):
            boxes = boxes.reshape(0, 4)
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("xyxy must have shape (N, 4)")
        if classes.shape != (len(boxes),) or scores.shape != (len(boxes),):
            raise ValueError("class_id and confidence must have shape (N,) matching xyxy")
        if not np.isfinite(boxes).all() or not np.isfinite(classes).all() or not np.isfinite(scores).all():
            raise ValueError("detections must contain finite boxes, classes, and confidence")
        if (classes < 0).any() or (classes != np.floor(classes)).any() or (classes > np.iinfo(np.int32).max).any():
            raise ValueError("class_id must contain nonnegative integer indices")
        if ((scores < 0) | (scores > 1)).any():
            raise ValueError("confidence must be between 0 and 1")
        if (boxes[:, 2:] < boxes[:, :2]).any():
            raise ValueError("xyxy maximum coordinates must not precede minimum coordinates")
        boxes = boxes.copy()
        height, width = frame.shape[:2]
        boxes[:, (0, 2)] = boxes[:, (0, 2)].clip(0, width - 1)
        boxes[:, (1, 3)] = boxes[:, (1, 3)].clip(0, height - 1)
        valid = (boxes[:, 2:] > boxes[:, :2]).all(axis=1)
        indices = np.flatnonzero(valid)
        return boxes[valid].astype(np.float32), classes[valid].astype(np.int32), scores[valid].astype(np.float32), indices, len(boxes)

    def update(self, xyxy, class_id, confidence, frame):
        boxes, classes, scores, original_indices, count = self._validate(xyxy, class_id, confidence, frame)
        dimensions = frame.shape[:2]
        if self._dimensions is not None and dimensions != self._dimensions:
            self._retired_tracks += len(self._histories)
            self._histories.clear()
            # Construct a fresh backend without reset(): its reset also changes
            # the package-global counter used by any other tracker instances.
            self._backend = self._backend_class(self._settings)
            self._resets += 1
        self._dimensions = dimensions
        self._backend.update(_Detections(boxes, classes, scores), img=frame)
        self._frames += 1
        frame_id = self._backend.frame_id

        # The backend can leave newly removed tracks in lost_stracks for one
        # update. Exclude them immediately and enforce the configured lifetime.
        current = [track for track in self._backend.tracked_stracks
                   if track.state in (self._states.New, self._states.Tracked)]
        lost = []
        for track in self._backend.lost_stracks:
            if track.state == self._states.Lost and frame_id - track.frame_id <= self.max_missed:
                lost.append(track)
            elif track.state == self._states.Lost:
                track.mark_removed()
        self._backend.lost_stracks = lost
        live = {int(track.track_id): track for track in current + lost}
        for backend_id in set(self._histories) - set(live):
            self._retired_tracks += 1
            del self._histories[backend_id]

        detection_ids = [None] * count
        views = []
        new_count = recovered_count = visible_count = lost_count = 0
        for backend_id, track in live.items():
            history = self._histories.get(backend_id)
            if history is None:
                self._total_ids += 1
                history = _History(self._total_ids, deque(maxlen=self.trail_length))
                self._histories[backend_id] = history
                new_count += 1
            missed = int(frame_id - track.frame_id)
            if track.state == self._states.Lost:
                state = "LOST"
                history.was_lost = True
                lost_count += 1
            else:
                visible_count += 1
                recovered = history.was_lost
                history.was_lost = False
                state = "RECOVERED" if recovered else "NEW" if track.state == self._states.New or history.hits == 0 else "TRACKED"
                if recovered:
                    recovered_count += 1
                    self._recovered_events += 1
                history.hits += 1
                index = int(track.idx)
                observation = boxes[index] if 0 <= index < len(boxes) else track.xyxy
                history.box = tuple(float(value) for value in observation)
                x1, y1, x2, y2 = history.box
                history.trail.append(((x1 + x2) / 2, y2))
                if 0 <= index < len(original_indices):
                    detection_ids[int(original_indices[index])] = history.display_id
            if history.box:
                views.append(TrackView(history.display_id, int(track.cls), float(track.score), history.box,
                                       tuple(history.trail), history.hits, missed, state))
        views.sort(key=lambda track: track.track_id)
        self._max_visible = max(self._max_visible, visible_count)
        self._snapshot = TrackingSnapshot(views, tuple(detection_ids), visible_count, lost_count,
                                          new_count, recovered_count, self._total_ids)
        return self._snapshot

    def summary(self):
        """JSON-safe diagnostics for the complete session, including resets."""
        return {"tracker": "TrackTrack", "implementation": "ultralytics",
                "with_reid": False, "gmc_method": "none", "frames": self._frames,
                "unique_ids": self._total_ids, "recovered_events": self._recovered_events,
                "retired_tracks": self._retired_tracks, "dimension_resets": self._resets,
                "visible_tracks": self._snapshot.visible, "lost_tracks": self._snapshot.lost,
                "max_visible_tracks": self._max_visible, "trail_length": self.trail_length,
                "track_buffer": self.max_missed, "low_confidence_threshold": self.low_confidence_threshold,
                "high_confidence_threshold": float(self._settings.track_high_thresh),
                "new_track_threshold": float(self._settings.new_track_thresh),
                "time_basis": "processed frames"}
