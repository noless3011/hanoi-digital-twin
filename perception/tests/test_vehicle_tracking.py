"""Exercise actual TrackTrack association without loading a detector or GPU."""

import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parents[1] / ".ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from vehicle_tracking import VehicleTracker


class VehicleTrackingTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((160, 320, 3), dtype=np.uint8)

    def update(self, tracker, x=30, *, score=.9):
        return tracker.update([[x, 50, x + 50, 100]], [0], [score], self.frame)

    def confirm(self, tracker):
        return [self.update(tracker, x=30 + step) for step in range(4)]

    def test_stable_identity_trail_and_actual_backend_with_no_appearance_model(self):
        tracker = VehicleTracker(trail_length=3)
        snapshots = self.confirm(tracker)
        snapshots.extend(self.update(tracker, x=35 + step) for step in range(5))
        ids = [snapshot.detection_ids[0] for snapshot in snapshots]
        self.assertEqual(ids, [1] * len(snapshots))
        self.assertEqual(snapshots[0].tracks[0].state, "NEW")
        self.assertEqual(snapshots[-1].tracks[0].state, "TRACKED")
        self.assertEqual(snapshots[-1].tracks[0].hits, 9)
        self.assertEqual(len(snapshots[-1].tracks[0].trail), 3)
        self.assertEqual(snapshots[-1].total_ids, 1)
        self.assertEqual(tracker._backend.__class__.__name__, "TRACKTRACK")
        self.assertIsNone(tracker._backend.encoder)
        self.assertIsNone(tracker._backend.gmc.method)
        self.assertEqual(tracker.summary()["unique_ids"], 1)
        json.dumps(tracker.summary())

    def test_empty_frames_mark_lost_without_drawing_motion_then_recover_same_id(self):
        tracker = VehicleTracker(max_missed=3)
        before = self.confirm(tracker)[-1].tracks[0]
        missing = tracker.update([], [], [], self.frame)
        self.assertEqual((missing.visible, missing.lost, missing.new, missing.recovered), (0, 1, 0, 0))
        lost = missing.tracks[0]
        self.assertEqual((lost.track_id, lost.state, lost.hits, lost.missed), (1, "LOST", 4, 1))
        self.assertEqual(lost.box, before.box)
        self.assertEqual(lost.trail, before.trail)
        recovered = self.update(tracker, x=35)
        self.assertEqual(recovered.detection_ids, (1,))
        self.assertEqual((recovered.visible, recovered.lost, recovered.recovered), (1, 0, 1))
        self.assertEqual((recovered.tracks[0].state, recovered.tracks[0].hits), ("RECOVERED", 5))
        self.assertEqual(self.update(tracker, x=36).tracks[0].state, "TRACKED")
        self.assertEqual(tracker.summary()["recovered_events"], 1)

    def test_expired_track_is_retired_and_history_pruned_then_gets_new_identity(self):
        tracker = VehicleTracker(trail_length=4, max_missed=2)
        self.confirm(tracker)
        for _ in range(2):
            self.assertEqual(tracker.update([], [], [], self.frame).lost, 1)
        expired = tracker.update([], [], [], self.frame)
        self.assertEqual(expired.tracks, [])
        self.assertEqual(expired.lost, 0)
        self.assertEqual(tracker._histories, {})
        self.assertEqual(tracker.summary()["retired_tracks"], 1)
        fresh = self.update(tracker, x=35)
        self.assertEqual((fresh.detection_ids, fresh.total_ids, fresh.new), ((2,), 2, 1))

    def test_low_confidence_keeps_existing_track_and_does_not_start_new_track(self):
        tracker = VehicleTracker()
        self.confirm(tracker)
        low = self.update(tracker, x=34, score=.25)
        self.assertEqual((low.detection_ids, low.new, low.total_ids), ((1,), 0, 1))
        self.assertAlmostEqual(low.tracks[0].confidence, .25)
        unassigned = tracker.update([[200, 50, 250, 100]], [1], [.25], self.frame)
        self.assertEqual(unassigned.detection_ids, (None,))
        self.assertEqual(unassigned.total_ids, 1)

    def test_two_vehicles_keep_ids_when_detection_order_changes(self):
        tracker = VehicleTracker()
        first = tracker.update([[30, 50, 80, 100], [200, 50, 250, 100]], [0, 1], [.9, .95], self.frame)
        self.assertEqual(first.detection_ids, (1, 2))
        for step in range(1, 6):
            next_snapshot = tracker.update([[200 - step, 50, 250 - step, 100], [30 + step, 50, 80 + step, 100]],
                                           [1, 0], [.95, .9], self.frame)
            self.assertEqual(next_snapshot.detection_ids, (2, 1))
        self.assertEqual(next_snapshot.total_ids, 2)

    def test_new_traffic_classes_keep_labels_and_ids_when_order_changes(self):
        tracker = VehicleTracker()
        boxes = [[20, 50, 50, 120], [120, 50, 170, 100], [250, 30, 270, 60]]
        first = tracker.update(boxes, [4, 5, 6], [.9, .9, .9], self.frame)
        self.assertEqual(first.detection_ids, (1, 2, 3))
        for _ in range(4):
            snapshot = tracker.update(boxes[::-1], [6, 5, 4], [.9, .9, .9], self.frame)
            self.assertEqual(snapshot.detection_ids, (3, 2, 1))
        self.assertEqual({track.track_id: track.class_id for track in snapshot.tracks},
                         {1: 4, 2: 5, 3: 6})
        self.assertEqual((snapshot.visible, snapshot.total_ids), (3, 3))

    def test_frame_resize_retires_old_coordinate_history_and_preserves_session_ids(self):
        tracker = VehicleTracker()
        self.confirm(tracker)
        resized = np.zeros((320, 640, 3), dtype=np.uint8)
        snapshot = tracker.update([[60, 100, 160, 200]], [0], [.9], resized)
        self.assertEqual(snapshot.detection_ids, (2,))
        self.assertEqual(snapshot.total_ids, 2)
        self.assertEqual(len(snapshot.tracks[0].trail), 1)
        self.assertEqual(snapshot.tracks[0].hits, 1)
        summary = tracker.summary()
        self.assertEqual((summary["dimension_resets"], summary["retired_tracks"], summary["frames"]), (1, 1, 5))

    def test_independent_tracker_instances_and_resizes_do_not_reuse_active_backend_ids(self):
        first = VehicleTracker()
        self.confirm(first)
        second = VehicleTracker()
        self.confirm(second)
        second.update([[60, 100, 160, 200]], [0], [.9], np.zeros((320, 640, 3), dtype=np.uint8))
        self.assertEqual(self.update(first, x=34).detection_ids, (1,))
        two_vehicles = first.update([[35, 50, 85, 100], [200, 50, 250, 100]], [0, 1], [.9, .95], self.frame)
        self.assertEqual(two_vehicles.detection_ids, (1, 2))
        self.assertEqual(two_vehicles.total_ids, 2)
        self.assertEqual(second.summary()["unique_ids"], 2)

    def test_clipping_preserves_original_detection_indices_and_never_mutates_inputs(self):
        tracker = VehicleTracker()
        boxes = np.array([[-5, 50, 45, 200], [400, 50, 450, 100], [80, 50, 80, 100]], dtype=float)
        original = boxes.copy()
        snapshot = tracker.update(boxes, [0, 0, 0], [.9, .9, .9], self.frame)
        self.assertEqual(snapshot.detection_ids, (1, None, None))
        self.assertEqual(snapshot.tracks[0].box, (0, 50, 45, 159))
        np.testing.assert_array_equal(boxes, original)
        self.assertFalse(self.frame.any())

    def test_zero_buffer_retires_confirmed_track_on_first_missing_update(self):
        tracker = VehicleTracker(max_missed=0)
        self.confirm(tracker)
        self.assertEqual(tracker.update([], [], [], self.frame).tracks, [])
        self.assertEqual(tracker.summary()["retired_tracks"], 1)

    def test_invalid_detections_do_not_advance_or_corrupt_tracking_state(self):
        tracker = VehicleTracker()
        self.confirm(tracker)
        cases = [
            ([[30, 50, float("nan"), 100]], [0], [.9]),
            ([[30, 50, 80, 100]], [float("inf")], [.9]),
            ([[30, 50, 80, 100]], [0], [float("nan")]),
            ([[80, 50, 30, 100]], [0], [.9]),
            ([[30, 50, 80, 100]], [-1], [.9]),
            ([[30, 50, 80, 100]], [.5], [.9]),
            ([[30, 50, 80, 100]], [0], [1.1]),
            ([[30, 50, 80, 100]], [0], [-.1]),
            ([[30, 50, 80]], [0], [.9]),
            ([[30, 50, 80, 100]], [], [.9]),
        ]
        for boxes, classes, scores in cases:
            with self.subTest(boxes=boxes, classes=classes, scores=scores), self.assertRaises(ValueError):
                tracker.update(boxes, classes, scores, self.frame)
        self.assertEqual(tracker.summary()["frames"], 4)
        self.assertEqual(self.update(tracker, x=34).detection_ids, (1,))

    def test_invalid_frame_and_constructor_parameters(self):
        tracker = VehicleTracker()
        for frame in (None, np.zeros((0, 320, 3)), np.zeros((160, 320)), np.zeros((160, 320, 4))):
            with self.subTest(shape=getattr(frame, "shape", None)), self.assertRaises(ValueError):
                tracker.update([], [], [], frame)
        for kwargs in ({"trail_length": 0}, {"trail_length": 1.5}, {"trail_length": True},
                       {"max_missed": -1}, {"max_missed": .5}, {"max_missed": False}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                VehicleTracker(**kwargs)


if __name__ == "__main__":
    unittest.main()
