"""Verify identity diagnostics drawn by the shared tracking renderer."""

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tracking_visualization import annotate_tracking, track_color
from vehicle_tracking import TrackView, TrackingSnapshot


def drawing_spy():
    return SimpleNamespace(
        rectangle=Mock(wraps=cv2.rectangle), line=Mock(wraps=cv2.line),
        circle=Mock(wraps=cv2.circle), arrowedLine=Mock(wraps=cv2.arrowedLine),
        putText=Mock(wraps=cv2.putText), getTextSize=cv2.getTextSize,
        FONT_HERSHEY_SIMPLEX=cv2.FONT_HERSHEY_SIMPLEX, LINE_AA=cv2.LINE_AA,
    )


class TrackingVisualizationTests(unittest.TestCase):
    def test_identity_state_trails_panel_and_source_pixels(self):
        frame = np.zeros((100, 160, 3), dtype=np.uint8)
        boxes = np.array(((20, 40, 60, 85),), dtype=float)
        classes, scores = np.array((0,)), np.array((.9,))
        original = frame.copy(), boxes.copy(), classes.copy(), scores.copy()
        track = TrackView(7, 0, .9, tuple(boxes[0]), ((32, 82), (35, 84), (40, 85)), 12, 0, "TRACKED")
        snapshot = TrackingSnapshot([track], (7,), 1, 0, 0, 0, 7)
        drawing = drawing_spy()
        result = annotate_tracking(frame, boxes, classes, scores, {0: "car"}, snapshot,
                                   50, 2, drawing, dropped_frames=3)
        self.assertEqual(result.shape, (420, 460, 3))
        for actual, expected in zip((frame, boxes, classes, scores), original):
            np.testing.assert_array_equal(actual, expected)
        texts = [call.args[1] for call in drawing.putText.call_args_list]
        self.assertIn("#7 car", texts)
        self.assertIn("TRACKED", texts)
        self.assertIn("TRACKING LIVE", texts)
        self.assertIn("IDs created this session: 7", texts)
        self.assertIn("12 observations | missed 0", texts)
        self.assertIn("Tracker 2.0 ms | Skipped frames: 3", texts)
        self.assertEqual(drawing.arrowedLine.call_count, 1)
        self.assertEqual(drawing.arrowedLine.call_args.args[3], track_color(7))
        self.assertTrue(result[:, 160:].any())

    def test_missing_track_keeps_dashed_last_box_and_has_no_motion_arrow(self):
        frame = np.zeros((180, 320, 3), dtype=np.uint8)
        track = TrackView(2, 0, .8, (30, 70, 100, 140), ((50, 138), (65, 140)), 8, 3, "LOST")
        snapshot = TrackingSnapshot([track], (), 0, 1, 0, 0, 2)
        drawing = drawing_spy()
        annotate_tracking(frame, [], [], [], {0: "car"}, snapshot, 30, 1, drawing)
        texts = [call.args[1] for call in drawing.putText.call_args_list]
        self.assertIn("#2 car LOST", texts)
        self.assertIn("8 observations | missed 3", texts)
        self.assertIn("VISIBLE    0     LOST    1", texts)
        drawing.arrowedLine.assert_not_called()
        boxes = [(call.args[1], call.args[2]) for call in drawing.rectangle.call_args_list]
        self.assertNotIn(((30, 70), (100, 140)), boxes)
        dashed_segments = [call for call in drawing.line.call_args_list
                           if call.args[1] == (30, 70) and call.args[2] == (37, 70)]
        self.assertEqual(len(dashed_segments), 1)

    def test_small_feeds_have_bounded_labels_and_uncovered_header_pixels(self):
        for width, height in ((160, 100), (320, 180), (480, 270)):
            with self.subTest(size=(width, height)):
                frame = np.full((height, width, 3), 80, dtype=np.uint8)
                drawing = drawing_spy()
                result = annotate_tracking(
                    frame, ((width - 30, height - 30, width - 1, height - 1),),
                    (0,), (.9,), {0: "a very long traffic object class name " * 4},
                    None, 10, 0, drawing)
                np.testing.assert_array_equal(result[:20, :width], frame[:20])
                fills = [call.args for call in drawing.rectangle.call_args_list
                         if call.args[0].shape[:2] == (height, width) and call.args[4] == -1]
                self.assertEqual(len(fills), 1)
                _, (left, top), (right, bottom), _, _ = fills[0]
                self.assertLessEqual(right - left, width // 2)
                self.assertLessEqual(bottom - top, height * .15)
                self.assertTrue(0 <= left <= right < width)
                self.assertTrue(0 <= top <= bottom < height)
                label = next(call.args for call in drawing.putText.call_args_list
                             if call.args[0].shape[:2] == (height, width))
                self.assertLess(label[4], .46)
                self.assertNotIn("0.90", label[1])
                self.assertTrue(label[1].endswith("..."))

    def test_unassigned_detections_and_detection_only_mode_are_explicit(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        snapshot = TrackingSnapshot([], (None, None), 0, 0, 0, 0, 0)
        for current, suffix, title in ((snapshot, "unassigned", "TRACKING LIVE"),
                                       (None, "detection", "DETECTION ONLY")):
            with self.subTest(title=title):
                drawing = drawing_spy()
                result = annotate_tracking(frame, ((10, 50, 70, 110), (80, 50, 130, 110)),
                                           (0, 0), (.6, .2), {0: "car"}, current, 10, 0,
                                           drawing, display_confidence=.35)
                self.assertEqual(result.shape, (480, 940, 3))
                texts = [call.args[1] for call in drawing.putText.call_args_list]
                self.assertIn(title, texts)
                self.assertIn(f"car 0.60 | {suffix}", texts)
                self.assertFalse(any("car 0.20" in text for text in texts))

    def test_traffic_classes_render_with_tracking_and_detection_only(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        labels = {0: "car", 1: "bus", 2: "van", 3: "others", 4: "person",
                  5: "motorcycle", 6: "traffic light"}
        boxes = ((20, 80, 60, 160), (120, 80, 200, 160), (250, 50, 280, 90))
        classes, scores = (4, 5, 6), (.9, .9, .9)
        tracks = [TrackView(index + 1, cls, .9, box, (), 1, 0, "NEW")
                  for index, (cls, box) in enumerate(zip(classes, boxes))]
        snapshot = TrackingSnapshot(tracks, (1, 2, 3), 3, 0, 3, 0, 3)
        for current in (snapshot, None):
            with self.subTest(tracking=current is not None):
                drawing = drawing_spy()
                annotate_tracking(frame, boxes, classes, scores, labels, current, 10, 0, drawing)
                texts = [call.args[1] for call in drawing.putText.call_args_list]
                for name in ("person", "motorcycle", "traffic light"):
                    self.assertTrue(any(f"{name} 0.90 |" in text for text in texts))
                self.assertFalse(any("vehicle" in text for text in texts))


if __name__ == "__main__":
    unittest.main()
