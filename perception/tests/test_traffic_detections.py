"""Regression cases for riders versus nearby pedestrians."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from traffic_detections import filter_motorcycle_riders, pedestrian_labels


class RiderFilterTests(unittest.TestCase):
    labels = {0: "person", 3: "motorcycle", 7: "truck"}

    def filter(self, boxes, classes, scores=None, labels=None):
        return filter_motorcycle_riders(np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                                        np.asarray(classes), np.asarray(scores if scores is not None else [.9] * len(classes)),
                                        labels or self.labels, motorcycle_confidence=.25)

    def test_rider_removed_bike_and_adjacent_pedestrian_preserved(self):
        boxes = [(40, 10, 70, 85), (20, 60, 100, 110), (105, 15, 130, 110)]
        kept, classes, scores, count = self.filter(boxes, [0, 3, 0])
        np.testing.assert_array_equal(kept, boxes[1:])
        np.testing.assert_array_equal(classes, [3, 0])
        self.assertEqual(count, 1)
        self.assertEqual(len(scores), 2)

    def test_pedestrian_extending_below_bike_or_only_slightly_overlapping_kept(self):
        for pedestrian in [(40, 10, 70, 130), (90, 10, 120, 85), (40, 10, 70, 65)]:
            with self.subTest(pedestrian=pedestrian):
                self.assertEqual(self.filter([pedestrian, (20, 60, 100, 110)], [0, 3])[3], 0)

    def test_missing_or_low_confidence_bike_does_not_remove_person(self):
        self.assertEqual(self.filter([(40, 10, 70, 85)], [0])[3], 0)
        self.assertEqual(self.filter([(40, 10, 70, 85), (20, 60, 100, 110)], [0, 3], [.9, .1])[3], 0)

    def test_multiple_passengers_removed_without_duplicating_motorcycle(self):
        result = self.filter([(35, 5, 60, 90), (60, 10, 85, 90), (20, 60, 100, 110)], [0, 0, 3])
        np.testing.assert_array_equal(result[1], [3])
        self.assertEqual(result[3], 2)

    def test_custom_class_indices_explicit_riders_and_pedestrian_semantics(self):
        labels = {4: "person", 5: "motorbike", 6: "rider", 8: "pedestrian"}
        result = self.filter([(40, 10, 70, 85)] * 3 + [(20, 60, 100, 110)], [4, 6, 8, 5], labels=labels)
        np.testing.assert_array_equal(result[1], [8, 5])
        self.assertEqual(result[3], 2)
        self.assertEqual(pedestrian_labels(labels)[4], "pedestrian")
        self.assertEqual(labels[4], "person")
        self.assertEqual(pedestrian_labels(["person", "car"]), {0: "pedestrian", 1: "car"})

    def test_empty_frames_and_other_classes(self):
        self.assertEqual(self.filter([], [])[0].shape, (0, 4))
        result = self.filter([(40, 10, 70, 85), (20, 60, 100, 110)], [7, 3])
        self.assertEqual(result[3], 0)


if __name__ == "__main__":
    unittest.main()
