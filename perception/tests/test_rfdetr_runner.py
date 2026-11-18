"""Verify local RF-DETR preview behavior without loading weights or a GPU."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_rfdetr


def detections(boxes=(), classes=(), scores=()):
    return SimpleNamespace(xyxy=np.asarray(boxes, dtype=float).reshape(-1, 4),
                           class_id=np.asarray(classes, dtype=int),
                           confidence=np.asarray(scores, dtype=float))


class PredictionTests(unittest.TestCase):
    def test_prediction_converts_bgr_to_rgb_and_returns_detections_directly(self):
        frame = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        original = frame.copy()
        result = detections(((0, 0, 1, 1),), (0,), (.9,))
        model = SimpleNamespace(predict=Mock(return_value=result))
        self.assertIs(run_rfdetr.predict_frame(model, frame, .35, cv2), result)
        args, kwargs = model.predict.call_args
        np.testing.assert_array_equal(args[0], np.array([[[30, 20, 10], [60, 50, 40]]], dtype=np.uint8))
        np.testing.assert_array_equal(frame, original)
        self.assertEqual(kwargs, {"threshold": .35, "include_source_image": False})


class ForegroundDetectionTests(unittest.TestCase):
    def setUp(self):
        self.labels = {0: "car", 1: "bus", 2: "van", 3: "others"}

    def test_filters_only_background_and_preserves_input_arrays(self):
        result = detections(((10, 40, 30, 60), (35, 50, 70, 85), (5, 5, 75, 95)),
                            (0, 3, 4), (.9, .8, .99))
        original = {name: getattr(result, name).copy() for name in ("xyxy", "class_id", "confidence")}

        foreground = run_rfdetr.foreground_detections(result, self.labels)

        np.testing.assert_array_equal(foreground.xyxy, original["xyxy"][:2])
        np.testing.assert_array_equal(foreground.class_id, [0, 3])
        np.testing.assert_array_equal(foreground.confidence, [.9, .8])
        self.assertEqual(foreground.filtered_background_count, 1)
        for name, expected in original.items():
            np.testing.assert_array_equal(getattr(result, name), expected)

    def test_background_only_predictions_produce_consistent_empty_arrays(self):
        result = detections(((10, 40, 30, 60), (35, 50, 70, 85)), (4, 4), (.99, .98))
        foreground = run_rfdetr.foreground_detections(result, self.labels)
        self.assertEqual(foreground.xyxy.shape, (0, 4))
        self.assertEqual(foreground.class_id.shape, (0,))
        self.assertEqual(foreground.confidence.shape, (0,))
        self.assertEqual(foreground.filtered_background_count, 2)

    def test_accepts_actual_supervision_empty_detections(self):
        from supervision import Detections

        foreground = run_rfdetr.foreground_detections(Detections.empty(), self.labels)
        self.assertEqual(foreground.xyxy.shape, (0, 4))
        self.assertEqual(foreground.class_id.shape, (0,))
        self.assertEqual(foreground.confidence.shape, (0,))
        self.assertEqual(foreground.filtered_background_count, 0)

    def test_invalid_class_indices_are_rejected_with_the_offending_values(self):
        for class_id in (-1, 1.5, float("nan"), float("inf"), float("-inf"), 5, 64):
            result = SimpleNamespace(xyxy=np.array([[10, 40, 30, 60]], dtype=float),
                                     class_id=np.array([class_id]), confidence=np.array([.9]))
            with self.subTest(class_id=class_id):
                with self.assertRaises(ValueError) as error:
                    run_rfdetr.foreground_detections(result, self.labels)
                self.assertIn(str(class_id), str(error.exception).lower())

    def test_detection_arrays_must_have_consistent_shapes(self):
        cases = [
            SimpleNamespace(xyxy=np.zeros((1, 3)), class_id=np.array([0]), confidence=np.array([.9])),
            SimpleNamespace(xyxy=np.zeros((1, 4)), class_id=np.array([0, 4]), confidence=np.array([.9])),
            SimpleNamespace(xyxy=np.zeros((1, 4)), class_id=np.array([[0]]), confidence=np.array([.9])),
            SimpleNamespace(xyxy=np.zeros((1, 4)), class_id=np.array([0]), confidence=np.array([])),
            SimpleNamespace(xyxy=np.zeros((1, 4)), class_id=np.array([0]), confidence=np.array([[.9]])),
        ]
        for result in cases:
            with self.subTest(shapes=(result.xyxy.shape, result.class_id.shape, result.confidence.shape)), \
                    self.assertRaisesRegex(ValueError, "shape|matching"):
                run_rfdetr.foreground_detections(result, self.labels)


class ClassMappingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.weights = Path(self.temp.name) / "checkpoint_best_total.pth"

    def test_default_traffic_bundle_has_all_seven_foreground_labels(self):
        args = run_rfdetr.parse_args([])
        self.assertEqual(args.model.parent.name, "rfdetr_nano_traffic7_bundle")
        self.assertEqual(run_rfdetr.load_labels(args.model, SimpleNamespace()),
                         {0: "car", 1: "bus", 2: "van", 3: "others",
                          4: "person", 5: "motorcycle", 6: "traffic light"})

    def test_bundle_mapping_is_zero_based_and_takes_precedence_over_model_names(self):
        (self.weights.parent / "class_names.json").write_text(json.dumps({
            "prediction_index_to_name": {"0": "car", "1": "bus", "2": "van", "3": "others"},
            "coco_category_id_to_name": {"1": "car", "2": "bus", "3": "van", "4": "others"},
        }))
        self.assertEqual(run_rfdetr.load_labels(self.weights, SimpleNamespace(class_names=["wrong"])),
                         {0: "car", 1: "bus", 2: "van", 3: "others"})

    def test_checkpoint_names_are_fallback_and_missing_or_gapped_mappings_fail(self):
        self.assertEqual(run_rfdetr.load_labels(self.weights, SimpleNamespace(class_names=["car", "bus"])),
                         {0: "car", 1: "bus"})
        self.assertEqual(run_rfdetr.load_labels(self.weights, SimpleNamespace(class_names={"0": "car"})),
                         {0: "car"})
        with self.assertRaisesRegex(ValueError, "no class names"):
            run_rfdetr.load_labels(self.weights, SimpleNamespace())
        for mapping in ({"1": "car"}, {"0": "car", "2": "bus"}, {"0": ""}, {"0": 123}):
            with self.subTest(mapping=mapping):
                (self.weights.parent / "class_names.json").write_text(json.dumps({"prediction_index_to_name": mapping}))
                with self.assertRaisesRegex(ValueError, "contiguous indices"):
                    run_rfdetr.load_labels(self.weights, SimpleNamespace(class_names=["fallback"]))


class FakeReader:
    def __init__(self, frames, *, error=None, terminal_state="finished"):
        self.frames = iter(frames)
        self.error, self.terminal_state = error, terminal_state
        self.source_fps = 24
        self.adapter_status = {"mode": "video", "state": terminal_state, "fps": 24, "frames": len(frames)}
        self.closed = False
        self.read_count = 0
        self.dropped = 1

    def read(self, timeout):
        self.read_count += 1
        frame = next(self.frames, None)
        return (False, None) if frame is None else (True, frame.copy())

    def close(self):
        self.closed = True

    def stats(self):
        if not self.closed:
            raise AssertionError("Reader statistics must be finalized after cleanup")
        return {"received_frames": 3, "dropped_frames": 1, "reconnections": 0}


class HeadlessRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.weights = self.root / "checkpoint_best_total.pth"
        self.weights.write_bytes(b"fake checkpoint; never loaded")
        self.args = run_rfdetr.parse_args([
            "--model", str(self.weights), "--device", "cpu", "--headless", "--no-tracking",
            "--runs-dir", str(self.root / "runs"), "--max-frames", "2",
        ])

    def fake_dependencies(self, model):
        torch_module = ModuleType("torch")
        torch_module.float32 = "fake float32"
        torch_module.cuda = SimpleNamespace(is_available=Mock(return_value=False))
        detector_class = SimpleNamespace(from_checkpoint=Mock(return_value=model))
        rfdetr_module = ModuleType("rfdetr")
        detr_module = ModuleType("rfdetr.detr")
        detr_module.RFDETR = detector_class
        return {"torch": torch_module, "rfdetr": rfdetr_module, "rfdetr.detr": detr_module}, detector_class

    def summary(self):
        paths = list(self.args.runs_dir.glob("*/summary.json"))
        self.assertEqual(len(paths), 1)
        return json.loads(paths[0].read_text()), paths[0].parent

    def test_headless_smoke_saves_preview_counts_summary_and_closes_reader(self):
        result = detections(((10, 40, 30, 60), (35, 50, 70, 85)), (0, 1), (.9, .8))
        model = SimpleNamespace(class_names=["car", "bus"], inference=Mock(), predict=Mock(return_value=result))
        modules, detector = self.fake_dependencies(model)
        reader = FakeReader([np.zeros((100, 80, 3), dtype=np.uint8) for _ in range(3)])
        with patch.dict(sys.modules, modules), patch.object(run_rfdetr, "LatestFrameReader", return_value=reader) as make_reader, \
                patch.object(cv2, "namedWindow", side_effect=AssertionError("Headless mode opened a window")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_rfdetr.run(self.args), 0)
        self.assertTrue(reader.closed)
        self.assertEqual(reader.read_count, 2)
        self.assertEqual(model.predict.call_count, 2)
        detector.from_checkpoint.assert_called_once_with(str(self.weights), device="cpu", compile=False)
        model.inference.assert_called_once_with(compile=False, dtype="fake float32")
        self.assertEqual(make_reader.call_args.kwargs, {"require_status": True})
        summary, run_dir = self.summary()
        self.assertEqual(summary["status"], "frame_limit")
        self.assertEqual(summary["frames"], 2)
        self.assertEqual(summary["detections_by_class"], {"car": 2, "bus": 2})
        self.assertEqual(summary["classes"], {"0": "car", "1": "bus"})
        self.assertEqual(summary["source_fps"], 24)
        self.assertEqual(summary["dropped_frames"], 1)
        self.assertEqual(summary["adapter"]["mode"], "video")
        self.assertTrue(summary["preview_has_detections"])
        self.assertIsNotNone(cv2.imread(str(run_dir / "preview.jpg")))
        self.assertGreaterEqual(summary["mean_inference_ms"], 0)

    def test_traffic7_labels_are_counted_and_tracked_with_only_slot_seven_filtered(self):
        from vehicle_tracking import VehicleTracker

        labels = run_rfdetr.load_labels(run_rfdetr.DEFAULT_MODEL, SimpleNamespace())
        (self.weights.parent / "class_names.json").write_text(json.dumps({
            "prediction_index_to_name": labels,
        }), encoding="utf-8")
        boxes = [(10 + 70 * index, 80, 50 + 70 * index, 160) for index in range(8)]
        result = detections(boxes, tuple(range(8)), (.9,) * 7 + (.99,))
        model = SimpleNamespace(class_names=["wrong"], model_config=SimpleNamespace(num_classes=7),
                                inference=Mock(), predict=Mock(return_value=result))
        modules, _ = self.fake_dependencies(model)
        tracker = VehicleTracker()
        self.args.no_tracking = False
        self.args.max_frames = 3
        reader = FakeReader([np.zeros((240, 640, 3), dtype=np.uint8) for _ in range(3)])

        with patch.dict(sys.modules, modules), patch("vehicle_tracking.VehicleTracker", return_value=tracker), \
                patch.object(tracker, "update", wraps=tracker.update) as update, \
                patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_rfdetr.run(self.args), 0)

        for call in update.call_args_list:
            np.testing.assert_array_equal(call.args[1], np.arange(7))
        summary, _ = self.summary()
        self.assertTrue(reader.closed)
        self.assertEqual(summary["status"], "frame_limit")
        self.assertEqual(summary["frames"], 3)
        expected_labels = {str(index): "pedestrian" if name == "person" else name for index, name in labels.items()}
        self.assertEqual(summary["classes"], expected_labels)
        self.assertEqual(summary["background_class_id"], 7)
        self.assertEqual(summary["background_predictions_filtered"], 3)
        self.assertEqual(summary["detections_by_class"], {name: 3 for name in expected_labels.values()})
        self.assertEqual(summary["tracking"]["unique_ids"], 7)

    def test_motorcycle_rider_is_removed_before_tracking_and_counting(self):
        from vehicle_tracking import VehicleTracker
        tracker = VehicleTracker()
        self.args.no_tracking = False
        result = detections(((40, 10, 70, 85), (20, 60, 100, 110), (110, 10, 140, 110)), (0, 1, 0), (.9, .9, .9))
        model = SimpleNamespace(class_names=["person", "motorcycle"], inference=Mock(), predict=Mock(return_value=result))
        modules, _ = self.fake_dependencies(model)
        reader = FakeReader([np.zeros((120, 160, 3), dtype=np.uint8)] * 2)
        with patch.dict(sys.modules, modules), patch("vehicle_tracking.VehicleTracker", return_value=tracker), \
                patch.object(tracker, "update", wraps=tracker.update) as update, \
                patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), redirect_stdout(io.StringIO()):
            self.assertEqual(run_rfdetr.run(self.args), 0)
        for call in update.call_args_list:
            np.testing.assert_array_equal(call.args[1], [1, 0])
        summary, _ = self.summary()
        self.assertEqual(summary["detections_by_class"], {"motorcycle": 2, "pedestrian": 2})
        self.assertEqual(summary["rider_detections_filtered"], 2)
        self.assertEqual(summary["tracking"]["unique_ids"], 2)

    def test_inference_failure_closes_reader_and_records_failure_without_counting_frame(self):
        model = SimpleNamespace(class_names=["car"], inference=Mock(),
                                predict=Mock(side_effect=RuntimeError("Synthetic inference failure")))
        modules, _ = self.fake_dependencies(model)
        reader = FakeReader([np.zeros((100, 80, 3), dtype=np.uint8)])
        with patch.dict(sys.modules, modules), patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "Synthetic inference failure"):
            run_rfdetr.run(self.args)
        self.assertTrue(reader.closed)
        summary, _ = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["detections_by_class"], {})
        self.assertIsNone(summary["mean_inference_ms"])
        self.assertIn("Synthetic inference failure", summary["error"])

    def test_feed_completion_and_feed_error_produce_distinct_summary_statuses(self):
        model = SimpleNamespace(class_names=["car"], inference=Mock(), predict=Mock())
        modules, _ = self.fake_dependencies(model)
        for terminal_state, error in (("finished", None), ("stopped", None), ("error", "Feed failed")):
            with self.subTest(terminal_state=terminal_state):
                self.args.runs_dir = self.root / terminal_state
                reader = FakeReader([], error=error, terminal_state=terminal_state)
                with patch.dict(sys.modules, modules), patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                        redirect_stdout(io.StringIO()):
                    if error:
                        with self.assertRaisesRegex(RuntimeError, "Feed failed"):
                            run_rfdetr.run(self.args)
                    else:
                        self.assertEqual(run_rfdetr.run(self.args), 0)
                self.assertTrue(reader.closed)
                summary, _ = self.summary()
                expected = "failed" if error else "completed" if terminal_state == "finished" else "stopped"
                self.assertEqual(summary["status"], expected)
                self.assertEqual(summary["frames"], 0)
        model.predict.assert_not_called()

    def test_tracking_gap_recovers_same_id_and_recording_preserves_panel(self):
        # Load the actual tracker before replacing torch for the detector fixture.
        from vehicle_tracking import VehicleTracker
        tracker = VehicleTracker(trail_length=4, max_missed=3)
        results = [detections() if index in (5, 6)
                   else detections(((10 + 2 * index, 40, 40 + 2 * index, 80),), (0,), (.9,))
                   for index in range(10)]
        model = SimpleNamespace(class_names=["car"], inference=Mock(), predict=Mock(side_effect=results))
        modules, _ = self.fake_dependencies(model)
        self.args.no_tracking = False
        self.args.max_frames = 10
        self.args.output = self.root / "tracking.mp4"
        reader = FakeReader([np.zeros((100, 160, 3), dtype=np.uint8) for _ in results])
        snapshots = []
        original_update = tracker.update

        def collect_update(*args):
            snapshot = original_update(*args)
            snapshots.append(snapshot)
            return snapshot

        with patch.dict(sys.modules, modules), patch("vehicle_tracking.VehicleTracker", return_value=tracker), \
                patch.object(tracker, "update", side_effect=collect_update), \
                patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                patch.object(cv2, "namedWindow", side_effect=AssertionError("Headless mode opened a window")), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_rfdetr.run(self.args), 0)
        self.assertTrue(reader.closed)
        summary, run_dir = self.summary()
        self.assertEqual(summary["detections_by_class"], {"car": 8})
        self.assertTrue(summary["tracking_enabled"])
        self.assertEqual(summary["prediction_confidence"], .1)
        self.assertEqual(summary["tracking"]["unique_ids"], 1)
        self.assertEqual(summary["tracking"]["recovered_events"], 1)
        self.assertEqual(summary["tracking"]["frames"], 10)
        self.assertGreaterEqual(summary["mean_tracking_ms"], 0)
        self.assertEqual(snapshots[5].tracks[0].state, "LOST")
        self.assertEqual(snapshots[7].tracks[0].state, "RECOVERED")
        self.assertEqual(snapshots[0].tracks[0].track_id, snapshots[7].tracks[0].track_id)
        self.assertEqual(len(snapshots[-1].tracks[0].trail), 4)
        self.assertEqual(cv2.imread(str(run_dir / "preview.jpg")).shape, (420, 460, 3))
        self.assertEqual(cv2.imread(summary["latest_preview"]).shape, (420, 460, 3))
        recording = cv2.VideoCapture(str(self.args.output))
        self.addCleanup(recording.release)
        self.assertTrue(recording.isOpened())
        self.assertEqual(recording.get(cv2.CAP_PROP_FRAME_COUNT), 10)
        ok, recorded = recording.read()
        self.assertTrue(ok)
        self.assertEqual(recorded.shape, (420, 460, 3))
        recording.release()

    def test_background_predictions_do_not_create_ids_or_interrupt_a_100_frame_run(self):
        # Load the actual tracker before replacing torch for the detector fixture.
        from vehicle_tracking import VehicleTracker

        tracker = VehicleTracker(trail_length=4, max_missed=3)
        results = []
        for index in range(100):
            vehicle_box = (10 + .4 * index, 40, 40 + .4 * index, 80)
            background_box = (100, 40, 140, 80)
            results.append(detections((background_box,), (4,), (.99,)) if index in (40, 41)
                           else detections((vehicle_box, background_box), (0, 4), (.9, .99)))
        model = SimpleNamespace(class_names=["car", "bus", "van", "others"], inference=Mock(),
                                predict=Mock(side_effect=results))
        modules, _ = self.fake_dependencies(model)
        self.args.no_tracking = False
        self.args.max_frames = 100
        reader = FakeReader([np.zeros((100, 160, 3), dtype=np.uint8) for _ in results])
        snapshots = []
        original_update = tracker.update

        def collect_update(boxes, classes, scores, frame):
            self.assertTrue(np.all(np.asarray(classes) == 0))
            snapshot = original_update(boxes, classes, scores, frame)
            snapshots.append(snapshot)
            return snapshot

        with patch.dict(sys.modules, modules), patch("vehicle_tracking.VehicleTracker", return_value=tracker), \
                patch.object(tracker, "update", side_effect=collect_update), \
                patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                redirect_stdout(io.StringIO()):
            self.assertEqual(run_rfdetr.run(self.args), 0)

        summary, _ = self.summary()
        self.assertTrue(reader.closed)
        self.assertEqual(summary["status"], "frame_limit")
        self.assertEqual(summary["frames"], 100)
        self.assertEqual(summary["background_class_id"], 4)
        self.assertEqual(summary["background_predictions_filtered"], 100)
        self.assertEqual(summary["detections_by_class"], {"car": 98})
        self.assertEqual(summary["tracking"]["frames"], 100)
        self.assertEqual(summary["tracking"]["unique_ids"], 1)
        self.assertEqual(summary["tracking"]["recovered_events"], 1)
        self.assertEqual(snapshots[40].detection_ids, ())
        self.assertEqual(snapshots[41].detection_ids, ())
        self.assertEqual(snapshots[40].tracks[0].state, "LOST")
        self.assertEqual(snapshots[42].tracks[0].state, "RECOVERED")
        self.assertEqual(snapshots[0].tracks[0].track_id, snapshots[42].tracks[0].track_id)
        self.assertTrue(all(track.class_id == 0 for snapshot in snapshots for track in snapshot.tracks))

    def test_unexpected_class_fails_before_the_tracker_receives_a_frame(self):
        from vehicle_tracking import VehicleTracker

        tracker = VehicleTracker()
        model = SimpleNamespace(class_names=["car", "bus", "van", "others"], inference=Mock(),
                                predict=Mock(return_value=detections(((10, 40, 30, 60),), (5,), (.99,))))
        modules, _ = self.fake_dependencies(model)
        self.args.no_tracking = False
        reader = FakeReader([np.zeros((100, 80, 3), dtype=np.uint8)])
        with patch.dict(sys.modules, modules), patch("vehicle_tracking.VehicleTracker", return_value=tracker), \
                patch.object(tracker, "update", wraps=tracker.update) as update, \
                patch.object(run_rfdetr, "LatestFrameReader", return_value=reader), \
                redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "5"):
            run_rfdetr.run(self.args)

        update.assert_not_called()
        self.assertTrue(reader.closed)
        summary, _ = self.summary()
        self.assertEqual(summary["status"], "failed")
        self.assertEqual(summary["frames"], 0)
        self.assertEqual(summary["tracking"]["frames"], 0)
        self.assertEqual(summary["tracking"]["unique_ids"], 0)
        self.assertEqual(summary["detections_by_class"], {})
        self.assertEqual(summary["background_predictions_filtered"], 0)
        self.assertIn("5", summary["error"])


class CliTests(unittest.TestCase):
    def test_default_feed_and_custom_loopback_endpoint(self):
        self.assertEqual(run_rfdetr.parse_args([]).endpoint, "http://127.0.0.1:8766/stream.mjpg")
        custom = run_rfdetr.parse_args(["--source", "https://[::1]:8443/camera", "--device", "cuda:2", "--conf", "0"])
        self.assertEqual(custom.endpoint, "https://[::1]:8443/camera")
        self.assertEqual(custom.device, "cuda:2")
        self.assertEqual(custom.conf, 0)

    def test_cli_rejects_invalid_devices_confidence_and_feed_endpoints(self):
        cases = (["--device", "mps"], ["--device", "-1"], ["--device", "cuda:"],
                 ["--conf", "nan"], ["--conf", "inf"], ["--conf", "1.1"], ["--conf", "-0.1"],
                 ["--source", "video.mp4"], ["--source", "https://www.youtube.com/watch?v=abc"],
                 ["--source", "http://192.168.0.10/video"], ["--source", "http://user:password@localhost/stream"],
                 ["--port", "0"], ["--port", "65536"], ["--max-frames", "-1"],
                 ["--trail-length", "0"], ["--trail-length", "-2"], ["--track-buffer", "-1"])
        for argv in cases:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                run_rfdetr.parse_args(argv)
            self.assertEqual(error.exception.code, 2)

    def test_device_resolution_normalizes_gpu_indices_and_rejects_unavailable_cuda(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=Mock(return_value=False)))
        self.assertEqual(run_rfdetr.resolve_device("auto", torch), "cpu")
        self.assertEqual(run_rfdetr.resolve_device("cpu", torch), "cpu")
        for device in ("0", "cuda", "cuda:2"):
            with self.subTest(device=device), self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                run_rfdetr.resolve_device(device, torch)
        torch.cuda.is_available.return_value = True
        self.assertEqual(run_rfdetr.resolve_device("auto", torch), "cuda:0")
        self.assertEqual(run_rfdetr.resolve_device("2", torch), "cuda:2")


if __name__ == "__main__":
    unittest.main()
