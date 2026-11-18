"""Endpoint pairs, source-video timing, calibration, and the finite export CLI."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
from vehicle_endpoints import Calibration, EndpointCollector
import export_vehicle_endpoints as exporter
from traffic_detections import pedestrian_labels


NAMES = {0: "Car", 1: "person", 2: "motorbike", 3: "bus", 4: "truck", 5: "van", 6: "bicycle"}


def observed(identity=1, box=(30, 30, 50, 60), *, cls=0, missed=0, state="TRACKED"):
    return SimpleNamespace(track_id=identity, class_id=cls, box=box, missed=missed, state=state)


def snapshot(*tracks):
    return SimpleNamespace(tracks=list(tracks))


def calibration_data(**overrides):
    data = {"image_points": [[0, 0], [1, 0], [1, 1], [0, 1]],
            "world_points": [[0, 2, 0], [100, 2, 0], [100, 2, 100], [0, 2, 100]],
            "end_nodes": [{"id": "north-exit", "position": [40, 20, 60], "radius_m": 3}]}
    data.update(overrides)
    return data


class EndpointCollectorTests(unittest.TestCase):
    def test_temporary_gap_recovers_same_pair_and_lost_predictions_do_not_become_endpoints(self):
        collector = EndpointCollector(100, 100, min_hits=3)
        collector.update(snapshot(observed(box=(20, 20, 40, 50))), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed(box=(25, 20, 45, 50))), NAMES, frame=1, time_s=.04)
        collector.update(snapshot(observed(box=(80, 70, 99, 99), missed=1, state="LOST")), NAMES, frame=2, time_s=.08)
        self.assertEqual(collector.vehicles, [], "A temporary miss must not finish the identity.")
        collector.update(snapshot(observed(box=(35, 20, 55, 55), state="RECOVERED")), NAMES, frame=3, time_s=.12)
        collector.update(snapshot(observed(box=(80, 70, 99, 99), missed=1, state="LOST")), NAMES, frame=4, time_s=.16)
        collector.update(snapshot(), NAMES, frame=5, time_s=.20)
        vehicle, = collector.finish()
        self.assertEqual(vehicle["spawn"], {"time_s": 0, "frame": 0, "point": [.3, .5]})
        self.assertEqual(vehicle["disappearance"], {"time_s": .12, "frame": 3, "point": [.45, .55], "reason": "tracking_lost"})
        self.assertEqual(set(vehicle), {"track_id", "kind", "spawn", "disappearance"})
        self.assertEqual(collector.active, {})

    def test_retirement_near_border_finishes_at_last_real_observation_once(self):
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(observed(box=(0, 20, 10, 50))), NAMES, frame=0, time_s=.125)
        collector.update(snapshot(observed(box=(50, 20, 60, 50), missed=1)), NAMES, frame=1, time_s=.25)
        collector.update(snapshot(), NAMES, frame=2, time_s=.375)
        vehicle, = collector.finish()
        self.assertEqual(vehicle["disappearance"]["reason"], "out_of_view")
        self.assertEqual(vehicle["disappearance"]["frame"], 0)
        self.assertEqual(vehicle["disappearance"]["time_s"], .125)
        self.assertEqual(collector.finish(), [vehicle])

    def test_end_of_clip_censors_both_visible_and_pending_lost_tracks_even_at_border(self):
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(observed(1, (0, 10, 10, 30)), observed(2)), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed(1, missed=1, state="LOST"), observed(2)), NAMES, frame=1, time_s=.1)
        vehicles = collector.finish()
        self.assertEqual([v["disappearance"]["reason"] for v in vehicles], ["clip_end", "clip_end"])
        self.assertEqual([v["disappearance"]["frame"] for v in vehicles], [0, 1])

    def test_nonvehicle_classes_are_excluded_and_vehicle_aliases_canonicalized(self):
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(*(observed(index + 1, cls=index) for index in range(8))), NAMES, frame=0, time_s=0)
        self.assertEqual([v["kind"] for v in collector.finish()], ["car", "motorcycle", "bus", "truck", "van"])
        self.assertEqual(collector.rejected_tracks, 0)

    def test_minimum_hits_counts_only_real_observations_not_lost_grace_frames(self):
        collector = EndpointCollector(100, 100, min_hits=3)
        collector.update(snapshot(observed()), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed()), NAMES, frame=1, time_s=.1)
        for frame in range(2, 6):
            collector.update(snapshot(observed(missed=frame - 1, state="LOST")), NAMES, frame=frame, time_s=frame / 10)
        collector.update(snapshot(), NAMES, frame=6, time_s=.6)
        self.assertEqual(collector.finish(), [])
        self.assertEqual(collector.rejected_tracks, 1)

    def test_marked_terminal_uses_ground_plane_distance_and_unmarked_loss_stays_failed(self):
        calibration = Calibration(calibration_data())
        collector = EndpointCollector(100, 100, min_hits=1, calibration=calibration)
        collector.update(snapshot(observed(1), observed(2, (65, 30, 75, 70))), NAMES, frame=0, time_s=0)
        collector.update(snapshot(), NAMES, frame=1, time_s=.1)
        exit_vehicle, failed_vehicle = collector.finish()
        self.assertEqual(exit_vehicle["disappearance"]["reason"], "end_node")
        self.assertEqual(exit_vehicle["disappearance"]["node_id"], "north-exit")
        np.testing.assert_allclose(exit_vehicle["spawn"]["world_position"], [40, 2, 60], atol=1e-8)
        self.assertEqual(failed_vehicle["disappearance"]["reason"], "tracking_lost")
        self.assertNotIn("node_id", failed_vehicle["disappearance"])

    def test_calibrated_eof_does_not_turn_a_visible_terminal_into_a_fabricated_exit(self):
        collector = EndpointCollector(100, 100, min_hits=1, calibration=Calibration(calibration_data()))
        collector.update(snapshot(observed()), NAMES, frame=0, time_s=0)
        end = collector.finish()[0]["disappearance"]
        self.assertEqual(end["reason"], "clip_end")
        self.assertNotIn("node_id", end)

    def test_output_sorting_uses_first_observation_time_and_identity(self):
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(observed(9), observed(2)), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed(5)), NAMES, frame=1, time_s=.1)
        self.assertEqual([v["track_id"] for v in collector.finish()], [2, 9, 5])

    def test_invalid_frame_time_and_frame_indices_reject_without_advancing_state(self):
        for bad_frame, bad_time in [(0, .1), (-1, .1), (.5, .1), (True, .1), (1, -.1), (1, math.nan), (1, math.inf)]:
            with self.subTest(frame=bad_frame, time=bad_time):
                collector = EndpointCollector(100, 100, min_hits=1)
                collector.update(snapshot(observed()), NAMES, frame=0, time_s=0)
                with self.assertRaises(ValueError):
                    collector.update(snapshot(), NAMES, frame=bad_frame, time_s=bad_time)
                collector.update(snapshot(observed()), NAMES, frame=1, time_s=.1)
                self.assertEqual(collector.finish()[0]["disappearance"]["frame"], 1)
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(observed()), NAMES, frame=0, time_s=1)
        with self.assertRaises(ValueError):
            collector.update(snapshot(), NAMES, frame=1, time_s=.9)

    def test_dimensions_threshold_and_margin_are_validated(self):
        for arguments in [(1, 100, 1), (100, 1, 1), (100, 100, 0), (math.nan, 100, 1), (100.5, 100, 1), (100, 100, 1.5), (True, 100, 1)]:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                EndpointCollector(arguments[0], arguments[1], min_hits=arguments[2])
        for margin in [-.1, .5, math.nan, math.inf]:
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                EndpointCollector(100, 100, edge_margin=margin)


class TrajectoryTests(unittest.TestCase):
    def test_paths_keep_every_real_observation_and_skip_lost_predictions(self):
        collector = EndpointCollector(100, 100, min_hits=2, keep_paths=True)
        collector.update(snapshot(observed(box=(20, 20, 40, 50))), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed(box=(80, 70, 99, 99), missed=1, state="LOST")), NAMES, frame=1, time_s=.1)
        collector.update(snapshot(observed(box=(30, 20, 50, 60), state="RECOVERED")), NAMES, frame=2, time_s=.2)
        collector.update(snapshot(), NAMES, frame=3, time_s=.3)
        vehicle, = collector.finish()
        path, = collector.trajectories
        self.assertEqual((path["track_id"], path["kind"], path["termination"]), (1, "car", "tracking_lost"))
        self.assertEqual([o["frame"] for o in path["observations"]], [0, 2])
        self.assertEqual(path["observations"][1]["point"], [.4, .6])
        self.assertEqual(path["observations"][1]["box"], [.3, .2, .5, .6])
        self.assertEqual(path["observations"][0]["point"], vehicle["spawn"]["point"])
        self.assertEqual(set(vehicle), {"track_id", "kind", "spawn", "disappearance"})

    def test_paths_follow_endpoint_filtering_sorting_and_calibration(self):
        collector = EndpointCollector(100, 100, min_hits=2, keep_paths=True, calibration=Calibration(calibration_data()))
        collector.update(snapshot(observed(9)), NAMES, frame=0, time_s=0)
        collector.update(snapshot(observed(9), observed(2), observed(5, cls=1)), NAMES, frame=1, time_s=.1)
        collector.update(snapshot(observed(2)), NAMES, frame=2, time_s=.2)
        collector.finish()
        self.assertEqual([p["track_id"] for p in collector.trajectories], [9, 2])
        self.assertEqual(collector.rejected_tracks, 0)
        np.testing.assert_allclose(collector.trajectories[0]["observations"][0]["world_position"], [40, 2, 60], atol=1e-8)

    def test_confidence_is_recorded_when_the_tracker_provides_it(self):
        collector = EndpointCollector(100, 100, min_hits=1, keep_paths=True)
        track = observed(); track.confidence = .87654
        collector.update(snapshot(track), NAMES, frame=0, time_s=0)
        collector.finish()
        self.assertEqual(collector.trajectories[0]["observations"][0]["confidence"], .8765)

    def test_paths_are_not_collected_by_default(self):
        collector = EndpointCollector(100, 100, min_hits=1)
        collector.update(snapshot(observed()), NAMES, frame=0, time_s=0)
        collector.finish()
        self.assertEqual(collector.trajectories, [])
        self.assertEqual(collector.active, {})

class CalibrationTests(unittest.TestCase):
    def test_nearest_explicit_terminal_wins_and_radius_is_respected(self):
        data = calibration_data(end_nodes=[{"id": "far", "position": [40, 0, 62], "radius_m": 3},
                                          {"id": "near", "position": [40, 10, 60], "radius_m": 1}])
        calibration = Calibration(data)
        self.assertEqual(calibration.end_node([40, 2, 60]), "near")
        self.assertIsNone(calibration.end_node([40, 2, 80]))

    def test_calibration_rejects_degenerate_nonfinite_and_invalid_terminals(self):
        invalid = [calibration_data(image_points=[[0, 0], [.2, .2], [.4, .4], [.8, .8]]),
                   calibration_data(image_points=[[0, 0], [2, 0], [1, 1], [0, 1]]),
                   calibration_data(ground_y=math.nan),
                   calibration_data(end_nodes=[{"id": "", "position": [0, 0, 0]}]),
                   calibration_data(end_nodes=[{"id": "end", "position": [0, 0, math.inf]}]),
                   calibration_data(end_nodes=[{"id": "end", "position": [0, 0, 0], "radius_m": 0}])]
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                Calibration(data)

    def test_projection_horizon_is_rejected(self):
        calibration = Calibration(calibration_data())
        calibration.matrix[2] = [1, 0, -.5]
        with self.assertRaises(ValueError):
            calibration.project([.5, .2])


class ActualTrackerEndpointTests(unittest.TestCase):
    def test_actual_tracktrack_recovers_then_retires_using_the_last_detection(self):
        from vehicle_tracking import VehicleTracker
        tracker = VehicleTracker(trail_length=1, max_missed=2)
        collector = EndpointCollector(320, 160, min_hits=3)
        frame = np.zeros((160, 320, 3), dtype=np.uint8)
        boxes = [[[30 + n, 50, 80 + n, 100]] for n in range(4)] + [[], [[35, 50, 85, 100]], [], [], []]
        for index, detections in enumerate(boxes):
            views = tracker.update(detections, [0] if detections else [], [.9] if detections else [], frame)
            collector.update(views, {0: "car"}, frame=index, time_s=index / 25)
        vehicle, = collector.finish()
        self.assertEqual(vehicle["track_id"], 1)
        self.assertEqual(vehicle["spawn"]["frame"], 0)
        self.assertEqual(vehicle["disappearance"]["frame"], 5)
        self.assertEqual(vehicle["disappearance"]["time_s"], .2)
        self.assertEqual(vehicle["disappearance"]["point"], [60 / 320, 100 / 160])
        self.assertEqual(vehicle["disappearance"]["reason"], "tracking_lost")
        self.assertEqual(tracker.summary()["recovered_events"], 1)
        self.assertEqual(tracker.summary()["retired_tracks"], 1)


class ExportCliTests(unittest.TestCase):
    def test_invalid_cli_values_are_rejected_before_loading_models(self):
        cases = [["--duration", "0"], ["--duration", "nan"], ["--lost-seconds", "-1"],
                 ["--lost-seconds", "inf"], ["--conf", "1.1"], ["--conf", "nan"],
                 ["--edge-margin", ".5"], ["--edge-margin", "-1"], ["--min-hits", "0"],
                 ["--feed", " "]]
        for arguments in cases:
            with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                exporter.parse_args(["input.mp4", "--output", "new.json", *arguments])
            self.assertEqual(raised.exception.code, 2)

    def test_file_validation_protects_existing_inputs_and_output(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            video, model = root / "input.mp4", root / "model.pt"
            arguments = exporter.parse_args([str(video), "--model", str(model), "--output", str(root / "new.json")])
            with self.assertRaises(FileNotFoundError):
                exporter.export(arguments)
            video.write_bytes(b"video"); model.write_bytes(b"model")
            arguments.output.write_bytes(b"previous-export")
            with self.assertRaises(FileExistsError):
                exporter.export(arguments)
            self.assertEqual(arguments.output.read_bytes(), b"previous-export")
            arguments.output = root / "new.csv"
            with self.assertRaisesRegex(ValueError, "json"):
                exporter.export(arguments)
            arguments.output = root / "new.json"; arguments.output.unlink()
            arguments.annotated = root / "new.avi"
            with self.assertRaisesRegex(ValueError, "mp4"):
                exporter.export(arguments)

    def fake_export(self, root, *, fps=10, dimensions=None, duration=.31, labels=None, trajectories=False):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(8)] if dimensions is None else [np.zeros(size, dtype=np.uint8) for size in dimensions]
        state = SimpleNamespace(reads=0, released=False, capture_created=False, predict_calls=[], tracker_options=None,
                                tracked_classes=[])

        class Capture:
            def __init__(self): state.capture_created = True
            def isOpened(self): return True
            def get(self, _): return fps
            def read(self):
                if state.reads >= len(frames): return False, None
                result = frames[state.reads]; state.reads += 1
                return True, result
            def release(self): state.released = True

        model_labels = dict(enumerate(["car", "person", "motorcycle"] if labels is None else labels))

        def load_model(weights, device, torch):
            return "model", model_labels, pedestrian_labels(model_labels)

        def detect(model, frame, confidence, labels, model_labels, motorcycle_confidence, cv2):
            # One vehicle and one pedestrian; only the vehicle may reach the tracker.
            state.predict_calls.append({"conf": confidence, "motorcycle_conf": motorcycle_confidence})
            return (SimpleNamespace(xyxy=np.array([[30., 30, 50, 60], [70, 10, 80, 40]]),
                                    class_id=np.array([0, 1]), confidence=np.array([.9, .9])), 0, 0)

        class Tracker:
            low_confidence_threshold = .1
            def __init__(self, **options): state.tracker_options = options
            def update(self, boxes, classes, *_):
                state.tracked_classes.append(list(classes))
                return snapshot(observed())
            def summary(self): return {"frames": len(state.predict_calls)}

        video, model = root / "input.mp4", root / "model.pt"
        video.write_bytes(b"video"); model.write_bytes(b"model")
        args = exporter.parse_args([str(video), "--model", str(model), "--output", str(root / "new.json"), "--min-hits", "1"])
        args.duration = duration
        if trajectories:
            args.trajectories = root / "paths.json"
        modules = {"cv2": SimpleNamespace(VideoCapture=lambda _: Capture(), CAP_PROP_FPS=1),
                   "torch": SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
                   "vehicle_tracking": SimpleNamespace(VehicleTracker=Tracker),
                   "tracking_visualization": SimpleNamespace(annotate_tracking=lambda *args, **kwargs: None)}
        with patch.dict(sys.modules, modules), patch.object(exporter.run_rfdetr, "load_model", load_model),                 patch.object(exporter.run_rfdetr, "detect", detect), patch.object(exporter.time, "perf_counter", side_effect=[100., 500.]), redirect_stdout(io.StringIO()):
            try:
                payload = exporter.export(args)
            except Exception:
                if state.capture_created:
                    self.assertTrue(state.released, "Decoder must release resources after validation/inference failure.")
                self.assertFalse(args.output.exists())
                raise
        self.assertTrue(state.released)
        self.assertEqual(json.loads(args.output.read_text(encoding="utf-8")), payload)
        return payload, state

    def test_export_processes_every_source_frame_and_uses_video_time_despite_slow_inference(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            payload, state = self.fake_export(Path(temporary))
        self.assertEqual((payload["frames"], payload["duration_s"], payload["fps"]), (4, .4, 10))
        self.assertEqual(state.reads, 4)
        self.assertEqual(state.tracker_options, {"trail_length": 1, "max_missed": 10})
        self.assertEqual(state.tracked_classes, [[0]] * 4)
        self.assertEqual(state.predict_calls, [{"conf": .1, "motorcycle_conf": .35}] * 4)
        vehicle, = payload["vehicles"]
        self.assertEqual(vehicle["spawn"]["time_s"], 0)
        self.assertEqual(vehicle["disappearance"]["time_s"], .3)
        self.assertEqual(vehicle["disappearance"]["reason"], "clip_end")
        self.assertEqual(payload["diagnostics"]["elapsed_seconds"], 400)
        self.assertEqual(payload["diagnostics"]["termination_counts"], {"clip_end": 1})
        self.assertEqual(set(vehicle), {"track_id", "kind", "spawn", "disappearance"})

    def test_export_writes_trajectories_with_the_same_identities(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            payload, _ = self.fake_export(root, trajectories=True)
            paths = json.loads((root / "paths.json").read_text(encoding="utf-8"))
        self.assertEqual(paths["schema"], "vehicle-trajectories")
        self.assertEqual((paths["fps"], paths["frames"], paths["endpoints_file"]), (10, 4, "new.json"))
        self.assertIsNone(paths["world_coordinate_system"])
        path, = paths["vehicles"]
        self.assertEqual(path["track_id"], payload["vehicles"][0]["track_id"])
        self.assertEqual([o["frame"] for o in path["observations"]], [0, 1, 2, 3])
        self.assertNotIn("trajectories", payload)

    def test_existing_trajectory_file_is_protected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary:
            root = Path(temporary)
            (root / "paths.json").write_bytes(b"previous")
            with self.assertRaises(FileExistsError):
                self.fake_export(root, trajectories=True)
            self.assertEqual((root / "paths.json").read_bytes(), b"previous")

    def test_export_rejects_bad_fps_no_frames_dimension_changes_and_nonvehicle_checkpoints(self):
        for options in [{"fps": 0}, {"fps": math.nan}, {"dimensions": []},
                        {"dimensions": [(100, 100, 3), (100, 120, 3)]}, {"labels": ["person", "bicycle"]}]:
            with self.subTest(options=options), tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temporary, self.assertRaises(ValueError):
                self.fake_export(Path(temporary), **options)


if __name__ == "__main__":
    unittest.main()
