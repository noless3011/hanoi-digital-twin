"""Track every frame of a finite video and export vehicle spawn/last-seen points.

Detection uses the same RF-DETR model and rider filter as run_rfdetr.py.

With --trajectories, every real per-frame observation is also written to a second file.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
import time

import run_rfdetr
from vehicle_endpoints import Calibration, EndpointCollector, VEHICLE_KINDS


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Finite local video; every frame is processed without live-reader drops")
    parser.add_argument("--output", required=True, type=Path, help="New endpoint JSON file for Unity")
    parser.add_argument("--duration", type=float, help="Analyze at most this many source-video seconds; default entire clip")
    parser.add_argument("--feed", default="yenphu-hangthan", help="Feed/scene identifier stored in the export")
    parser.add_argument("--model", type=Path, default=run_rfdetr.DEFAULT_MODEL, help="RF-DETR checkpoint")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or a GPU index")
    parser.add_argument("--conf", type=float, default=.35)
    parser.add_argument("--lost-seconds", type=float, default=1., help="Source-video grace period for temporary tracking misses")
    parser.add_argument("--min-hits", type=int, default=3, help="Reject tracks with fewer actual observations")
    parser.add_argument("--edge-margin", type=float, default=.04, help="Image fraction near a border considered an exit (heuristic)")
    parser.add_argument("--calibration", type=Path, help="Optional planar landmark calibration JSON with authored end_nodes")
    parser.add_argument("--trajectories", type=Path, help="Optional new JSON file with every per-frame observation of each kept vehicle")
    parser.add_argument("--annotated", type=Path, help="Optional new diagnostic MP4 at the source FPS")
    args = parser.parse_args(argv)
    if (args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0)
            or not math.isfinite(args.lost_seconds) or args.lost_seconds < 0
            or not math.isfinite(args.conf) or not 0 <= args.conf <= 1
            or not math.isfinite(args.edge_margin) or not 0 <= args.edge_margin < .5
            or args.min_hits < 1 or not args.feed.strip()):
        parser.error("Use positive duration/min-hits, nonnegative lost-seconds, confidence in [0,1], edge-margin in [0,.5), and a feed name")
    return args


def export(args):
    for path in (args.video, args.model):
        if not path.is_file():
            raise FileNotFoundError(f"Input not found: {path}")
    trajectories = getattr(args, "trajectories", None)
    for path, suffix in ((args.output, ".json"), (trajectories, ".json"), (args.annotated, ".mp4")):
        if path is not None:
            if path.exists():
                raise FileExistsError(f"Choose a new output file: {path}")
            if path.suffix.lower() != suffix:
                raise ValueError(f"Output needs {suffix} extension: {path}")
    calibration = Calibration(json.loads(args.calibration.read_text(encoding="utf-8-sig"))) if args.calibration else None
    import cv2
    import numpy as np
    import torch
    from vehicle_tracking import VehicleTracker
    from tracking_visualization import annotate_tracking

    device = run_rfdetr.resolve_device(args.device, torch)
    model, model_labels, labels = run_rfdetr.load_model(args.model, device, torch)
    vehicle_ids = [key for key, label in labels.items() if str(label).lower() in VEHICLE_KINDS]
    if not vehicle_ids:
        raise ValueError("Checkpoint has no supported vehicle classes (car, motorcycle, bus, truck, van)")
    capture = cv2.VideoCapture(str(args.video))
    writer = None
    processed = 0
    started = time.perf_counter()
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode video: {args.video}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Video has no valid FPS; re-encode with a known source frame rate")
        tracker = VehicleTracker(trail_length=1, max_missed=math.ceil(args.lost_seconds * fps))
        collector = None
        frame_limit = math.ceil(args.duration * fps) if args.duration is not None else None
        while frame_limit is None or processed < frame_limit:
            ok, frame = capture.read()
            if not ok:
                break
            height, width = frame.shape[:2]
            if collector is None:
                collector = EndpointCollector(width, height, min_hits=args.min_hits,
                                              edge_margin=args.edge_margin, calibration=calibration,
                                              keep_paths=trajectories is not None)
            elif (width, height) != (collector.width, collector.height):
                raise ValueError("Video dimensions changed; use a fixed-resolution clip")
            inference_start = time.monotonic()
            detections, _, _ = run_rfdetr.detect(model, frame, min(args.conf, tracker.low_confidence_threshold),
                                                 labels, model_labels, args.conf, cv2)
            inference_ms = (time.monotonic() - inference_start) * 1000
            vehicles_only = np.isin(detections.class_id, vehicle_ids)
            xyxy = detections.xyxy[vehicles_only]
            classes = detections.class_id[vehicles_only]
            scores = detections.confidence[vehicles_only]
            snapshot = tracker.update(xyxy, classes, scores, frame)
            collector.update(snapshot, labels, frame=processed, time_s=processed / fps)
            if args.annotated:
                image = annotate_tracking(frame, xyxy, classes, scores, labels, snapshot,
                                          inference_ms, 0, cv2,
                                          detector="RF-DETR endpoints", display_confidence=args.conf)
                if writer is None:
                    args.annotated.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.annotated), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (image.shape[1], image.shape[0]))
                    if not writer.isOpened():
                        raise RuntimeError(f"Cannot write diagnostic video: {args.annotated}")
                writer.write(image)
            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed} source frames ({processed/fps:.2f}s)", flush=True)
        if collector is None:
            raise ValueError("Video contains no decodable frames")
        vehicles = collector.finish()
        payload = {"schema": "vehicle-endpoints", "version": 1, "feed": args.feed,
                   "duration_s": processed / fps, "frame_width": collector.width,
                   "frame_height": collector.height, "fps": fps, "frames": processed,
                   "coordinate_system": "normalized_image_top_left",
                   "vehicles": vehicles,
                   "diagnostics": {"termination_counts": dict(Counter(v["disappearance"]["reason"] for v in vehicles)),
                                   "rejected_short_tracks": collector.rejected_tracks,
                                   "lost_seconds": args.lost_seconds, "minimum_observations": args.min_hits,
                                   "calibrated_world_positions": calibration is not None,
                                   "elapsed_seconds": round(time.perf_counter() - started, 3),
                                   "device": device, "detector": "rfdetr", "model": args.model.name,
                                   "tracking": tracker.summary()}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation protects an existing export even if another process wrote it meanwhile.
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        print(f"Saved {len(vehicles)} vehicle endpoint pairs from {processed/fps:.2f}s: {args.output}", flush=True)
        print(f"Terminations: {payload['diagnostics']['termination_counts']}", flush=True)
        if trajectories is not None:
            paths = {"schema": "vehicle-trajectories", "version": 1, "feed": args.feed,
                     "duration_s": payload["duration_s"], "frame_width": collector.width,
                     "frame_height": collector.height, "fps": fps, "frames": processed,
                     "coordinate_system": "normalized_image_top_left",
                     "ground_point": "bottom_center_of_box",
                     "world_coordinate_system": "unity_xz_metres" if calibration else None,
                     "endpoints_file": args.output.name,
                     "vehicles": collector.trajectories}
            trajectories.parent.mkdir(parents=True, exist_ok=True)
            with trajectories.open("x", encoding="utf-8") as handle:
                json.dump(paths, handle, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            count = sum(len(path["observations"]) for path in collector.trajectories)
            print(f"Saved {count} observations for {len(collector.trajectories)} trajectories: {trajectories}", flush=True)
        return payload
    finally:
        capture.release()
        if writer is not None:
            writer.release()


def main(argv=None):
    args = parse_args(argv)
    try:
        export(args)
        return 0
    except Exception as error:
        print(f"Endpoint export failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
