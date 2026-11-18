"""Preview RF-DETR traffic object identities and trajectories from the local video feed."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace
from uuid import uuid4

from local_feed import DEFAULT_PORT, LatestFrameReader, decode_jpeg, local_endpoint, safe_message


PROJECT = Path(__file__).resolve().parent
DEFAULT_MODEL = PROJECT / "models" / "rfdetr_nano_traffic7_bundle" / "checkpoint_best_total.pth"
WINDOW = "RF-DETR + TrackTrack traffic object tracking - Q or Esc to stop"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--source", help="Loopback MJPEG endpoint; route YouTube/files/twin through feed/video_adapter.py")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, or GPU index")
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--no-tracking", action="store_true", help="Show detections only, without persistent IDs")
    parser.add_argument("--trail-length", type=int, default=40, help="Maximum observed positions in each track trail")
    parser.add_argument("--track-buffer", type=int, default=30, help="Processed frames to retain a missed track")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N processed frames; 0 keeps running")
    parser.add_argument("--output", type=Path, help="Optional annotated MP4; existing files are preserved")
    parser.add_argument("--runs-dir", type=Path, default=PROJECT.parent / "runs" / "rfdetr")
    args = parser.parse_args(argv)
    if not math.isfinite(args.conf) or not 0 <= args.conf <= 1:
        parser.error("--conf must be finite and between 0 and 1")
    if args.max_frames < 0:
        parser.error("--max-frames cannot be negative")
    if args.trail_length <= 0 or args.track_buffer < 0:
        parser.error("--trail-length must be positive; --track-buffer cannot be negative")
    if not re.fullmatch(r"auto|cpu|cuda(?::\d+)?|\d+", args.device):
        parser.error("--device must be auto, cpu, cuda, cuda:N, or a GPU index")
    try:
        args.endpoint = local_endpoint(args.source, args.port)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def resolve_device(requested, torch):
    device = ("cuda:0" if torch.cuda.is_available() else "cpu") if requested == "auto" else requested
    if device.isdecimal():
        device = f"cuda:{device}"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use --device cpu or restore the CUDA PyTorch build with setup.ps1.")
    return device


def load_labels(weights, model):
    mapping_path = weights.parent / "class_names.json"
    if mapping_path.is_file():
        payload = json.loads(mapping_path.read_text(encoding="utf-8"))
        mapping = payload["prediction_index_to_name"]
        labels = {int(index): name for index, name in mapping.items()}
    else:
        names = getattr(model, "class_names", None)
        if not names:
            raise ValueError("Checkpoint has no class names. Place class_names.json beside the weights.")
        labels = dict(enumerate(names)) if isinstance(names, (list, tuple)) else {int(k): v for k, v in names.items()}
    if sorted(labels) != list(range(len(labels))) or not labels or any(not isinstance(name, str) or not name for name in labels.values()):
        raise ValueError("Prediction class names must be nonempty strings with contiguous indices starting at zero.")
    return labels


def predict_frame(model, bgr_frame, confidence, cv2):
    # RF-DETR expects RGB arrays; OpenCV decoding and display use BGR.
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    return model.predict(rgb, threshold=confidence, include_source_image=False)


def foreground_detections(detections, labels):
    """Remove RF-DETR's final no-object slot before association or counting.

    Fine-tuned detection heads have N foreground logits plus background at N.
    predict() can return that background slot when its score clears threshold.
    Other unknown IDs remain errors rather than being hidden by this filter.
    """
    import numpy as np

    boxes = np.asarray(detections.xyxy, dtype=float)
    classes = np.asarray(detections.class_id, dtype=float)
    scores = np.asarray(detections.confidence, dtype=float)
    if (boxes.ndim != 2 or boxes.shape[1] != 4 or classes.shape != (len(boxes),)
            or scores.shape != (len(boxes),)):
        raise ValueError("Detection boxes must have shape (N, 4), with matching class and confidence arrays.")
    if not np.isfinite(classes).all():
        raise ValueError(f"Model returned non-finite class indices {np.unique(classes[~np.isfinite(classes)]).tolist()}.")
    if not np.isfinite(boxes).all() or not np.isfinite(scores).all():
        raise ValueError("Model returned non-finite boxes or confidence scores.")
    background_id = len(labels)
    invalid = ((classes != np.floor(classes)) | (classes < 0)
               | ~np.isin(classes, (*labels, background_id)))
    if invalid.any():
        raise ValueError(f"Model returned class indices {np.unique(classes[invalid]).tolist()}, absent from its "
                         f"class mapping {sorted(labels)} (background slot {background_id}).")
    foreground = classes != background_id
    return SimpleNamespace(xyxy=boxes[foreground], class_id=classes[foreground].astype(int),
                           confidence=scores[foreground], filtered_background_count=int((~foreground).sum()))


def load_model(weights, device, torch):
    """Load an RF-DETR checkpoint; return (model, model_labels, tracking_labels)."""
    from rfdetr.detr import RFDETR
    from traffic_detections import pedestrian_labels

    model = RFDETR.from_checkpoint(str(weights), device=device, compile=False)
    model.inference(compile=False, dtype=torch.float32)
    model_labels = load_labels(weights, model)
    labels = pedestrian_labels(model_labels)
    checkpoint_classes = getattr(getattr(model, "model_config", None), "num_classes", len(labels))
    if checkpoint_classes != len(labels):
        raise ValueError(f"Checkpoint has {checkpoint_classes} foreground classes but its class mapping has {len(labels)}.")
    return model, model_labels, labels


def detect(model, frame, confidence, labels, model_labels, motorcycle_confidence, cv2):
    """One frame -> foreground detections with motorcycle riders merged away.

    Returns (detections, background_filtered, riders_filtered).
    """
    from traffic_detections import filter_motorcycle_riders

    detections = foreground_detections(predict_frame(model, frame, confidence, cv2), labels)
    boxes, classes, scores, riders = filter_motorcycle_riders(
        detections.xyxy, detections.class_id, detections.confidence,
        model_labels, motorcycle_confidence=motorcycle_confidence)
    return (SimpleNamespace(xyxy=boxes, class_id=classes, confidence=scores),
            detections.filtered_background_count, riders)


def window_closed(cv2):
    key = cv2.waitKey(1) & 0xFF
    return key in {ord("q"), ord("Q"), 27} or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1


def run(args):
    if not args.model.is_file():
        raise FileNotFoundError(f"RF-DETR checkpoint not found: {args.model}")
    if args.output and (args.output.exists() or args.output.suffix.lower() != ".mp4"):
        raise ValueError("--output must be a new .mp4 file")
    import cv2
    import numpy as np
    import torch
    from tracking_visualization import annotate_tracking, annotate_waiting
    from vehicle_tracking import VehicleTracker

    device = resolve_device(args.device, torch)
    run_dir = args.runs_dir / (datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid4().hex[:6])
    run_dir.mkdir(parents=True, exist_ok=False)
    summary = {"started_at": datetime.now(timezone.utc).isoformat(), "model": str(args.model.resolve()),
               "backend": "rfdetr", "device": device, "precision": "float32", "confidence": args.conf,
               "source": args.endpoint, "frames": 0, "status": "loading", "headless": args.headless,
               "background_predictions_filtered": 0}
    reader = writer = None
    counts = Counter()
    timings = []
    tracking_timings = []
    tracker = None
    last_image = None
    started = time.perf_counter()
    window_created = False
    try:
        print(f"Loading RF-DETR: {args.model}; device: {device}", flush=True)
        model, model_labels, labels = load_model(args.model, device, torch)
        summary["model_classes"] = model_labels
        summary["rider_filter"] = "motorcycle box geometry"
        summary["rider_detections_filtered"] = 0
        summary["classes"] = labels
        summary["background_class_id"] = len(labels)
        tracker = None if args.no_tracking else VehicleTracker(trail_length=args.trail_length, max_missed=args.track_buffer)
        summary["tracking_enabled"] = tracker is not None
        summary["prediction_confidence"] = min(args.conf, tracker.low_confidence_threshold) if tracker else args.conf
        print("Tracking: TrackTrack; appearance matching OFF; camera motion compensation OFF" if tracker
              else "Tracking disabled: detection-only preview", flush=True)
        if device.startswith("cuda"):
            summary["gpu"] = torch.cuda.get_device_name(torch.device(device))
        print(f"Classes: {labels}; artifacts: {run_dir}", flush=True)
        last_image = np.zeros((480, 854, 3), dtype=np.uint8)
        cv2.putText(last_image, "Waiting for the video feed...", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        if not args.headless:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            window_created = True
            cv2.resizeWindow(WINDOW, 1300, 720)
            cv2.imshow(WINDOW, last_image)
            cv2.waitKey(1)
            summary["window_visible"] = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1
            print("OpenCV window ready. Press Q / Esc or close it to stop.", flush=True)
        reader = LatestFrameReader(args.endpoint, lambda jpeg: decode_jpeg(jpeg, cv2, np),
                                   require_status=args.source is None)
        summary["status"] = "waiting"
        print(f"Reading feed: {args.endpoint}", flush=True)
        waiting_since = None
        output_size = None
        while True:
            ok, frame = reader.read(timeout=0.05)
            if window_created and window_closed(cv2):
                summary["status"] = "stopped"
                break
            if ok is None:
                if waiting_since is None:
                    waiting_since = time.perf_counter()
                if window_created and time.perf_counter() - waiting_since > 1:
                    cv2.imshow(WINDOW, annotate_waiting(last_image, cv2))
                continue
            if not ok:
                if reader.error:
                    raise RuntimeError(reader.error)
                summary["status"] = "stopped" if reader.terminal_state == "stopped" else "completed"
                break
            waiting_since = None
            summary["status"] = "running"
            inference_start = time.perf_counter()
            detections, background, riders = detect(model, frame, summary["prediction_confidence"],
                                                    labels, model_labels, args.conf, cv2)
            if device.startswith("cuda"):
                torch.cuda.synchronize(torch.device(device))
            inference_ms = (time.perf_counter() - inference_start) * 1000
            timings.append(inference_ms)
            if background and not summary["background_predictions_filtered"]:
                print("Ignoring RF-DETR background/no-object predictions before tracking.", flush=True)
            summary["background_predictions_filtered"] += background
            summary["rider_detections_filtered"] += riders
            tracking_start = time.perf_counter()
            snapshot = tracker.update(detections.xyxy, detections.class_id, detections.confidence, frame) if tracker else None
            tracking_ms = (time.perf_counter() - tracking_start) * 1000 if tracker else 0.0
            if tracker:
                tracking_timings.append(tracking_ms)
            annotated = annotate_tracking(frame, detections.xyxy, detections.class_id, detections.confidence,
                                          labels, snapshot, inference_ms, tracking_ms, cv2, detector="RF-DETR",
                                          dropped_frames=getattr(reader, "dropped", 0), display_confidence=args.conf)
            # Counts remain detection observations, independent of tracker identities.
            frame_counts = Counter()
            height, width = frame.shape[:2]
            for box, cls, confidence in zip(detections.xyxy, detections.class_id, detections.confidence):
                x1, y1, x2, y2 = (int(round(float(value))) for value in box)
                if (float(confidence) >= args.conf and min(width - 1, x2) > max(0, x1)
                        and min(height - 1, y2) > max(0, y1)):
                    frame_counts[labels[int(cls)]] += 1
            counts.update(frame_counts)
            last_image = annotated
            summary["frames"] += 1
            summary["source_fps"] = reader.source_fps
            if (summary["frames"] in {1, 10}
                    or (frame_counts and not summary.get("preview_has_detections"))):
                preview = run_dir / "preview.jpg"
                if not cv2.imwrite(str(preview), annotated):
                    raise RuntimeError(f"Could not save preview: {preview}")
                summary.update({"preview": str(preview.resolve()), "preview_has_detections": bool(frame_counts)})
                print(f"Frame {summary['frames']}: {dict(frame_counts)}; {inference_ms:.1f} ms; preview: {preview}", flush=True)
            if summary["frames"] % 100 == 0:
                recent = timings[-100:]
                print(f"Processed {summary['frames']} frames; recent mean inference {sum(recent) / len(recent):.1f} ms", flush=True)
            if args.output:
                if writer is None:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    output_size = (annotated.shape[1], annotated.shape[0])
                    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), reader.source_fps, output_size)
                    if not writer.isOpened():
                        raise RuntimeError(f"Could not open output video: {args.output}")
                    summary["output"] = str(args.output.resolve())
                writer.write(cv2.resize(annotated, output_size) if (annotated.shape[1], annotated.shape[0]) != output_size else annotated)
            if window_created:
                cv2.imshow(WINDOW, annotated)
            if args.max_frames and summary["frames"] >= args.max_frames:
                summary["status"] = "frame_limit"
                break
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as exc:
        summary.update({"status": "failed", "error": safe_message(exc)})
        raise
    finally:
        if reader is not None:
            reader.close()
            summary.update(reader.stats())
            if reader.adapter_status:
                summary["adapter"] = {key: reader.adapter_status.get(key) for key in ("mode", "state", "fps", "frames")}
        if writer is not None:
            writer.release()
        if window_created:
            cv2.destroyWindow(WINDOW)
        if last_image is not None and summary["frames"]:
            latest = run_dir / "latest.jpg"
            if cv2.imwrite(str(latest), last_image):
                summary["latest_preview"] = str(latest.resolve())
        summary["tracking"] = tracker.summary() if tracker else {"enabled": False}
        summary["mean_tracking_ms"] = round(sum(tracking_timings) / len(tracking_timings), 3) if tracking_timings else None
        summary.update({"elapsed_seconds": round(time.perf_counter() - started, 3),
                        "detections_by_class": dict(counts),
                        "mean_inference_ms": round(sum(timings) / len(timings), 3) if timings else None,
                        "mean_inference_ms_after_first": round(sum(timings[1:]) / len(timings[1:]), 3) if len(timings) > 1 else None,
                        "finished_at": datetime.now(timezone.utc).isoformat()})
        summary_file = run_dir / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Processed {summary['frames']} frames; status: {summary['status']}; summary: {summary_file}", flush=True)
    return 0


def main(argv=None):
    args = parse_args(argv)
    try:
        return run(args)
    except ModuleNotFoundError as exc:
        print(f"Missing dependency: {exc.name}. Run setup.ps1 or install perception/requirements.txt and perception/rfdetr-requirements.txt using the shared .venv.", file=sys.stderr)
    except Exception as exc:
        print(f"Error: {safe_message(exc)}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
