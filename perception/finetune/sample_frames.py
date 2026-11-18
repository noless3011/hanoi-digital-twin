"""Pick frames from recorded camera videos for labelling, and pre-draw boxes with the current detector.

Frames are spread evenly over each camera's videos (so every day and time of day is represented),
taken at random moments at least --min-gap seconds apart within a video, and a frame almost identical
to one already taken from the same video (e.g. a stalled iHanoi feed) is skipped. Pre-labels come from
the current RF-DETR model; in the labelling UI you correct them rather than draw everything.

A camera is the part of a data folder name before its date: data/buoi-hht-20261106 -> buoi-hht.

    .venv\\Scripts\\python.exe perception\\finetune\\sample_frames.py --project data\\labeling\\hanoi-v1 --per-camera 100

Project layout (created, or extended when run again):
    project.json        classes and settings
    images/<id>.jpg     sampled frames, original resolution
    labels/<id>.json    boxes in pixels, status "todo" (pre-labelled) or "done"
    items.json          one row per frame: id, camera, video, time, size
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "perception"))

# Label classes, in the fixed order of the seven-class model, minus the ones not labelled here.
# "ignore" marks an area too crowded or too far to label box by box; it is masked out in training.
CLASSES = ["car", "bus", "van", "others", "motorcycle", "ignore"]
PRELABEL = {"car": "car", "bus": "bus", "van": "van", "others": "others", "motorcycle": "motorcycle"}
DUPLICATE_DIFF = 2.0     # mean absolute grey-level difference below which two frames count as the same


def camera_of(folder):
    m = re.match(r"(.+?)-\d{8}", folder.name)
    return m.group(1) if m else folder.name


def videos_by_camera(data_dir):
    out = {}
    for folder in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        vids = sorted(p for p in folder.glob("*.mp4") if not p.name.endswith(".partial.mp4"))
        if vids:
            out.setdefault(camera_of(folder), []).extend(vids)
    return out


def read_frame(cv2, path, t):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=ROOT / "data", help="Folder of recorded camera folders")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--per-camera", type=int, default=100, help="Frames to add per camera")
    parser.add_argument("--min-gap", type=float, default=15.0, help="Seconds between frames of one video")
    parser.add_argument("--cameras", nargs="*", help="Only these cameras (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-prelabel", action="store_true", help="Start every frame empty")
    parser.add_argument("--conf", type=float, default=0.35, help="Pre-label confidence threshold")
    args = parser.parse_args(argv)

    import cv2
    rng = np.random.default_rng(args.seed)
    project = args.project
    (project / "images").mkdir(parents=True, exist_ok=True)
    (project / "labels").mkdir(parents=True, exist_ok=True)
    (project / "project.json").write_text(json.dumps({"classes": CLASSES, "source": "hanoi"}, indent=1), encoding="utf-8")
    items_path = project / "items.json"
    items = json.loads(items_path.read_text(encoding="utf-8")) if items_path.exists() else []
    taken = {i["id"] for i in items}

    detector = None
    if not args.no_prelabel:
        import torch
        import run_rfdetr
        device = run_rfdetr.resolve_device("auto", torch)
        model, names, _ = run_rfdetr.load_model(run_rfdetr.DEFAULT_MODEL, device, torch)

        def detector(frame):
            d = run_rfdetr.foreground_detections(run_rfdetr.predict_frame(model, frame, args.conf, cv2), names)
            boxes = []
            for (x1, y1, x2, y2), c in zip(np.asarray(d.xyxy, float), d.class_id):
                cls = PRELABEL.get(str(names[int(c)]).lower())
                if cls:
                    boxes.append({"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1), "cls": cls})
            return boxes

    cameras = videos_by_camera(args.data)
    if args.cameras:
        cameras = {c: v for c, v in cameras.items() if c in args.cameras}
    for camera, videos in cameras.items():
        added, attempts = 0, 0
        # Round-robin over videos in random order so frames spread over every recording.
        order = list(rng.permutation(len(videos)))
        used = {}   # video -> list of (time, small grey frame)
        while added < args.per_camera and attempts < args.per_camera * 20:
            attempts += 1
            video = videos[order[attempts % len(order)]]
            cap = cv2.VideoCapture(str(video))
            duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
            cap.release()
            t = float(rng.uniform(0.5, max(duration - 0.5, 1.0)))
            if any(abs(t - u) < args.min_gap for u, _ in used.get(video, [])):
                continue
            frame = read_frame(cv2, video, t)
            if frame is None:
                continue
            small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (96, 54)).astype(float)
            if any(np.mean(np.abs(small - s)) < DUPLICATE_DIFF for _, s in used.get(video, [])):
                continue
            item_id = hashlib.sha1(f"{video.parent.name}/{video.name}@{t:.2f}".encode()).hexdigest()[:12]
            if item_id in taken:
                continue
            used.setdefault(video, []).append((t, small))
            cv2.imwrite(str(project / "images" / f"{item_id}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            boxes = detector(frame) if detector else []
            (project / "labels" / f"{item_id}.json").write_text(
                json.dumps({"status": "todo", "prelabelled": detector is not None, "boxes": boxes}), encoding="utf-8")
            items.append({"id": item_id, "camera": camera, "video": f"{video.parent.name}/{video.name}",
                          "time_s": round(t, 2), "width": frame.shape[1], "height": frame.shape[0]})
            taken.add(item_id)
            added += 1
        print(f"{camera}: {added} frames from {len(used)} of {len(videos)} videos", flush=True)
    items_path.write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"{len(items)} frames in {project}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
