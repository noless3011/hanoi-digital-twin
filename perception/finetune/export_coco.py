"""Export fine-tuning data as COCO detection folders in the seven-class layout of perception/training.

  hanoi   frames labelled in the local labeller (status "done") -> train/valid/test, split by video so
          neighbouring frames of one recording never land in two splits
  yolo    a YOLO-format dataset (e.g. Kaggle vehicle-danang-2025) -> the same layout, keeping its splits

Every image carries the partial-label metadata the training adapter (helpers/partial_labels.py) uses:
which classes its labels cover (known_class_ids) and whether its "car" also covers vans (car_or_van).
"ignore" boxes are blackened in the exported image and listed in ignored_regions_xywh; labelled boxes
at least half inside them are dropped, as for UA-DETRAC.

    .venv\\Scripts\\python.exe perception\\finetune\\export_coco.py hanoi --project data\\labeling\\hanoi-v1 --out perception\\models\\hanoi_coco
    .venv\\Scripts\\python.exe perception\\finetune\\export_coco.py yolo --src data\\external\\vehicle-danang-2025 --out perception\\models\\danang_coco
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

# Fixed class order of the seven-class model; COCO category id = position + 1.
CLASS_NAMES = ("car", "bus", "van", "others", "person", "motorcycle", "traffic light")
CAT = {name: i + 1 for i, name in enumerate(CLASS_NAMES)}
CATEGORIES = [{"id": CAT[n], "name": n, "supercategory": "traffic"} for n in CLASS_NAMES]
SPLITS = ("train", "valid", "test")
IGNORE_DROP = 0.5            # drop a box when this share of it lies in ignored regions
ID_BASE = {"hanoi": 50_000_000, "danang": 60_000_000}

# YOLO class names (as in the dataset's data.yaml) -> seven-class name.
YOLO_MAP = {"car": "car", "motorbike": "motorcycle", "motorcycle": "motorcycle", "motor": "motorcycle",
            "bus": "bus", "truck": "others", "van": "van", "person": "person", "pedestrian": "person",
            "bicycle": None, "traffic light": "traffic light"}
YOLO_DEFAULT_NAMES = ["car", "motorbike", "bus", "truck"]   # vehicle-danang-2025 / huancsw YOLO26n order


def ignored_share(box, regions):
    """Share of a box covered by the union of ignored rectangles (pixel raster, exact for small boxes)."""
    x1, y1, x2, y2 = box
    area = (x2 - x1) * (y2 - y1)
    if area <= 0 or not regions:
        return 0.0
    gx1, gy1, gx2, gy2 = math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2)
    covered = 0
    for py in range(gy1, gy2):
        for px in range(gx1, gx2):
            cx, cy = px + 0.5, py + 0.5
            if x1 <= cx < x2 and y1 <= cy < y2 and any(rx1 <= cx < rx2 and ry1 <= cy < ry2 for rx1, ry1, rx2, ry2 in regions):
                covered += 1
    pixels = sum(1 for py in range(gy1, gy2) for px in range(gx1, gx2) if x1 <= px + 0.5 < x2 and y1 <= py + 0.5 < y2)
    return covered / max(pixels, 1)


def write_image(src, dst, regions):
    if not regions:
        shutil.copyfile(src, dst)
        return
    from PIL import Image, ImageDraw
    im = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(im)
    for x1, y1, x2, y2 in regions:
        draw.rectangle([math.floor(x1), math.floor(y1), math.ceil(x2) - 1, math.ceil(y2) - 1], fill=(0, 0, 0))
    im.save(dst, quality=95)


class Writer:
    def __init__(self, out: Path, source: str, known: list, car_or_van: bool, description: str):
        if out.exists() and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not empty; choose a new --out")
        self.out, self.source, self.known, self.car_or_van = out, source, sorted(CAT[k] for k in known), car_or_van
        self.docs = {s: {"info": {"description": description, "version": "1"}, "licenses": [], "categories": CATEGORIES,
                         "images": [], "annotations": []} for s in SPLITS}
        self.next_image, self.next_ann = ID_BASE[source], ID_BASE[source]
        self.dropped = Counter()
        for s in SPLITS:
            (out / s).mkdir(parents=True, exist_ok=True)

    def add(self, split, src: Path, width, height, boxes, regions=(), extra=None):
        """boxes: [(x1, y1, x2, y2, seven-class name)] in pixels; regions: ignored [(x1, y1, x2, y2)]."""
        self.next_image += 1
        image_id = self.next_image
        name = f"{self.source}_{image_id}.jpg"
        write_image(src, self.out / split / name, list(regions))
        doc = self.docs[split]
        doc["images"].append({"id": image_id, "file_name": name, "width": width, "height": height,
                              "source": self.source, "known_class_ids": self.known, "car_or_van": self.car_or_van,
                              "ignored_regions_xywh": [[r[0], r[1], r[2] - r[0], r[3] - r[1]] for r in regions],
                              **(extra or {})})
        for x1, y1, x2, y2, cls in boxes:
            x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
            y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
            if x2 - x1 < 1 or y2 - y1 < 1:
                self.dropped["too small"] += 1
                continue
            if regions and ignored_share((x1, y1, x2, y2), regions) >= IGNORE_DROP:
                self.dropped["in ignored region"] += 1
                continue
            self.next_ann += 1
            doc["annotations"].append({"id": self.next_ann, "image_id": image_id, "category_id": CAT[cls],
                                       "bbox": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                                       "area": round((x2 - x1) * (y2 - y1), 2), "iscrowd": 0})

    def finish(self, extra_manifest=None):
        counts = {}
        for s, doc in self.docs.items():
            (self.out / s / "_annotations.coco.json").write_text(json.dumps(doc), encoding="utf-8")
            by_cls = Counter(CLASS_NAMES[a["category_id"] - 1] for a in doc["annotations"])
            counts[s] = {"images": len(doc["images"]), "annotations": len(doc["annotations"]), "per_class": dict(by_cls)}
        manifest = {"schema": "finetune-coco", "source": self.source, "class_names": list(CLASS_NAMES),
                    "known_class_names": [CLASS_NAMES[i - 1] for i in self.known], "car_or_van": self.car_or_van,
                    "ignored_region_policy": f"black mask; boxes with >= {IGNORE_DROP} of their area inside are dropped",
                    "counts": counts, "dropped_boxes": dict(self.dropped), **(extra_manifest or {})}
        (self.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
        for s in SPLITS:
            c = counts[s]
            print(f"  {s:<5} {c['images']:>6} images {c['annotations']:>7} boxes  {c['per_class']}")
        if self.dropped:
            print(f"  dropped boxes: {dict(self.dropped)}")
        print(f"Wrote {self.out}")


# ------------------------------------------------------------------ Hanoi labelling project

def split_videos(items, valid, test, seed):
    """Per camera, assign whole videos to splits by a stable hash, so each camera is in every split."""
    by_camera = {}
    for it in items:
        by_camera.setdefault(it["camera"], set()).add(it["video"])
    out = {}
    for camera, videos in by_camera.items():
        ordered = sorted(videos, key=lambda v: hashlib.sha1(f"{seed}:{v}".encode()).hexdigest())
        n = len(ordered)
        n_test = max(1, round(n * test)) if n >= 3 else 0
        n_valid = max(1, round(n * valid)) if n >= 3 else 0
        for k, v in enumerate(ordered):
            out[v] = "test" if k < n_test else "valid" if k < n_test + n_valid else "train"
    return out


def export_hanoi(args):
    project = args.project
    meta = json.loads((project / "project.json").read_text(encoding="utf-8"))
    items = json.loads((project / "items.json").read_text(encoding="utf-8"))
    done = []
    for it in items:
        lab_path = project / "labels" / f"{it['id']}.json"
        lab = json.loads(lab_path.read_text(encoding="utf-8")) if lab_path.exists() else {"status": "todo"}
        if lab["status"] == "done":
            done.append((it, lab))
    if not done:
        raise SystemExit("No frames marked done yet")
    labelled = [c for c in meta["classes"] if c in CAT]
    w = Writer(args.out, "hanoi", labelled, car_or_van=False,
               description="Hanoi traffic camera frames labelled locally")
    split_of = split_videos([it for it, _ in done], args.valid, args.test, args.seed)
    for it, lab in done:
        regions = [(b["x1"], b["y1"], b["x2"], b["y2"]) for b in lab["boxes"] if b["cls"] == "ignore"]
        boxes = [(b["x1"], b["y1"], b["x2"], b["y2"], b["cls"]) for b in lab["boxes"] if b["cls"] in CAT]
        w.add(split_of[it["video"]], project / "images" / f"{it['id']}.jpg", it["width"], it["height"], boxes, regions,
              {"camera": it["camera"], "video": it["video"], "time_s": it["time_s"]})
    print(f"{len(done)} done frames of {len(items)} ({len({it['video'] for it, _ in done})} videos)")
    w.finish({"split_policy": "whole videos per split, per camera", "videos": split_of})


# ------------------------------------------------------------------ YOLO-format dataset

SPLIT_ALIASES = {"train": "train", "training": "train", "val": "valid", "valid": "valid", "validation": "valid",
                 "test": "test", "testing": "test"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def yolo_names(src: Path):
    """Class names from data.yaml (list or mapping form) without a YAML dependency."""
    for yaml in list(src.rglob("data.yaml")) + list(src.rglob("*.yaml")):
        text = yaml.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"^names\s*:\s*\[(.*?)\]", text, re.M | re.S)
        if m:
            return [n.strip().strip("'\"") for n in m.group(1).split(",") if n.strip()]
        m = re.search(r"^names\s*:\s*\n((?:[ \t]+.*\n?)+)", text, re.M)
        if m:
            names = {}
            for line in m.group(1).splitlines():
                mm = re.match(r"\s*(?:-\s*(.+)|(\d+)\s*:\s*(.+))\s*$", line)
                if mm and mm.group(1):
                    names[len(names)] = mm.group(1).strip().strip("'\"")
                elif mm:
                    names[int(mm.group(2))] = mm.group(3).strip().strip("'\"")
            if names:
                return [names[k] for k in sorted(names)]
    return None


def yolo_pairs(src: Path):
    """{split: [(image, label file)]}, for the layouts <split>/images + <split>/labels and
    images/<split> + labels/<split>."""
    out = {}
    for images_dir in src.rglob("images"):
        if not images_dir.is_dir():
            continue
        candidates = [(images_dir.parent.name, images_dir, images_dir.parent / "labels")]
        candidates += [(d.name, d, images_dir.parent / "labels" / d.name) for d in images_dir.iterdir() if d.is_dir()]
        for name, img_dir, lab_dir in candidates:
            split = SPLIT_ALIASES.get(name.lower())
            if split and lab_dir.is_dir():
                for img in sorted(img_dir.iterdir()):
                    if img.suffix.lower() in IMAGE_EXT:
                        out.setdefault(split, []).append((img, lab_dir / (img.stem + ".txt")))
    return out


def export_yolo(args):
    from PIL import Image
    names = yolo_names(args.src) or YOLO_DEFAULT_NAMES
    mapping = {i: YOLO_MAP.get(n.lower()) for i, n in enumerate(names)}
    unknown = [n for n in names if n.lower() not in YOLO_MAP]
    if unknown:
        raise SystemExit(f"No mapping for YOLO classes {unknown}; add them to YOLO_MAP")
    pairs = yolo_pairs(args.src)
    if not pairs:
        raise SystemExit(f"No <split>/images + labels folders found under {args.src}")
    known = sorted({m for m in mapping.values() if m})
    print(f"classes {names} -> {[mapping[i] for i in range(len(names))]}")
    # Datasets that label vans as "car" supervise car-or-van, as BDD100K does.
    w = Writer(args.out, args.source, known, car_or_van="van" not in known,
               description=f"{args.src.name} converted from YOLO format")
    for split in SPLITS:
        rows = pairs.get(split, [])
        if args.limit:
            rows = rows[: args.limit]
        for k, (img, lab) in enumerate(rows):
            with Image.open(img) as im:
                width, height = im.size
            boxes = []
            if lab.exists():
                for line in lab.read_text(encoding="utf-8").split("\n"):
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    cls = mapping.get(int(float(parts[0])))
                    if cls is None:
                        continue
                    cx, cy, bw, bh = (float(p) for p in parts[1:5])
                    boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width,
                                  (cy + bh / 2) * height, cls))
            w.add(split, img, width, height, boxes, extra={"original_file": str(img.relative_to(args.src))})
            if (k + 1) % 2000 == 0:
                print(f"  {split}: {k + 1}/{len(rows)}", flush=True)
    w.finish({"yolo_class_names": names})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hanoi", help="Export the local labelling project")
    h.add_argument("--project", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    h.add_argument("--valid", type=float, default=0.15, help="Share of each camera's videos for validation")
    h.add_argument("--test", type=float, default=0.15, help="Share of each camera's videos for testing")
    h.add_argument("--seed", type=int, default=0)
    y = sub.add_parser("yolo", help="Convert a YOLO-format dataset")
    y.add_argument("--src", type=Path, required=True)
    y.add_argument("--out", type=Path, required=True)
    y.add_argument("--source", default="danang", choices=sorted(ID_BASE), help="Source tag stored per image")
    y.add_argument("--limit", type=int, default=0, help="At most this many images per split (0: all)")
    args = parser.parse_args(argv)
    (export_hanoi if args.cmd == "hanoi" else export_yolo)(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
