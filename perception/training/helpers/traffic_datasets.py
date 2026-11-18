"""Portable COCO 2017 / BDD100K detection preparation for RF-DETR.

Only Pillow and the Python standard library are required. Helpers are embedded
in the Colab notebook, so their private names are deliberately namespaced.
Source validation partitions provide custom, disjoint valid/test holdouts;
these are not either dataset's official test benchmark.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import random
import shutil
import stat
import tarfile
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image


TRAFFIC_CLASS_NAMES = ["car", "bus", "van", "others", "person", "motorcycle", "traffic light"]
TRAFFIC_CATEGORIES = [
    {"id": index + 1, "name": name,
     "supercategory": "vehicle" if index < 4 or name == "motorcycle" else "object"}
    for index, name in enumerate(TRAFFIC_CLASS_NAMES)
]
TRAFFIC_CLASS_IDS = {name: index + 1 for index, name in enumerate(TRAFFIC_CLASS_NAMES)}
TRAFFIC_FORMAT_VERSION = 2
TRAFFIC_SOURCE_METADATA = {
    "coco": {
        "name": "COCO 2017 object detection",
        "download_page": "https://cocodataset.org/#download",
        "license_page": "https://cocodataset.org/#termsofuse",
        "annotation_license": "Creative Commons Attribution 4.0; images retain their individual source licenses",
        "known_class_ids": [1, 2, 4, 5, 6, 7],
        "mapping": {"car": "car", "bus": "bus", "truck": "others", "person": "person",
                    "motorcycle": "motorcycle", "traffic light": "traffic light"},
    },
    "bdd100k": {
        "name": "BDD100K object detection",
        "download_page": "https://github.com/bdd100k/bdd100k/blob/master/doc/source/download.rst",
        "license_page": "https://bdd-data.berkeley.edu/license.html",
        "license": "BDD100K dataset license; review source terms before use or redistribution",
        "known_class_ids": [1, 2, 4, 5, 6, 7],
        "mapping": {"car": "car", "bus": "bus", "truck": "others", "pedestrian": "person",
                    "person": "person", "rider": "person", "motorcycle": "motorcycle",
                    "motor": "motorcycle", "traffic light": "traffic light"},
    },
    "ua_detrac": {
        "name": "UA-DETRAC TRAIN",
        "download_page": "https://sites.google.com/view/daweidu/projects/ua-detrac",
        "license": "Source UA-DETRAC research terms; retain original provenance and review usage terms",
        "known_class_ids": [1, 2, 3, 4],
    },
}


def _traffic_json(path):
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _traffic_integer(value, context, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{context}: expected an integer >= {minimum}")
    return value


def _traffic_safe_relative(name):
    """Reject portable path traversal, drive paths, and ZIP separator tricks."""
    if not isinstance(name, str) or not name or "\\" in name or ":" in name or "\x00" in name:
        raise ValueError(f"Unsafe source filename: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in {".", ".."} for part in name.split("/")):
        raise ValueError(f"Unsafe source filename: {name!r}")
    return path


def _traffic_safe_image_name(name):
    path = _traffic_safe_relative(name)
    if len(path.parts) != 1 or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise ValueError(f"Expected a plain image filename, got {name!r}")
    return name


def _traffic_clip_box(box, width, height, context):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{context}: bbox must have four coordinates")
    try:
        x, y, w, h = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context}: malformed bbox") from exc
    if not all(math.isfinite(value) for value in (x, y, w, h)) or w <= 0 or h <= 0:
        raise ValueError(f"{context}: bbox must be finite with positive width/height")
    left, top = max(0.0, x), max(0.0, y)
    right, bottom = min(float(width), x + w), min(float(height), y + h)
    if right <= left or bottom <= top:
        raise ValueError(f"{context}: bbox is outside the image")
    return [left, top, right - left, bottom - top]


def _traffic_record(source, split, source_id, name, width, height):
    return {
        "key": f"{source}:{split}:{name}",
        "source": source, "source_split": split, "source_image_id": source_id,
        "source_file_name": name, "file_name": name,
        "width": width, "height": height, "annotations": [],
        "known_class_ids": list(TRAFFIC_SOURCE_METADATA[source]["known_class_ids"]),
        "car_or_van": True,
    }


def load_coco_records(train_json, val_json):
    """Read official instance JSONs, remap target labels, and retain licenses.

    Entire images with target crowd/ignore annotations are excluded. Dropping
    only those boxes would incorrectly make visible target objects background.
    Malformed target annotations fail rather than silently corrupt supervision.
    Images with no target annotations are omitted by the subsequent sampler.
    """
    records = []
    seen_names = set()
    for split, path in (("train", train_json), ("val", val_json)):
        data = _traffic_json(path)
        if not isinstance(data, dict) or not all(key in data for key in ("images", "annotations", "categories")):
            raise ValueError(f"{path}: expected COCO images, annotations, and categories")
        category_map, category_ids = {}, set()
        aliases = TRAFFIC_SOURCE_METADATA["coco"]["mapping"]
        for category in data["categories"]:
            category_id = _traffic_integer(category.get("id"), "COCO category ID")
            if category_id in category_ids:
                raise ValueError(f"Duplicate COCO category ID {category_id}")
            category_ids.add(category_id)
            if category.get("name") in aliases:
                category_map[category_id] = TRAFFIC_CLASS_IDS[aliases[category["name"]]]
        licenses = {item["id"]: item for item in data.get("licenses", [])}
        images = {}
        for image in data["images"]:
            image_id = _traffic_integer(image.get("id"), "COCO image ID")
            name = _traffic_safe_image_name(image.get("file_name"))
            if image_id in images or name in seen_names:
                raise ValueError(f"Duplicate COCO image ID or filename: {image_id}, {name}")
            seen_names.add(name)
            width = _traffic_integer(image.get("width"), f"{name} width", 1)
            height = _traffic_integer(image.get("height"), f"{name} height", 1)
            record = _traffic_record("coco", split, image_id, name, width, height)
            record["source_license"] = licenses.get(image.get("license"), {"id": image.get("license")})
            record["source_urls"] = {key: image[key] for key in ("coco_url", "flickr_url") if image.get(key)}
            images[image_id] = record
        excluded, annotation_ids = set(), set()
        for annotation in data["annotations"]:
            annotation_id = _traffic_integer(annotation.get("id"), "COCO annotation ID")
            if annotation_id in annotation_ids:
                raise ValueError(f"Duplicate COCO annotation ID {annotation_id}")
            annotation_ids.add(annotation_id)
            image_id, category_id = annotation.get("image_id"), annotation.get("category_id")
            if image_id not in images or category_id not in category_ids:
                raise ValueError(f"COCO annotation {annotation_id}: unknown image/category")
            if category_id not in category_map:
                continue
            if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
                excluded.add(image_id)
                continue
            record = images[image_id]
            box = _traffic_clip_box(annotation.get("bbox"), record["width"], record["height"],
                                    f"COCO annotation {annotation_id}")
            record["annotations"].append({"category_id": category_map[category_id], "bbox": box,
                                          "source_annotation_id": annotation_id,
                                          "source_category_id": category_id})
        records.extend(record for image_id, record in images.items() if image_id not in excluded)
    return sorted(records, key=lambda record: record["key"])


def resolve_bdd_annotation_paths(root):
    """Find native aggregate labels or normalized Supervisely split folders."""
    root = Path(root).expanduser().resolve()
    native_pairs = [(root / "det_train.json", root / "det_val.json"),
                    (root / "bdd100k_labels_images_train.json", root / "bdd100k_labels_images_val.json")]
    supervisely = (root / "train" / "ann", root / "val" / "ann")
    native_found = [pair for pair in native_pairs if all(path.is_file() for path in pair)]
    supervisely_found = all(path.is_dir() for path in supervisely)
    if len(native_found) + int(supervisely_found) != 1:
        raise ValueError(f"{root}: expected exactly one BDD annotation format: det_train/val.json or train/ann + val/ann")
    return native_found[0] if native_found else supervisely


def _traffic_bdd_tags(tags, context):
    if not isinstance(tags, list):
        raise ValueError(f"{context}: tags must be a list")
    attributes = {}
    for tag in tags:
        if not isinstance(tag, dict) or not isinstance(tag.get("name"), str):
            raise ValueError(f"{context}: malformed tag")
        name, value = tag["name"], tag.get("value")
        if name == "attributes":
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    try:
                        value = ast.literal_eval(value)
                    except (ValueError, SyntaxError) as exc:
                        raise ValueError(f"{context}: malformed attributes tag") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{context}: attributes tag must contain a dictionary")
            # Validate JSON compatibility before retaining source metadata.
            try:
                json.dumps(value, allow_nan=False)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{context}: attributes must contain JSON-compatible values") from exc
            attributes.update(value)
        else:
            attributes[name] = value
    return attributes


def _traffic_load_bdd_supervisely(annotation_dir, split, seen_names, aliases):
    root = Path(annotation_dir).expanduser().resolve()
    paths = sorted(root.glob("*.json"))
    if not paths:
        raise ValueError(f"{root}: no per-image Supervisely JSON files")
    meta_path = root.parent.parent / "meta.json"
    class_names = {}
    if meta_path.is_file():
        for item in _traffic_json(meta_path).get("classes", []):
            if item.get("id") is not None:
                class_names[item["id"]] = item.get("title")
    records = []
    for path in paths:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Annotation escapes its folder through a symlink: {path}")
        name = _traffic_safe_image_name(path.name[:-5])
        if name in seen_names:
            raise ValueError(f"Duplicate BDD image filename across partitions: {name}")
        seen_names.add(name)
        data = _traffic_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("objects"), list):
            raise ValueError(f"{path}: expected Supervisely objects list")
        size = data.get("size") or {}
        width = _traffic_integer(size.get("width"), f"{name} width", 1)
        height = _traffic_integer(size.get("height"), f"{name} height", 1)
        record = _traffic_record("bdd100k", split, name, name, width, height)
        record["source_annotation_format"] = "supervisely"
        record["source_attributes"] = _traffic_bdd_tags(data.get("tags", []), name)
        excluded, annotation_ids = False, set()
        for label in data["objects"]:
            if not isinstance(label, dict):
                raise ValueError(f"{name}: malformed Supervisely object")
            category = label.get("classTitle")
            if label.get("classId") in class_names and class_names[label["classId"]] != category:
                raise ValueError(f"{name}: classId does not match classTitle")
            if category not in aliases:
                continue
            label_id = label.get("id")
            if label_id is None or str(label_id) in annotation_ids:
                raise ValueError(f"{name}: missing or duplicate target annotation ID")
            annotation_ids.add(str(label_id))
            attributes = _traffic_bdd_tags(label.get("tags", []), f"{name} object {label_id}")
            if any(attributes.get(key, False) for key in ("crowd", "ignore", "ignored")):
                excluded = True
                continue
            if label.get("geometryType") != "rectangle":
                raise ValueError(f"{name} object {label_id}: target category requires rectangle geometry")
            points = label.get("points") or {}
            exterior = points.get("exterior")
            try:
                if len(exterior) != 2 or any(len(point) != 2 for point in exterior):
                    raise ValueError("Expected two rectangle corners")
                (x1, y1), (x2, y2) = [[float(value) for value in point] for point in exterior]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} object {label_id}: malformed rectangle corners") from exc
            if x2 < x1 or y2 < y1:
                raise ValueError(f"{name} object {label_id}: reversed rectangle corners")
            box = _traffic_clip_box([x1, y1, x2 - x1 + 1, y2 - y1 + 1], width, height,
                                    f"{name} object {label_id}")
            record["annotations"].append({"category_id": TRAFFIC_CLASS_IDS[aliases[category]],
                                          "bbox": box, "source_annotation_id": label_id,
                                          "source_category": category, "source_attributes": attributes})
        if not excluded:
            records.append(record)
    return records


def load_bdd_records(train_json, val_json, *, image_width=1280, image_height=720):
    """Read native Scalabel JSON or Supervisely train/ann and val/ann folders.

    Legacy category aliases ``person``/``motor`` are accepted. ``pedestrian``
    and ``rider`` map to person; truck maps to others. BDD's inclusive maximum
    box endpoints become COCO widths/heights using +1, as in Scalabel's official
    ``box2d_to_bbox`` converter. The native 100K keyframes are 1280x720.
    """
    image_width = _traffic_integer(image_width, "BDD default width", 1)
    image_height = _traffic_integer(image_height, "BDD default height", 1)
    aliases = dict(TRAFFIC_SOURCE_METADATA["bdd100k"]["mapping"])
    aliases.update({"person": "person", "motor": "motorcycle"})
    records, seen_names = [], set()
    for split, path in (("train", train_json), ("val", val_json)):
        if Path(path).expanduser().is_dir():
            records.extend(_traffic_load_bdd_supervisely(path, split, seen_names, aliases))
            continue
        data = _traffic_json(path)
        frames = data.get("frames") if isinstance(data, dict) else data
        if not isinstance(frames, list):
            raise ValueError(f"{path}: expected a Scalabel frame list or frames object")
        for frame in frames:
            name = _traffic_safe_image_name(frame.get("name"))
            if name in seen_names:
                raise ValueError(f"Duplicate BDD image filename across partitions: {name}")
            seen_names.add(name)
            size = frame.get("size") or {}
            width = _traffic_integer(size.get("width", image_width), f"{name} width", 1)
            height = _traffic_integer(size.get("height", image_height), f"{name} height", 1)
            record = _traffic_record("bdd100k", split, name, name, width, height)
            record["source_annotation_format"] = "scalabel"
            record["source_attributes"] = frame.get("attributes") or {}
            labels = frame.get("labels")
            if labels is None:
                raise ValueError(f"{name}: labels missing; provide annotated detection train/val files")
            if not isinstance(labels, list):
                raise ValueError(f"{name}: labels must be a list")
            excluded, annotation_ids = False, set()
            for label in labels:
                category = label.get("category")
                if category not in aliases:
                    continue
                label_id = label.get("id")
                if label_id is None or str(label_id) in annotation_ids:
                    raise ValueError(f"{name}: missing or duplicate target annotation ID")
                annotation_ids.add(str(label_id))
                attributes = label.get("attributes") or {}
                if any(attributes.get(key, False) for key in ("crowd", "ignore", "ignored")) or label.get("ignore", False):
                    excluded = True
                    continue
                box2d = label.get("box2d")
                if not isinstance(box2d, dict):
                    raise ValueError(f"{name} label {label_id}: missing box2d")
                try:
                    x1, y1, x2, y2 = [float(box2d[key]) for key in ("x1", "y1", "x2", "y2")]
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"{name} label {label_id}: malformed box2d") from exc
                if x2 < x1 or y2 < y1:
                    raise ValueError(f"{name} label {label_id}: reversed box2d")
                box = _traffic_clip_box([x1, y1, x2 - x1 + 1, y2 - y1 + 1], width, height,
                                        f"{name} label {label_id}")
                record["annotations"].append({"category_id": TRAFFIC_CLASS_IDS[aliases[category]],
                                              "bbox": box, "source_annotation_id": label_id,
                                              "source_category": category,
                                              "source_attributes": attributes})
            if not excluded:
                records.append(record)
    return sorted(records, key=lambda record: record["key"])


def _traffic_sample_balanced(records, cap, seed, balanced_classes):
    ordered = sorted(records, key=lambda record: record["key"])
    random.Random(seed).shuffle(ordered)
    if cap is None or cap >= len(ordered):
        return sorted(ordered, key=lambda record: record["key"])
    cap = _traffic_integer(cap, "image cap", 1)
    target_ids = [TRAFFIC_CLASS_IDS[name] for name in balanced_classes]
    pools = {category_id: [record for record in ordered
                          if any(item["category_id"] == category_id for item in record["annotations"])]
             for category_id in target_ids}
    selected, keys = [], set()
    # Reserve equal per-class image quotas before filling with other targets.
    # Multi-class images satisfy several quotas without duplicating the image.
    quota = max(1, cap // max(1, len(target_ids)))
    coverage = Counter()
    offsets = Counter()
    while len(selected) < cap:
        advanced = False
        for category_id in sorted(target_ids, key=lambda item: (len(pools[item]), item)):
            if coverage[category_id] >= quota:
                continue
            pool = pools[category_id]
            while offsets[category_id] < len(pool):
                record = pool[offsets[category_id]]
                offsets[category_id] += 1
                if record["key"] in keys:
                    continue
                keys.add(record["key"])
                selected.append(record)
                for present in {item["category_id"] for item in record["annotations"]}:
                    coverage[present] += 1
                advanced = True
                break
            if len(selected) >= cap:
                break
        if not advanced:
            break
    for record in ordered:
        if len(selected) >= cap:
            break
        if record["key"] not in keys:
            selected.append(record)
            keys.add(record["key"])
    return sorted(selected, key=lambda record: record["key"])


def select_source_records(records, *, train_cap=None, valid_cap=None, test_cap=None, seed=42,
                          balanced_classes=("person", "motorcycle", "traffic light")):
    """Deterministically cap images, balancing the three added target classes.

    All boxes for every selected image are retained. Official training images
    only enter train. Official val images are shuffled then split in half into
    distinct candidate valid/test pools before either cap is applied.
    ``None`` or 0 means uncapped; positive caps are per source and output split.
    """
    _traffic_integer(seed, "seed")
    for cap in (train_cap, valid_cap, test_cap):
        if cap is not None:
            _traffic_integer(cap, "image cap")
    train_cap, valid_cap, test_cap = train_cap or None, valid_cap or None, test_cap or None
    if len(set(balanced_classes)) != len(balanced_classes) or any(name not in TRAFFIC_CLASS_IDS for name in balanced_classes):
        raise ValueError("balanced_classes must contain distinct target class names")
    keys, train, holdout = set(), [], []
    for record in records:
        if record["key"] in keys:
            raise ValueError(f"Duplicate source image {record['key']}")
        keys.add(record["key"])
        if record["source_split"] not in {"train", "val"}:
            raise ValueError("Only official train/val source partitions are supported")
        if record["annotations"]:
            (train if record["source_split"] == "train" else holdout).append(record)
    if not train or len(holdout) < 2:
        raise ValueError("Source requires target boxes in train and at least 2 distinct validation images")
    holdout.sort(key=lambda record: record["key"])
    random.Random(seed).shuffle(holdout)
    midpoint = (len(holdout) + 1) // 2
    return {
        "train": _traffic_sample_balanced(train, train_cap, seed + 1, balanced_classes),
        "valid": _traffic_sample_balanced(holdout[:midpoint], valid_cap, seed + 2, balanced_classes),
        "test": _traffic_sample_balanced(holdout[midpoint:], test_cap, seed + 3, balanced_classes),
    }


def _traffic_flatten(records):
    return [record for split in ("train", "valid", "test") for record in records.get(split, [])] if isinstance(records, dict) else list(records)


def _traffic_signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def _traffic_image_path(root, record):
    source = record.get("source")
    if source not in {"coco", "bdd100k"} or record.get("source_split") not in {"train", "val"}:
        raise ValueError("Unexpected source or source split")
    root = Path(root).resolve()
    result = root / source / record["source_split"] / _traffic_safe_image_name(record["source_file_name"])
    if not result.resolve().is_relative_to(root):
        raise ValueError(f"Staged image escapes its root through a symlink: {result}")
    return result


def _traffic_source_path(value, split):
    return value.get(split) if isinstance(value, dict) else value


def _traffic_check_image(path, record):
    try:
        with Image.open(path) as image:
            if image.size != (record["width"], record["height"]):
                raise ValueError(f"Image/annotation dimensions differ for {path}: {image.size} vs {(record['width'], record['height'])}")
            image.verify()
    except OSError as exc:
        raise ValueError(f"Unreadable image {path}: {exc}") from exc


def _traffic_source_identity(value):
    if isinstance(value, dict):
        return {split: _traffic_source_identity(path) for split, path in sorted(value.items())}
    if value is None:
        return None
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image source does not exist: {path}")
    status = path.stat()
    return {"path": str(path), "size": status.st_size, "mtime_ns": status.st_mtime_ns,
            "kind": "folder" if path.is_dir() else "file"}


def _traffic_download_image(url, target, *, progress=None):
    """Retry brief transient network failures without disabling TLS validation."""
    for attempt in range(1, 4):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "traffic-training-colab/1"})
            with urllib.request.urlopen(request, timeout=90) as response, target.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return
        except urllib.error.HTTPError as error:
            if error.code != 429 and not 500 <= error.code < 600:
                raise
            failure = error
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            failure = error
        if attempt == 3:
            raise failure
        if progress:
            progress(f"Transient COCO download failure; retry {attempt + 1}/3 for {Path(urllib.parse.urlparse(url).path).name}")
        if target.exists():
            target.unlink()
        time.sleep(attempt)


def _traffic_archive_member_path(name, is_directory):
    # TAR writers commonly include an empty root or ./ prefix.
    if is_directory and name in {"", ".", "./"}:
        return None
    if not isinstance(name, str) or "\\" in name or ":" in name or "\x00" in name or name.startswith("/"):
        raise ValueError(f"Unsafe archive filename: {name!r}")
    parts = name.rstrip("/").split("/")
    if ".." in parts:
        raise ValueError(f"Unsafe archive filename: {name!r}")
    return _traffic_safe_relative("/".join(part for part in parts if part != "."))


def stage_selected_images(records, output_root, *, images_root=None, images_archive=None, images_zip=None,
                          allow_coco_download=False, progress=None):
    """Stage selected images from folders, ZIP/TAR archives, or COCO HTTP.

    Folder/archive arguments can be one path or ``{'train': path, 'val': path}``.
    TAR compression is detected automatically. ``images_zip`` is a legacy alias.
    Wrapper directories in archives are allowed; duplicate candidate files,
    unsafe archive members, symlinks, and missing selected images are refused.
    COCO download uses the official images.cocodataset.org public S3 bucket
    through its working HTTPS path endpoint, retaining TLS verification. BDD
    archives may be downloaded by the notebook's URL/Drive staging cell first.
    Reusing a source stage requires identical selected record metadata and
    source identity (resolved path, size, mtime). Completed image files are
    reused after an interrupted download using the saved staging signature.
    """
    if images_zip is not None:
        if images_archive is not None:
            raise ValueError("Choose images_archive or its legacy images_zip alias")
        images_archive = images_zip
    if images_root is not None and images_archive is not None:
        raise ValueError("Choose image folders or archives, not both")
    records = sorted(_traffic_flatten(records), key=lambda record: record["key"])
    if not records or len({record["key"] for record in records}) != len(records):
        raise ValueError("Selected images must be nonempty and have unique keys")
    root = Path(output_root).expanduser().resolve()
    by_source = {}
    for record in records:
        _traffic_image_path(root, record)
        by_source.setdefault(record["source"], []).append(record)
    identity = {"images_root": _traffic_source_identity(images_root),
                "images_archive": _traffic_source_identity(images_archive),
                "allow_coco_download": bool(allow_coco_download)}
    signatures = {source: _traffic_signature({"records": source_records, "source_identity": identity})
                  for source, source_records in by_source.items()}
    for source, signature in signatures.items():
        marker = root / f"_stage_{source}.json"
        source_dir = root / source
        if marker.is_file():
            if _traffic_json(marker).get("signature") != signature:
                raise ValueError(f"{source}: selected images changed; choose a fresh staging directory")
        elif source_dir.exists() and any(source_dir.iterdir()):
            raise FileExistsError(f"Incomplete or untracked source stage: {source_dir}; choose a fresh directory")
    root.mkdir(parents=True, exist_ok=True)
    for source, signature in signatures.items():
        (root / f"_stage_{source}.json").write_text(
            json.dumps({"signature": signature, "source_identity": identity,
                        "images": len(by_source[source]), "complete": False}, indent=2), encoding="utf-8")
    archive_handles, archive_indices, folder_indices = {}, {}, {}
    try:
        for index, record in enumerate(records, 1):
            destination = _traffic_image_path(root, record)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.is_file():
                _traffic_check_image(destination, record)
                continue
            split, name = record["source_split"], record["source_file_name"]
            folder_value = _traffic_source_path(images_root, split)
            archive_value = _traffic_source_path(images_archive, split)
            temp = destination.with_suffix(destination.suffix + ".partial")
            try:
                if folder_value is not None:
                    folder = Path(folder_value).expanduser().resolve()
                    if not folder.is_dir():
                        raise FileNotFoundError(f"Image folder not found: {folder}")
                    folder_key = str(folder)
                    if folder_key not in folder_indices:
                        folder_index = {}
                        for candidate in folder.rglob("*"):
                            if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                resolved = candidate.resolve()
                                if not resolved.is_relative_to(folder):
                                    raise ValueError(f"Image path escapes folder through a symlink: {candidate}")
                                folder_index.setdefault(candidate.name, []).append(resolved)
                        folder_indices[folder_key] = folder_index
                    matches = folder_indices[folder_key].get(name, [])
                    if len(matches) != 1:
                        raise FileNotFoundError(f"Expected one selected image {name} under {folder}; found {len(matches)}")
                    shutil.copyfile(matches[0], temp)
                elif archive_value is not None:
                    archive_path = Path(archive_value).expanduser().resolve()
                    archive_key = str(archive_path)
                    if archive_key not in archive_handles:
                        if zipfile.is_zipfile(archive_path):
                            archive = zipfile.ZipFile(archive_path)
                            kind, members = "zip", archive.infolist()
                        elif tarfile.is_tarfile(archive_path):
                            archive = tarfile.open(archive_path, "r:*")
                            kind, members = "tar", archive
                        else:
                            raise ValueError(f"Expected ZIP or TAR image archive: {archive_path}")
                        archive_handles[archive_key] = (kind, archive)
                        image_index = {}
                        for member in members:
                            member_name = member.filename if kind == "zip" else member.name
                            is_directory = member.is_dir() if kind == "zip" else member.isdir()
                            relative = _traffic_archive_member_path(member_name, is_directory)
                            if relative is None:
                                continue
                            if kind == "zip" and stat.S_ISLNK(member.external_attr >> 16):
                                raise ValueError(f"Symlink ZIP entry is unsupported: {member_name}")
                            if kind == "tar" and not (member.isfile() or is_directory):
                                raise ValueError(f"Link or special TAR entry is unsupported: {member_name}")
                            if not is_directory and relative.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                                image_index.setdefault(relative.name, []).append(member)
                        archive_indices[archive_key] = image_index
                    matches = archive_indices[archive_key].get(name, [])
                    if len(matches) != 1:
                        raise FileNotFoundError(f"Expected one selected image {name} in {archive_path}; found {len(matches)}")
                    kind, archive = archive_handles[archive_key]
                    image_handle = archive.open(matches[0]) if kind == "zip" else archive.extractfile(matches[0])
                    if image_handle is None:
                        raise ValueError(f"Image is not a regular archive file: {name}")
                    with image_handle as source, temp.open("wb") as target:
                        shutil.copyfileobj(source, target)
                elif allow_coco_download and record["source"] == "coco":
                    partition = "train2017" if split == "train" else "val2017"
                    url = f"https://s3.amazonaws.com/images.cocodataset.org/{partition}/{urllib.parse.quote(name)}"
                    _traffic_download_image(url, temp, progress=progress)
                else:
                    raise ValueError(f"Supply image folders/archives for {record['source']} or enable COCO HTTP downloads")
                _traffic_check_image(temp, record)
                temp.replace(destination)
            finally:
                if temp.exists():
                    temp.unlink()
            if progress and (index == 1 or index % 100 == 0 or index == len(records)):
                progress(f"Staged {index}/{len(records)} selected images")
        for source, signature in signatures.items():
            (root / f"_stage_{source}.json").write_text(
                json.dumps({"signature": signature, "source_identity": identity,
                            "images": len(by_source[source]), "complete": True}, indent=2), encoding="utf-8")
    finally:
        for _, archive in archive_handles.values():
            archive.close()
    return root


def _traffic_ua_records(dataset_dir):
    root = Path(dataset_dir).expanduser().resolve()
    selected = {split: [] for split in ("train", "valid", "test")}
    seen_names = set()
    for split in selected:
        data = _traffic_json(root / split / "_annotations.coco.json")
        mapping = {item["id"]: TRAFFIC_CLASS_IDS[item["name"]]
                   for item in data["categories"] if item.get("name") in TRAFFIC_CLASS_IDS}
        if set(mapping.values()) != {1, 2, 3, 4}:
            raise ValueError("Prepared UA-DETRAC must contain car, bus, van, others category definitions")
        images = {}
        for image in data["images"]:
            name = str(_traffic_safe_relative(image["file_name"]))
            if name in seen_names or image["id"] in images:
                raise ValueError(f"Duplicate UA-DETRAC image name or ID: {name}")
            seen_names.add(name)
            source_path = (root / split / name).resolve()
            if not source_path.is_relative_to(root / split):
                raise ValueError("UA-DETRAC image escapes its split folder")
            record = {**image, "key": f"ua_detrac:{name}", "source": "ua_detrac", "source_split": split,
                      "source_file_name": name, "source_image_id": image["id"],
                      "source_path": str(source_path), "known_class_ids": [1, 2, 3, 4],
                      "car_or_van": False, "annotations": []}
            _traffic_check_image(source_path, record)
            images[image["id"]] = record
        for annotation in data["annotations"]:
            if annotation["category_id"] not in mapping or annotation["image_id"] not in images:
                raise ValueError("UA-DETRAC annotation has an unknown image/category")
            if annotation.get("iscrowd", 0) or annotation.get("ignore", 0):
                raise ValueError("Prepared UA-DETRAC must already mask ignored/crowd regions")
            record = images[annotation["image_id"]]
            box = _traffic_clip_box(annotation["bbox"], record["width"], record["height"], "UA-DETRAC target")
            record["annotations"].append({**annotation, "category_id": mapping[annotation["category_id"]],
                                          "bbox": box, "source_annotation_id": annotation["id"]})
        selected[split] = list(images.values())
    return selected


def prepare_traffic_dataset(coco_records, bdd_records, output_dir, staged_images_root, *,
                            ua_dataset_dir=None, seed=42, source_details=None, progress=None):
    """Combine selected sources into RF-DETR Roboflow COCO split folders.

    ``coco_records`` and ``bdd_records`` are sampler outputs; pass ``None`` to
    omit a source. UA-DETRAC retains
    its sequence-level splits and supplies distinct van supervision. Extra
    image fields ``known_class_ids`` / ``car_or_van`` must be consumed by the
    notebook's partial-label training adapter; plain RF-DETR ignores them.
    A complete manifest permits reuse only for identical selected inputs.
    """
    output = Path(output_dir).expanduser().resolve()
    stage = Path(staged_images_root).expanduser().resolve()
    if output == stage or output.is_relative_to(stage) or stage.is_relative_to(output):
        raise ValueError("Output and staged source images must use separate directories")
    selected = {split: [] for split in ("train", "valid", "test")}
    for source_records in (coco_records, bdd_records):
        if source_records is None:
            continue
        if not isinstance(source_records, dict) or set(source_records) != set(selected):
            raise ValueError("Each source needs selected train, valid, and test record lists")
        for split in selected:
            selected[split].extend(source_records[split])
    if ua_dataset_dir is not None:
        ua_root = Path(ua_dataset_dir).expanduser().resolve()
        if output == ua_root or output.is_relative_to(ua_root) or ua_root.is_relative_to(output):
            raise ValueError("Output and prepared UA-DETRAC must use separate directories")
        ua_selected = _traffic_ua_records(ua_root)
        for split in selected:
            selected[split].extend(ua_selected[split])
    keys, input_paths = set(), {}
    counts = {}
    for split, records in selected.items():
        if not records:
            raise ValueError(f"Combined {split} split is empty")
        records.sort(key=lambda record: record["key"])
        annotation_counts, source_counts = Counter(), Counter()
        for record in records:
            if record["key"] in keys:
                raise ValueError(f"Image leakage across combined splits: {record['key']}")
            keys.add(record["key"])
            source_path = Path(record["source_path"]) if record["source"] == "ua_detrac" else _traffic_image_path(stage, record)
            _traffic_check_image(source_path, record)
            input_paths[record["key"]] = source_path
            source_counts[record["source"]] += 1
            annotation_counts.update(TRAFFIC_CLASS_NAMES[item["category_id"] - 1] for item in record["annotations"])
        counts[split] = {"images": len(records), "annotations": sum(annotation_counts.values()),
                         "images_per_source": dict(source_counts),
                         "annotations_per_class": {name: annotation_counts[name] for name in TRAFFIC_CLASS_NAMES}}
    missing = [name for name, count in counts["train"]["annotations_per_class"].items() if not count]
    if ua_dataset_dir is not None and missing:
        raise ValueError(f"No training annotations for {missing}; increase sampling caps or supply missing labeled data")
    details = dict(source_details or {})
    stage_signatures = {source: _traffic_json(stage / f"_stage_{source}.json").get("signature")
                        for source in {record["source"] for records in selected.values() for record in records}
                        if source != "ua_detrac" and (stage / f"_stage_{source}.json").is_file()}
    ua_manifest = (_traffic_json(Path(ua_dataset_dir) / "manifest.json")
                   if ua_dataset_dir is not None and (Path(ua_dataset_dir) / "manifest.json").is_file() else None)
    signature = _traffic_signature({"version": TRAFFIC_FORMAT_VERSION, "selected": selected,
                                    "source_details": details, "seed": seed,
                                    "stage_signatures": stage_signatures, "ua_manifest": ua_manifest})
    manifest_path = output / "manifest.json"
    if manifest_path.is_file():
        manifest = _traffic_json(manifest_path)
        if manifest.get("inputs_signature") != signature:
            raise ValueError("Dataset inputs/configuration changed; choose a fresh output directory")
        for split in selected:
            saved = _traffic_json(output / split / "_annotations.coco.json")
            for image in saved["images"]:
                _traffic_check_image(output / split / image["file_name"], image)
        return manifest
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Output is not empty: {output}; choose a fresh output directory")
    manifest = {
        "format_version": TRAFFIC_FORMAT_VERSION, "inputs_signature": signature,
        "categories": TRAFFIC_CATEGORIES, "counts": counts,
        "config": {"seed": seed, "ua_detrac_included": ua_dataset_dir is not None,
                   "sampling": "per-source balanced caps; each official validation partition split into disjoint halves"},
        "sources": {source: {**TRAFFIC_SOURCE_METADATA[source], **details.get(source, {})}
                    for source in sorted({record["source"] for records in selected.values() for record in records})},
        "crowd_policy": "exclude entire source image with any target crowd/ignore box",
        "evaluation_note": "Custom detection holdouts, not official source test benchmarks. "
                           "Source labels are partial and broad car includes vans; use the notebook's source-aware evaluation.",
        "partial_label_policy": {"image_fields": ["known_class_ids", "car_or_van"],
                                 "requires_training_adapter": True,
                                 "others_mapping": "UA-DETRAC others; road-dataset truck"},
        "missing_training_classes": missing,
    }
    image_id, annotation_id = 0, 0
    for split, records in selected.items():
        split_dir = output / split
        split_dir.mkdir(parents=True, exist_ok=True)
        images, annotations = [], []
        for record in records:
            image_id += 1
            prefix = record["source"] + "_" + record["source_split"]
            basename = Path(record["source_file_name"]).name
            digest = hashlib.sha256(record["key"].encode()).hexdigest()[:12]
            filename = f"{prefix}_{digest}_{basename}"
            shutil.copyfile(input_paths[record["key"]], split_dir / filename)
            image = {"id": image_id, "file_name": filename, "width": record["width"], "height": record["height"],
                     "source": record["source"], "source_split": record["source_split"],
                     "source_image_id": record["source_image_id"], "source_file_name": record["source_file_name"],
                     "known_class_ids": record["known_class_ids"], "car_or_van": record["car_or_van"]}
            for extra in ("source_license", "source_urls", "source_attributes", "source_annotation_format",
                          "video_id", "frame_id", "sequence"):
                if extra in record:
                    image[extra] = record[extra]
            images.append(image)
            for item in record["annotations"]:
                annotation_id += 1
                box = _traffic_clip_box(item["bbox"], record["width"], record["height"], "Combined target")
                annotation = {**item, "id": annotation_id, "image_id": image_id,
                              "bbox": box, "area": box[2] * box[3], "iscrowd": 0,
                              "source": record["source"]}
                annotations.append(annotation)
        (split_dir / "_annotations.coco.json").write_text(
            json.dumps({"info": {"description": "Seven-class traffic detector; partial source labels require adapter"},
                        "images": images, "annotations": annotations, "categories": TRAFFIC_CATEGORIES}, indent=2,
                       allow_nan=False), encoding="utf-8")
        if progress:
            progress(f"Prepared {split}: {counts[split]['images']} images, {counts[split]['annotations']} boxes")
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest

