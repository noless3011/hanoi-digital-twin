"""Convert UA-DETRAC TRAIN XML/frame archives to RF-DETR's COCO layout.

Only the Python standard library and Pillow are required. Supply the official
TRAIN annotations/images; the resulting sequence-level holdouts are custom,
sampled detection splits, not the official UA-DETRAC tracking benchmark.
"""

from __future__ import annotations

import json
import math
import random
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image


CATEGORIES = [
    {"id": 1, "name": "car", "supercategory": "vehicle"},
    {"id": 2, "name": "bus", "supercategory": "vehicle"},
    {"id": 3, "name": "van", "supercategory": "vehicle"},
    {"id": 4, "name": "others", "supercategory": "vehicle"},
]
CLASS_IDS = {item["name"]: item["id"] for item in CATEGORIES}
CLASS_IDS["other"] = CLASS_IDS["others"]
FORMAT_VERSION = 1


def _integer(value, context, minimum=0):
    if value is None or not re.fullmatch(r"\d+", str(value)):
        raise ValueError(f"{context}: expected an integer, got {value!r}")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{context}: must be >= {minimum}")
    return result


def _one(parent, tag, context):
    matches = parent.findall(tag)
    if len(matches) != 1:
        raise ValueError(f"{context}: expected one <{tag}>, found {len(matches)}")
    return matches[0]


def _box(element, context):
    values = []
    for key in ("left", "top", "width", "height"):
        try:
            value = float(element.attrib[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{context}: missing or malformed box {key}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{context}: box {key} must be finite")
        values.append(value)
    if values[2] <= 0 or values[3] <= 0:
        raise ValueError(f"{context}: box width and height must be positive")
    return values


def _parse_sequence(xml_path):
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"{xml_path}: malformed XML: {exc}") from exc
    sequence = root.attrib.get("name", "")
    if root.tag != "sequence" or not re.fullmatch(r"MVI_\d+", sequence):
        raise ValueError(f"{xml_path}: expected <sequence name='MVI_...'>")
    ignored_elements = root.findall("ignored_region")
    if len(ignored_elements) > 1:
        raise ValueError(f"{xml_path}: duplicate <ignored_region>")
    ignored = []
    if ignored_elements:
        ignored = [_box(box, f"{sequence} ignored region")
                   for box in ignored_elements[0].findall("box")]
    frames, seen = [], set()
    for frame in root.findall("frame"):
        frame_id = _integer(frame.attrib.get("num"), f"{sequence} frame num", 1)
        context = f"{sequence} frame {frame_id}"
        if frame_id in seen:
            raise ValueError(f"{context}: duplicate frame")
        seen.add(frame_id)
        target_list = _one(frame, "target_list", context)
        targets, target_ids = [], set()
        for target in target_list.findall("target"):
            track_id = _integer(target.attrib.get("id"), f"{context} target id")
            target_context = f"{context} target {track_id}"
            if track_id in target_ids:
                raise ValueError(f"{target_context}: duplicate target ID")
            target_ids.add(track_id)
            box = _box(_one(target, "box", target_context), target_context)
            attribute = _one(target, "attribute", target_context)
            label = attribute.attrib.get("vehicle_type")
            if label is None or not label.strip():
                raise ValueError(f"{target_context}: missing vehicle_type")
            normalized = label.strip().lower()
            if normalized not in CLASS_IDS:
                raise ValueError(f"{target_context}: unknown vehicle_type {label!r}")
            targets.append({"track_id": track_id, "bbox": box,
                            "category_id": CLASS_IDS[normalized],
                            "source_attributes": dict(attribute.attrib)})
        frames.append({"frame_id": frame_id, "targets": targets})
    if not frames:
        raise ValueError(f"{xml_path}: sequence has no frames")
    return {"name": sequence, "ignored_regions": ignored,
            "frames": sorted(frames, key=lambda frame: frame["frame_id"]),
            "source_attributes": [dict(item.attrib) for item in root.findall("sequence_attribute")]}


def _clip(box, width, height):
    left, top, box_width, box_height = box
    x1, y1 = max(0.0, left), max(0.0, top)
    x2, y2 = min(float(width), left + box_width), min(float(height), top + box_height)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2 - x1, y2 - y1]


def _intersection_union_area(box, regions):
    """Area covered by the union of ignored rectangles inside one object box."""
    left, top, width, height = box
    intersections = []
    for x, y, w, h in regions:
        x1, y1 = max(left, x), max(top, y)
        x2, y2 = min(left + width, x + w), min(top + height, y + h)
        if x2 > x1 and y2 > y1:
            intersections.append((x1, y1, x2, y2))
    xs = sorted({x for rect in intersections for x in (rect[0], rect[2])})
    area = 0.0
    for x1, x2 in zip(xs, xs[1:]):
        intervals = sorted((y1, y2) for rx1, y1, rx2, y2 in intersections
                           if rx1 < x2 and rx2 > x1)
        total_height, start, end = 0.0, None, None
        for y1, y2 in intervals:
            if start is None:
                start, end = y1, y2
            elif y1 > end:
                total_height += end - start
                start, end = y1, y2
            else:
                end = max(end, y2)
        if start is not None:
            total_height += end - start
        area += (x2 - x1) * total_height
    return area


def _split_sequences(names, seed, val_fraction, test_fraction):
    if len(names) < 3:
        raise ValueError("At least 3 TRAIN sequences are required for sequence-level train/valid/test splits")
    if not (0 < val_fraction < 1 and 0 < test_fraction < 1 and val_fraction + test_fraction < 1):
        raise ValueError("val_fraction/test_fraction must be positive and sum to less than 1")
    shuffled = sorted(names)
    random.Random(seed).shuffle(shuffled)
    valid_count = max(1, round(len(names) * val_fraction))
    test_count = max(1, round(len(names) * test_fraction))
    while valid_count + test_count >= len(names):
        if valid_count >= test_count and valid_count > 1:
            valid_count -= 1
        elif test_count > 1:
            test_count -= 1
        else:
            raise ValueError("Split fractions leave no training sequence")
    return {
        "train": sorted(shuffled[valid_count + test_count:]),
        "valid": sorted(shuffled[:valid_count]),
        "test": sorted(shuffled[valid_count:valid_count + test_count]),
    }


def _selected_frames(sequence, frame_stride, cap):
    selected = [frame for frame in sequence["frames"] if (frame["frame_id"] - 1) % frame_stride == 0]
    return selected if cap is None else selected[:cap]


def _find_frame_directories(frames_root):
    directories = {}
    candidates = [frames_root] + sorted(frames_root.rglob("MVI_*"))
    for candidate in candidates:
        if candidate.is_dir() and re.fullmatch(r"MVI_\d+", candidate.name):
            if candidate.name in directories:
                raise ValueError(f"Duplicate image sequence folder {candidate.name}: {directories[candidate.name]} and {candidate}")
            directories[candidate.name] = candidate
    return directories


def prepare_ua_detrac(
    frames_root,
    annotations_root,
    output_dir,
    *,
    frame_stride=5,
    max_frames_per_sequence=None,
    seed=42,
    val_fraction=0.2,
    test_fraction=0.1,
    ignored_overlap_threshold=0.5,
    jpeg_quality=95,
    progress=None,
):
    """Write train/valid/test images and ``_annotations.coco.json`` files.

    Source paths may contain archive wrapper directories. ``annotations_root``
    must contain only official TRAIN XML files. Each whole sequence belongs to
    exactly one split; frame numbers 1, 1+stride, ... are retained, optionally
    capped per sequence. The cap selects the first sampled frames.

    Ignored regions are blackened in output JPEGs. Targets with >= threshold
    union intersection-over-clipped-object area are dropped. Remaining boxes
    retain their clipped geometry. Floating XML coordinates are used as supplied
    without a speculative 1-pixel offset. Track IDs remain source-sequence-local.

    A nonempty output directory is refused to protect existing runs. Preflight
    validates all XML targets and selected image files before output is created.
    ``manifest.json`` is written last and indicates complete conversion.
    If provided, ``progress`` is called with one readable status string after
    each converted sequence (for example, ``progress=print`` in Colab).
    """
    frames_root = Path(frames_root).expanduser().resolve()
    annotations_root = Path(annotations_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    frame_stride = _integer(frame_stride, "frame_stride", 1)
    if max_frames_per_sequence is not None:
        max_frames_per_sequence = _integer(max_frames_per_sequence, "max_frames_per_sequence", 1)
    jpeg_quality = _integer(jpeg_quality, "jpeg_quality", 1)
    if jpeg_quality > 100:
        raise ValueError("jpeg_quality must be <= 100")
    if not (0 < ignored_overlap_threshold <= 1):
        raise ValueError("ignored_overlap_threshold must be in (0, 1]")
    if not frames_root.is_dir() or not annotations_root.is_dir():
        raise FileNotFoundError("frames_root and annotations_root must be existing directories")
    for source_root in (frames_root, annotations_root):
        if output_dir == source_root or output_dir.is_relative_to(source_root) or source_root.is_relative_to(output_dir):
            raise ValueError("output_dir must be separate from, and outside, both input directories")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Output is not an empty directory: {output_dir}; choose a fresh output_dir")
    xml_paths = sorted(annotations_root.rglob("*.xml"))
    if not xml_paths:
        raise FileNotFoundError(f"No XML annotations found under {annotations_root}")
    frame_dirs = _find_frame_directories(frames_root)
    sequence_paths, image_sizes = {}, {}
    # Preflight one sequence at a time, avoiding all-dataset XML trees in RAM.
    for xml_path in xml_paths:
        sequence = _parse_sequence(xml_path)
        name = sequence["name"]
        if name in sequence_paths:
            raise ValueError(f"Duplicate XML sequence {name}: {sequence_paths[name]} and {xml_path}")
        sequence_paths[name] = xml_path
        if name not in frame_dirs:
            raise FileNotFoundError(f"No image folder named {name} under {frames_root}")
        selected = _selected_frames(sequence, frame_stride, max_frames_per_sequence)
        if not selected:
            raise ValueError(f"{name}: no frames match frame_stride={frame_stride}")
        for frame in selected:
            image_path = frame_dirs[name] / f"img{frame['frame_id']:05d}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing annotated image: {image_path}")
            try:
                with Image.open(image_path) as source_image:
                    image_sizes[(name, frame["frame_id"])] = source_image.size
                    source_image.verify()
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable image {image_path}: {exc}") from exc
    splits = _split_sequences(sequence_paths, seed, val_fraction, test_fraction)
    video_ids = {name: index + 1 for index, name in enumerate(sorted(sequence_paths))}
    config = {"frame_stride": frame_stride, "max_frames_per_sequence": max_frames_per_sequence,
              "seed": seed, "val_fraction": val_fraction, "test_fraction": test_fraction,
              "ignored_overlap_threshold": ignored_overlap_threshold, "jpeg_quality": jpeg_quality}
    manifest = {
        "format_version": FORMAT_VERSION,
        "source": {"frames_root": str(frames_root), "annotations_root": str(annotations_root),
                   "expected_partition": "official TRAIN"},
        "config": config,
        "categories": CATEGORIES,
        "class_aliases": {"other": "others"},
        "split_sequences": splits,
        "ignored_region_policy": {
            "name": "black_mask_and_drop_overlapping_targets",
            "drop_at_intersection_over_object": ignored_overlap_threshold,
            "overlap_computation": "union of clipped ignored boxes divided by clipped object area",
            "mask_rasterization": "floor left/top and ceil right/bottom; half-open pixel rectangle",
            "reason": "Unlabeled vehicles in ignored regions should not become ordinary background negatives",
        },
        "coordinate_policy": "XML floating left/top/width/height as supplied; clip to image; no offset",
        "track_id_policy": "Original integer IDs are local to each sequence; key tracks by (sequence, track_id)",
        "evaluation_note": "Custom sequence holdouts with frame sampling and ignored-region masking; not the official UA-DETRAC benchmark. Use original complete unmasked sequences and official toolkit for tracking evaluation.",
        "counts": {},
        "sequences": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    image_id, annotation_id = 1, 1
    for split, names in splits.items():
        split_dir = output_dir / split
        split_dir.mkdir()
        coco = {"info": {"description": "UA-DETRAC custom sequence-split detection dataset",
                         "version": str(FORMAT_VERSION)},
                "licenses": [], "categories": CATEGORIES, "images": [], "annotations": [],
                "videos": [{"id": video_ids[name], "name": name} for name in names]}
        counts = {"sequences": len(names), "images": 0, "annotations": 0,
                  "masked_images": 0, "dropped_ignored_annotations": 0,
                  "dropped_outside_image_annotations": 0,
                  "annotations_per_class": {item["name"]: 0 for item in CATEGORIES}}
        for name in names:
            sequence = _parse_sequence(sequence_paths[name])
            selected = _selected_frames(sequence, frame_stride, max_frames_per_sequence)
            manifest["sequences"][name] = {
                "split": split, "video_id": video_ids[name],
                "annotation_file": str(sequence_paths[name].relative_to(annotations_root)),
                "source_frame_count": len(sequence["frames"]),
                "selected_frame_ids": [frame["frame_id"] for frame in selected],
                "ignored_regions_xywh": sequence["ignored_regions"],
                "source_attributes": sequence["source_attributes"],
            }
            for frame in selected:
                frame_id = frame["frame_id"]
                image_path = frame_dirs[name] / f"img{frame_id:05d}.jpg"
                file_name = f"{name}_img{frame_id:05d}.jpg"
                width, height = image_sizes[(name, frame_id)]
                ignored = [clipped for box in sequence["ignored_regions"]
                           if (clipped := _clip(box, width, height)) is not None]
                coco["images"].append({"id": image_id, "file_name": file_name,
                    "width": width, "height": height, "sequence": name,
                    "video_id": video_ids[name], "frame_id": frame_id,
                    "source_file_name": str(image_path.relative_to(frames_root)),
                    "ignored_regions_xywh": ignored})
                for target in frame["targets"]:
                    box = _clip(target["bbox"], width, height)
                    if box is None:
                        counts["dropped_outside_image_annotations"] += 1
                        continue
                    area = box[2] * box[3]
                    overlap = _intersection_union_area(box, ignored) / area
                    if overlap >= ignored_overlap_threshold:
                        counts["dropped_ignored_annotations"] += 1
                        continue
                    coco["annotations"].append({"id": annotation_id, "image_id": image_id,
                        "category_id": target["category_id"], "bbox": box, "area": area,
                        "iscrowd": 0, "sequence": name, "video_id": video_ids[name],
                        "frame_id": frame_id, "track_id": target["track_id"],
                        "source_bbox_xywh": target["bbox"],
                        "source_attributes": target["source_attributes"],
                        "ignored_overlap_fraction": overlap})
                    annotation_id += 1
                    counts["annotations"] += 1
                    category_name = CATEGORIES[target["category_id"] - 1]["name"]
                    counts["annotations_per_class"][category_name] += 1
                if ignored:
                    with Image.open(image_path) as source_image:
                        masked = source_image.convert("RGB")
                        for x, y, w, h in ignored:
                            masked.paste((0, 0, 0), (math.floor(x), math.floor(y),
                                                   math.ceil(x + w), math.ceil(y + h)))
                        masked.save(split_dir / file_name, "JPEG", quality=jpeg_quality)
                    counts["masked_images"] += 1
                else:
                    shutil.copyfile(image_path, split_dir / file_name)
                counts["images"] += 1
                image_id += 1
            if progress is not None:
                progress(f"{split}: {name} — converted {len(selected)} frames")
        if counts["annotations"] == 0:
            raise ValueError(f"{split} has no retained annotations; choose other sequences or review ignored regions")
        with (split_dir / "_annotations.coco.json").open("w", encoding="utf-8") as handle:
            json.dump(coco, handle, ensure_ascii=False, indent=2, allow_nan=False)
        manifest["counts"][split] = counts
    with (output_dir / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return manifest
