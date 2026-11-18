"""Small standalone data-staging helpers embedded in the Colab notebook."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import stat
import tarfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath
from urllib.request import Request, urlopen


def archive_format(archive):
    """Detect ZIP or TAR (including compressed TAR) from contents, not its extension."""
    archive = Path(archive)
    if zipfile.is_zipfile(archive):
        return "zip"
    try:
        if tarfile.is_tarfile(archive):
            return "tar"
    except (OSError, tarfile.TarError):
        pass
    raise ValueError(f"Not a readable ZIP or TAR archive: {archive}")


def download_http_archive(url, destination):
    """Download a public ZIP/TAR atomically, rejecting access/quota pages."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        try:
            archive_format(destination)
        except ValueError:
            pass
        else:
            print(f"Reusing {destination.name}")
            return destination
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("Use an HTTPS dataset ZIP/TAR URL.")
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(Request(url, headers={"User-Agent": "traffic-training/1.0"}), timeout=90) as response:
            length = int(response.headers.get("Content-Length", 0))
            if length and shutil.disk_usage(destination.parent).free < length + 512 * 1024**2:
                raise OSError(f"Insufficient space to download {destination.name}.")
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
        archive_format(temporary)
        temporary.replace(destination)
    except Exception as exc:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download {destination.name}. Use the Drive archive/folder option. {exc}"
        ) from exc
    return destination


def download_http_zip(url, destination):
    """Compatibility wrapper for existing ZIP download configurations."""
    return download_http_archive(url, destination)


def extract_json_zip(archive, destination, required_names):
    """Compatibility wrapper accepting ZIP and TAR annotation archives."""
    return extract_json_archive(archive, destination, required_names)


def extract_json_archive(archive, destination, required_names):
    """Select required JSON basenames from a ZIP/TAR with optional wrapper folders."""
    archive = Path(archive)
    wanted = set(required_names)
    if not wanted:
        raise ValueError("Specify at least one required annotation JSON filename.")
    entries = []
    found = set()
    kind, members = _archive_members(archive)
    for info, name in members:
        if _archive_is_dir(info, kind) or name.name not in wanted:
            continue
        if name.name in found:
            raise ValueError(f"Duplicate annotation JSON: {name.name}")
        found.add(name.name)
        entries.append((info, Path(name.name)))
    if found != wanted:
        raise FileNotFoundError(f"Annotation archive lacks {sorted(wanted - found)}")
    return _copy_archive_entries(archive, destination, entries, {"kind": "json_annotations"}, kind)


def extract_bdd_annotation_archive(archive, destination):
    """Extract BDD aggregate labels or Supervisely train/ann and val/ann JSON files.

    Images and unlabeled official test data stay in the archive. Wrapper folders
    are removed so per-image annotations retain their train/ann or val/ann paths.
    """
    archive = Path(archive)
    kind, members = _archive_members(archive)
    aggregate_names = {
        "det_train.json", "det_val.json",
        "bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json",
    }
    entries, used = [], set()
    annotation_splits = set()
    aggregate_found = set()
    for info, name in members:
        if _archive_is_dir(info, kind) or name.suffix.lower() != ".json":
            continue
        relative = None
        if name.name in aggregate_names:
            relative = Path(name.name)
            aggregate_found.add(name.name)
        elif name.name == "meta.json":
            relative = Path("meta.json")
        else:
            for index in range(len(name.parts) - 2):
                if name.parts[index] in {"train", "val"} and name.parts[index + 1] == "ann":
                    relative = Path(*name.parts[index:])
                    annotation_splits.add(name.parts[index])
                    break
        if relative is None:
            continue
        if relative in used:
            raise ValueError(f"Duplicate BDD annotation filename in archive: {relative}")
        used.add(relative)
        entries.append((info, relative))
    has_aggregate_pair = (
        {"det_train.json", "det_val.json"}.issubset(aggregate_found)
        or {"bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json"}.issubset(aggregate_found)
    )
    if not has_aggregate_pair and annotation_splits != {"train", "val"}:
        raise FileNotFoundError(
            "BDD annotation archive needs det_train.json/det_val.json, "
            "bdd100k_labels_images_train.json/bdd100k_labels_images_val.json, "
            "or Supervisely train/ann and val/ann JSON directories."
        )
    return _copy_archive_entries(archive, destination, entries, {"kind": "bdd_annotations"}, kind)


def download_drive_zip(file_id, destination):
    import gdown

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and zipfile.is_zipfile(destination):
        print(f"Reusing {destination.name}")
        return destination
    result = gdown.download(id=file_id, output=str(destination), quiet=False, resume=True)
    if not result or not zipfile.is_zipfile(destination):
        raise RuntimeError(
            "Google Drive did not return a complete ZIP. Public download quotas may apply. "
            "Download the author-linked archives to your own Drive, select DATA_SOURCE='drive_archives', "
            "and set the two ZIP paths in the configuration cell."
        )
    return destination


def _checked_archive_name(filename):
    name = PurePosixPath(filename.replace("\\", "/"))
    if name.is_absolute() or ".." in name.parts or any(":" in part for part in name.parts):
        raise ValueError(f"Invalid archive path: {filename}")
    return name


def _checked_zip_name(info):
    name = _checked_archive_name(info.filename)
    if not name.parts and not info.is_dir():
        raise ValueError("Archive contains a file with an empty name")
    if stat.S_ISLNK(info.external_attr >> 16):
        raise ValueError(f"Archive contains a symlink: {info.filename}")
    file_type = stat.S_IFMT(info.external_attr >> 16)
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"Archive contains a special file: {info.filename}")
    return name


def _checked_tar_name(info):
    name = _checked_archive_name(info.name)
    if not name.parts and not info.isdir():
        raise ValueError("Archive contains a file with an empty name")
    if not (info.isfile() or info.isdir()):
        raise ValueError(f"Archive contains a link or special file: {info.name}")
    return name


def _archive_members(archive):
    kind = archive_format(archive)
    if kind == "zip":
        with zipfile.ZipFile(archive) as handle:
            return kind, [(info, _checked_zip_name(info)) for info in handle.infolist()]
    with tarfile.open(archive, "r:*") as handle:
        return kind, [(info, _checked_tar_name(info)) for info in handle]


def _archive_is_dir(info, kind):
    return info.is_dir() if kind == "zip" else info.isdir()


def _archive_file_size(info, kind):
    return info.file_size if kind == "zip" else info.size


def _copy_zip_entries(archive, destination, entries, signature):
    """Compatibility helper for the existing UA-DETRAC ZIP extraction paths."""
    return _copy_archive_entries(archive, destination, entries, signature, "zip")


def _copy_archive_entries(archive, destination, entries, signature, kind):
    archive = Path(archive)
    destination = Path(destination)
    archive_stat = archive.stat()
    selected_members = [
        [info.filename if kind == "zip" else info.name, relative.as_posix(),
         _archive_file_size(info, kind),
         info.CRC if kind == "zip" else info.mtime]
        for info, relative in entries
    ]
    signature = {
        **signature, "archive_format": kind, "archive_path": str(archive.resolve()),
        "archive_size": archive_stat.st_size, "archive_mtime_ns": archive_stat.st_mtime_ns,
        "selection_sha256": hashlib.sha256(
            json.dumps(selected_members, separators=(",", ":")).encode()
        ).hexdigest(),
    }
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / ".extraction.json"
    if marker.is_symlink():
        raise ValueError(f"Extraction marker must not be a symlink: {marker}")
    targets = []
    for info, relative in entries:
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()) or (destination / relative).is_symlink():
            raise ValueError(f"Archive target escaped destination or is a symlink: {relative}")
        targets.append((info, target))
    reusable = False
    if marker.is_file():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        if saved != signature:
            raise FileExistsError(f"Extraction settings changed for {destination}; choose a new WORK_DIR.")
        reusable = True
        if all(target.is_file() and target.stat().st_size == _archive_file_size(info, kind)
               for info, target in targets):
            print(f"Reusing {len(entries):,} extracted files in {destination.name}")
            return destination
    pending = [(info, target) for info, target in targets if not (
        reusable and target.is_file() and target.stat().st_size == _archive_file_size(info, kind)
    )]
    required = sum(_archive_file_size(info, kind) for info, _ in pending)
    if shutil.disk_usage(destination).free < required + 512 * 1024**2:
        raise OSError(f"Insufficient free space to extract {required / 1024**3:.2f} GiB.")
    # Persist the signature before writing files so interrupted extraction can
    # reuse completed files only when the source and selection are unchanged.
    marker_temporary = marker.with_suffix(".json.partial")
    if marker_temporary.is_symlink():
        raise ValueError(f"Extraction marker must not be a symlink: {marker_temporary}")
    marker_temporary.write_text(json.dumps(signature, indent=2), encoding="utf-8")
    marker_temporary.replace(marker)
    handle = zipfile.ZipFile(archive) if kind == "zip" else tarfile.open(archive, "r:*")
    with handle:
        for index, (info, target) in enumerate(pending, 1):
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".partial")
            if temporary.is_symlink():
                raise ValueError(f"Extraction temporary file must not be a symlink: {temporary}")
            try:
                source = handle.open(info) if kind == "zip" else handle.extractfile(info)
                if source is None:
                    raise ValueError(f"Could not read archive member: {target.name}")
                with source, temporary.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1024 * 1024)
                if temporary.stat().st_size != _archive_file_size(info, kind):
                    raise ValueError(f"Incomplete archive member: {target.name}")
                temporary.replace(target)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
            if index % 1000 == 0 or index == len(pending):
                print(f"Extracted {index:,}/{len(pending):,} files")
    return destination


def extract_annotation_zip(archive, destination):
    archive = Path(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a complete annotation ZIP: {archive}")
    entries = []
    used = set()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            name = _checked_zip_name(info)
            if info.is_dir() or name.suffix.lower() != ".xml":
                continue
            relative = Path(name.name)
            if relative in used:
                raise ValueError(f"Duplicate XML filename in archive: {relative}")
            used.add(relative)
            entries.append((info, relative))
    if not entries:
        raise ValueError("Annotation ZIP contains no XML files.")
    return _copy_zip_entries(archive, destination, entries, {
        "kind": "annotations", "archive_size": archive.stat().st_size,
        "files": sorted(str(relative) for _, relative in entries),
    })


def selected_frame_keys(annotations_root, frame_stride, max_frames_per_sequence):
    wanted, sequences = set(), set()
    for xml_path in sorted(Path(annotations_root).rglob("*.xml")):
        root = ET.parse(xml_path).getroot()
        name = root.get("name", "")
        if root.tag != "sequence" or not name.startswith("MVI_") or name in sequences:
            raise ValueError(f"Invalid or duplicate sequence in {xml_path}")
        sequences.add(name)
        frame_ids = sorted(int(frame.get("num")) for frame in root.findall("frame"))
        chosen = [number for number in frame_ids if (number - 1) % frame_stride == 0]
        if max_frames_per_sequence is not None:
            chosen = chosen[:max_frames_per_sequence]
        wanted.update(f"{name}/img{number:05d}.jpg" for number in chosen)
    if len(sequences) < 3 or not wanted:
        raise ValueError("Supply TRAIN XML for at least three sequences with selected frames.")
    return wanted, sequences


def extract_selected_images(archive, destination, wanted):
    """Extract only sampled TRAIN frames, even if the ZIP also holds TEST images."""
    archive = Path(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a complete image ZIP: {archive}")
    entries, found = [], set()
    with zipfile.ZipFile(archive) as handle:
        for info in handle.infolist():
            name = _checked_zip_name(info)
            if info.is_dir() or len(name.parts) < 2:
                continue
            key = "/".join(name.parts[-2:])
            if key not in wanted:
                continue
            if key in found:
                raise ValueError(f"Duplicate selected image in archive: {key}")
            found.add(key)
            entries.append((info, Path(*name.parts[-2:])))
    missing = wanted - found
    if missing:
        raise FileNotFoundError(f"Image ZIP lacks {len(missing)} selected frames, e.g. {sorted(missing)[:3]}")
    signature = {
        "kind": "selected_train_images", "archive_size": archive.stat().st_size,
        "selection_sha256": hashlib.sha256("\n".join(sorted(wanted)).encode()).hexdigest(),
    }
    return _copy_zip_entries(archive, destination, entries, signature)


def validate_prepared_dataset(dataset_dir, class_names=None):
    """Check paths, IDs, boxes and disjoint sequence splits before expensive training."""
    dataset_dir = Path(dataset_dir)
    sequence_sets = {}
    source_sets = {}
    all_image_ids, all_annotation_ids = set(), set()
    class_names = class_names or ["car", "bus", "van", "others"]
    expected_categories = list(enumerate(class_names, 1))
    summary = {}
    for split in ("train", "valid", "test"):
        document = json.loads((dataset_dir / split / "_annotations.coco.json").read_text(encoding="utf-8"))
        if [(c["id"], c["name"]) for c in document["categories"]] != expected_categories:
            raise ValueError(f"Unexpected category mapping in {split}")
        images = {item["id"]: item for item in document["images"]}
        if len(images) != len(document["images"]) or not images or not document["annotations"]:
            raise ValueError(f"Empty split or duplicate image ID: {split}")
        annotation_ids = set()
        source_sets[split] = set()
        for image in images.values():
            image_path = (dataset_dir / split / image["file_name"]).resolve()
            if not image_path.is_relative_to((dataset_dir / split).resolve()) or not image_path.is_file():
                raise FileNotFoundError(f"Missing or invalid prepared image: {image_path}")
            if image["id"] in all_image_ids:
                raise ValueError("Duplicate image ID across splits")
            all_image_ids.add(image["id"])
            if len(class_names) == 7:
                source = image.get("source")
                expected_known = [1, 2, 3, 4] if source == "ua_detrac" else [1, 2, 4, 5, 6, 7]
                if source not in {"coco", "bdd100k", "ua_detrac"} or image.get("known_class_ids") != expected_known:
                    raise ValueError(f"Invalid source/known_class_ids for image {image['id']}")
                if image.get("car_or_van") is not (source != "ua_detrac"):
                    raise ValueError(f"Invalid car_or_van metadata for image {image['id']}")
                name = image.get("source_file_name")
                if not isinstance(name, str) or not name:
                    raise ValueError("Missing source image identity")
                source_sets[split].add((source, name))
                if source != "ua_detrac" and image.get("source_split") != ("train" if split == "train" else "val"):
                    raise ValueError("Official source train/validation partition leakage")
        for annotation in document["annotations"]:
            if annotation["id"] in annotation_ids or annotation["category_id"] not in range(1, len(class_names) + 1):
                raise ValueError(f"Invalid annotation ID/category in {split}")
            annotation_ids.add(annotation["id"])
            if annotation["id"] in all_annotation_ids:
                raise ValueError("Duplicate annotation ID across splits")
            all_annotation_ids.add(annotation["id"])
            image = images[annotation["image_id"]]
            if len(class_names) == 7 and annotation["category_id"] not in image["known_class_ids"]:
                raise ValueError("Annotation category is not labeled by this source")
            x, y, w, h = annotation["bbox"]
            if not all(math.isfinite(value) for value in (x, y, w, h)) or x < 0 or y < 0 or w <= 0 or h <= 0 or x + w > image["width"] + 1e-6 or y + h > image["height"] + 1e-6:
                raise ValueError(f"Invalid bounding box in {split}: {annotation['id']}")
        sequence_sets[split] = {image["sequence"] for image in images.values() if "sequence" in image}
        summary[split] = {"sequences": len(sequence_sets[split]), "images": len(images),
                          "annotations": len(document["annotations"])}
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if sequence_sets[left] & sequence_sets[right]:
            raise ValueError(f"Sequence leakage between {left} and {right}")
        if source_sets[left] & source_sets[right]:
            raise ValueError(f"Source image leakage between {left} and {right}")
    return summary


def prepared_dataset_hash(dataset_dir):
    """Fingerprint training labels and image bytes independently of host paths.

    Manifests retain staging provenance (archive paths/mtimes), which must not
    make an identical Kaggle/Colab dataset look different in a later session.
    Canonical annotation JSON and every referenced image are hashed instead.
    """
    root = Path(dataset_dir).resolve()
    digest = hashlib.sha256()
    digest.update(b"training_content_v2\x00")
    for split in ("train", "valid", "test"):
        name = f"{split}/_annotations.coco.json"
        document = json.loads((root / name).read_text(encoding="utf-8"))
        digest.update(name.encode())
        digest.update(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        for image in sorted(document["images"], key=lambda item: item["file_name"]):
            image_path = (root / split / image["file_name"]).resolve()
            if not image_path.is_relative_to(root / split):
                raise ValueError("Prepared image escapes its split folder")
            digest.update(image["file_name"].encode())
            with image_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
    return digest.hexdigest()
