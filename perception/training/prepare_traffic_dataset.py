"""Prepare the notebook's seven-class traffic dataset on a local machine.

Pass ``--bdd-dir`` and ``--ua-dir`` for the two extracted dataset folders.
Images and labels are discovered inside those folders, including wrappers.
Only BDD100K and UA-DETRAC are used. Inputs are read-only; preparation is
fully offline. Pillow is the only dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import uuid
import zipfile
from pathlib import Path

# Also support importlib loading by the repository's standalone unit tests.
NOTEBOOK_ROOT = Path(__file__).resolve().parent
if str(NOTEBOOK_ROOT) not in sys.path:
    sys.path.insert(0, str(NOTEBOOK_ROOT))

from helpers import colab_runtime as runtime
from helpers.hosted_runtime import (
    _hosted_has_symlink_component,
    _hosted_ua_container,
    _hosted_ua_excluded,
)

SPLITS = ("train", "valid", "test")
PARTIAL_SUFFIXES = (".part", ".partial", ".crdownload", ".download", ".tmp")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Prepare BDD100K + UA-DETRAC from two extracted folders into a seven-class RF-DETR dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("Inputs must contain both images and annotations. Preparation is fully offline. "
                "Inputs and existing output files are never overwritten. "
                'Example: python perception/training/prepare_traffic_dataset.py --bdd-dir "D:/datasets/BDD100K" '
                '--ua-dir "D:/datasets/UA-DETRAC" --output-dir "D:/datasets/traffic_clean" --archive'),
    )
    models = NOTEBOOK_ROOT.parent / "models"
    folders = parser.add_argument_group("extracted dataset inputs")
    folders.add_argument("--bdd-dir", type=Path, help="Extracted BDD100K root containing images and native/Supervisely train/val labels")
    folders.add_argument("--ua-dir", type=Path, help="Extracted UA-DETRAC root containing image sequences and official TRAIN XML annotations")
    parser.add_argument("--output-dir", "--output", dest="output", type=Path, default=models / "traffic7_coco",
                        help="Location of the final prepared dataset folder (COCO annotation format)")
    parser.add_argument("--work-dir", type=Path, default=models / ".traffic7_work", help="Reusable staging cache")
    sampling = parser.add_argument_group("sampling and conversion")
    for source in ("bdd",):
        for split, default in (("train", 6000), ("valid", 600), ("test", 600)):
            sampling.add_argument(f"--{source}-{split}-cap", type=int, default=default, help="Maximum images, 0 means uncapped")
    sampling.add_argument("--frame-stride", type=int, default=5, help="UA frames 1, 1+stride, ...")
    sampling.add_argument("--max-frames-per-sequence", type=int, default=100, help="Cap sampled UA frames per sequence; 0 means uncapped")
    sampling.add_argument("--expected-ua-sequences", type=int, default=60, help="Require all official TRAIN sequences; 0 permits custom inputs with at least 3")
    sampling.add_argument("--seed", type=int, default=42, help="Deterministic source sampling and sequence splitting seed")
    sampling.add_argument("--val-fraction", type=float, default=0.2, help="UA validation sequence fraction")
    sampling.add_argument("--test-fraction", type=float, default=0.1, help="UA test sequence fraction")
    sampling.add_argument("--bdd-image-width", type=int, default=1280, help="Native BDD image width; Supervisely reads dimensions from JSON")
    sampling.add_argument("--bdd-image-height", type=int, default=720, help="Native BDD image height")
    parser.add_argument("--archive", nargs="?", const="__default__", metavar="ZIP",
                        help="Also package the finished dataset into an upload ZIP. Omit to keep only the folder; without a path, saves beside the output folder")
    return parser


def _walk(root, excluded=()):
    """Walk metadata only, pruning generated datasets and image descendants."""
    excluded = tuple(Path(path).resolve() for path in excluded)
    for current, directories, filenames in os.walk(root, followlinks=False):
        current = Path(current)
        directories[:] = sorted(name for name in directories if not (
            (current / name).is_symlink() or _hosted_ua_excluded(name)
            or name.startswith(".traffic7")
            or any((current / name).resolve() == path or (current / name).resolve().is_relative_to(path)
                   for path in excluded)
        ))
        yield current, directories, filenames


def discover_local_sources(root, *, excluded=()):
    """Discover extracted BDD/UA layouts; source archives are ignored."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    found = {name: [] for name in ("ua_images", "ua_annotations", "bdd_images", "bdd_annotations")}
    ua_groups = {"ua_images": {}, "ua_annotations": {}}
    bdd_annotations, bdd_images = {}, {}
    for current, directories, filenames in _walk(root, excluded):
        names = {name.casefold(): name for name in filenames}
        has_images = any(name.endswith((".jpg", ".jpeg", ".png")) for name in names)
        if any(re.fullmatch(r"mvi_\d+\.xml", name) for name in names):
            container = _hosted_ua_container(current, root, "annotations") or current
            ua_groups["ua_annotations"].setdefault(container, set()).add(current)
        if has_images and re.fullmatch(r"mvi_\d+", current.name.casefold()):
            container = _hosted_ua_container(current, root, "images") or current.parent
            ua_groups["ua_images"].setdefault(container, set()).add(current.parent)
            directories[:] = []
        for train, val in (("det_train.json", "det_val.json"),
                           ("bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json")):
            if {train, val} <= names.keys():
                bdd_annotations.setdefault(current, set()).update(("train", "val"))
        if current.name.casefold() == "ann" and current.parent.name.casefold() in {"train", "val"}:
            if any(name.endswith(".json") for name in names):
                bdd_annotations.setdefault(current.parent.parent, set()).add(current.parent.name.casefold())
        if has_images and current.name.casefold() in {"train", "val"} and current.parent.name.casefold() in {"100k", "images"}:
            bdd_images.setdefault(current.parent, set()).add(current.name.casefold())
            directories[:] = []
        if has_images and current.name.casefold() == "img" and current.parent.name.casefold() in {"train", "val"}:
            bdd_images.setdefault(current.parent.parent, set()).add(current.parent.name.casefold())
            directories[:] = []
    for role, groups in ua_groups.items():
        for container, leaves in groups.items():
            found[role].append(next(iter(leaves)) if len(leaves) == 1 else container)
    found["bdd_annotations"].extend(path for path, splits in bdd_annotations.items() if splits == {"train", "val"})
    found["bdd_images"].extend(path for path, splits in bdd_images.items() if splits == {"train", "val"})
    return {role: sorted(set(paths), key=str) for role, paths in found.items()}


def _checked_input(path):
    path = Path(path).expanduser().absolute()
    if _hosted_has_symlink_component(path):
        raise ValueError(f"Use a real source path instead of a symlink: {path}")
    if path.name.casefold().endswith(PARTIAL_SUFFIXES):
        raise ValueError(f"Incomplete download: {path}. Wait for the final archive.")
    partials = [] if path.is_dir() else [candidate for candidate in path.parent.glob(path.name + ".*")
                                       if candidate.is_file() and candidate.name.casefold().endswith(PARTIAL_SUFFIXES)]
    if partials:
        raise ValueError(f"Download still has partial files: {', '.join(map(str, partials))}. Wait for completion before preparing {path.name}.")
    if not path.exists():
        raise FileNotFoundError(f"Input does not exist: {path}. Finish the download or correct the path.")
    if path.is_file() and path.stat().st_size == 0:
        raise ValueError(f"Empty/incomplete input: {path}. Wait for the download to finish.")
    if not (path.is_file() or path.is_dir()):
        raise ValueError(f"Input must be a regular file or directory: {path}")
    return path.resolve()


def _dataset_folder_sources(root, dataset, roles):
    """Resolve images and labels only within one extracted dataset root."""
    root = _checked_input(root)
    flag = "--bdd-dir" if dataset == "BDD100K" else "--ua-dir"
    if not root.is_dir():
        raise ValueError(f"{flag} must be an extracted dataset directory, not an archive or file: {root}")
    found = discover_local_sources(root)
    selected = {}
    for role in roles:
        candidates = found[role]
        description = "images" if role.endswith("images") else "TRAIN XML annotations" if role == "ua_annotations" else "train/val annotations"
        if not candidates:
            raise FileNotFoundError(f"{dataset} folder lacks extracted {description}: {root}. Put images and labels under the same {flag} root.")
        if len(candidates) != 1:
            raise ValueError(f"Ambiguous {dataset} {description} under {root}: {', '.join(map(str, candidates))}. Point {flag} to one extracted dataset.")
        selected[role] = candidates[0]
    return selected


def resolve_sources(args):
    if args.bdd_dir is None or args.ua_dir is None:
        raise ValueError("Supply both --bdd-dir and --ua-dir for the two extracted dataset folders.")
    sources = _dataset_folder_sources(args.bdd_dir, "BDD100K", ("bdd_images", "bdd_annotations"))
    sources.update(_dataset_folder_sources(args.ua_dir, "UA-DETRAC", ("ua_images", "ua_annotations")))
    return sources


def _digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity(value):
    if isinstance(value, dict):
        return {key: _source_identity(path) for key, path in value.items()}
    status = value.stat()
    identity = {"path": str(value), "kind": "folder" if value.is_dir() else "file",
                "size": status.st_size, "mtime_ns": status.st_mtime_ns}
    if value.is_dir():
        digest, count = hashlib.sha256(), 0
        for current, directories, filenames in os.walk(value, followlinks=False):
            directories.sort()
            for name in sorted([*directories, *filenames]):
                path = Path(current) / name
                if path.is_symlink():
                    raise ValueError(f"Source tree contains a symlink: {path}")
                if path.is_file():
                    item = path.stat()
                    digest.update(json.dumps([path.relative_to(value).as_posix(), item.st_size, item.st_mtime_ns]).encode())
                    count += 1
        identity.update(files=count, tree_metadata_sha256=digest.hexdigest())
    elif value.suffix.casefold() == ".json":
        identity["sha256"] = _file_digest(value)
    return identity


def _ua_xml_name(name):
    return re.fullmatch(r"MVI_\d+\.xml", name, re.IGNORECASE) is not None


def _ua_folder_annotations(source, destination):
    paths = []
    for current, _, filenames in _walk(source):
        paths.extend(current / name for name in filenames if _ua_xml_name(name))
    if not paths:
        raise FileNotFoundError(f"No TRAIN MVI_*.xml files in {source}")
    used = set()
    for path in paths:
        if path.name.casefold() in used:
            raise ValueError(f"Duplicate UA TRAIN XML filename: {path.name}")
        used.add(path.name.casefold())
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = destination / path.name
        if not target.is_file():
            shutil.copyfile(path, target)
    return destination


def _fresh_conversion(parent, name):
    path = parent / name
    if path.exists() and not (path / "manifest.json").is_file():
        # Keep interrupted output for inspection; converters require an empty path.
        path = parent / f"{name}-{uuid.uuid4().hex[:8]}"
    return path


def _write_json(path, value):
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _validate_output(output, preparation):
    from helpers.traffic_datasets import TRAFFIC_CLASS_NAMES
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Prepared dataset manifest must be a JSON object: {output}")
    if manifest.get("local_preparation") != preparation:
        raise FileExistsError(f"Existing dataset inputs/configuration differ: {output}. Choose a fresh --output.")
    runtime.validate_prepared_dataset(output, TRAFFIC_CLASS_NAMES)
    if manifest.get("dataset_hash_policy") != "training_content_v2" or runtime.prepared_dataset_hash(output) != manifest.get("prepared_dataset_sha256"):
        raise ValueError(f"Prepared dataset contents changed: {output}. Choose a fresh --output and rebuild.")
    for split in SPLITS:
        if _file_digest(output / split / "_annotations.coco.json") != manifest.get("prepared_annotations_sha256", {}).get(split):
            raise ValueError(f"Prepared {split} annotations changed: {output}")
    return manifest


def _validate_upload_archive(output, destination, manifest):
    """Read the entire existing bundle before deciding it is safe to reuse."""
    digest = hashlib.sha256(b"training_content_v2\x00")
    prefix = output.name + "/"
    expected = {prefix + "manifest.json"}
    with zipfile.ZipFile(destination) as handle:
        names = []
        for info in handle.infolist():
            runtime._checked_zip_name(info)
            if not info.is_dir():
                names.append(info.filename)
        if len(names) != len(set(names)):
            raise ValueError("Upload archive contains duplicate member names")
        saved_manifest = json.loads(handle.read(prefix + "manifest.json"))
        if saved_manifest != manifest:
            raise ValueError("Upload archive manifest differs from the prepared dataset")
        for split in SPLITS:
            relative = f"{split}/_annotations.coco.json"
            expected.add(prefix + relative)
            annotation_bytes = handle.read(prefix + relative)
            if hashlib.sha256(annotation_bytes).hexdigest() != manifest["prepared_annotations_sha256"][split]:
                raise ValueError(f"Upload archive has changed {split} annotations")
            document = json.loads(annotation_bytes)
            digest.update(relative.encode())
            digest.update(json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
            for image in sorted(document["images"], key=lambda item: item["file_name"]):
                runtime._checked_archive_name(image["file_name"])
                member = prefix + split + "/" + image["file_name"]
                expected.add(member)
                digest.update(image["file_name"].encode())
                with handle.open(member) as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(chunk)
        if set(names) != expected or digest.hexdigest() != manifest["prepared_dataset_sha256"]:
            raise ValueError("Upload archive members or dataset content differ")


def create_upload_archive(output, destination, manifest):
    """Bundle just the finalized dataset, keeping JPEG/PNG bytes uncompressed."""
    destination = Path(destination).expanduser().absolute()
    if _hosted_has_symlink_component(destination) or destination.resolve().is_relative_to(output):
        raise ValueError("Upload archive must be outside the dataset and cannot use a symlink path.")
    if destination.exists():
        try:
            _validate_upload_archive(output, destination, manifest)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            raise FileExistsError(f"Upload archive already exists and does not match: {destination}. Choose another --archive path. {exc}") from exc
        print(f"Reusing fully validated upload ZIP: {destination}")
        return destination
    files = [output / "manifest.json"]
    for split in SPLITS:
        annotation_path = output / split / "_annotations.coco.json"
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        files.append(annotation_path)
        files.extend(output / split / image["file_name"] for image in document["images"])
    destination.parent.mkdir(parents=True, exist_ok=True)
    required = sum(path.stat().st_size for path in files)
    if shutil.disk_usage(destination.parent).free < required + 512 * 1024**2:
        raise OSError(f"Insufficient free space for upload ZIP ({required / 1024**3:.2f} GiB).")
    temporary = destination.with_name(destination.name + ".partial")
    if temporary.exists():
        raise FileExistsError(f"Interrupted ZIP exists: {temporary}. Choose another --archive path.")
    print(f"Writing upload ZIP: {destination}")
    with zipfile.ZipFile(temporary, "w", allowZip64=True) as handle:
        for index, path in enumerate(files, 1):
            compression = zipfile.ZIP_DEFLATED if path.suffix == ".json" else zipfile.ZIP_STORED
            handle.write(path, (Path(output.name) / path.relative_to(output)).as_posix(), compress_type=compression)
            if index % 1000 == 0 or index == len(files):
                print(f"Archived {index:,}/{len(files):,} files")
    temporary.rename(destination)
    print(f"Upload archive: {destination} ({destination.stat().st_size / 1024**3:.2f} GiB)")
    return destination


def prepare(args):
    """Validate sources, prepare locally, and return the completed manifest."""
    for name in ("frame_stride", "bdd_image_width", "bdd_image_height"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("max_frames_per_sequence", "expected_ua_sequences", *(
            f"{source}_{split}_cap" for source in ("bdd",) for split in SPLITS)):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} cannot be negative.")
    if not (0 < args.val_fraction < 1 and 0 < args.test_fraction < 1 and args.val_fraction + args.test_fraction < 1):
        raise ValueError("UA validation/test fractions must be positive and sum to less than 1.")
    output = args.output.expanduser().absolute()
    work = args.work_dir.expanduser().absolute()
    for path in (output, work):
        if _hosted_has_symlink_component(path):
            raise ValueError(f"Output/cache cannot use a symlink path: {path}")
    output, work = output.resolve(), work.resolve()
    if output == work or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError("--output and --work-dir must be separate directories.")
    if output.exists() and (not output.is_dir() or not (output / "manifest.json").is_file()):
        raise FileExistsError(f"Output already exists without a complete manifest: {output}. Choose a fresh --output.")
    archive = output.with_suffix(".zip") if args.archive == "__default__" else (Path(args.archive) if args.archive else None)
    if archive is not None and archive.expanduser().absolute().exists() and not output.exists():
        raise FileExistsError(f"Upload archive already exists: {archive}. Choose another --archive path.")
    sources = resolve_sources(args)
    for value in sources.values():
        for path in (value.values() if isinstance(value, dict) else (value,)):
            if path is not None and path.is_dir() and any(
                    dest == path or dest.is_relative_to(path) or path.is_relative_to(dest) for dest in (output, work)):
                raise ValueError(f"Dataset output/cache must be outside source folders: {path}")
    config = {name: getattr(args, name) for name in (
        "seed", "frame_stride", "max_frames_per_sequence", "expected_ua_sequences", "val_fraction", "test_fraction",
        "bdd_image_width", "bdd_image_height", *(
            f"{source}_{split}_cap" for source in ("bdd",) for split in SPLITS))}
    config.update(classes=["car", "bus", "van", "others", "person", "motorcycle", "traffic light"],
                  ignored_overlap_threshold=0.5, jpeg_quality=95, partial_label_policy="source_mask_car_van_union_v1")
    preparation = {"format_version": 1, "config": config,
                   "source_identities": {role: _source_identity(value) for role, value in sources.items()}}
    if output.exists():
        manifest = _validate_output(output, preparation)
        print(f"Reusing fully validated prepared dataset: {output}")
    else:
        try:
            from helpers.traffic_datasets import (
                TRAFFIC_CLASS_NAMES, load_bdd_records, resolve_bdd_annotation_paths,
                prepare_traffic_dataset, select_source_records, stage_selected_images,
            )
            from helpers.ua_detrac import prepare_ua_detrac
        except ModuleNotFoundError as exc:
            if exc.name == "PIL":
                raise ValueError("Install the local dependency first: python -m pip install Pillow") from exc
            raise
        cache = work / _digest_json(preparation)[:16]
        cache.mkdir(parents=True, exist_ok=True)
        print("Source inputs:", json.dumps({key: str(value) for key, value in sources.items()}, indent=2))
        annotation_root = _ua_folder_annotations(sources["ua_annotations"], cache / "ua_annotations")
        wanted, sequences = runtime.selected_frame_keys(annotation_root, args.frame_stride, args.max_frames_per_sequence or None)
        if args.expected_ua_sequences and len(sequences) != args.expected_ua_sequences:
            raise ValueError(f"Expected {args.expected_ua_sequences} official UA TRAIN sequences, found {len(sequences)}. Supply TRAIN annotations; --expected-ua-sequences 0 permits custom data.")
        print(f"Selected {len(wanted):,} UA frames across {len(sequences)} sequences.")
        frames_root = sources["ua_images"]
        bdd_train, bdd_val = resolve_bdd_annotation_paths(sources["bdd_annotations"])
        records = load_bdd_records(bdd_train, bdd_val, image_width=args.bdd_image_width, image_height=args.bdd_image_height)
        bdd_selected = select_source_records(records, seed=args.seed,
                                             **{f"{split}_cap": getattr(args, f"bdd_{split}_cap") for split in SPLITS})
        print("Selected BDD100K images:", {split: len(rows) for split, rows in bdd_selected.items()})
        image_stage = cache / "traffic_images"
        stage_selected_images(bdd_selected, image_stage, progress=print, images_root=sources["bdd_images"])
        ua_output = _fresh_conversion(cache, "ua_coco")
        if not (ua_output / "manifest.json").is_file():
            prepare_ua_detrac(frames_root, annotation_root, ua_output, frame_stride=args.frame_stride,
                             max_frames_per_sequence=args.max_frames_per_sequence or None, seed=args.seed,
                             val_fraction=args.val_fraction, test_fraction=args.test_fraction,
                             ignored_overlap_threshold=0.5, jpeg_quality=95, progress=print)
        runtime.validate_prepared_dataset(ua_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pending = output.with_name(f".{output.name}.preparing-{uuid.uuid4().hex[:8]}")
        manifest = prepare_traffic_dataset(
            None, bdd_selected, pending, image_stage, ua_dataset_dir=ua_output,
            seed=args.seed, source_details={
                "bdd100k": {"train_annotations": str(bdd_train), "val_annotations": str(bdd_val),
                            "annotation_format": "supervisely" if bdd_train.is_dir() else "scalabel"},
                "ua_detrac": {"train_annotations": str(annotation_root)},
            }, progress=print,
        )
        summary = runtime.validate_prepared_dataset(pending, TRAFFIC_CLASS_NAMES)
        print("Validating complete dataset content and computing portable fingerprint...")
        manifest.update(local_preparation=preparation, dataset_hash_policy="training_content_v2",
                        prepared_dataset_sha256=runtime.prepared_dataset_hash(pending),
                        prepared_annotations_sha256={split: _file_digest(pending / split / "_annotations.coco.json") for split in SPLITS},
                        local_validation=summary)
        _write_json(pending / "manifest.json", manifest)
        if output.exists():
            raise FileExistsError(f"Another process created {output}; prepared data retained at {pending}.")
        pending.rename(output)
        print(f"Prepared dataset: {output}")
    print("Annotation counts by class:")
    for split in SPLITS:
        print(split, manifest["counts"][split]["annotations_per_class"])
    print("Dataset SHA256:", manifest["prepared_dataset_sha256"])
    if archive is not None:
        create_upload_archive(output, archive, manifest)
    print("Attach the prepared folder to Kaggle, or upload/extract the ZIP once into Drive for Colab.")
    return manifest


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        prepare(args)
    except (OSError, ValueError, zipfile.BadZipFile, ModuleNotFoundError) as exc:
        print(f"Preparation failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
