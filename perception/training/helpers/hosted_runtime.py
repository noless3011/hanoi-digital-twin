"""Standalone Colab/Kaggle helpers embedded in the training notebook.

Only the Python standard library is used, so discovery and checkpoint restore
can run before the notebook installs its training dependencies.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


def detect_notebook_runtime(requested="auto", *, kaggle_root="/kaggle"):
    """Select Kaggle when its input mount or kernel environment is present."""
    if requested not in {"auto", "colab", "kaggle"}:
        raise ValueError("NOTEBOOK_RUNTIME must be auto, colab, or kaggle.")
    if requested != "auto":
        return requested
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or (Path(kaggle_root) / "input").is_dir():
        return "kaggle"
    return "colab"


def _hosted_unique(paths, description):
    paths = sorted(set(Path(path) for path in paths), key=str)
    if len(paths) > 1:
        names = ", ".join(str(path) for path in paths)
        raise ValueError(
            f"Multiple {description} found under Kaggle input: {names}. "
            "Set AUTO_DISCOVER_INPUTS=False and enter the dataset paths manually."
        )
    return paths[0] if paths else None


def _hosted_archive_name(name):
    """Return the filename without a supported archive extension."""
    lower = name.casefold()
    for suffix in (".tar.gz", ".tar.bz2", ".tar.xz", ".tgz", ".tbz2", ".txz", ".tar", ".zip"):
        if lower.endswith(suffix):
            return lower[:-len(suffix)]
    return None


def _hosted_bdd_archive_contents(archive, *, maximum_members=4096):
    """Inspect bounded archive headers; never unpack images to detect a layout."""
    flags = {"train_labels": False, "val_labels": False, "train_images": False, "val_images": False}

    def inspect(name, is_file):
        if not is_file:
            return
        # Detection is deliberately conservative. The extraction helper performs
        # full archive safety validation before it writes any data.
        normalized = name.replace("\\", "/")
        path = PurePosixPath(normalized)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in BDD archive {archive}: {name}")
        parts = tuple(part.casefold() for part in path.parts if part != ".")
        basename = parts[-1] if parts else ""
        if basename in {"det_train.json", "bdd100k_labels_images_train.json"}:
            flags["train_labels"] = True
        elif basename in {"det_val.json", "bdd100k_labels_images_val.json"}:
            flags["val_labels"] = True
        for split in ("train", "val"):
            for index in range(len(parts) - 1):
                if parts[index:index + 2] == (split, "ann") and basename.endswith(".json"):
                    flags[f"{split}_labels"] = True
                if parts[index:index + 2] == (split, "img") and basename.endswith((".jpg", ".jpeg", ".png")):
                    flags[f"{split}_images"] = True
            if split in parts[:-1] and basename.endswith((".jpg", ".jpeg", ".png")):
                flags[f"{split}_images"] = True

    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as handle:
                for info in handle.infolist()[:maximum_members]:
                    inspect(info.filename, not info.is_dir())
                    if all(flags.values()):
                        break
        else:
            with tarfile.open(archive, "r:*") as handle:
                for index, info in enumerate(handle):
                    if index >= maximum_members:
                        break
                    if info.issym() or info.islnk():
                        raise ValueError(f"Links are unsupported in BDD archives: {archive}: {info.name}")
                    inspect(info.name, info.isfile())
                    if all(flags.values()):
                        break
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as exc:
        raise ValueError(f"Could not inspect attached BDD archive: {archive}: {exc}") from exc
    return flags


def _hosted_folder_key(name):
    return re.sub(r"[^a-z0-9]", "", name.casefold())


def _hosted_ua_excluded(name):
    key = _hosted_folder_key(name)
    return key.startswith("detractestannot") or "detracmottoolkit" in key


def _hosted_ua_container(path, input_root, kind):
    """Group wrapped XML/image folders under their official source container."""
    prefix = "detractrainannot" if kind == "annotations" else "detracimages"
    current = input_root
    for part in path.relative_to(input_root).parts:
        current = current / part
        if _hosted_folder_key(part).startswith(prefix):
            return current
    return None


def _hosted_ua_folder(official, fallback, description):
    # Prefer official containers over unrelated MVI examples. A single leaf is
    # the narrowest root; multiple leaves in one TRAIN container are all needed.
    if official:
        container = _hosted_unique(official, description)
        leaves = official[container]
        return next(iter(leaves)) if len(leaves) == 1 else container
    return _hosted_unique(fallback, description)


def discover_kaggle_sources(input_root):
    """Find complete, unambiguous dataset sources attached under /kaggle/input.

    Values are notebook configuration strings. Missing/incomplete datasets do
    not override the public/manual settings. Image directories are visited once
    and their descendants are pruned to avoid repeatedly scanning every frame.
    """
    input_root = Path(input_root)
    if not input_root.is_dir():
        raise FileNotFoundError(f"Kaggle input directory does not exist: {input_root}")
    if input_root.is_symlink():
        raise ValueError(f"Use the dataset directory itself, not a symlink: {input_root}")
    archives = []
    ua_frames, ua_annotations = set(), set()
    official_ua_frames, official_ua_annotations = {}, {}
    bdd_annotations, bdd_images, coco_annotations, coco_images = {}, {}, {}, {}

    def split_entry(mapping, root, split, value):
        mapping.setdefault(root, {})[split] = value

    for current, directories, filenames in os.walk(input_root, followlinks=False):
        current = Path(current)
        # Official TEST annotations and MOT toolkit examples are not training
        # sources. Prune them before looking for MVI XMLs or example images.
        directories[:] = sorted(name for name in directories
                                if not (current / name).is_symlink() and not _hosted_ua_excluded(name))
        if any(_hosted_ua_excluded(part) for part in current.relative_to(input_root).parts):
            directories[:] = []
            continue
        file_names = {name.casefold(): name for name in filenames}
        has_images = any(name.endswith((".jpg", ".jpeg", ".png")) for name in file_names)
        for filename in filenames:
            if _hosted_archive_name(filename) is not None:
                path = current / filename
                if path.is_symlink():
                    raise ValueError(f"Attached dataset archive is a symlink: {path}")
                archives.append(path)
        if any(re.fullmatch(r"mvi[^/]*\.xml", name) for name in file_names):
            container = _hosted_ua_container(current, input_root, "annotations")
            if container:
                official_ua_annotations.setdefault(container, set()).add(current)
            else:
                ua_annotations.add(current)
        if has_images and re.fullmatch(r"mvi[^/]*", current.name.casefold()):
            container = _hosted_ua_container(current, input_root, "images")
            if container:
                official_ua_frames.setdefault(container, set()).add(current.parent)
            else:
                ua_frames.add(current.parent)
            directories[:] = []

        for train_name, val_name in (("det_train.json", "det_val.json"),
                                     ("bdd100k_labels_images_train.json", "bdd100k_labels_images_val.json")):
            if {train_name, val_name} <= file_names.keys():
                if current in bdd_annotations:
                    raise ValueError(f"Multiple BDD native annotation pairs in {current}. Set AUTO_DISCOVER_INPUTS=False and choose paths manually.")
                bdd_annotations[current] = {
                    "train": current / file_names[train_name], "val": current / file_names[val_name],
                }
        if current.name.casefold() == "ann" and current.parent.name.casefold() in {"train", "val"}:
            if any(name.endswith(".json") for name in file_names):
                split_entry(bdd_annotations, current.parent.parent, current.parent.name.casefold(), current)
        if current.name.casefold() == "img" and current.parent.name.casefold() in {"train", "val"} and has_images:
            root = current.parent.parent
            split_entry(bdd_images, root, current.parent.name.casefold(), root)
            directories[:] = []
        if current.parent.name.casefold() in {"100k", "images"} and current.name.casefold() in {"train", "val"} and has_images:
            split_entry(bdd_images, current.parent, current.name.casefold(), current.parent)
            directories[:] = []

        if {"instances_train2017.json", "instances_val2017.json"} <= file_names.keys():
            coco_annotations[current] = {
                "train": current / file_names["instances_train2017.json"],
                "val": current / file_names["instances_val2017.json"],
            }
        if current.name.casefold() in {"train2017", "val2017"} and has_images:
            split_entry(coco_images, current.parent, current.name.casefold()[:-4], current.parent)
            directories[:] = []

    named_archives = {}
    for path in archives:
        named_archives.setdefault(path.name.casefold(), []).append(path)
    overrides = {}

    ua_images_archive = _hosted_unique(named_archives.get("detrac-images.zip", []), "UA-DETRAC image archives")
    ua_labels_archive = _hosted_unique(
        named_archives.get("detrac-train-annotations-xml.zip", []), "UA-DETRAC annotation archives"
    )
    ua_frame_folder = _hosted_ua_folder(official_ua_frames, ua_frames, "UA-DETRAC frame folders")
    ua_label_folder = _hosted_ua_folder(official_ua_annotations, ua_annotations, "UA-DETRAC TRAIN annotation folders")
    if ua_images_archive and ua_labels_archive:
        if ua_frame_folder and ua_label_folder:
            raise ValueError("Both UA-DETRAC folders and archives are attached. Set AUTO_DISCOVER_INPUTS=False and choose paths manually.")
        overrides.update(DATA_SOURCE="input_archives", IMAGES_ZIP=str(ua_images_archive), TRAIN_XML_ZIP=str(ua_labels_archive))
    elif ua_frame_folder and ua_label_folder:
        overrides.update(DATA_SOURCE="input_folders", FRAMES_FOLDER=str(ua_frame_folder), TRAIN_XML_FOLDER=str(ua_label_folder))

    bdd_image_archives, bdd_label_archives = [], []
    for path in archives:
        name = _hosted_archive_name(path.name)
        if "bdd" not in name:
            continue
        if "datasetninja" in name and "images-100k" in name:
            # The official DatasetNinja Images 100K download is a combined
            # Supervisely export; its archive can contain hundreds of thousands
            # of members, so do not scan all image headers just for discovery.
            bdd_image_archives.append(path)
            bdd_label_archives.append(path)
        elif "images" in name and "100k" in name and "sample" not in name:
            bdd_image_archives.append(path)
        elif "label" in name or "det_20" in name:
            bdd_label_archives.append(path)
        else:
            flags = _hosted_bdd_archive_contents(path)
            if flags["train_labels"] and flags["val_labels"]:
                bdd_label_archives.append(path)
            if flags["train_images"] and flags["val_images"]:
                bdd_image_archives.append(path)
    bdd_image_archive = _hosted_unique(bdd_image_archives, "BDD100K image archives")
    bdd_label_archive = _hosted_unique(bdd_label_archives, "BDD100K annotation archives")
    bdd_annotation_roots = [root for root, splits in bdd_annotations.items() if {"train", "val"} <= splits.keys()]
    bdd_image_roots = [root for root, splits in bdd_images.items() if {"train", "val"} <= splits.keys()]
    bdd_annotation_root = _hosted_unique(bdd_annotation_roots, "BDD100K annotation folders")
    bdd_image_root = _hosted_unique(bdd_image_roots, "BDD100K image folders")
    if bdd_image_archive and bdd_label_archive:
        if bdd_annotation_root and bdd_image_root:
            raise ValueError("Both BDD100K folders and archives are attached. Set AUTO_DISCOVER_INPUTS=False and choose paths manually.")
        overrides.update(BDD_SOURCE="input_archives", BDD_IMAGES_ARCHIVE=str(bdd_image_archive), BDD_ANNOTATIONS_ARCHIVE=str(bdd_label_archive))
    elif bdd_annotation_root and bdd_image_root:
        splits = bdd_annotations[bdd_annotation_root]
        overrides.update(BDD_SOURCE="input_folders", BDD_IMAGES_FOLDER=str(bdd_image_root), BDD_TRAIN_JSON=str(splits["train"]), BDD_VAL_JSON=str(splits["val"]))

    def named_coco(stem, description):
        return _hosted_unique([path for path in archives if _hosted_archive_name(path.name) == stem], description)

    coco_label_archive = named_coco("annotations_trainval2017", "COCO annotation archives")
    coco_train_archive = named_coco("train2017", "COCO training image archives")
    coco_val_archive = named_coco("val2017", "COCO validation image archives")
    coco_annotation_root = _hosted_unique(coco_annotations, "COCO annotation folders")
    coco_image_root = _hosted_unique(
        [root for root, splits in coco_images.items() if {"train", "val"} <= splits.keys()], "COCO image folders"
    )
    if coco_label_archive and coco_train_archive and coco_val_archive:
        if coco_annotation_root and coco_image_root:
            raise ValueError("Both COCO folders and archives are attached. Set AUTO_DISCOVER_INPUTS=False and choose paths manually.")
        overrides.update(COCO_SOURCE="input_archives", COCO_ANNOTATIONS_ZIP=str(coco_label_archive), COCO_TRAIN_IMAGES_ZIP=str(coco_train_archive), COCO_VAL_IMAGES_ZIP=str(coco_val_archive))
    elif coco_annotation_root and coco_image_root:
        splits = coco_annotations[coco_annotation_root]
        overrides.update(COCO_SOURCE="input_folders", COCO_IMAGES_FOLDER=str(coco_image_root), COCO_TRAIN_JSON=str(splits["train"]), COCO_VAL_JSON=str(splits["val"]))
    return overrides


def _hosted_file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hosted_has_symlink_component(path):
    absolute = Path(path).absolute()
    return any(part.is_symlink() for part in (absolute, *absolute.parents))


def restore_training_run(source_dir, output_dir, *, run_name):
    """Copy a saved run from read-only Kaggle input to its writable output.

    A run signature and full resume checkpoint are mandatory. Only top-level
    training artifacts are copied; datasets and nested directories are ignored.
    All destination conflicts are checked before any file is written.
    """
    if not isinstance(run_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_name):
        raise ValueError("RUN_NAME must use letters, digits, underscores or hyphens (at most 80 characters).")
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Saved run directory does not exist: {source_dir}")
    if _hosted_has_symlink_component(source_dir) or _hosted_has_symlink_component(output_dir):
        raise ValueError("Use actual run directories, not symlinks.")
    if output_dir.resolve().is_relative_to(source_dir.resolve()):
        raise ValueError("The restored output directory must be separate from the saved input run.")
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Run output is not a directory: {output_dir}")
    config_path = source_dir / "run_config.json"
    if config_path.is_symlink() or not config_path.is_file():
        raise ValueError("The saved run must include run_config.json with its original signature.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not isinstance(config.get("signature"), dict) or not config["signature"]:
        raise ValueError("The saved run_config.json must contain its original nonempty signature object.")
    if "run_name" in config and config["run_name"] != run_name:
        raise ValueError("The saved run name differs. Keep RUN_NAME unchanged when resuming.")
    allowed_names = {
        "run_config.json", "training_config.json", "class_names.json", "dataset_manifest.json",
        "partial_labels.py", "requirements-colab.txt", "requirements-runtime.txt", "signature.json",
    }
    selected = []
    for path in sorted(source_dir.iterdir(), key=lambda item: item.name):
        if path.name in allowed_names or path.suffix in {".ckpt", ".pth"}:
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Saved training artifact is not a regular file: {path}")
            selected.append(path)
    if not any(path.suffix == ".ckpt" for path in selected):
        raise ValueError("The saved run needs a full .ckpt checkpoint (usually last.ckpt) to continue training.")
    to_copy = []
    for source in selected:
        destination = output_dir / source.name
        if destination.is_symlink():
            raise ValueError(f"Restored artifact path is a symlink: {destination}")
        if destination.exists():
            if not destination.is_file() or _hosted_file_digest(source) != _hosted_file_digest(destination):
                raise FileExistsError(f"Existing run artifact differs: {destination}. Choose an empty output run directory.")
        else:
            temporary = destination.with_name(destination.name + ".restore-part")
            if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
                raise ValueError(f"Restoration temporary path is not a regular file: {temporary}")
            to_copy.append((source, destination))
    required_bytes = sum(source.stat().st_size for source, _ in to_copy)
    existing_parent = output_dir
    while not existing_parent.exists():
        existing_parent = existing_parent.parent
    if shutil.disk_usage(existing_parent).free < required_bytes + 1024 * 1024:
        raise OSError("Insufficient free space to restore this training run.")
    output_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for source, destination in to_copy:
        temporary = destination.with_name(destination.name + ".restore-part")
        if temporary.is_symlink():
            raise ValueError(f"Restoration temporary path is a symlink: {temporary}")
        try:
            shutil.copy2(source, temporary)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
        copied.append(destination)
    return copied
