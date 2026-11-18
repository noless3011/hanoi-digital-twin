"""Build the training-only Kaggle/Colab notebook for a prepared traffic dataset."""

import ast
import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent


def selected_functions(path, names):
    """Embed only training-related functions from the tested runtime helper."""
    source = path.read_text(encoding="utf-8")
    functions = {node.name: ast.get_source_segment(source, node)
                 for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
    return "\n\n\n".join(functions[name] for name in names) + "\n"


def build_notebook():
    cells = []

    def markdown(cell_id, text):
        cells.append({"cell_type": "markdown", "id": cell_id, "metadata": {},
                      "source": dedent(text).strip() + "\n"})

    def code(cell_id, text, *, hidden=False):
        metadata = {"cellView": "form"} if hidden else {}
        cells.append({"cell_type": "code", "id": cell_id, "metadata": metadata,
                      "source": dedent(text).strip() + "\n", "execution_count": None, "outputs": []})

    markdown("intro", """
        # Train RF-DETR Nano on a locally prepared traffic dataset

        Fine-tune a COCO-pretrained **RF-DETR Nano** on **car, bus, van, others,
        person, motorcycle, traffic light** using BDD100K and UA-DETRAC.

        1. Extract BDD100K and UA-DETRAC on your machine, including their images and labels.
           Run `python perception/training/prepare_traffic_dataset.py --bdd-dir "D:/datasets/BDD100K" --ua-dir "D:/datasets/UA-DETRAC" --archive`
           from the project root, replacing the two example paths with your dataset folders.
           Preparation runs fully offline using only these two extracted datasets.
           The default prepared output is `perception/models/traffic7_coco`.
        2. Upload the complete prepared **folder** as a Kaggle Dataset and attach it with **Add Input**,
           or copy it to `/content/drive/MyDrive/traffic7_coco` for Colab.
           If you transfer a ZIP/TAR, unpack it before running this notebook.
        3. Choose a GPU, enable Internet for packages/pretrained weights, and set
           `PREPARED_DATASET_DIR` to the folder containing `manifest.json` and `train`, `valid`, `test`.
           A single prepared Kaggle input is detected automatically.
        4. Run the cells in order to train, inspect metrics, evaluate, and save/download the results.
           The notebook is self-contained; no repository clone is needed in the hosted runtime.

        The prepared folder is used directly. Images and labels must stay unchanged after local
        preparation. Annotation fingerprints are checked here; the offline image/label fingerprint
        ensures resume compatibility without reading all images again.

        Defaults train 30 epochs at 512px, batch 4, accumulation 4. For a quick check use **2 epochs**
        and a new run name. Change sampling or cleanup settings by running local preparation into
        a new output directory before uploading.

        Kaggle saves runs under `/kaggle/working/traffic_training/runs`; Colab saves them in Drive.
        Save a Kaggle notebook version **with outputs** to retain full checkpoints.

        **Partial labels:** the embedded training adapter suppresses negative supervision for the
        three missing UA-DETRAC classes. On BDD100K, broad car boxes supervise car or van;
        UA-DETRAC teaches the distinction. Source-aware evaluation uses the same rules. `person`
        includes pedestrians/riders, `others` includes trucks, and `traffic light` does not classify color.
        Metrics describe the prepared detection holdouts. TrackTrack associates detections afterwards
        and is not trained here.

        [RF-DETR 1.11.1 release](https://github.com/roboflow/rf-detr/releases/tag/1.11.1)
    """)
    markdown("install-heading", """
        ## 1. Install the pinned training package

        Enable **Internet** in Kaggle's settings. Run this before importing RF-DETR or PyTorch.
        The hosted runtime supplies GPU-enabled PyTorch. If a restart is requested, restart and
        continue from the GPU check. RF-DETR is pinned to the verified **1.11.1** release.
    """)
    code("install", """
        import subprocess
        import sys

        subprocess.run([
            sys.executable, "-m", "pip", "install", "--quiet",
            "--upgrade-strategy", "only-if-needed",
            "rfdetr[train,loggers]==1.11.1", "Pillow", "matplotlib",
        ], check=True)
        print("Installation complete.")
    """)
    code("gpu-check", """
        import sys
        import importlib.metadata
        import torch

        assert importlib.metadata.version("rfdetr") == "1.11.1"
        if not torch.cuda.is_available():
            raise RuntimeError("Enable a GPU in Kaggle Settings or Colab Runtime → Change runtime type, then rerun.")
        properties = torch.cuda.get_device_properties(0)
        print(f"GPU: {properties.name}; VRAM: {properties.total_memory / 1024**3:.1f} GiB")
        print(f"Python: {sys.version.split()[0]}; PyTorch: {torch.__version__}; RF-DETR: 1.11.1")
        print(f"Visible GPUs: {torch.cuda.device_count()}; training uses GPU 0 in one process.")
    """)
    runtime_imports = "import hashlib\nimport json\nimport os\nimport re\nimport shutil\nfrom pathlib import Path\n\n"
    code("runtime-helpers", "# @title Hosted runtime and checkpoint restore\n" + runtime_imports +
         selected_functions(ROOT / "helpers" / "hosted_runtime.py", (
             "detect_notebook_runtime", "_hosted_file_digest", "_hosted_has_symlink_component",
             "restore_training_run")), hidden=True)
    markdown("drive-heading", """
        ## 2. Set up runtime storage

        `auto` detects Kaggle and skips Google Drive mounting. The prepared Kaggle Dataset is read
        directly from `/kaggle/input`. Runs and bundles go to `/kaggle/working`; save outputs before
        the session ends. Colab mounts Drive for the prepared dataset and persistent runs.

        To resume a saved Kaggle run, attach that notebook's outputs using **Add Input** and set
        `RESUME_RUN_DIR` to the folder containing `last.ckpt` and `run_config.json`.
        Attach the same prepared dataset again.
    """)
    code("mount-drive", """
        from pathlib import Path

        RUNTIME = "auto" # @param ["auto", "kaggle", "colab"]
        KAGGLE_INPUT_ROOT = "/kaggle/input" # @param {type:"string"}
        KAGGLE_OUTPUT_ROOT = "/kaggle/working" # @param {type:"string"}
        RUNTIME = detect_notebook_runtime(RUNTIME)
        if RUNTIME == "kaggle":
            INPUT_ROOT = Path(KAGGLE_INPUT_ROOT)
            DEFAULT_PROJECT_DIR = Path(KAGGLE_OUTPUT_ROOT) / "traffic_training"
            print("Attach the prepared traffic7_coco folder using Add Input.")
        else:
            from google.colab import drive
            drive.mount("/content/drive")
            INPUT_ROOT = Path("/content/drive/MyDrive")
            DEFAULT_PROJECT_DIR = INPUT_ROOT / "traffic_training"
        print(f"Runtime: {RUNTIME}; prepared inputs: {INPUT_ROOT}")
        print(f"Saved runs: {DEFAULT_PROJECT_DIR / 'runs'}")
    """)
    markdown("config-heading", """
        ## 3. Configuration

        `PREPARED_DATASET_DIR` points to the prepared folder with `manifest.json`,
        `train/_annotations.coco.json`, `valid/_annotations.coco.json`, and `test/_annotations.coco.json`.
        Keep every image alongside its split annotations and upload the offline manifest.
        On Kaggle, an empty path discovers one complete prepared folder under `/kaggle/input`.
        Set the path explicitly when several prepared datasets are attached. On Colab, an empty path
        uses `/content/drive/MyDrive/traffic7_coco`.

        Dataset sampling and cleanup settings belong to the local preparation script. Keep the
        seven-class order fixed. The prepared folder is read in place, including read-only Kaggle inputs.

        For resume, keep `RUN_NAME`, data, resolution, learning rates, seed and batch/accumulation
        unchanged. `EPOCHS` is the **total target**, so it can increase when continuing a run.
        Set `RESUME_RUN_DIR` when restoring a saved run into a new writable session.
        Old runs using the same `training_content_v2` fingerprint remain compatible with identical data.
        Choose a fresh run name when changing the recipe or dataset.
    """)
    code("prepared-data-helpers", r'''
        # @title Locate and check prepared dataset metadata
        import hashlib
        import json
        import math
        import os
        import re
        from collections import Counter
        from pathlib import Path, PurePosixPath


        def discover_prepared_datasets(input_root):
            """Find complete prepared folders without traversing their images."""
            root = Path(input_root)
            if not root.is_dir():
                raise FileNotFoundError(f"Input directory does not exist: {root}")
            candidates = []
            for current, directories, filenames in os.walk(root, followlinks=False):
                current = Path(current)
                directories[:] = sorted(name for name in directories if not (current / name).is_symlink())
                if "manifest.json" in filenames and all(
                    (current / split / "_annotations.coco.json").is_file()
                    for split in ("train", "valid", "test")
                ):
                    directories[:] = []
                    try:
                        manifest = json.loads((current / "manifest.json").read_text(encoding="utf-8"))
                    except (OSError, ValueError):
                        continue
                    if (isinstance(manifest, dict)
                        and isinstance(manifest.get("sources"), dict)
                        and set(manifest["sources"]) == {"bdd100k", "ua_detrac"}
                        and manifest.get("dataset_hash_policy") == "training_content_v2"
                        and isinstance(manifest.get("prepared_dataset_sha256"), str)
                        and re.fullmatch(r"[0-9a-f]{64}", manifest["prepared_dataset_sha256"])
                        and isinstance(manifest.get("prepared_annotations_sha256"), dict)
                        and set(manifest["prepared_annotations_sha256"]) == {"train", "valid", "test"}):
                        candidates.append(current)
            return sorted(candidates, key=str)


        def load_prepared_dataset(dataset_dir, class_names):
            """Check offline manifest/label metadata without opening image bytes."""
            root = Path(dataset_dir)
            manifest_path = root / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing prepared manifest: {manifest_path}. Run local preparation first.")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest.get("sources"), dict) or set(manifest["sources"]) != {"bdd100k", "ua_detrac"}:
                raise ValueError("Prepared data must contain only BDD100K and UA-DETRAC. Rebuild with the two-folder preparation script.")
            fingerprint = manifest.get("prepared_dataset_sha256")
            if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
                raise ValueError("Manifest needs prepared_dataset_sha256 from the local preparation script.")
            if manifest.get("dataset_hash_policy") != "training_content_v2":
                raise ValueError("Unsupported prepared dataset fingerprint policy.")
            annotation_hashes = manifest.get("prepared_annotations_sha256")
            if not isinstance(annotation_hashes, dict) or set(annotation_hashes) != {"train", "valid", "test"}:
                raise ValueError("Manifest needs prepared_annotations_sha256 for train, valid, and test.")
            expected_categories = list(enumerate(class_names, 1))
            if [(item["id"], item["name"]) for item in manifest.get("categories", [])] != expected_categories:
                raise ValueError("Prepared manifest has an unexpected seven-class mapping.")
            if manifest.get("partial_label_policy", {}).get("requires_training_adapter") is not True:
                raise ValueError("Prepared manifest must declare the source-aware training adapter policy.")
            if not isinstance(manifest.get("evaluation_note"), str):
                raise ValueError("Prepared manifest is missing its evaluation provenance.")
            summary, image_ids, annotation_ids = {}, set(), set()
            source_images, sequences = {}, {}
            for split in ("train", "valid", "test"):
                path = root / split / "_annotations.coco.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing prepared annotations: {path}")
                raw_annotations = path.read_bytes()
                if hashlib.sha256(raw_annotations).hexdigest() != annotation_hashes[split]:
                    raise ValueError(f"Prepared {split} labels changed after local preparation. Upload the unchanged prepared folder.")
                document = json.loads(raw_annotations)
                if [(item["id"], item["name"]) for item in document["categories"]] != expected_categories:
                    raise ValueError(f"Unexpected category mapping in {split}.")
                images = {image["id"]: image for image in document["images"]}
                if not images or len(images) != len(document["images"]) or not document["annotations"]:
                    raise ValueError(f"Empty split or duplicate image ID: {split}")
                source_images[split], sequences[split] = set(), set()
                for image in images.values():
                    image_id = image["id"]
                    if image_id in image_ids:
                        raise ValueError("Duplicate global image ID across splits.")
                    image_ids.add(image_id)
                    filename = image["file_name"]
                    relative = PurePosixPath(filename.replace("\\", "/")) if isinstance(filename, str) else None
                    if relative is None or relative.is_absolute() or not relative.parts or any(
                        part in {"..", "."} or ":" in part for part in relative.parts
                    ):
                        raise ValueError(f"Unsafe prepared image filename: {filename!r}")
                    if image["width"] <= 0 or image["height"] <= 0:
                        raise ValueError(f"Invalid prepared image dimensions: {filename}")
                    source = image.get("source")
                    expected_known = [1, 2, 3, 4] if source == "ua_detrac" else [1, 2, 4, 5, 6, 7]
                    if source not in {"bdd100k", "ua_detrac"} or image.get("known_class_ids") != expected_known:
                        raise ValueError(f"Invalid source/known_class_ids for image {image_id}.")
                    if image.get("car_or_van") is not (source != "ua_detrac"):
                        raise ValueError(f"Invalid car_or_van policy for image {image_id}.")
                    name = image.get("source_file_name")
                    if not isinstance(name, str) or not name:
                        raise ValueError(f"Missing source identity for image {image_id}.")
                    identity = (source, name)
                    if identity in source_images[split]:
                        raise ValueError(f"Duplicate source image in {split}: {identity}")
                    source_images[split].add(identity)
                    if source != "ua_detrac" and image.get("source_split") != ("train" if split == "train" else "val"):
                        raise ValueError("Official train/validation partition leakage.")
                    if source == "ua_detrac":
                        sequences[split].add(image["sequence"])
                class_counts = Counter()
                for annotation in document["annotations"]:
                    if annotation["id"] in annotation_ids:
                        raise ValueError("Duplicate annotation ID across splits.")
                    annotation_ids.add(annotation["id"])
                    image = images.get(annotation["image_id"])
                    category_id = annotation["category_id"]
                    if image is None or category_id not in image["known_class_ids"]:
                        raise ValueError(f"Invalid annotation image/category in {split}.")
                    x, y, width, height = annotation["bbox"]
                    if (not all(math.isfinite(value) for value in (x, y, width, height)) or x < 0 or y < 0
                        or width <= 0 or height <= 0 or x + width > image["width"] + 1e-6
                        or y + height > image["height"] + 1e-6 or annotation.get("iscrowd", 0)
                        or annotation.get("ignore", 0)):
                        raise ValueError(f"Invalid prepared bounding box in {split}: {annotation['id']}")
                    class_counts[class_names[category_id - 1]] += 1
                counts = manifest.get("counts", {}).get(split, {})
                expected_counts = {name: class_counts[name] for name in class_names}
                if (counts.get("images") != len(images) or counts.get("annotations") != len(document["annotations"])
                    or counts.get("annotations_per_class") != expected_counts):
                    raise ValueError(f"Prepared manifest counts disagree with {split} labels.")
                summary[split] = {"images": len(images), "annotations": len(document["annotations"]),
                                  "annotations_per_class": expected_counts}
            for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
                if source_images[left] & source_images[right] or sequences[left] & sequences[right]:
                    raise ValueError(f"Prepared image/sequence leakage between {left} and {right}.")
            missing = [name for name, count in summary["train"]["annotations_per_class"].items() if not count]
            if missing:
                raise ValueError(f"No prepared training examples for {missing}. Prepare new data with higher source/frame caps.")
            return manifest, summary
    ''', hidden=True)
    code("config", """
        # @title Prepared dataset and training settings
        from pathlib import Path
        import json
        import re

        PREPARED_DATASET_DIR = "" # @param {type:"string"}
        AUTO_DISCOVER_PREPARED_DATASET = True # @param {type:"boolean"}
        RUN_NAME = "rfdetr_nano_traffic7_v2" # @param {type:"string"}
        RESUME_RUN_DIR = "" # @param {type:"string"}
        SEED = 42
        EPOCHS = 30 # @param {type:"integer"}
        RESOLUTION = 512 # @param [384, 512] {type:"raw"}
        BATCH_SIZE = 4 # @param {type:"integer"}
        GRAD_ACCUM_STEPS = 4 # @param {type:"integer"}
        NUM_WORKERS = 2 # @param {type:"integer"}
        AUTO_RESUME = True # @param {type:"boolean"}
        LR = 1e-4
        LR_ENCODER = 1.5e-4

        if not PREPARED_DATASET_DIR and RUNTIME == "kaggle" and AUTO_DISCOVER_PREPARED_DATASET:
            candidates = discover_prepared_datasets(INPUT_ROOT)
            if len(candidates) != 1:
                raise ValueError(f"Found {len(candidates)} prepared datasets: {[str(path) for path in candidates]}. Set PREPARED_DATASET_DIR explicitly.")
            PREPARED_DATASET_DIR = str(candidates[0])
        DATASET_DIR = Path(PREPARED_DATASET_DIR or INPUT_ROOT / "traffic7_coco").expanduser().resolve()
        PROJECT_DIR = DEFAULT_PROJECT_DIR
        OUTPUT_DIR = PROJECT_DIR / "runs" / RUN_NAME
        CLASS_NAMES = ["car", "bus", "van", "others", "person", "motorcycle", "traffic light"]
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", RUN_NAME):
            raise ValueError("RUN_NAME must use letters, digits, underscores or hyphens (at most 80 characters).")
        if EPOCHS < 1 or BATCH_SIZE < 1 or GRAD_ACCUM_STEPS < 1 or NUM_WORKERS < 0 or RESOLUTION not in {384, 512}:
            raise ValueError("Invalid epochs, batch, worker, or resolution setting.")
        if LR <= 0 or LR_ENCODER <= 0:
            raise ValueError("Learning rates must be positive.")
        if OUTPUT_DIR.resolve().is_relative_to(DATASET_DIR):
            raise ValueError("The result folder must be outside the prepared dataset.")
        if RUNTIME == "kaggle" and OUTPUT_DIR.resolve().is_relative_to(INPUT_ROOT.resolve()):
            raise ValueError("Kaggle input is read-only. Put the run folder outside INPUT_ROOT.")
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Prepared data: {DATASET_DIR}")
        print(f"Run folder: {OUTPUT_DIR}")
        print(f"Nominal effective batch: {BATCH_SIZE * GRAD_ACCUM_STEPS}")
    """)
    markdown("helpers-heading", """
        ## 4. Load prepared data and source-aware training support

        This checks the offline manifest, split annotations, and source-label policies without
        rewriting data or reading all image contents. Preparation already validated and fingerprinted
        images on your computer. The partial-label adapter is required for correct losses, validation
        metrics, and model selection on these mixed sources.
    """)
    code("validate-data", """
        manifest_path = DATASET_DIR / "manifest.json"
        manifest, dataset_summary = load_prepared_dataset(DATASET_DIR, CLASS_NAMES)
        print(json.dumps(dataset_summary, indent=2))
        print(f"Offline dataset fingerprint: {manifest['prepared_dataset_sha256']}")
        print(manifest["evaluation_note"])
    """)
    code("partial-label-support", "# @title Source-aware training and evaluation\n" +
         (ROOT / "helpers" / "partial_labels.py").read_text(encoding="utf-8"), hidden=True)

    markdown("preview-heading", """
        ## 5. Inspect ground-truth labels before training

        Confirm boxes and all seven class counts. UA-DETRAC black rectangles are ignored regions.
        BDD100K car boxes are coarse car-or-van labels. The adapter preserves that distinction.
        Small traffic lights may need more resolution and representative local examples.
    """)
    code("preview-groundtruth", """
        import random
        from collections import defaultdict
        from PIL import Image, ImageDraw
        import matplotlib.pyplot as plt

        train_coco = json.loads((DATASET_DIR / "train" / "_annotations.coco.json").read_text())
        annotations_by_image = defaultdict(list)
        for annotation in train_coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)
        examples = random.Random(SEED).sample(train_coco["images"], min(6, len(train_coco["images"])))
        figure, axes = plt.subplots(2, 3, figsize=(18, 10))
        for axis in axes.flat:
            axis.axis("off")
        for axis, record in zip(axes.flat, examples):
            with Image.open(DATASET_DIR / "train" / record["file_name"]) as source:
                preview = source.convert("RGB")
            draw = ImageDraw.Draw(preview)
            for annotation in annotations_by_image[record["id"]]:
                x, y, w, h = annotation["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline="lime", width=2)
                draw.text((x, max(0, y - 12)), CLASS_NAMES[annotation["category_id"] - 1], fill="lime")
            axis.imshow(preview)
            axis.set_title(f"{record['source']} / {record.get('source_file_name', record['file_name'])}")
        plt.tight_layout()
        plt.show()
    """)
    markdown("training-heading", """
        ## 6. Fine-tune RF-DETR Nano

        Pretrained weights download automatically. Checkpoints/logs go to the run folder printed above.
        `AUTO_RESUME=True` restores `last.ckpt` if present, including optimizer and scheduler state.
        Colab keeps completed epochs in Drive. On Kaggle, save a version with outputs before the session
        ends. Attach the saved outputs to a new session and set `RESUME_RUN_DIR`; the selected run files
        are copied into the writable output directory. Attach the same prepared dataset and keep the run
        name. The offline fingerprint is independent of the upload path; label hashes are checked above
        without reading image bytes. Unsaved checkpoints do not survive a Kaggle session reset.

        On a CUDA out-of-memory error, choose a **new run name**, halve `BATCH_SIZE`, and increase
        `GRAD_ACCUM_STEPS` if you want to retain the nominal effective batch. These batch configurations
        are not mathematically identical training runs. If loading data fails, try `NUM_WORKERS=0`.

        RF-DETR 1.11.1 saves `last.ckpt` each epoch with `checkpoint_interval=5`, and periodic full
        checkpoints every five epochs. The selected inference weights are `checkpoint_best_total.pth`.
        [Released checkpoint implementation](https://github.com/roboflow/rf-detr/blob/1.11.1/src/rfdetr/training/trainer.py)
    """)
    code("train", """
        import importlib.metadata
        import subprocess
        from datetime import datetime, timezone
        from rfdetr import RFDETRNano

        # Install after rfdetr[train] is available, before constructing the trainer.
        # Keep this adapter active for evaluation, including EMA model selection.
        if RESUME_RUN_DIR:
            if not AUTO_RESUME:
                raise ValueError("Enable AUTO_RESUME when specifying RESUME_RUN_DIR.")
            # Once restored, later reruns use the newer writable checkpoints.
            if (OUTPUT_DIR / "run_config.json").is_file() and any(OUTPUT_DIR.glob("*.ckpt")):
                print("Using the existing writable run; saved-input restoration is already complete.")
            else:
                restore_training_run(RESUME_RUN_DIR, OUTPUT_DIR, run_name=RUN_NAME)
        install_partial_label_support(DATASET_DIR)
        run_signature = {
            "model": "RFDETRNano", "rfdetr_version": "1.11.1",
            "resolution": RESOLUTION, "class_names": CLASS_NAMES,
            "dataset_sha256": manifest["prepared_dataset_sha256"],
            "dataset_hash_policy": manifest["dataset_hash_policy"],
            "batch_size": BATCH_SIZE, "grad_accum_steps": GRAD_ACCUM_STEPS,
            "lr": LR, "lr_encoder": LR_ENCODER, "seed": SEED,
            "multi_scale": False, "expanded_scales": False,
            "partial_label_policy": "source_mask_car_van_union_v1",
        }
        run_config_path = OUTPUT_DIR / "run_config.json"
        if run_config_path.is_file():
            prior = json.loads(run_config_path.read_text(encoding="utf-8"))
            if prior["signature"] != run_signature:
                raise ValueError("Run settings/data differ from this saved run. Choose a new RUN_NAME.")

        resume_path = OUTPUT_DIR / "last.ckpt"
        numbered_checkpoints = list(OUTPUT_DIR.glob("checkpoint_*.ckpt"))
        if not resume_path.is_file() and numbered_checkpoints:
            resume_path = max(numbered_checkpoints, key=lambda path: int(path.stem.rsplit("_", 1)[-1]))
        resume = str(resume_path) if AUTO_RESUME and resume_path.is_file() else None
        existing_weights = list(OUTPUT_DIR.glob("*.pth"))
        existing_full_checkpoints = list(OUTPUT_DIR.glob("*.ckpt"))
        if (existing_weights or existing_full_checkpoints) and resume is None:
            raise FileExistsError("This run already has weights. Enable AUTO_RESUME or choose a new RUN_NAME.")
        print(f"Resume checkpoint: {resume or 'new run from COCO-pretrained weights'}")

        run_config = {
            "signature": run_signature, "target_epochs": EPOCHS,
            "updated_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": torch.cuda.get_device_name(0), "torch_version": torch.__version__,
            "runtime": RUNTIME, "run_name": RUN_NAME,
            "dataset_note": manifest["evaluation_note"],
        }
        run_config_path.write_text(json.dumps(run_config, indent=2), encoding="utf-8")
        (OUTPUT_DIR / "class_names.json").write_text(json.dumps({
            "prediction_index_to_name": {str(index): name for index, name in enumerate(CLASS_NAMES)},
            "coco_category_id_to_name": {str(index + 1): name for index, name in enumerate(CLASS_NAMES)},
        }, indent=2), encoding="utf-8")
        shutil.copyfile(manifest_path, OUTPUT_DIR / "dataset_manifest.json")
        # Save a standalone adapter so a training run can be reproduced outside this notebook.
        (OUTPUT_DIR / "partial_labels.py").write_text(PARTIAL_LABEL_SOURCE, encoding="utf-8")
        frozen_packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
        (OUTPUT_DIR / "requirements-runtime.txt").write_text(frozen_packages, encoding="utf-8")

        model = RFDETRNano(
            resolution=RESOLUTION, num_classes=len(CLASS_NAMES), device="cuda", compile=False,
            cuda_graphs=False,
        )
        model.train(
            dataset_dir=str(DATASET_DIR), dataset_file="roboflow", output_dir=str(OUTPUT_DIR),
            epochs=EPOCHS, batch_size=BATCH_SIZE, grad_accum_steps=GRAD_ACCUM_STEPS,
            lr=LR, lr_encoder=LR_ENCODER, num_workers=NUM_WORKERS, class_names=CLASS_NAMES,
            multi_scale=False, expanded_scales=False, amp_dtype="auto",
            checkpoint_interval=5, resume=resume, seed=SEED,
            devices=1,
            run_test=False, tensorboard=True, wandb=False, progress_bar="tqdm",
        )
        print(f"Training finished. Artifacts: {OUTPUT_DIR}")
    """)
    markdown("metrics-heading", """
        ## 7. Inspect training curves (optional)

        TensorBoard shows the training and validation metrics. Use validation results for model selection;
        use the held-out set below once the recipe is selected.
    """)
    code("tensorboard", """
        if RUNTIME == "colab":
            from IPython import get_ipython
            shell = get_ipython()
            shell.run_line_magic("load_ext", "tensorboard")
            shell.run_line_magic("tensorboard", f"--logdir {OUTPUT_DIR}")
        else:
            # Inline scalar plots avoid opening a TensorBoard server in Kaggle.
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            import matplotlib.pyplot as plt
            from collections import defaultdict
            scalar_points = defaultdict(dict)
            for event_file in sorted(OUTPUT_DIR.rglob("events.out.tfevents.*")):
                events = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
                events.Reload()
                for tag in events.Tags().get("scalars", []):
                    for point in events.Scalars(tag):
                        scalar_points[tag][point.step] = point.value
            metric_tags = [tag for tag in sorted(scalar_points) if any(term in tag.lower() for term in ("loss", "map", "precision", "recall"))][:8]
            if not metric_tags:
                metric_tags = sorted(scalar_points)[:8]
            if metric_tags:
                figure, axes = plt.subplots((len(metric_tags) + 1) // 2, 2, figsize=(14, 4 * ((len(metric_tags) + 1) // 2)), squeeze=False)
                for axis in axes.flat:
                    axis.axis("off")
                for axis, tag in zip(axes.flat, metric_tags):
                    points = sorted(scalar_points[tag].items())
                    axis.axis("on")
                    axis.plot([step for step, _ in points], [value for _, value in points])
                    axis.set_title(tag)
                    axis.set_xlabel("step")
                plt.tight_layout()
                plt.show()
            else:
                print(f"No scalar events yet. Training logs: {OUTPUT_DIR}")
    """)
    markdown("evaluation-heading", """
        ## 8. Evaluate the selected weights on held-out source images

        This reloads the best checkpoint rather than evaluating the final epoch's weights. Test metrics
        are detector metrics (AP, precision/recall), not tracking HOTA/IDF1 or calibrated speed error.
        The active adapter scores only each source's known classes and merges car/van predictions
        for BDD100K's coarse car labels. UA-DETRAC scores car and van separately. Pooled AP
        is source-aware and does not prove performance on a fully labeled seven-class camera dataset.
    """)
    code("evaluate", """
        import gc
        from rfdetr.detr import RFDETR

        install_partial_label_support(DATASET_DIR)
        if "model" in globals():
            del model
        gc.collect()
        torch.cuda.empty_cache()
        BEST_WEIGHTS = OUTPUT_DIR / "checkpoint_best_total.pth"
        if not BEST_WEIGHTS.is_file():
            raise FileNotFoundError("Best weights were not saved; inspect training/validation output before proceeding.")
        best_model = RFDETR.from_checkpoint(str(BEST_WEIGHTS), device="cuda")
        test_metrics = best_model.evaluate(
            split="test", dataset_dir=str(DATASET_DIR), dataset_file="roboflow",
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, device="cuda", amp_dtype="auto",
        )
        def metric_json(value):
            if hasattr(value, "item"):
                return value.item()
            raise TypeError(f"Cannot serialize metric type {type(value)}")
        metrics_text = json.dumps(test_metrics, indent=2, default=metric_json)
        (OUTPUT_DIR / "heldout_metrics.json").write_text(metrics_text, encoding="utf-8")
        print(metrics_text)
    """)
    markdown("prediction-heading", """
        ## 9. Preview predictions

        Test a held-out frame and compare predictions with its ground-truth boxes. Prediction class IDs
        are 0–6. Prediction preview shows the original seven-class outputs, including source-unlabeled
        classes; evaluation applies the source-aware rules separately. Change confidence to inspect misses.
    """)
    code("predict", """
        CONFIDENCE = 0.35 # @param {type:"number"}
        test_coco = json.loads((DATASET_DIR / "test" / "_annotations.coco.json").read_text())
        sample_record = random.Random(SEED).choice(test_coco["images"])
        SAMPLE_IMAGE = DATASET_DIR / "test" / sample_record["file_name"]
        detections = best_model.predict(str(SAMPLE_IMAGE), threshold=CONFIDENCE)
        with Image.open(SAMPLE_IMAGE) as source:
            truth_image = source.convert("RGB")
        predicted_image = truth_image.copy()
        truth_draw, prediction_draw = ImageDraw.Draw(truth_image), ImageDraw.Draw(predicted_image)
        for annotation in test_coco["annotations"]:
            if annotation["image_id"] == sample_record["id"]:
                x, y, w, h = annotation["bbox"]
                truth_draw.rectangle((x, y, x + w, y + h), outline="lime", width=2)
                truth_draw.text((x, max(0, y - 12)), CLASS_NAMES[annotation["category_id"] - 1], fill="lime")
        names = detections.data.get("class_name")
        for index, box in enumerate(detections.xyxy):
            class_id = int(detections.class_id[index])
            name = str(names[index]) if names is not None else CLASS_NAMES[class_id]
            confidence = float(detections.confidence[index])
            x1, y1, x2, y2 = map(float, box)
            prediction_draw.rectangle((x1, y1, x2, y2), outline="cyan", width=2)
            prediction_draw.text((x1, max(0, y1 - 12)), f"{name} {confidence:.2f}", fill="cyan")
        figure, axes = plt.subplots(1, 2, figsize=(18, 6))
        for axis, preview, title in zip(axes, [truth_image, predicted_image], ["Ground truth", "Predictions"]):
            axis.imshow(preview)
            axis.set_title(title)
            axis.axis("off")
        plt.tight_layout()
        plt.show()
        predicted_image.save(OUTPUT_DIR / "prediction_preview.jpg")
        print(f"Preview: {OUTPUT_DIR / 'prediction_preview.jpg'}")
    """)
    markdown("bundle-heading", """
        ## 10. Save a portable deployment bundle

        The ZIP contains the selected detector weights, class mapping, training/data configuration,
        held-out metrics, preview, and a minimal inference script. Full resume checkpoints remain in the run folder
        and are deliberately excluded from the small deployment bundle. Run the inference script in an
        environment with `rfdetr==1.11.1`; these `.pth` weights require an RF-DETR runner.
        Inference needs no training adapter; `partial_labels.py` is included for reproducible training/evaluation.
    """)
    code("bundle", r'''
        import zipfile
        from textwrap import dedent

        inference_script = dedent("""
            import argparse
            from pathlib import Path
            from rfdetr.detr import RFDETR

            parser = argparse.ArgumentParser()
            parser.add_argument("image")
            parser.add_argument("--device", default="cuda")
            parser.add_argument("--threshold", type=float, default=0.35)
            args = parser.parse_args()
            weights = Path(__file__).resolve().with_name("checkpoint_best_total.pth")
            model = RFDETR.from_checkpoint(str(weights), device=args.device)
            detections = model.predict(args.image, threshold=args.threshold)
            for index, box in enumerate(detections.xyxy):
                print(detections.data["class_name"][index],
                      float(detections.confidence[index]), box.tolist())
        """).strip() + "\n"
        (OUTPUT_DIR / "infer_image.py").write_text(inference_script, encoding="utf-8")
        BUNDLE_PATH = OUTPUT_DIR / "rfdetr_nano_traffic7_bundle.zip"
        bundle_files = [
            "checkpoint_best_total.pth", "class_names.json", "run_config.json",
            "training_config.json", "dataset_manifest.json", "heldout_metrics.json", "requirements-runtime.txt",
            "prediction_preview.jpg", "infer_image.py", "partial_labels.py",
        ]
        with zipfile.ZipFile(BUNDLE_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name in bundle_files:
                path = OUTPUT_DIR / name
                if not path.is_file():
                    raise FileNotFoundError(f"Missing bundle artifact: {path}")
                archive.write(path, arcname=name)
        print(f"Saved {BUNDLE_PATH} ({BUNDLE_PATH.stat().st_size / 1024**2:.1f} MiB)")
    ''')
    code("download-bundle", """
        if RUNTIME == "colab":
            from google.colab import files
            files.download(str(BUNDLE_PATH))
        else:
            from IPython.display import FileLink, display
            print(f"Bundle: {BUNDLE_PATH}")
            print("Download from Kaggle's Output panel. Save a notebook version with outputs to retain the full run.")
            print(f"Resume checkpoint: {OUTPUT_DIR / 'last.ckpt'}")
            display(FileLink(str(BUNDLE_PATH.resolve().relative_to(Path(KAGGLE_OUTPUT_ROOT).resolve()))))
    """)
    markdown("next-steps", """
        ## Next steps

        - Run the detector on representative real traffic clips and feed boxes/confidences/classes to TrackTrack.
        - Tune association on labeled continuous video; report HOTA/IDF1, identity switches, and count error.
        - Use complete original sequences for tracking evaluation. The sampled/masked detection set above is
          not a replacement for the official tracking evaluation protocol.
        - Metric positions/speeds require camera calibration, physical scale, and source timestamps.
        - Validate motorcycle-heavy traffic and small signals on fully annotated local camera footage.
        - The native source labels are incomplete/coarse; source-aware AP is not full seven-class validation.

        **Files to keep:** `checkpoint_best_total.pth` for inference, `class_names.json` for class mapping,
        `run_config.json` and `dataset_manifest.json` for provenance, and `last.ckpt` for continuing training.
    """)
    # Keep the exact embedded adapter available for export without repository access.
    adapter_source = (ROOT / "helpers" / "partial_labels.py").read_text(encoding="utf-8")
    adapter_cell = next(cell for cell in cells if cell["id"] == "partial-label-support")
    adapter_cell["source"] += "\nPARTIAL_LABEL_SOURCE = " + repr(adapter_source) + "\n"
    return {
        "cells": cells,
        "metadata": {"accelerator": "GPU", "colab": {"name": "train_rfdetr_nano_ua_detrac.ipynb",
                         "provenance": [], "toc_visible": True},
                     "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


if __name__ == "__main__":
    destination = ROOT / "train_rfdetr_nano_ua_detrac.ipynb"
    destination.write_text(json.dumps(build_notebook(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(destination)
