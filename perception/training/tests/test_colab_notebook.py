"""Exercise the training-only notebook, resume guards and deployment bundling."""

import ast
import builtins
import hashlib
import importlib.util
import json
import shutil
import sys
import tempfile
import types
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, NOTEBOOK_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module("notebook_builder", "build_training_notebook.py")
runtime = load_module("colab_runtime", "helpers/colab_runtime.py")
converter = load_module("detrac_converter", "helpers/ua_detrac.py")
traffic = load_module("notebook_traffic", "helpers/traffic_datasets.py")


def code_cell(cell_id):
    return next(cell["source"] for cell in builder.build_notebook()["cells"] if cell["id"] == cell_id)


class ColabNotebookTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def dataset(self):
        frames, annotations = self.root / "frames", self.root / "annotations"
        frames.mkdir()
        annotations.mkdir()
        for index in range(6):
            name = f"MVI_{index + 1:05d}"
            (frames / name).mkdir()
            Image.new("RGB", (80, 60), "white").save(frames / name / "img00001.jpg")
            (annotations / f"{name}.xml").write_text(
                f'<sequence name="{name}"><frame num="1"><target_list>' +
                ''.join(f'<target id="{number}"><box left="10" top="10" width="20" height="15"/>'
                        f'<attribute vehicle_type="{kind}"/></target>'
                        for number, kind in enumerate(('car', 'bus', 'van', 'others'), 1)) +
                '</target_list></frame></sequence>',
                encoding="utf-8",
            )
        prepared = self.root / "prepared"
        manifest = converter.prepare_ua_detrac(frames, annotations, prepared, frame_stride=1)
        return prepared, manifest

    def test_notebook_is_portable_and_every_cell_compiles(self):
        notebook = builder.build_notebook()
        saved = json.loads((NOTEBOOK_ROOT / "train_rfdetr_nano_ua_detrac.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(saved, notebook, "Rebuild after changing a helper or builder")
        self.assertEqual(notebook["nbformat"], 4)
        self.assertEqual(notebook["metadata"]["accelerator"], "GPU")
        ids = set()
        for cell in notebook["cells"]:
            self.assertNotIn(cell["id"], ids)
            ids.add(cell["id"])
            self.assertIsInstance(cell["source"], str)
            if cell["cell_type"] == "code":
                ast.parse(cell["source"], filename=cell["id"])
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
        self.assertIn("def load_prepared_dataset", code_cell("prepared-data-helpers"))
        self.assertIn("def install_partial_label_support", code_cell("partial-label-support"))
        self.assertIn("import sys", code_cell("gpu-check"))
        removed_ids = {"staging-helpers", "converter", "traffic-converter", "stage-data",
                       "stage-traffic-data", "prepare-data"}
        self.assertFalse(ids & removed_ids)
        definition_names = {node.name for cell in notebook["cells"] if cell["cell_type"] == "code"
                            for node in ast.walk(ast.parse(cell["source"]))
                            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        self.assertFalse(definition_names & {"prepare_ua_detrac", "prepare_traffic_dataset",
                         "stage_selected_images", "extract_annotation_zip", "download_http_archive",
                         "download_drive_zip", "download_http_zip", "prepared_dataset_hash"})

    def test_embedded_helpers_load_without_a_repository_import(self):
        namespace = {"__name__": "__colab__", "CLASS_NAMES": traffic.TRAFFIC_CLASS_NAMES}
        with patch.dict(sys.modules, {"torch": types.ModuleType("torch")}):
            for cell_id in ("runtime-helpers", "prepared-data-helpers", "partial-label-support"):
                exec(compile(code_cell(cell_id), cell_id, "exec"), namespace)
        for name in ("detect_notebook_runtime", "restore_training_run", "discover_prepared_datasets",
                     "load_prepared_dataset", "install_partial_label_support"):
            self.assertTrue(callable(namespace[name]))
        self.assertEqual(list(namespace["CLASS_NAMES"]), traffic.TRAFFIC_CLASS_NAMES)
        self.assertEqual(namespace["PARTIAL_LABEL_SOURCE"],
                         (NOTEBOOK_ROOT / "helpers" / "partial_labels.py").read_text(encoding="utf-8"))

    def mixed_dataset(self):
        ua_prepared, _ = self.dataset()
        stage, images = self.root / "stage", self.root / "source_images"
        selected_sources = []
        for source in ("bdd100k",):
            selected = {split: [] for split in ("train", "valid", "test")}
            for split in selected:
                name = f"{source}_{split}.jpg"
                source_split = "train" if split == "train" else "val"
                image = images / source_split / name
                image.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (80, 60), "white").save(image)
                selected[split].append({"key": f"{source}:{source_split}:{name}", "source": source,
                                       "source_split": source_split, "source_image_id": 1,
                                       "file_name": name, "source_file_name": name,
                                       "width": 80, "height": 60, "known_class_ids": [1, 2, 4, 5, 6, 7],
                                       "car_or_van": True,
                                       "annotations": [{"category_id": category, "bbox": [10, 10, 20, 15],
                                                        "source_annotation_id": category}
                                                       for category in (1, 2, 4, 5, 6, 7)]})
            traffic.stage_selected_images(selected, stage, images_root=images)
            selected_sources.append(selected)
        output = self.root / "mixed_prepared"
        manifest = traffic.prepare_traffic_dataset(None, selected_sources[0], output, stage, ua_dataset_dir=ua_prepared)
        manifest["prepared_dataset_sha256"] = runtime.prepared_dataset_hash(output)
        manifest["dataset_hash_policy"] = "training_content_v2"
        manifest["prepared_annotations_sha256"] = {
            split: hashlib.sha256((output / split / "_annotations.coco.json").read_bytes()).hexdigest()
            for split in ("train", "valid", "test")}
        (output / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return output, manifest

    def test_notebook_rejects_prepared_data_with_a_coco_source(self):
        dataset, manifest = self.mixed_dataset()
        manifest["sources"]["coco"] = {"name": "COCO 2017"}
        (dataset / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        namespace = {"__name__": "__notebook__"}
        exec(compile(code_cell("prepared-data-helpers"), "prepared-data-helpers", "exec"), namespace)
        self.assertEqual(namespace["discover_prepared_datasets"](dataset.parent), [])
        with self.assertRaisesRegex(ValueError, "only BDD100K and UA-DETRAC"):
            namespace["load_prepared_dataset"](dataset, traffic.TRAFFIC_CLASS_NAMES)

    def test_kaggle_storage_and_configuration_use_prepared_inputs_without_colab(self):
        inputs = self.root / "kaggle" / "input"
        output = self.root / "kaggle" / "working"
        scratch = self.root / "scratch"
        inputs.mkdir(parents=True)
        prepared, _ = self.mixed_dataset()
        attached = inputs / "traffic-training" / "traffic7_coco"
        shutil.copytree(prepared, attached)
        before = {path.relative_to(inputs): path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
        namespace = {"__name__": "__kaggle__"}
        for cell_id in ("runtime-helpers", "prepared-data-helpers"):
            exec(compile(code_cell(cell_id), cell_id, "exec"), namespace)
        storage = ast.parse(code_cell("mount-drive"))
        replacements = {"KAGGLE_INPUT_ROOT": inputs, "KAGGLE_OUTPUT_ROOT": output,
                        "KAGGLE_SCRATCH_ROOT": scratch}
        for statement in ast.walk(storage):
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target = statement.targets[0]
                if isinstance(target, ast.Name) and target.id in replacements:
                    statement.value = ast.Constant(str(replacements[target.id]))
        ast.fix_missing_locations(storage)
        original_import = builtins.__import__

        def import_without_colab(name, *args, **kwargs):
            if name == "google.colab" or name.startswith("google.colab."):
                raise AssertionError("Kaggle must not import Google Colab")
            return original_import(name, *args, **kwargs)

        with patch.dict("os.environ", {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}), \
                patch("builtins.__import__", side_effect=import_without_colab), \
                patch("urllib.request.urlopen", side_effect=AssertionError("Prepared inputs must work offline")):
            exec(compile(storage, "mount-drive", "exec"), namespace)
            exec(compile(code_cell("config"), "config", "exec"), namespace)
            exec(compile(code_cell("validate-data"), "validate-data", "exec"), namespace)
        self.assertEqual(namespace["RUNTIME"], "kaggle")
        self.assertEqual(Path(namespace["INPUT_ROOT"]), inputs)
        self.assertEqual(namespace["DATASET_DIR"], attached)
        self.assertTrue(namespace["OUTPUT_DIR"].is_relative_to(output))
        self.assertTrue(namespace["OUTPUT_DIR"].is_dir())
        self.assertEqual(before, {path.relative_to(inputs): path.read_bytes()
                                  for path in inputs.rglob("*") if path.is_file()})
        self.assertNotIn("DATA_SOURCE", namespace)
        self.assertNotIn("FRAME_STRIDE", namespace)
        self.assertEqual(sum(row["images"] for row in namespace["dataset_summary"].values()), 9)

    def test_prepared_metadata_survives_relocation_and_rejects_changed_annotations(self):
        prepared, manifest = self.mixed_dataset()
        namespace = {"__name__": "__colab__"}
        exec(compile(code_cell("prepared-data-helpers"), "prepared-data-helpers", "exec"), namespace)
        relocated = self.root / "attached" / "traffic7_coco"
        shutil.copytree(prepared, relocated)
        original_open = Path.open
        with patch.object(Path, "open", autospec=True, side_effect=original_open) as opened:
            # Validation reads small metadata; it must never scan image contents on Drive.
            namespace["load_prepared_dataset"](relocated, traffic.TRAFFIC_CLASS_NAMES)
            self.assertFalse(any(Path(call.args[0]).suffix.lower() in {".jpg", ".jpeg", ".png"}
                                 for call in opened.call_args_list))
        annotation = relocated / "train" / "_annotations.coco.json"
        document = json.loads(annotation.read_text())
        document["annotations"][0]["bbox"][0] += 1
        annotation.write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "changed|digest|checksum|fingerprint|SHA|hash"):
            namespace["load_prepared_dataset"](relocated, traffic.TRAFFIC_CLASS_NAMES)

    def test_prepared_discovery_excludes_ua_intermediate_and_unrelated_manifest_roots(self):
        prepared, _ = self.mixed_dataset()
        unrelated = self.root / "unrelated_dataset"
        shutil.copytree(prepared, unrelated)
        (unrelated / "manifest.json").write_text('{"purpose": "unrelated export"}', encoding="utf-8")
        malformed = self.root / "unfinished_dataset"
        shutil.copytree(prepared, malformed)
        (malformed / "manifest.json").write_text("{not finished", encoding="utf-8")
        namespace = {"__name__": "__colab__"}
        exec(compile(code_cell("prepared-data-helpers"), "prepared-data-helpers", "exec"), namespace)
        # mixed_dataset also leaves its valid four-class UA intermediate under prepared/.
        self.assertTrue((self.root / "prepared" / "manifest.json").is_file())
        self.assertEqual(namespace["discover_prepared_datasets"](self.root), [prepared])

    def test_selective_extraction_skips_test_and_unsampled_images(self):
        annotations_zip = self.root / "annotations.zip"
        with zipfile.ZipFile(annotations_zip, "w") as archive:
            for index in range(3):
                name = f"MVI_{index + 1:05d}"
                archive.writestr(f"wrapper/{name}.xml", f'<sequence name="{name}">'
                                 '<frame num="1"/><frame num="2"/><frame num="6"/></sequence>')
        annotations = runtime.extract_annotation_zip(annotations_zip, self.root / "raw_annotations")
        wanted, sequences = runtime.selected_frame_keys(annotations, 5, 1)
        self.assertEqual(len(sequences), 3)
        self.assertEqual(len(wanted), 3)
        images_zip = self.root / "images.zip"
        with zipfile.ZipFile(images_zip, "w") as archive:
            for key in wanted:
                archive.writestr(f"DETRAC-Images/{key}", b"selected")
                second_frame = key.rsplit("/", 1)[0] + "/img00002.jpg"
                archive.writestr(f"DETRAC-Images/{second_frame}", b"unused")
            archive.writestr("DETRAC-Images/MVI_99999/img00001.jpg", b"official test")
        output = runtime.extract_selected_images(images_zip, self.root / "raw_frames", wanted)
        self.assertEqual(len(list(output.rglob("*.jpg"))), 3)
        self.assertFalse((output / "MVI_99999").exists())
        runtime.extract_selected_images(images_zip, output, wanted)
        with self.assertRaises(FileExistsError):
            extra = {key.rsplit("/", 1)[0] + "/img00002.jpg" for key in wanted}
            runtime.extract_selected_images(images_zip, output, wanted | extra)

    def test_archive_validation_rejects_html_missing_frames_and_traversal(self):
        html = self.root / "not-a-zip.zip"
        html.write_text("<html>download quota exceeded</html>")
        with self.assertRaises(ValueError):
            runtime.extract_annotation_zip(html, self.root / "html_output")
        bad = self.root / "unsafe.zip"
        with zipfile.ZipFile(bad, "w") as archive:
            archive.writestr("../escape.xml", "<sequence/>")
        with self.assertRaises(ValueError):
            runtime.extract_annotation_zip(bad, self.root / "bad_output")
        missing = self.root / "missing.zip"
        with zipfile.ZipFile(missing, "w") as archive:
            archive.writestr("MVI_00001/img00002.jpg", b"wrong frame")
        with self.assertRaises(FileNotFoundError):
            runtime.extract_selected_images(missing, self.root / "missing_output", {"MVI_00001/img00001.jpg"})

    def test_coco_validation_detects_sequence_leakage(self):
        dataset, _ = self.dataset()
        summary = runtime.validate_prepared_dataset(dataset)
        self.assertEqual(sum(item["images"] for item in summary.values()), 6)
        valid_json = dataset / "valid" / "_annotations.coco.json"
        valid = json.loads(valid_json.read_text())
        train = json.loads((dataset / "train" / "_annotations.coco.json").read_text())
        valid["images"][0]["sequence"] = train["images"][0]["sequence"]
        valid_json.write_text(json.dumps(valid))
        with self.assertRaisesRegex(ValueError, "leakage"):
            runtime.validate_prepared_dataset(dataset)

    def test_resume_dataset_hash_survives_host_relocation_and_detects_content_changes(self):
        dataset, _ = self.dataset()
        original_hash = runtime.prepared_dataset_hash(dataset)
        relocated = self.root / "kaggle_working" / "prepared"
        shutil.copytree(dataset, relocated)
        manifest_path = relocated / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["source_paths"] = {"frames": "/kaggle/input/ua/frames"}
        manifest["updated_utc"] = "later session"
        manifest_path.write_text(json.dumps(manifest, indent=4))
        self.assertEqual(runtime.prepared_dataset_hash(relocated), original_hash)

        annotation_path = relocated / "train" / "_annotations.coco.json"
        original_document = json.loads(annotation_path.read_text())
        # Whitespace and dictionary ordering do not affect training semantics.
        annotation_path.write_text(json.dumps(original_document, indent=5, sort_keys=True))
        self.assertEqual(runtime.prepared_dataset_hash(relocated), original_hash)
        changed_document = json.loads(annotation_path.read_text())
        changed_document["annotations"][0]["bbox"][0] += 1
        annotation_path.write_text(json.dumps(changed_document))
        self.assertNotEqual(runtime.prepared_dataset_hash(relocated), original_hash)
        annotation_path.write_text(json.dumps(original_document))
        image_path = relocated / "train" / original_document["images"][0]["file_name"]
        Image.new("RGB", (80, 60), "black").save(image_path)
        self.assertNotEqual(runtime.prepared_dataset_hash(relocated), original_hash)

    def training_namespace(self):
        dataset, manifest = self.mixed_dataset()
        output = self.root / "run"
        output.mkdir()
        calls = []

        class FakeNano:
            def __init__(self, **kwargs):
                calls.append(("constructor", kwargs))

            def train(self, **kwargs):
                calls.append(("train", kwargs))
                (output / "last.ckpt").write_bytes(b"fake resumable checkpoint")
                (output / "checkpoint_best_total.pth").write_bytes(b"fake inference weights")
                (output / "training_config.json").write_text(json.dumps(kwargs))

        fake_rfdetr = types.ModuleType("rfdetr")
        fake_rfdetr.RFDETRNano = FakeNano
        namespace = {
            "DATASET_DIR": dataset, "OUTPUT_DIR": output,
            "RESOLUTION": 384,
            "CLASS_NAMES": ["car", "bus", "van", "others", "person", "motorcycle", "traffic light"],
            "BATCH_SIZE": 4, "GRAD_ACCUM_STEPS": 4, "EPOCHS": 2,
            "LR": 1e-4, "LR_ENCODER": 1.5e-4, "SEED": 42, "NUM_WORKERS": 2,
            "AUTO_RESUME": True, "manifest": manifest, "manifest_path": dataset / "manifest.json",
            "RUNTIME": "colab", "RESUME_RUN_DIR": "", "RUN_NAME": "test_run",
            "restore_training_run": Mock(return_value=[]),
            "prepared_dataset_hash": runtime.prepared_dataset_hash,
            "install_partial_label_support": Mock(return_value=None),
            "PARTIAL_LABEL_SOURCE": "# Synthetic adapter bundled by the portable notebook test.\n",
            "json": json, "shutil": runtime.shutil, "sys": sys,
            "torch": types.SimpleNamespace(__version__="test", cuda=types.SimpleNamespace(get_device_name=lambda _: "test GPU")),
        }
        return namespace, fake_rfdetr, calls

    def test_training_cell_fresh_resume_and_configuration_guards(self):
        namespace, fake_rfdetr, calls = self.training_namespace()
        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}), patch("subprocess.check_output", return_value="rfdetr==1.11.1\n"):
            exec(compile(code_cell("train"), "train", "exec"), namespace)
            train_call = calls[-1][1]
            self.assertIsNone(train_call["resume"])
            self.assertEqual(train_call["checkpoint_interval"], 5)
            self.assertFalse(train_call["run_test"])
            self.assertEqual(train_call["dataset_file"], "roboflow")
            self.assertEqual(calls[-2][1]["num_classes"], 7)
            self.assertEqual(train_call["class_names"], namespace["CLASS_NAMES"])
            namespace["install_partial_label_support"].assert_called_once_with(namespace["DATASET_DIR"])
            exec(compile(code_cell("train"), "train", "exec"), namespace)
            self.assertEqual(calls[-1][1]["resume"], str(namespace["OUTPUT_DIR"] / "last.ckpt"))
            mapping = json.loads((namespace["OUTPUT_DIR"] / "class_names.json").read_text())
            self.assertEqual(mapping["prediction_index_to_name"]["0"], "car")
            self.assertEqual(mapping["coco_category_id_to_name"]["4"], "others")
            self.assertEqual(mapping["prediction_index_to_name"]["6"], "traffic light")
            self.assertEqual(mapping["coco_category_id_to_name"]["7"], "traffic light")
            namespace["BATCH_SIZE"] = 2
            with self.assertRaisesRegex(ValueError, "settings/data differ"):
                exec(compile(code_cell("train"), "train", "exec"), namespace)
            namespace["BATCH_SIZE"] = 4
            namespace["AUTO_RESUME"] = False
            with self.assertRaises(FileExistsError):
                exec(compile(code_cell("train"), "train", "exec"), namespace)

    def test_kaggle_training_restores_saved_run_before_configuration_and_resume_checks(self):
        namespace, fake_rfdetr, calls = self.training_namespace()
        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}), patch("subprocess.check_output", return_value="rfdetr==1.11.1\n"):
            exec(compile(code_cell("train"), "train", "exec"), namespace)
            source = self.root / "kaggle_input" / "previous_run"
            shutil.copytree(namespace["OUTPUT_DIR"], source)
            for child in namespace["OUTPUT_DIR"].iterdir():
                child.unlink()
            hosted = load_module("hosted_runtime_resume", "helpers/hosted_runtime.py")
            namespace["restore_training_run"] = Mock(wraps=hosted.restore_training_run)
            namespace["RUNTIME"] = "kaggle"
            namespace["RESUME_RUN_DIR"] = str(source)
            exec(compile(code_cell("train"), "train", "exec"), namespace)
            namespace["restore_training_run"].assert_called_once()
            arguments = namespace["restore_training_run"].call_args
            self.assertEqual(Path(arguments.args[0]), source)
            self.assertEqual(Path(arguments.args[1]), namespace["OUTPUT_DIR"])
            self.assertEqual(arguments.kwargs["run_name"], namespace["RUN_NAME"])
            self.assertEqual(calls[-1][1]["resume"], str(namespace["OUTPUT_DIR"] / "last.ckpt"))
            self.assertTrue((namespace["OUTPUT_DIR"] / "requirements-runtime.txt").is_file())
            # A later cell rerun must use the newer writable checkpoint rather
            # than copying the older read-only input over it again.
            (namespace["OUTPUT_DIR"] / "last.ckpt").write_bytes(b"newer writable optimizer state")
            namespace["EPOCHS"] += 1
            exec(compile(code_cell("train"), "train", "exec"), namespace)
            namespace["restore_training_run"].assert_called_once()
            self.assertEqual(calls[-1][1]["resume"], str(namespace["OUTPUT_DIR"] / "last.ckpt"))

    def test_deployment_bundle_and_generated_inference_script(self):
        namespace, fake_rfdetr, _ = self.training_namespace()
        with patch.dict(sys.modules, {"rfdetr": fake_rfdetr}), patch("subprocess.check_output", return_value="rfdetr==1.11.1\n"):
            exec(compile(code_cell("train"), "train", "exec"), namespace)
        output = namespace["OUTPUT_DIR"]
        (output / "heldout_metrics.json").write_text('{"test/mAP_50_95": 0.5}')
        Image.new("RGB", (80, 60)).save(output / "prediction_preview.jpg")
        exec(compile(code_cell("bundle"), "bundle", "exec"), namespace)
        ast.parse((output / "infer_image.py").read_text(), filename="infer_image.py")
        with zipfile.ZipFile(namespace["BUNDLE_PATH"]) as archive:
            self.assertIn("checkpoint_best_total.pth", archive.namelist())
            self.assertIn("training_config.json", archive.namelist())
            self.assertIn("requirements-runtime.txt", archive.namelist())
            self.assertIn("class_names.json", archive.namelist())
            self.assertNotIn("last.ckpt", archive.namelist())
            self.assertIsNone(archive.testzip())

    def test_kaggle_bundle_download_uses_a_relative_file_link_without_colab(self):
        output_root = self.root / "kaggle" / "working"
        bundle = output_root / "traffic_training" / "runs" / "test_run" / "bundle.zip"
        bundle.parent.mkdir(parents=True)
        bundle.write_bytes(b"download artifact")
        display_module = types.ModuleType("IPython.display")
        display_module.FileLink = Mock(return_value="download link")
        display_module.display = Mock()
        namespace = {"RUNTIME": "kaggle", "BUNDLE_PATH": bundle, "OUTPUT_DIR": bundle.parent,
                     "KAGGLE_OUTPUT_ROOT": str(output_root), "Path": Path}
        original_import = builtins.__import__

        def import_without_colab(name, *args, **kwargs):
            if name == "google.colab" or name.startswith("google.colab."):
                raise AssertionError("Kaggle downloads must not import Google Colab")
            return original_import(name, *args, **kwargs)

        with patch.dict(sys.modules, {"IPython.display": display_module}), \
                patch("builtins.__import__", side_effect=import_without_colab):
            exec(compile(code_cell("download-bundle"), "download-bundle", "exec"), namespace)
        display_module.FileLink.assert_called_once()
        linked_path = Path(display_module.FileLink.call_args.args[0])
        self.assertFalse(linked_path.is_absolute())
        self.assertEqual(linked_path, bundle.relative_to(output_root))
        display_module.display.assert_called_once_with("download link")

    def test_training_curves_use_inline_scalars_on_kaggle_and_tensorboard_magic_on_colab(self):
        output = self.root / "run"
        output.mkdir()
        event_file = output / "events.out.tfevents.test"
        event_file.write_bytes(b"event file represented by the fake accumulator")
        event_module = types.ModuleType("tensorboard.backend.event_processing.event_accumulator")
        events = Mock()
        events.Tags.return_value = {"scalars": ["train/loss", "valid/mAP"]}
        events.Scalars.return_value = [types.SimpleNamespace(step=2, value=.5),
                                       types.SimpleNamespace(step=1, value=.8)]
        event_module.EventAccumulator = Mock(return_value=events)
        plot_module = types.ModuleType("matplotlib.pyplot")
        axes = [Mock(), Mock()]
        plot_module.subplots = Mock(return_value=(Mock(), types.SimpleNamespace(flat=axes)))
        plot_module.tight_layout = Mock()
        plot_module.show = Mock()
        shell = Mock()
        ipython_module = types.ModuleType("IPython")
        ipython_module.get_ipython = Mock(return_value=shell)
        with patch.dict(sys.modules, {
            "tensorboard.backend.event_processing.event_accumulator": event_module,
            "matplotlib.pyplot": plot_module, "IPython": ipython_module,
        }):
            exec(compile(code_cell("tensorboard"), "tensorboard", "exec"),
                 {"RUNTIME": "kaggle", "OUTPUT_DIR": output})
            for axis in axes:
                axis.plot.assert_called_once_with([1, 2], [.8, .5])
            plot_module.show.assert_called_once()
            shell.run_line_magic.assert_not_called()
            exec(compile(code_cell("tensorboard"), "tensorboard", "exec"),
                 {"RUNTIME": "colab", "OUTPUT_DIR": output})
        self.assertEqual(shell.run_line_magic.call_args_list[0].args, ("load_ext", "tensorboard"))
        self.assertEqual(shell.run_line_magic.call_args_list[1].args,
                         ("tensorboard", f"--logdir {output}"))


if __name__ == "__main__":
    unittest.main()
