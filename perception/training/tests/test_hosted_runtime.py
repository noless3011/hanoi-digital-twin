"""Validate Kaggle input discovery and copying saved runs into writable storage."""

import importlib.util
import io
import json
import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def load_hosted_runtime():
    spec = importlib.util.spec_from_file_location(
        "hosted_runtime", NOTEBOOK_ROOT / "helpers" / "hosted_runtime.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class HostedRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.runtime = load_hosted_runtime()

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, path, members):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "w") as archive:
                for name, content in members.items():
                    archive.writestr(name, content)
        else:
            with tarfile.open(path, "w") as archive:
                for name, content in members.items():
                    data = content.encode() if isinstance(content, str) else content
                    member = tarfile.TarInfo(name)
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
        return path

    def saved_run(self, root=None):
        root = root or self.root / "saved" / "test_run"
        root.mkdir(parents=True)
        (root / "run_config.json").write_text(json.dumps({
            "run_name": "test_run", "signature": {"model": "RFDETRNano", "dataset_sha256": "abc"}
        }), encoding="utf-8")
        (root / "last.ckpt").write_bytes(b"optimizer and model state")
        (root / "checkpoint_best_total.pth").write_bytes(b"best inference weights")
        (root / "requirements-runtime.txt").write_text("rfdetr==1.11.1\n", encoding="utf-8")
        (root / "class_names.json").write_text('{"0":"car"}', encoding="utf-8")
        return root

    def test_runtime_auto_detection_and_explicit_override(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(self.runtime.detect_notebook_runtime(kaggle_root=self.root), "colab")
            (self.root / "input").mkdir()
            self.assertEqual(self.runtime.detect_notebook_runtime(kaggle_root=self.root), "kaggle")
            self.assertEqual(self.runtime.detect_notebook_runtime("colab", kaggle_root=self.root), "colab")
        with patch.dict(os.environ, {"KAGGLE_KERNEL_RUN_TYPE": "Interactive"}, clear=True):
            self.assertEqual(self.runtime.detect_notebook_runtime(kaggle_root=self.root / "absent"), "kaggle")
        with self.assertRaises(ValueError):
            self.runtime.detect_notebook_runtime("unsupported")

    def test_discover_attached_bdd_tar_and_ua_zips_without_modifying_inputs(self):
        inputs = self.root / "input"
        bdd = self.archive(inputs / "bdd" / "bdd100k_-images-100k-DatasetNinja.tar", {
            "dataset/train/ann/train.jpg.json": '{"size":{"width":10,"height":10},"objects":[]}',
            "dataset/val/ann/val.jpg.json": '{"size":{"width":10,"height":10},"objects":[]}',
            "dataset/train/img/train.jpg": b"image",
            "dataset/val/img/val.jpg": b"image",
        })
        images = self.archive(inputs / "ua" / "DETRAC-Images.zip", {
            "DETRAC-Images/MVI_20011/img00001.jpg": b"image"
        })
        labels = self.archive(inputs / "ua" / "DETRAC-Train-Annotations-XML.zip", {
            "DETRAC-Train-Annotations-XML/MVI_20011.xml": '<sequence name="MVI_20011"/>'
        })
        before = {path.relative_to(inputs): path.read_bytes() for path in inputs.rglob("*") if path.is_file()}
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["BDD_SOURCE"], "input_archives")
        self.assertEqual(result["BDD_IMAGES_ARCHIVE"], str(bdd))
        self.assertEqual(result["BDD_ANNOTATIONS_ARCHIVE"], str(bdd))
        self.assertEqual(result["DATA_SOURCE"], "input_archives")
        self.assertEqual(result["IMAGES_ZIP"], str(images))
        self.assertEqual(result["TRAIN_XML_ZIP"], str(labels))
        self.assertNotIn("COCO_SOURCE", result)
        self.assertEqual(before, {path.relative_to(inputs): path.read_bytes()
                                 for path in inputs.rglob("*") if path.is_file()})

    def test_discover_extracted_bdd_supervisely_and_ua_folders(self):
        inputs = self.root / "input"
        bdd = inputs / "bdd" / "export"
        for split in ("train", "val"):
            (bdd / split / "ann").mkdir(parents=True)
            (bdd / split / "img").mkdir()
            (bdd / split / "ann" / f"{split}.jpg.json").write_text('{"objects":[]}')
            (bdd / split / "img" / f"{split}.jpg").write_bytes(b"image")
        frames = inputs / "ua" / "DETRAC-Images"
        (frames / "MVI_20011").mkdir(parents=True)
        (frames / "MVI_20011" / "img00001.jpg").write_bytes(b"image")
        xml = inputs / "ua" / "DETRAC-Train-Annotations-XML"
        xml.mkdir()
        (xml / "MVI_20011.xml").write_text('<sequence name="MVI_20011"/>')
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["BDD_SOURCE"], "input_folders")
        self.assertEqual(Path(result["BDD_TRAIN_JSON"]), bdd / "train" / "ann")
        self.assertEqual(Path(result["BDD_VAL_JSON"]), bdd / "val" / "ann")
        self.assertEqual(result["DATA_SOURCE"], "input_folders")
        self.assertEqual(Path(result["FRAMES_FOLDER"]), frames)
        self.assertEqual(Path(result["TRAIN_XML_FOLDER"]), xml)

    def test_discover_double_nested_ua_train_ignores_test_and_toolkit_examples(self):
        inputs = self.root / "input"
        ua = inputs / "ua-detrac"
        frames = ua / "DETRAC-Images" / "DETRAC-Images"
        train = ua / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
        test = ua / "DETRAC-Test-Annotations-XML" / "DETRAC-Test-Annotations-XML"
        toolkit = ua / "DETRAC-MOT-toolkit" / "DETRAC-MOT-toolkit" / "examples"
        for directory, sequences in ((train, ("MVI_20011", "MVI_20012")),
                                     (test, ("MVI_40011",)), (toolkit, ("MVI_20011",))):
            directory.mkdir(parents=True)
            for sequence in sequences:
                (directory / f"{sequence}.xml").write_text(f'<sequence name="{sequence}"/>')
                image = (toolkit if directory == toolkit else frames) / sequence / "img00001.jpg"
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"image")
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["DATA_SOURCE"], "input_folders")
        self.assertEqual(Path(result["FRAMES_FOLDER"]), frames)
        self.assertEqual(Path(result["TRAIN_XML_FOLDER"]), train)

    def test_discover_ua_train_container_with_distributed_annotation_subfolders(self):
        inputs = self.root / "input"
        ua = inputs / "ua-detrac"
        train = ua / "DETRAC-Train-Annotations-XML" / "DETRAC-Train-Annotations-XML"
        frames = ua / "DETRAC-Images" / "DETRAC-Images"
        for group, sequence in (("sunny", "MVI_20011"), ("rainy", "MVI_20012")):
            annotations = train / group
            annotations.mkdir(parents=True)
            (annotations / f"{sequence}.xml").write_text(f'<sequence name="{sequence}"/>')
            image = frames / sequence / "img00001.jpg"
            image.parent.mkdir(parents=True)
            image.write_bytes(b"image")
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["DATA_SOURCE"], "input_folders")
        # The chosen container must include both groups, not a single XML leaf.
        self.assertEqual({path.stem for path in Path(result["TRAIN_XML_FOLDER"]).rglob("MVI_*.xml")},
                         {"MVI_20011", "MVI_20012"})
        self.assertEqual(Path(result["FRAMES_FOLDER"]), frames)

    def test_discover_nested_bdd_export_skips_unlabelled_test_split(self):
        inputs = self.root / "input"
        export = inputs / "bdd100k" / "bdd100k" / "-images-100k-DatasetNinja"
        for split in ("train", "val", "test"):
            (export / split / "ann").mkdir(parents=True)
            (export / split / "img").mkdir()
            objects = [] if split == "test" else [{"classTitle": "person"}]
            (export / split / "ann" / f"{split}.jpg.json").write_text(json.dumps({"objects": objects}))
            (export / split / "img" / f"{split}.jpg").write_bytes(b"image")
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["BDD_SOURCE"], "input_folders")
        self.assertEqual(Path(result["BDD_IMAGES_FOLDER"]), export)
        self.assertEqual(Path(result["BDD_TRAIN_JSON"]), export / "train" / "ann")
        self.assertEqual(Path(result["BDD_VAL_JSON"]), export / "val" / "ann")

    def test_discover_native_bdd_2018_annotations_with_images_train_val(self):
        inputs = self.root / "input"
        bdd = inputs / "bdd100k" / "bdd100k"
        labels = bdd / "labels"
        labels.mkdir(parents=True)
        for split in ("train", "val"):
            (labels / f"bdd100k_labels_images_{split}.json").write_text("[]", encoding="utf-8")
            images = bdd / "images" / split
            images.mkdir(parents=True)
            (images / f"{split}.jpg").write_bytes(b"image")
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertEqual(result["BDD_SOURCE"], "input_folders")
        self.assertEqual(Path(result["BDD_IMAGES_FOLDER"]), bdd / "images")
        self.assertEqual(Path(result["BDD_TRAIN_JSON"]), labels / "bdd100k_labels_images_train.json")
        self.assertEqual(Path(result["BDD_VAL_JSON"]), labels / "bdd100k_labels_images_val.json")

    def test_missing_and_ambiguous_sources_are_not_silently_selected(self):
        inputs = self.root / "input"
        inputs.mkdir()
        self.assertEqual(self.runtime.discover_kaggle_sources(inputs), {})
        self.archive(inputs / "incomplete" / "DETRAC-Images.zip", {"MVI_20011/img00001.jpg": b"image"})
        result = self.runtime.discover_kaggle_sources(inputs)
        self.assertNotIn("DATA_SOURCE", result)
        for export_name in ("first", "second"):
            export = inputs / export_name
            for split in ("train", "val"):
                (export / split / "ann").mkdir(parents=True)
                (export / split / "img").mkdir()
                (export / split / "ann" / f"{split}.jpg.json").write_text('{"objects":[]}')
                (export / split / "img" / f"{split}.jpg").write_bytes(b"image")
        with self.assertRaisesRegex(ValueError, "(?i)(ambiguous|multiple|manual)"):
            self.runtime.discover_kaggle_sources(inputs)

    def test_restore_checkpoint_and_provenance_without_copying_unrelated_inputs(self):
        source = self.saved_run()
        (source / "unrelated.jpg").write_bytes(b"not a training artifact")
        output = self.root / "working" / "runs" / "test_run"
        copied = self.runtime.restore_training_run(source, output, run_name="test_run")
        self.assertTrue(copied)
        self.assertEqual((output / "last.ckpt").read_bytes(), (source / "last.ckpt").read_bytes())
        self.assertEqual((output / "run_config.json").read_bytes(), (source / "run_config.json").read_bytes())
        self.assertTrue((output / "requirements-runtime.txt").is_file())
        self.assertFalse((output / "unrelated.jpg").exists())
        self.runtime.restore_training_run(source, output, run_name="test_run")
        self.assertEqual((output / "last.ckpt").read_bytes(), b"optimizer and model state")

    def test_restore_conflict_fails_before_copying_missing_artifacts(self):
        source = self.saved_run()
        output = self.root / "working" / "runs" / "test_run"
        output.mkdir(parents=True)
        (output / "last.ckpt").write_bytes(b"newer checkpoint already in writable output")
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises((ValueError, FileExistsError)):
            self.runtime.restore_training_run(source, output, run_name="test_run")
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_restore_requires_checkpoint_and_matching_run_name(self):
        source = self.saved_run()
        with self.assertRaises(ValueError):
            self.runtime.restore_training_run(source, self.root / "wrong_run", run_name="wrong_run")
        with self.assertRaises(ValueError):
            self.runtime.restore_training_run(source, self.root / "unsafe", run_name="../unsafe")
        (source / "last.ckpt").unlink()
        with self.assertRaises((ValueError, FileNotFoundError)):
            self.runtime.restore_training_run(source, self.root / "weights_only", run_name="test_run")

    def test_restore_rejects_overlapping_input_and_invalid_provenance_before_writing(self):
        source = self.saved_run()
        with self.assertRaises(ValueError):
            self.runtime.restore_training_run(source, source / "nested_output", run_name="test_run")
        self.assertFalse((source / "nested_output").exists())
        (source / "run_config.json").write_text('{"signature":{}}')
        output = self.root / "invalid_signature_output"
        with self.assertRaises(ValueError):
            self.runtime.restore_training_run(source, output, run_name="test_run")
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
