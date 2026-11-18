"""Synthetic conversion checks for the combined Colab traffic dataset."""

import copy
import importlib.util
import io
import json
import tempfile
import unittest
import urllib.error
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, NOTEBOOK_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


traffic = load_module("traffic_datasets_test", "helpers/traffic_datasets.py")
ua_detrac = load_module("ua_detrac_traffic_test", "helpers/ua_detrac.py")
runtime = load_module("colab_runtime_traffic_test", "helpers/colab_runtime.py")

CLASS_NAMES = ["car", "bus", "van", "others", "person", "motorcycle", "traffic light"]
COCO_CATEGORIES = [
    {"id": 1, "name": "person"}, {"id": 3, "name": "car"},
    {"id": 4, "name": "motorcycle"}, {"id": 6, "name": "bus"},
    {"id": 8, "name": "truck"}, {"id": 10, "name": "traffic light"},
    {"id": 18, "name": "dog"},
]


class TrafficDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def json_file(self, name, document):
        path = self.root / name
        path.write_text(json.dumps(document), encoding="utf-8")
        return path

    def coco_document(self, count=6, prefix="coco", kinds=(1, 3, 4, 6, 8, 10)):
        images, annotations = [], []
        for number in range(1, count + 1):
            images.append({"id": number, "file_name": f"{prefix}_{number:04d}.jpg", "width": 100,
                           "height": 80, "license": 3,
                           "coco_url": f"https://images.cocodataset.org/{prefix}/{prefix}_{number:04d}.jpg"})
            for category in kinds:
                annotations.append({"id": len(annotations) + 1, "image_id": number,
                                    "category_id": category, "bbox": [10, 12, 20, 25],
                                    "area": 500, "iscrowd": 0})
        return {"images": images, "annotations": annotations, "categories": COCO_CATEGORIES,
                "licenses": [{"id": 3, "name": "CC BY", "url": "https://creativecommons.org/licenses/by/4.0/"}]}

    def bdd_document(self, count=6, prefix="bdd", kinds=None):
        kinds = kinds or ("car", "bus", "truck", "pedestrian", "rider", "motorcycle", "traffic light")
        return [{"name": f"{prefix}_{number:04d}.jpg", "labels": [
            {"id": index, "category": kind, "attributes": {"occluded": False, "truncated": False},
             "box2d": {"x1": 10, "y1": 12, "x2": 29, "y2": 36}}
            for index, kind in enumerate(kinds, 1)
        ]} for number in range(1, count + 1)]

    def coco_records(self, train=None, val=None):
        return traffic.load_coco_records(
            self.json_file("instances_train2017.json", train or self.coco_document(prefix="train2017")),
            self.json_file("instances_val2017.json", val or self.coco_document(prefix="val2017")),
        )

    def bdd_records(self, train=None, val=None):
        return traffic.load_bdd_records(
            self.json_file("bdd100k_labels_images_train.json", train or self.bdd_document(prefix="train")),
            self.json_file("bdd100k_labels_images_val.json", val or self.bdd_document(prefix="val")),
            image_width=100, image_height=80,
        )

    def selected(self, records):
        return traffic.select_source_records(records, train_cap=3, valid_cap=2, test_cap=2, seed=42)

    def write_record_images(self, records, image_root):
        for record in records:
            path = image_root / record["source_split"] / record["file_name"]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (record["width"], record["height"]), "white").save(path)

    def all_coco(self, output):
        return [json.loads((output / split / "_annotations.coco.json").read_text())
                for split in ("train", "valid", "test")]

    def test_coco_and_bdd_mapping_and_partial_label_metadata(self):
        self.assertEqual(list(traffic.TRAFFIC_CLASS_NAMES), CLASS_NAMES)
        self.assertEqual([item["id"] for item in traffic.TRAFFIC_CATEGORIES], list(range(1, 8)))
        coco = self.coco_records()
        bdd = self.bdd_records()
        for records in (coco, bdd):
            self.assertEqual(len(records), 12)
            self.assertEqual({record["source_split"] for record in records}, {"train", "val"})
            for record in records:
                self.assertEqual({ann["category_id"] for ann in record["annotations"]}, {1, 2, 4, 5, 6, 7})
                self.assertNotIn(3, record["known_class_ids"])
                self.assertTrue(record["car_or_van"])
        rider_person = [ann for ann in bdd[0]["annotations"] if ann["category_id"] == 5]
        self.assertEqual(len(rider_person), 2, "Both pedestrian and rider map to person")
        # BDD's official inclusive maximum coordinates become COCO widths/heights.
        self.assertEqual(bdd[0]["annotations"][0]["bbox"], [10.0, 12.0, 20.0, 25.0])
        self.assertEqual(coco[0]["annotations"][0]["bbox"], [10.0, 12.0, 20.0, 25.0])
        self.assertEqual(coco[0]["source_license"]["name"], "CC BY")
        self.assertIn("images.cocodataset.org", coco[0]["source_urls"]["coco_url"])
        aliases = self.bdd_records(train=self.bdd_document(kinds=("person", "motor")))
        self.assertEqual({item["category_id"] for item in aliases[0]["annotations"]}, {5, 6})

    def test_crowd_and_ignore_images_are_excluded_and_nonfinite_boxes_rejected(self):
        coco_train = self.coco_document()
        coco_train["annotations"][0]["iscrowd"] = 1
        coco_train["annotations"][6]["ignore"] = 1
        records = self.coco_records(train=coco_train)
        self.assertEqual(len([r for r in records if r["source_split"] == "train"]), 4)
        bdd_train = self.bdd_document()
        bdd_train[0]["labels"][0]["attributes"]["ignore"] = True
        records = self.bdd_records(train=bdd_train)
        self.assertEqual(len([r for r in records if r["source_split"] == "train"]), 5)
        coco_train = self.coco_document()
        coco_train["annotations"][0]["bbox"][0] = float("nan")
        with self.assertRaises(ValueError):
            self.coco_records(train=coco_train)
        bdd_train = self.bdd_document()
        bdd_train[0]["labels"][0]["box2d"]["x2"] = float("inf")
        with self.assertRaises(ValueError):
            self.bdd_records(train=bdd_train)

    def test_selection_is_deterministic_disjoint_and_balances_new_classes(self):
        records = self.coco_records()
        first = self.selected(records)
        second = self.selected(list(reversed(records)))
        keys = {split: {record["key"] for record in rows} for split, rows in first.items()}
        self.assertEqual(keys, {split: {record["key"] for record in rows} for split, rows in second.items()})
        self.assertEqual({split: len(rows) for split, rows in first.items()}, {"train": 3, "valid": 2, "test": 2})
        self.assertFalse(keys["train"] & keys["valid"] or keys["train"] & keys["test"] or keys["valid"] & keys["test"])
        self.assertTrue(all(record["source_split"] == "train" for record in first["train"]))
        self.assertTrue(all(record["source_split"] == "val" for record in first["valid"] + first["test"]))
        uncapped = traffic.select_source_records(records, train_cap=0, valid_cap=0, test_cap=0, seed=42)
        self.assertEqual({split: len(rows) for split, rows in uncapped.items()}, {"train": 6, "valid": 3, "test": 3})

        # With abundant cars and only one example of each new class, a 3-image
        # budget must not quietly sample away the motorcycle or traffic light.
        rare = self.coco_document(count=20, kinds=(3,))
        for image_id, category in ((1, 1), (2, 4), (3, 10)):
            rare["annotations"].append({"id": len(rare["annotations"]) + 1, "image_id": image_id,
                                        "category_id": category, "bbox": [20, 20, 5, 5], "iscrowd": 0})
        balanced = self.selected(self.coco_records(train=rare))
        selected_categories = {ann["category_id"] for row in balanced["train"] for ann in row["annotations"]}
        self.assertTrue({5, 6, 7}.issubset(selected_categories))

    def test_selective_zip_staging_rejects_missing_images_and_traversal(self):
        selection = self.selected(self.coco_records())
        selected_records = [row for rows in selection.values() for row in rows]
        image_bytes = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(image_bytes, format="JPEG")
        archive_path = self.root / "coco_images.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for row in selected_records:
                archive.writestr(f"wrapper/{row['source_split']}/{row['file_name']}", image_bytes.getvalue())
            archive.writestr("wrapper/train/unselected.jpg", image_bytes.getvalue())
        staged = self.root / "staged"
        traffic.stage_selected_images(selection, staged, images_zip=archive_path)
        self.assertEqual(len(list(staged.rglob("*.jpg"))), len(selected_records))
        self.assertFalse(list(staged.rglob("unselected.jpg")))
        traffic.stage_selected_images(selection, staged, images_zip=archive_path)

        missing = self.root / "missing.zip"
        with zipfile.ZipFile(missing, "w") as archive:
            archive.writestr("unselected.jpg", image_bytes.getvalue())
        with self.assertRaises(FileNotFoundError):
            traffic.stage_selected_images(selection, self.root / "missing_output", images_zip=missing)
        malicious = self.root / "malicious.zip"
        with zipfile.ZipFile(malicious, "w") as archive:
            archive.writestr("../escape.jpg", image_bytes.getvalue())
        with self.assertRaises(ValueError):
            traffic.stage_selected_images(selection, self.root / "malicious_output", images_zip=malicious)
        self.assertFalse((self.root / "escape.jpg").exists())

    def test_interrupted_staging_reuses_files_and_refuses_a_changed_archive(self):
        selection = self.selected(self.coco_records())
        rows = [row for split_rows in selection.values() for row in split_rows]
        image_bytes = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(image_bytes, format="JPEG")
        archive_path = self.root / "resume.zip"
        with zipfile.ZipFile(archive_path, "w") as archive:
            for row in rows:
                archive.writestr(row["file_name"], image_bytes.getvalue())
        staged = self.root / "resume_stage"

        def interrupt(_message):
            raise RuntimeError("Simulated Colab disconnect")

        with self.assertRaisesRegex(RuntimeError, "disconnect"):
            traffic.stage_selected_images(selection, staged, images_zip=archive_path, progress=interrupt)
        completed = list(staged.rglob("*.jpg"))
        self.assertEqual(len(completed), 1)
        completed_mtime = completed[0].stat().st_mtime_ns
        self.assertFalse(list(staged.rglob("*.partial")))
        traffic.stage_selected_images(selection, staged, images_zip=archive_path)
        self.assertEqual(completed[0].stat().st_mtime_ns, completed_mtime)
        self.assertEqual(len(list(staged.rglob("*.jpg"))), len(rows))
        with zipfile.ZipFile(archive_path, "a") as archive:
            archive.writestr("new_unselected.jpg", image_bytes.getvalue())
        with self.assertRaisesRegex(ValueError, "changed"):
            traffic.stage_selected_images(selection, staged, images_zip=archive_path)

    def test_public_coco_download_retries_transient_failures_and_uses_trusted_endpoint(self):
        record = copy.deepcopy(self.coco_records()[0])
        record["source_urls"]["coco_url"] = "https://untrusted.example/image.jpg"
        image_bytes = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(image_bytes, format="JPEG")
        throttled = urllib.error.HTTPError("https://example.invalid", 429, "rate limited", {}, None)
        with patch.object(traffic.urllib.request, "urlopen", side_effect=[
            throttled, urllib.error.URLError("temporary connection failure"), io.BytesIO(image_bytes.getvalue())
        ]) as download, patch.object(traffic.time, "sleep") as backoff:
            traffic.stage_selected_images([record], self.root / "downloaded", allow_coco_download=True)
        self.assertEqual(download.call_count, 3)
        self.assertEqual([call.args[0] for call in backoff.call_args_list], [1, 2])
        expected = f"https://s3.amazonaws.com/images.cocodataset.org/train2017/{record['file_name']}"
        self.assertEqual(download.call_args.args[0].full_url, expected)
        self.assertEqual(len(list((self.root / "downloaded").rglob("*.jpg"))), 1)

        not_found = urllib.error.HTTPError(expected, 404, "not found", {}, None)
        with patch.object(traffic.urllib.request, "urlopen", side_effect=not_found) as download, \
                patch.object(traffic.time, "sleep") as backoff:
            with self.assertRaises(urllib.error.HTTPError):
                traffic.stage_selected_images([record], self.root / "missing_http", allow_coco_download=True)
        download.assert_called_once()
        backoff.assert_not_called()
        self.assertFalse(list((self.root / "missing_http").rglob("*.partial")))

    def ua_dataset(self):
        frames, annotations = self.root / "ua_frames", self.root / "ua_annotations"
        frames.mkdir()
        annotations.mkdir()
        for index in range(1, 7):
            name = f"MVI_{index:05d}"
            (frames / name).mkdir()
            Image.new("RGB", (100, 80), "white").save(frames / name / "img00001.jpg")
            targets = "".join(
                f'<target id="{number}"><box left="10" top="12" width="20" height="25"/>'
                f'<attribute vehicle_type="{kind}"/></target>'
                for number, kind in enumerate(("car", "bus", "van", "others"), 1)
            )
            (annotations / f"{name}.xml").write_text(
                f'<sequence name="{name}"><frame num="1"><target_list>{targets}</target_list>'
                '</frame></sequence>', encoding="utf-8"
            )
        output = self.root / "ua_prepared"
        ua_detrac.prepare_ua_detrac(frames, annotations, output, frame_stride=1)
        return output

    def test_merge_preserves_ua_tracking_adds_all_classes_and_validates_metadata(self):
        coco, bdd = self.selected(self.coco_records()), self.selected(self.bdd_records())
        staged = self.root / "staged"
        for source, records in (("coco", coco), ("bdd", bdd)):
            image_root = self.root / f"{source}_images"
            self.write_record_images([row for rows in records.values() for row in rows], image_root)
            traffic.stage_selected_images(records, staged, images_root=image_root)
        output = self.root / "combined"
        manifest = traffic.prepare_traffic_dataset(
            coco, bdd, output, staged, ua_dataset_dir=self.ua_dataset(), seed=42,
            source_details={"coco": {"license": "source licenses"}, "bdd100k": {"labels": "synthetic"}},
        )
        documents = self.all_coco(output)
        images = [row for document in documents for row in document["images"]]
        annotations = [row for document in documents for row in document["annotations"]]
        self.assertEqual(len({row["id"] for row in images}), len(images))
        self.assertEqual(len({row["id"] for row in annotations}), len(annotations))
        for split, document in zip(("train", "valid", "test"), documents):
            self.assertEqual([row["name"] for row in document["categories"]], CLASS_NAMES)
            self.assertEqual(set(manifest["counts"][split]["annotations_per_class"]), set(CLASS_NAMES))
            self.assertTrue(all(value > 0 for value in manifest["counts"][split]["annotations_per_class"].values()))
            for image in document["images"]:
                self.assertTrue((output / split / image["file_name"]).is_file())
                if image["source"] == "ua_detrac":
                    self.assertEqual(image["known_class_ids"], [1, 2, 3, 4])
                    self.assertEqual(image["frame_id"], 1)
                    self.assertTrue(image["sequence"].startswith("MVI_"))
                else:
                    self.assertNotIn(3, image["known_class_ids"])
                    self.assertTrue(image["car_or_van"])
            for annotation in document["annotations"]:
                image = next(row for row in document["images"] if row["id"] == annotation["image_id"])
                if image["source"] == "ua_detrac":
                    self.assertIn(annotation["track_id"], (1, 2, 3, 4))
        summary = runtime.validate_prepared_dataset(output, class_names=CLASS_NAMES)
        self.assertEqual(sum(item["images"] for item in summary.values()), len(images))
        saved_manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(saved_manifest, manifest)
        self.assertEqual(manifest["sources"]["bdd100k"]["labels"], "synthetic")
        self.assertTrue(manifest["partial_label_policy"]["requires_training_adapter"])

        # Corrupt a source image's label coverage and make sure preflight catches it.
        valid_path = output / "valid" / "_annotations.coco.json"
        damaged = copy.deepcopy(documents[1])
        damaged["images"][0]["known_class_ids"] = [99]
        valid_path.write_text(json.dumps(damaged))
        with self.assertRaises(ValueError):
            runtime.validate_prepared_dataset(output, class_names=CLASS_NAMES)

        valid_path.write_text(json.dumps(documents[1]))
        duplicated = copy.deepcopy(documents[2])
        origin = next(row for row in documents[1]["images"] if row["source"] == "coco")
        target = next(row for row in duplicated["images"] if row["source"] == "coco")
        for name in ("source_split", "source_file_name", "source_image_id"):
            target[name] = origin[name]
        (output / "test" / "_annotations.coco.json").write_text(json.dumps(duplicated))
        with self.assertRaisesRegex(ValueError, "leakage"):
            runtime.validate_prepared_dataset(output, class_names=CLASS_NAMES)


if __name__ == "__main__":
    unittest.main()
