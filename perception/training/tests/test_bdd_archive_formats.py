"""Check DatasetNinja/Supervisely BDD exports and selective TAR staging."""

import copy
import importlib.util
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from PIL import Image


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, NOTEBOOK_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


traffic = load_module("traffic_bdd_archive_test", "helpers/traffic_datasets.py")
runtime = load_module("runtime_bdd_archive_test", "helpers/colab_runtime.py")
ua_detrac = load_module("ua_bdd_archive_test", "helpers/ua_detrac.py")


class BddArchiveFormatTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        image = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(image, format="JPEG")
        self.image_bytes = image.getvalue()

    def tearDown(self):
        self.temp.cleanup()

    def document(self, kinds=None):
        kinds = kinds or ("car", "bus", "truck", "person", "rider", "motor", "traffic light")
        objects = []
        for number, kind in enumerate(kinds, 1):
            objects.append({
                "id": number, "classId": 1000 + number, "classTitle": kind,
                "geometryType": "rectangle",
                "points": {"exterior": [[10, 12], [29, 36]], "interior": []},
                "tags": [{"name": "attributes", "value":
                          "{'occluded': False, 'truncated': False, 'trafficLightColor': 'none'}"}],
            })
        # Detection preparation must ignore non-target polygons in the same file.
        objects.append({"id": 999, "classTitle": "drivable area", "geometryType": "polygon",
                        "points": {"exterior": [[0, 0], [99, 79], [20, 79]], "interior": []},
                        "tags": []})
        return {"description": "", "size": {"width": 100, "height": 80},
                "tags": [{"name": "weather", "value": "clear"}], "objects": objects}

    def annotation_folders(self, document=None, count=4):
        for split in ("train", "val"):
            folder = self.root / "supervisely" / split / "ann"
            folder.mkdir(parents=True, exist_ok=True)
            for index in range(count):
                (folder / f"bdd_{split}_{index}.jpg.json").write_text(
                    json.dumps(document or self.document()), encoding="utf-8")
        return self.root / "supervisely" / "train" / "ann", self.root / "supervisely" / "val" / "ann"

    def write_tar(self, name, entries, mode="w", extra_members=()):
        path = self.root / name
        with tarfile.open(path, mode) as archive:
            directory = tarfile.TarInfo("./wrapper/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)
            for member_name, data in entries:
                data = data.encode("utf-8") if isinstance(data, str) else data
                info = tarfile.TarInfo(member_name)
                info.size = len(data)
                archive.addfile(info, io.BytesIO(data))
            for info in extra_members:
                archive.addfile(info)
        return path

    def combined_entries(self, count=4):
        entries = [("./wrapper/meta.json", json.dumps({"classes": []}))]
        for split in ("train", "val"):
            for index in range(count):
                name = f"bdd_{split}_{index}.jpg"
                entries.append((f"./wrapper/{split}/ann/{name}.json", json.dumps(self.document())))
                entries.append((f"./wrapper/{split}/img/{name}", self.image_bytes))
        # Original BDD test images are unannotated and must never become our test holdout.
        entries.append(("./wrapper/test/img/official_test.jpg", self.image_bytes))
        entries.append(("./wrapper/train/img/unselected.jpg", self.image_bytes))
        return entries

    def selection(self, records):
        return traffic.select_source_records(records, train_cap=3, valid_cap=2, test_cap=2, seed=42)

    def test_supervisely_aliases_attributes_sizes_and_inclusive_rectangles(self):
        document = self.document()
        document["objects"][5]["points"]["exterior"] = [[10, 12], [29, 12]]
        train, val = self.annotation_folders(document)
        records = traffic.load_bdd_records(train, val)
        self.assertEqual(len(records), 8)
        self.assertEqual({row["source_split"] for row in records}, {"train", "val"})
        for row in records:
            self.assertEqual((row["width"], row["height"]), (100, 80))
            self.assertEqual(row["known_class_ids"], [1, 2, 4, 5, 6, 7])
            self.assertTrue(row["car_or_van"])
            self.assertEqual({item["category_id"] for item in row["annotations"]}, {1, 2, 4, 5, 6, 7})
            self.assertEqual(len([item for item in row["annotations"] if item["category_id"] == 5]), 2)
            car = next(item for item in row["annotations"] if item["category_id"] == 1)
            self.assertEqual(car["bbox"], [10.0, 12.0, 20.0, 25.0])
            motor = next(item for item in row["annotations"] if item["category_id"] == 6)
            self.assertEqual(motor["bbox"], [10.0, 12.0, 20.0, 1.0])
            self.assertIs(motor["source_attributes"]["occluded"], False)
        self.assertTrue(all(row["file_name"].endswith(".jpg") for row in records))

    def test_supervisely_target_geometry_and_attribute_expressions_fail_safely(self):
        document = self.document()
        document["objects"][0]["geometryType"] = "polygon"
        train, val = self.annotation_folders(document)
        with self.assertRaises(ValueError):
            traffic.load_bdd_records(train, val)

        marker = self.root / "must_not_execute.txt"
        document = self.document()
        document["objects"][0]["tags"][0]["value"] = (
            f"__import__('pathlib').Path({str(marker)!r}).touch()")
        (train / "bdd_train_0.jpg.json").write_text(json.dumps(document), encoding="utf-8")
        with self.assertRaises(ValueError):
            traffic.load_bdd_records(train, val)
        self.assertFalse(marker.exists())

    def test_tar_and_compressed_tar_annotations_are_detected_from_contents(self):
        for index, mode in enumerate(("w", "w:gz", "w:bz2", "w:xz")):
            with self.subTest(mode=mode):
                # The extension deliberately does not identify the format.
                archive = self.write_tar(f"native_{index}.data", [
                    ("./wrapper/labels/det_20/det_train.json", "[]"),
                    ("./wrapper/labels/det_20/det_val.json", "[]"),
                    ("./wrapper/train/img/unselected.jpg", self.image_bytes),
                ], mode=mode)
                self.assertEqual(runtime.archive_format(archive), "tar")
                destination = self.root / f"native_out_{index}"
                runtime.extract_json_archive(archive, destination, ("det_train.json", "det_val.json"))
                self.assertEqual(json.loads((destination / "det_train.json").read_text()), [])
                self.assertEqual(json.loads((destination / "det_val.json").read_text()), [])
                self.assertEqual(traffic.resolve_bdd_annotation_paths(destination),
                                 (destination / "det_train.json", destination / "det_val.json"))
                self.assertFalse(list(destination.rglob("*.jpg")))
                runtime.extract_json_archive(archive, destination, ("det_train.json", "det_val.json"))

    def test_combined_supervisely_tar_extracts_annotations_and_only_selected_images(self):
        archive = self.write_tar("dataset.tar", self.combined_entries())
        labels = runtime.extract_bdd_annotation_archive(archive, self.root / "labels")
        self.assertEqual(len(list(labels.rglob("*.jpg.json"))), 8)
        self.assertFalse(list(labels.rglob("*.jpg")))
        paths = traffic.resolve_bdd_annotation_paths(labels)
        self.assertEqual(paths, (labels / "train" / "ann", labels / "val" / "ann"))
        selection = self.selection(traffic.load_bdd_records(*paths))
        staged = traffic.stage_selected_images(selection, self.root / "staged", images_archive=archive)
        selected = [row for rows in selection.values() for row in rows]
        self.assertEqual(len(list(staged.rglob("*.jpg"))), len(selected))
        self.assertFalse(list(staged.rglob("official_test.jpg")))
        self.assertFalse(list(staged.rglob("unselected.jpg")))
        mtimes = {path: path.stat().st_mtime_ns for path in staged.rglob("*.jpg")}
        runtime.extract_bdd_annotation_archive(archive, labels)
        traffic.stage_selected_images(selection, staged, images_archive=archive)
        self.assertEqual({path: path.stat().st_mtime_ns for path in staged.rglob("*.jpg")}, mtimes)
        for row in selected:
            path = staged / "bdd100k" / row["source_split"] / row["file_name"]
            self.assertEqual(path.read_bytes(), self.image_bytes)

    def test_tar_traversal_links_and_special_members_are_rejected(self):
        valid_entries = self.combined_entries()
        malicious = [tarfile.TarInfo("../escape.jpg"), tarfile.TarInfo("/absolute.jpg")]
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.CHRTYPE):
            info = tarfile.TarInfo(f"./wrapper/unsafe_{kind.decode()}")
            info.type = kind
            info.linkname = "../../escape.jpg"
            malicious.append(info)
        train, val = self.annotation_folders()
        selection = self.selection(traffic.load_bdd_records(train, val))
        for index, info in enumerate(malicious):
            with self.subTest(member=info.name, type=info.type):
                archive = self.write_tar(f"malicious_{index}.tar", valid_entries, extra_members=(info,))
                with self.assertRaises(ValueError):
                    runtime.extract_bdd_annotation_archive(archive, self.root / f"unsafe_labels_{index}")
                with self.assertRaises(ValueError):
                    traffic.stage_selected_images(selection, self.root / f"unsafe_images_{index}", images_archive=archive)
        self.assertFalse((self.root / "escape.jpg").exists())

    def test_tar_missing_selected_image_and_duplicate_annotation_are_rejected(self):
        archive = self.write_tar("missing.tar", [("./wrapper/train/img/unselected.jpg", self.image_bytes)])
        train, val = self.annotation_folders()
        selection = self.selection(traffic.load_bdd_records(train, val))
        with self.assertRaises(FileNotFoundError):
            traffic.stage_selected_images(selection, self.root / "missing_images", images_archive=archive)
        duplicate = self.write_tar("duplicate.tar", self.combined_entries() + [
            ("./another/train/ann/bdd_train_0.jpg.json", json.dumps(self.document()))])
        with self.assertRaises(ValueError):
            runtime.extract_bdd_annotation_archive(duplicate, self.root / "duplicate_labels")

    def ua_dataset(self):
        frames, annotations = self.root / "ua_frames", self.root / "ua_xml"
        frames.mkdir()
        annotations.mkdir()
        for index in range(1, 7):
            name = f"MVI_{index:05d}"
            (frames / name).mkdir()
            (frames / name / "img00001.jpg").write_bytes(self.image_bytes)
            targets = "".join(
                f'<target id="{number}"><box left="10" top="12" width="20" height="25"/>'
                f'<attribute vehicle_type="{kind}"/></target>'
                for number, kind in enumerate(("car", "bus", "van", "others"), 1))
            (annotations / f"{name}.xml").write_text(
                f'<sequence name="{name}"><frame num="1"><target_list>{targets}</target_list>'
                '</frame></sequence>', encoding="utf-8")
        output = self.root / "ua_prepared"
        ua_detrac.prepare_ua_detrac(frames, annotations, output, frame_stride=1)
        return output

    def test_tar_supervisely_records_merge_with_coco_and_ua_detrac(self):
        archive = self.write_tar("combined.tar.gz", self.combined_entries(), mode="w:gz")
        labels = runtime.extract_bdd_annotation_archive(archive, self.root / "merge_labels")
        bdd = self.selection(traffic.load_bdd_records(labels / "train" / "ann", labels / "val" / "ann"))
        staged = traffic.stage_selected_images(bdd, self.root / "merge_stage", images_archive=archive)
        coco = copy.deepcopy(bdd)
        coco_images = self.root / "coco_images"
        for rows in coco.values():
            for row in rows:
                row["source"] = "coco"
                row["file_name"] = row["source_file_name"] = "coco_" + row["file_name"]
                row["key"] = f"coco:{row['source_split']}:{row['file_name']}"
                image = coco_images / row["source_split"] / row["file_name"]
                image.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(self.image_bytes)
        traffic.stage_selected_images(coco, staged, images_root=coco_images)
        output = self.root / "merged"
        manifest = traffic.prepare_traffic_dataset(coco, bdd, output, staged, ua_dataset_dir=self.ua_dataset())
        summary = runtime.validate_prepared_dataset(output, class_names=traffic.TRAFFIC_CLASS_NAMES)
        self.assertEqual(sum(item["images"] for item in summary.values()), 20)
        for split in ("train", "valid", "test"):
            self.assertTrue(all(count > 0 for count in manifest["counts"][split]["annotations_per_class"].values()))
            document = json.loads((output / split / "_annotations.coco.json").read_text())
            self.assertEqual({item["source"] for item in document["images"]}, {"coco", "bdd100k", "ua_detrac"})


if __name__ == "__main__":
    unittest.main()
