"""Exercise the offline CLI using tiny archives and extracted dataset folders."""

import contextlib
import hashlib
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image


NOTEBOOK_ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, NOTEBOOK_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


prepare = load_module("local_preparation", "prepare_traffic_dataset.py")
runtime = load_module("local_preparation_runtime", "helpers/colab_runtime.py")
traffic = load_module("local_preparation_traffic", "helpers/traffic_datasets.py")


class LocalPreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.inputs = self.root / "models"
        self.inputs.mkdir()
        buffer = io.BytesIO()
        Image.new("RGB", (100, 80), "white").save(buffer, "JPEG")
        self.image_bytes = buffer.getvalue()

    def tearDown(self):
        self.temp.cleanup()

    def archive(self, name, entries):
        path = self.inputs / name
        if path.suffix == ".zip":
            with zipfile.ZipFile(path, "w") as handle:
                for member, payload in entries:
                    handle.writestr(member, payload)
        else:
            mode = "w:gz" if name.endswith((".tar.gz", ".tgz")) else "w"
            with tarfile.open(path, mode) as handle:
                for member, payload in entries:
                    if isinstance(payload, str):
                        payload = payload.encode()
                    info = tarfile.TarInfo(member)
                    info.size = len(payload)
                    handle.addfile(info, io.BytesIO(payload))
        return path

    def sources(self, *, ua_tar=False, native_bdd=False):
        """Raw fixtures for the user's extraction step; never create COCO data."""
        ua_xml, ua_images = [], []
        targets = "".join(
            f'<target id="{number}"><box left="10" top="12" width="20" height="25"/>'
            f'<attribute vehicle_type="{kind}"/></target>'
            for number, kind in enumerate(("car", "bus", "van", "others"), 1)
        )
        for index in range(1, 7):
            name = f"MVI_{index:05d}"
            ua_xml.append((f"wrapper/DETRAC-Train-Annotations-XML/{name}.xml",
                           f'<sequence name="{name}"><frame num="1"><target_list>{targets}</target_list>'
                           '</frame><frame num="2"><target_list/></frame></sequence>'))
            ua_images.extend((
                (f"DETRAC-Images/{name}/img00001.jpg", self.image_bytes),
                (f"DETRAC-Images/{name}/img00002.jpg", self.image_bytes),
            ))
        ua_images.append(("DETRAC-Images/MVI_99999/img00001.jpg", self.image_bytes))
        suffix = ".tar" if ua_tar else ".zip"
        ua_images_path = self.archive("DETRAC-Images" + suffix, ua_images)
        ua_xml_path = self.archive("DETRAC-Train-Annotations-XML" + suffix, ua_xml)
        bdd_annotations, bdd_images = [], []
        for partition in ("train", "val"):
            frames = []
            for index in range(1, 5):
                name = f"bdd_{partition}_{index}.jpg"
                labels = [{"id": number, "category": category, "attributes": {},
                           "box2d": {"x1": 10, "y1": 12, "x2": 30, "y2": 37}}
                          for number, category in enumerate(
                              ("car", "bus", "truck", "pedestrian", "motorcycle", "traffic light"), 1)]
                if native_bdd:
                    bdd_images.append((f"wrapper/images/100k/{partition}/{name}", self.image_bytes))
                    frames.append({"name": name, "labels": labels})
                else:
                    objects = [{"id": label["id"], "classTitle": label["category"],
                                "geometryType": "rectangle", "tags": [],
                                "points": {"exterior": [[10, 12], [30, 37]], "interior": []}}
                               for label in labels]
                    bdd_annotations.append((f"dataset/{partition}/ann/{name}.json",
                                            json.dumps({"size": {"width": 100, "height": 80},
                                                        "tags": [], "objects": objects})))
                    bdd_images.append((f"dataset/{partition}/img/{name}", self.image_bytes))
            if native_bdd:
                bdd_annotations.append((f"bdd100k/labels/det_{partition}.json", json.dumps(frames)))
        bdd_images.append(("dataset/test/img/official_test.jpg", self.image_bytes))
        if native_bdd:
            images = self.archive("bdd100k_images_100k.zip", bdd_images)
            annotations = self.archive("bdd100k_labels.zip", bdd_annotations)
        else:
            images = annotations = self.archive("bdd100k_-images-100k-DatasetNinja.tar.gz", bdd_images + bdd_annotations)
        return ["--ua-images", str(ua_images_path), "--ua-annotations", str(ua_xml_path),
                "--bdd-images", str(images), "--bdd-annotations", str(annotations)]

    def arguments(self, source_flags):
        return ["--output-dir", str(self.root / "prepared"),
                "--work-dir", str(self.root / "work"), "--expected-ua-sequences", "0",
                "--frame-stride", "1", "--max-frames-per-sequence", "1",
                "--bdd-train-cap", "3", "--bdd-valid-cap", "2", "--bdd-test-cap", "2",
                *source_flags]

    def extracted_sources(self, *, native_bdd=False):
        """Extract fixture downloads as a user would, with independent dataset roots."""
        archive_flags = self.sources(ua_tar=True, native_bdd=native_bdd)
        ua_root = self.root / "extracted_ua"
        bdd_root = self.root / "extracted_bdd"
        for root in (ua_root, bdd_root):
            root.mkdir()
        for root in (ua_root, bdd_root):
            root.with_name(root.name + ".zip.part").write_bytes(b"a separate unfinished archive download")

        extracted = set()
        for flag, destination in (("--ua-images", ua_root), ("--ua-annotations", ua_root),
                                  ("--bdd-images", bdd_root / "outer_wrapper"),
                                  ("--bdd-annotations", bdd_root / "outer_wrapper")):
            archive = Path(archive_flags[archive_flags.index(flag) + 1])
            if archive in extracted:
                continue
            extracted.add(archive)
            if zipfile.is_zipfile(archive):
                with zipfile.ZipFile(archive) as handle:
                    handle.extractall(destination)
            else:
                with tarfile.open(archive) as handle:
                    handle.extractall(destination, filter="data")

        # Folder-mode discovery must ignore archives, even known source names.
        # Leaving originals under --input-dir also catches accidental fallback.
        for flag, destination in (("--ua-images", ua_root), ("--ua-annotations", ua_root),
                                  ("--bdd-images", bdd_root), ("--bdd-annotations", bdd_root)):
            archive = Path(archive_flags[archive_flags.index(flag) + 1])
            if not (destination / archive.name).exists():
                shutil.copyfile(archive, destination / archive.name)
        return ["--ua-dir", str(ua_root), "--bdd-dir", str(bdd_root),
                "--bdd-image-width", "100", "--bdd-image-height", "80"]

    def run_cli(self, argv):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr), \
                patch("urllib.request.urlopen", side_effect=AssertionError("Offline preparation contacted the network")):
            result = prepare.main(argv)
        return result, stdout.getvalue(), stderr.getvalue()

    def assert_dataset_contract(self, output):
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        summary = runtime.validate_prepared_dataset(output, traffic.TRAFFIC_CLASS_NAMES)
        self.assertEqual(sum(row["images"] for row in summary.values()), 13)
        self.assertEqual(manifest["dataset_hash_policy"], "training_content_v2")
        self.assertEqual(manifest["prepared_dataset_sha256"], runtime.prepared_dataset_hash(output))
        self.assertEqual(set(manifest["prepared_annotations_sha256"]), {"train", "valid", "test"})
        self.assertTrue(manifest["partial_label_policy"]["requires_training_adapter"])
        self.assertEqual(set(manifest["sources"]), {"ua_detrac", "bdd100k"})
        self.assertEqual(set(manifest["local_preparation"]["source_identities"]), {"ua_images", "ua_annotations", "bdd_images", "bdd_annotations"})
        for split in ("train", "valid", "test"):
            path = output / split / "_annotations.coco.json"
            self.assertEqual(manifest["prepared_annotations_sha256"][split],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            document = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual({row["source"] for row in document["images"]},
                             {"ua_detrac", "bdd100k"})
            self.assertTrue(all(count > 0 for count in manifest["counts"][split]["annotations_per_class"].values()))
            self.assertFalse(any("99999" in row["file_name"] or "official_test" in row["file_name"]
                                 or "img00002" in row["file_name"] for row in document["images"]))
            policies = {row["id"]: row for row in document["images"]}
            for row in document["annotations"]:
                self.assertIn(row["category_id"], policies[row["image_id"]]["known_class_ids"])
                if row["source"] == "ua_detrac":
                    self.assertIn("track_id", row)
        return manifest

    def test_two_folder_merge_creates_portable_archive_and_reuses_identical_output(self):
        argv = self.arguments(self.extracted_sources()) + ["--archive"]
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        output = self.root / "prepared"
        manifest = self.assert_dataset_contract(output)
        self.assertIn("local_preparation", manifest)
        with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
            self.assertIn("prepared/manifest.json", archive.namelist())
            self.assertIn("prepared/train/_annotations.coco.json", archive.namelist())
            self.assertIsNone(archive.testzip())
            self.assertFalse(any("work/" in name or ".extraction" in name for name in archive.namelist()))
            relocated = self.root / "relocated" / "prepared"
            archive.extractall(relocated.parent)
        self.assertEqual(runtime.prepared_dataset_hash(relocated), manifest["prepared_dataset_sha256"])
        self.assertEqual(runtime.validate_prepared_dataset(relocated, traffic.TRAFFIC_CLASS_NAMES),
                         runtime.validate_prepared_dataset(output, traffic.TRAFFIC_CLASS_NAMES))
        archive_path = output.with_suffix(".zip")
        before = {path: (path.stat().st_mtime_ns, path.read_bytes())
                  for path in [archive_path, *(path for path in output.rglob("*") if path.is_file())]}
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        self.assertEqual(before, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before})

    def test_existing_different_or_malformed_archive_is_refused_without_overwrite(self):
        argv = self.arguments(self.extracted_sources()) + ["--archive"]
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        output = self.root / "prepared"
        archive_path = output.with_suffix(".zip")
        output_snapshot = {path: (path.stat().st_mtime_ns, path.read_bytes())
                           for path in output.rglob("*") if path.is_file()}
        with zipfile.ZipFile(archive_path) as archive:
            original_entries = [(info.filename, archive.read(info.filename)) for info in archive.infolist()]
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            changed_image = False
            for name, payload in original_entries:
                if not changed_image and name.endswith(".jpg"):
                    image = io.BytesIO()
                    Image.new("RGB", (100, 80), "black").save(image, "JPEG")
                    payload = image.getvalue()
                    changed_image = True
                archive.writestr(name, payload)
        for payload in (buffer.getvalue(), b"<html>not a complete archive</html>"):
            with self.subTest(valid_zip=zipfile.is_zipfile(io.BytesIO(payload))):
                archive_path.write_bytes(payload)
                snapshot = archive_path.stat().st_mtime_ns, archive_path.read_bytes()
                result, _, stderr = self.run_cli(argv)
                self.assertEqual(result, 1)
                self.assertRegex(stderr, "archive.*does not match|archive.*exists")
                self.assertEqual(snapshot, (archive_path.stat().st_mtime_ns, archive_path.read_bytes()))
                self.assertEqual(output_snapshot,
                                 {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in output_snapshot})


    def assert_extracted_folder_preparation(self, *, native_bdd):
        source_flags = self.extracted_sources(native_bdd=native_bdd)
        before = {path: (path.stat().st_mtime_ns, path.read_bytes())
                  for name in ("extracted_ua", "extracted_bdd")
                  for path in (self.root / name).rglob("*") if path.is_file()}
        before.update({path: (path.stat().st_mtime_ns, path.read_bytes())
                       for path in self.root.glob("extracted_*.zip.part")})
        result, stdout, stderr = self.run_cli(self.arguments(source_flags))
        self.assertEqual(result, 0, stderr or stdout)
        manifest = self.assert_dataset_contract(self.root / "prepared")
        identities = manifest["local_preparation"]["source_identities"]
        for role in ("ua_images", "ua_annotations", "bdd_images", "bdd_annotations"):
            self.assertEqual(identities[role]["kind"], "folder")
            self.assertTrue(Path(identities[role]["path"]).is_relative_to(
                self.root / ("extracted_ua" if role.startswith("ua_") else "extracted_bdd")))
        self.assertEqual(before, {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in before})
        self.assertFalse(any(path.name == ".extraction" for path in (self.root / "work").rglob("*")))

    def test_two_extracted_folders_support_nested_supervisely_bdd_and_ignore_raw_archives(self):
        self.assert_extracted_folder_preparation(native_bdd=False)

    def test_two_extracted_folders_support_native_bdd_without_coco(self):
        self.assert_extracted_folder_preparation(native_bdd=True)

    def test_extracted_folder_arguments_must_be_paired_before_creating_outputs(self):
        source_flags = self.extracted_sources()
        for omitted in ("--ua-dir", "--bdd-dir"):
            with self.subTest(omitted=omitted):
                flags = source_flags.copy()
                index = flags.index(omitted)
                del flags[index:index + 2]
                result, _, stderr = self.run_cli(self.arguments(flags))
                self.assertEqual(result, 1, stderr)
                self.assertRegex(stderr, r"(?s)(--ua-dir.*--bdd-dir|--bdd-dir.*--ua-dir)")
                self.assertFalse((self.root / "prepared").exists())
                self.assertFalse((self.root / "work").exists())

    def test_extracted_folder_arguments_reject_missing_paths_and_files_before_creating_outputs(self):
        source_flags = self.extracted_sources()
        regular_file = self.root / "dataset.zip"
        regular_file.write_bytes(b"a file is not an extracted dataset directory")
        for flag in ("--ua-dir", "--bdd-dir"):
            for invalid in (self.root / "missing_dataset", regular_file):
                with self.subTest(flag=flag, invalid=invalid.name):
                    flags = source_flags.copy()
                    flags[flags.index(flag) + 1] = str(invalid)
                    result, _, stderr = self.run_cli(self.arguments(flags))
                    self.assertEqual(result, 1, stderr)
                    self.assertRegex(stderr, "(?i)folder|directory|does not exist|missing")
                    self.assertIn(str(invalid), stderr)
                    self.assertFalse((self.root / "prepared").exists())
                    self.assertFalse((self.root / "work").exists())

    def test_extracted_folders_with_missing_annotations_do_not_fall_back_to_archives(self):
        source_flags = self.extracted_sources()
        ua_labels = self.root / "extracted_ua" / "wrapper" / "DETRAC-Train-Annotations-XML"
        bdd_root = self.root / "extracted_bdd"
        shutil.rmtree(ua_labels)
        result, _, stderr = self.run_cli(self.arguments(source_flags))
        self.assertEqual(result, 1, stderr)
        self.assertRegex(stderr, "(?i)ua.*(annotation|xml)|annotation.*ua")
        self.assertFalse((self.root / "prepared").exists())
        self.assertFalse((self.root / "work").exists())

        # Restore the UA labels so the missing BDD annotation pair is evaluated.
        archive = self.inputs / "DETRAC-Train-Annotations-XML.tar"
        with tarfile.open(archive) as handle:
            handle.extractall(self.root / "extracted_ua", filter="data")
        for directory in bdd_root.rglob("ann"):
            shutil.rmtree(directory)
        result, _, stderr = self.run_cli(self.arguments(source_flags))
        self.assertEqual(result, 1, stderr)
        self.assertRegex(stderr, "(?i)bdd.*(annotation|label)|annotation.*bdd")
        self.assertFalse((self.root / "prepared").exists())
        self.assertFalse((self.root / "work").exists())

    def test_same_input_reuses_prepared_files_and_changed_configuration_refuses_overwrite(self):
        argv = self.arguments(self.extracted_sources())
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        output = self.root / "prepared"
        before = {path.relative_to(output): (path.stat().st_mtime_ns, path.read_bytes())
                  for path in output.rglob("*") if path.is_file()}
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        after = {path.relative_to(output): (path.stat().st_mtime_ns, path.read_bytes())
                 for path in output.rglob("*") if path.is_file()}
        self.assertEqual(before, after)
        changed = argv.copy()
        changed[changed.index("--frame-stride") + 1] = "2"
        result, _, stderr = self.run_cli(changed)
        self.assertEqual(result, 1)
        self.assertRegex(stderr, "fresh|changed|different|configuration")
        self.assertEqual({path.relative_to(output): path.read_bytes()
                          for path in output.rglob("*") if path.is_file()},
                         {path: content for path, (_, content) in before.items()})

    def test_reuse_refuses_modified_image_and_missing_manifest_output(self):
        argv = self.arguments(self.extracted_sources())
        result, stdout, stderr = self.run_cli(argv)
        self.assertEqual(result, 0, stderr or stdout)
        output = self.root / "prepared"
        image = next((output / "train").glob("*.jpg"))
        original = image.read_bytes()
        Image.new("RGB", (100, 80), "black").save(image)
        result, _, stderr = self.run_cli(argv)
        self.assertEqual(result, 1)
        self.assertRegex(stderr, "changed|fingerprint|content|modified|checksum")
        image.write_bytes(original)
        (output / "manifest.json").unlink()
        result, _, stderr = self.run_cli(argv)
        self.assertEqual(result, 1)
        self.assertRegex(stderr, "not empty|manifest|fresh|incomplete")

    def test_empty_extracted_folder_refuses_before_creating_outputs(self):
        flags = self.extracted_sources()
        empty = self.root / "empty"
        empty.mkdir()
        flags[flags.index("--ua-dir") + 1] = str(empty)
        result, _, stderr = self.run_cli(self.arguments(flags))
        self.assertEqual(result, 1, stderr)
        self.assertFalse((self.root / "prepared").exists())
        self.assertFalse((self.root / "work").exists())

    def test_discovery_is_extracted_only_and_excludes_coco(self):
        flags = self.extracted_sources()
        for flag, roles in (("--ua-dir", {"ua_images", "ua_annotations"}),
                            ("--bdd-dir", {"bdd_images", "bdd_annotations"})):
            found = prepare.discover_local_sources(flags[flags.index(flag) + 1])
            self.assertEqual(set(found), {"ua_images", "ua_annotations", "bdd_images", "bdd_annotations"})
            for role in roles:
                self.assertEqual(len(found[role]), 1)
                self.assertTrue(found[role][0].is_dir())
        result, stdout, stderr = self.run_cli(self.arguments(flags))
        self.assertEqual(result, 0, stderr or stdout)
        self.assert_dataset_contract(self.root / "prepared")

    def test_coco_input_and_download_flags_are_removed(self):
        for flag in ("--download-coco", "--coco-annotations", "--coco-train-cap"):
            with self.subTest(flag=flag), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    prepare.main([flag])
                self.assertEqual(error.exception.code, 2)
        self.assertNotIn("--download-coco", prepare.build_parser().format_help())


if __name__ == "__main__":
    unittest.main()
