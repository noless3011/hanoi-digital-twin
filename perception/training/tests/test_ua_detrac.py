"""Synthetic checks for the standalone Colab dataset converter."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image


HELPER = Path(__file__).resolve().parents[1] / "helpers" / "ua_detrac.py"
SPEC = importlib.util.spec_from_file_location("ua_detrac", HELPER)
ua_detrac = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ua_detrac)


def target(track_id, kind="car", box=(20, 20, 20, 20)):
    x, y, w, h = box
    return (f'<target id="{track_id}"><box left="{x}" top="{y}" width="{w}" height="{h}"/>'
            f'<attribute vehicle_type="{kind}" truncation_ratio="0"/></target>')


def frame(frame_id, targets):
    return f'<frame num="{frame_id}"><target_list>{targets}</target_list></frame>'


class DetracConverterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.frames = self.root / "frames"
        self.annotations = self.root / "annotations"
        self.frames.mkdir()
        self.annotations.mkdir()
        self.output = self.root / "prepared"

    def tearDown(self):
        self.temp.cleanup()

    def write_sequence(self, name, frame_ids=(1, 2, 6), targets=None, ignored=""):
        sequence_dir = self.frames / "archive-wrapper" / name
        sequence_dir.mkdir(parents=True)
        for frame_id in frame_ids:
            Image.new("RGB", (100, 80), "white").save(sequence_dir / f"img{frame_id:05d}.jpg")
        targets = targets if targets is not None else target(7)
        xml = (f'<sequence name="{name}"><sequence_attribute camera_state="stable"/>'
               f'<ignored_region>{ignored}</ignored_region>'
               + "".join(frame(num, targets) for num in frame_ids) + '</sequence>')
        xml_dir = self.annotations / "nested-wrapper"
        xml_dir.mkdir(exist_ok=True)
        xml_path = xml_dir / f"{name}.xml"
        xml_path.write_text(xml, encoding="utf-8")
        return xml_path

    def populate(self, count=6, **kwargs):
        return [self.write_sequence(f"MVI_{index + 1:05d}", **kwargs) for index in range(count)]

    def all_coco(self, output=None):
        output = output or self.output
        return [json.loads((output / split / "_annotations.coco.json").read_text())
                for split in ("train", "valid", "test")]

    def test_sequence_split_sampling_categories_and_unique_ids(self):
        objects = ''.join(target(index, kind) for index, kind in enumerate(
            ("Car", "BUS", "van", "other", "others"), 1))
        self.populate(targets=objects)
        progress = []
        manifest = ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output,
                                             progress=progress.append)
        datasets = self.all_coco()
        self.assertEqual(len(progress), 6)
        self.assertTrue(all("converted 2 frames" in message for message in progress))
        self.assertNotIn("progress", manifest["config"])
        names = [set(value) for value in manifest["split_sequences"].values()]
        self.assertEqual(set.union(*names), {f"MVI_{index:05d}" for index in range(1, 7)})
        self.assertFalse(names[0] & names[1] or names[0] & names[2] or names[1] & names[2])
        images = [image for data in datasets for image in data["images"]]
        annotations = [ann for data in datasets for ann in data["annotations"]]
        self.assertEqual(len(images), 12)
        self.assertEqual({image["frame_id"] for image in images}, {1, 6})
        self.assertEqual(len({image["id"] for image in images}), len(images))
        self.assertEqual(len({ann["id"] for ann in annotations}), len(annotations))
        self.assertEqual({ann["category_id"] for ann in annotations}, {1, 2, 3, 4})
        for data in datasets:
            image_ids = {image["id"] for image in data["images"]}
            self.assertTrue(all(ann["image_id"] in image_ids for ann in data["annotations"]))
            self.assertEqual([cat["id"] for cat in data["categories"]], [1, 2, 3, 4])
            for image in data["images"]:
                self.assertTrue((self.output / manifest["sequences"][image["sequence"]]["split"]
                                 / image["file_name"]).is_file())
        second_output = self.root / "prepared_again"
        second = ua_detrac.prepare_ua_detrac(self.frames, self.annotations, second_output)
        self.assertEqual(manifest["split_sequences"], second["split_sequences"])

    def test_clip_mask_union_overlap_and_tracking_metadata(self):
        ignored = ('<box left="0" top="0" width="12" height="12"/>'
                   '<box left="0" top="0" width="12" height="12"/>')
        objects = (target(1, "car", (-5.5, 20.25, 15.5, 10.5))
                   + target(2, "bus", (0, 0, 10, 10))
                   + target(3, "van", (0, 0, 20, 20))
                   + target(4, "others", (101, 20, 5, 5)))
        self.populate(3, frame_ids=(1,), targets=objects, ignored=ignored)
        manifest = ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        for data, split in zip(self.all_coco(), ("train", "valid", "test")):
            self.assertEqual(len(data["annotations"]), 2)
            self.assertEqual(data["annotations"][0]["bbox"], [0.0, 20.25, 10.0, 10.5])
            self.assertEqual(data["annotations"][0]["area"], 105.0)
            self.assertEqual(data["annotations"][0]["track_id"], 1)
            # Duplicate ignore rectangles must not double-count overlapping area.
            self.assertAlmostEqual(data["annotations"][1]["ignored_overlap_fraction"], .36)
            self.assertEqual(data["annotations"][1]["track_id"], 3)
            self.assertEqual(manifest["counts"][split]["dropped_ignored_annotations"], 1)
            self.assertEqual(manifest["counts"][split]["dropped_outside_image_annotations"], 1)
            image = data["images"][0]
            self.assertEqual(len(image["ignored_regions_xywh"]), 2)
            with Image.open(self.output / split / image["file_name"]) as masked:
                self.assertLess(max(masked.getpixel((3, 3))), 5)
                self.assertGreater(min(masked.getpixel((70, 60))), 250)
        self.assertIn("not the official", manifest["evaluation_note"])

    def test_cap_preserves_original_frame_numbers(self):
        self.populate(3, frame_ids=(1, 2, 6, 11))
        manifest = ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output,
                                             max_frames_per_sequence=2)
        self.assertTrue(all(seq["selected_frame_ids"] == [1, 6]
                            for seq in manifest["sequences"].values()))

    def test_selected_only_staging_does_not_require_unsampled_or_capped_images(self):
        self.populate(3, frame_ids=(1, 2, 6, 11))
        for sequence_dir in (self.frames / "archive-wrapper").iterdir():
            (sequence_dir / "img00002.jpg").unlink()
            (sequence_dir / "img00011.jpg").unlink()
        manifest = ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output,
                                             max_frames_per_sequence=2)
        self.assertEqual(sum(counts["images"] for counts in manifest["counts"].values()), 6)
        self.assertTrue(all(seq["source_frame_count"] == 4
                            for seq in manifest["sequences"].values()))

    def test_missing_selected_image_fails_before_output(self):
        self.populate(3)
        (self.frames / "archive-wrapper" / "MVI_00001" / "img00006.jpg").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing annotated image"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        self.assertFalse(self.output.exists())

    def test_unknown_label_and_nonfinite_box_fail(self):
        paths = self.populate(3)
        original = paths[0].read_text()
        paths[0].write_text(original.replace('vehicle_type="car"', 'vehicle_type="truck"'))
        with self.assertRaisesRegex(ValueError, "unknown vehicle_type"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        paths[0].write_text(original.replace('left="20"', 'left="nan"'))
        with self.assertRaisesRegex(ValueError, "must be finite"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        paths[0].write_text(original.replace('vehicle_type="car"', 'color="red"'))
        with self.assertRaisesRegex(ValueError, "missing vehicle_type"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)

    def test_duplicate_sequence_frame_target_and_image_folder_fail(self):
        paths = self.populate(3)
        original = paths[0].read_text()
        copy_path = self.annotations / "duplicate.xml"
        copy_path.write_text(original)
        with self.assertRaisesRegex(ValueError, "Duplicate XML sequence"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        copy_path.unlink()
        paths[0].write_text(original.replace('</sequence>', frame(1, target(7)) + '</sequence>'))
        with self.assertRaisesRegex(ValueError, "duplicate frame"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        paths[0].write_text(original.replace('</target_list>', target(7) + '</target_list>'))
        with self.assertRaisesRegex(ValueError, "duplicate target ID"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        paths[0].write_text(original)
        (self.frames / "duplicate-wrapper" / "MVI_00001").mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "Duplicate image sequence folder"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)

    def test_unsafe_output_and_nonempty_output_are_refused(self):
        self.populate(3)
        with self.assertRaisesRegex(ValueError, "outside"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.frames / "prepared")
        self.output.mkdir()
        marker = self.output / "keep.txt"
        marker.write_text("keep")
        with self.assertRaises(FileExistsError):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        self.assertEqual(marker.read_text(), "keep")

    def test_three_sequences_minimum_and_fraction_validation(self):
        self.populate(2)
        with self.assertRaisesRegex(ValueError, "At least 3"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output)
        self.write_sequence("MVI_00003")
        with self.assertRaisesRegex(ValueError, "sum to less than 1"):
            ua_detrac.prepare_ua_detrac(self.frames, self.annotations, self.output,
                                     val_fraction=.6, test_fraction=.5)


if __name__ == "__main__":
    unittest.main()
