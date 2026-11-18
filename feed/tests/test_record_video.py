"""Verify camera capture timing and sanitized recording artifacts."""

from argparse import Namespace
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

FEED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEED))
import record_video as recorder


class RecordingTests(unittest.TestCase):
    def options(self, folder, **updates):
        args = Namespace(ihanoi="yenphu-hangthan", ihanoi_feeds=folder / "feeds.json",
                         ihanoi_ca_file=None, duration=0.2, fps=10,
                         output=folder / "source.mp4", startup_timeout=1,
                         stale_timeout=1, overwrite=False)
        args.ihanoi_feeds.write_text(json.dumps({args.ihanoi: "ws://camera.invalid/private-session-token"}))
        for key, value in updates.items():
            setattr(args, key, value)
        return args

    def test_fixed_duration_real_video_and_metadata_exclude_session_url(self):
        import cv2
        import numpy as np

        class Frame:
            def to_ndarray(self, format):
                self_format = format
                assert self_format == "bgr24"
                return np.full((32, 48, 3), 120, dtype=np.uint8)

        def stream(url, stopped, on_frame, **kwargs):
            while not stopped.is_set():
                on_frame(Frame())
                stopped.wait(0.025)

        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            args = self.options(folder)
            started = time.monotonic()
            with patch.object(recorder, "stream_frames", side_effect=stream):
                result = recorder.record(args)
            self.assertGreaterEqual(time.monotonic() - started, args.duration)
            self.assertEqual((result["frames"], result["width"], result["height"]), (2, 48, 32))
            self.assertEqual(result["duration_seconds"], 0.2)
            raw = args.output.with_suffix(".json").read_text()
            self.assertNotIn("private-session-token", raw)
            self.assertNotIn("camera.invalid", raw)
            video = cv2.VideoCapture(str(args.output))
            try:
                self.assertEqual(round(video.get(cv2.CAP_PROP_FRAME_COUNT)), 2)
                self.assertEqual(video.get(cv2.CAP_PROP_FPS), 10)
                self.assertTrue(video.read()[0])
            finally:
                video.release()
            self.assertTrue(args.output.with_suffix(".jpg").is_file())
            self.assertFalse((folder / "source.partial.mp4").exists())

    def test_error_is_sanitized_and_does_not_publish_a_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = self.options(Path(temporary))
            with patch.object(recorder, "stream_frames", side_effect=RuntimeError("failed ws://camera.invalid/private-session-token")):
                with self.assertRaisesRegex(RuntimeError, r"failed \[URL\]") as failure:
                    recorder.record(args)
            self.assertNotIn("private-session-token", str(failure.exception))
            self.assertFalse(args.output.exists())

    def test_invalid_options_and_existing_artifacts_fail_before_connect(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for key, value in (("duration", 0), ("duration", float("inf")), ("fps", -1), ("startup_timeout", 0)):
                args = self.options(folder, **{key: value})
                with patch.object(recorder, "stream_frames") as stream, self.assertRaises(ValueError):
                    recorder.record(args)
                stream.assert_not_called()
            args = self.options(folder)
            args.output.write_bytes(b"existing")
            with patch.object(recorder, "stream_frames") as stream, self.assertRaises(FileExistsError):
                recorder.record(args)
            stream.assert_not_called()
            self.assertEqual(args.output.read_bytes(), b"existing")


if __name__ == "__main__":
    unittest.main()
