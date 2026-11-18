"""Long recordings: one fresh connection per segment, retries, manifest, and end time."""

from argparse import Namespace
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

FEED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FEED))
import record_session
import record_video


class Frame:
    def to_ndarray(self, format):
        return np.full((32, 48, 3), 90, dtype=np.uint8)


class Camera:
    """Fake stream_frames that records each connection and when it was closed."""

    def __init__(self, fail_first=0):
        self.connections, self.open, self.overlaps, self.fail_first = 0, 0, 0, fail_first
        self.lock = threading.Lock()

    def __call__(self, url, stopped, on_frame, **kwargs):
        with self.lock:
            self.connections += 1
            self.overlaps += self.open > 0
            self.open += 1
            attempt = self.connections
        try:
            if attempt <= self.fail_first:
                raise RuntimeError("The iHanoi camera connection closed before delivering frames. ws://x/secret")
            while not stopped.is_set():
                on_frame(Frame())
                stopped.wait(.02)
        finally:
            with self.lock:
                self.open -= 1


def options(folder, **updates):
    feeds = folder / "feeds.json"
    feeds.write_text(json.dumps({"cam": "ws://camera.invalid/secret-session"}))
    args = Namespace(ihanoi="cam", ihanoi_feeds=feeds, ihanoi_ca_file=None, segment=.3, fps=10,
                     output_dir=folder / "session", reconnect_delay=.05, max_backoff=.01, min_free_gb=0,
                     startup_timeout=1, stale_timeout=1)
    vars(args).update(updates)
    return args


class SessionTests(unittest.TestCase):
    def test_every_segment_uses_its_own_connection(self):
        camera = Camera()
        with tempfile.TemporaryDirectory() as temporary, patch.object(record_video, "stream_frames", camera):
            args = options(Path(temporary))
            end = datetime.now().astimezone() + timedelta(seconds=1.1)
            manifest = record_session.Session(args, end).run(sleep=lambda s: __import__("time").sleep(min(s, .05)))
            self.assertEqual(manifest["status"], "complete")
            self.assertGreaterEqual(len(manifest["segments"]), 2)
            self.assertEqual(camera.connections, len(manifest["segments"]))
            self.assertEqual(camera.overlaps, 0)  # previous connection closed before the next opened
            for segment in manifest["segments"]:
                self.assertTrue((args.output_dir / segment["video_file"]).is_file())
            saved = (args.output_dir / "session.json").read_text()
            self.assertNotIn("secret", saved)
            self.assertEqual(json.loads(saved)["segments"], manifest["segments"])

    def test_failures_reconnect_and_are_logged(self):
        camera = Camera(fail_first=2)
        waits = []
        with tempfile.TemporaryDirectory() as temporary, patch.object(record_video, "stream_frames", camera):
            args = options(Path(temporary), max_backoff=8)
            end = datetime.now().astimezone() + timedelta(seconds=.8)
            manifest = record_session.Session(args, end).run(sleep=waits.append)
            self.assertEqual(len(manifest["failures"]), 2)
            self.assertNotIn("secret", json.dumps(manifest))
            self.assertEqual(waits[:2], [5, 8])  # exponential backoff, capped
            self.assertGreaterEqual(len(manifest["segments"]), 1)
            self.assertEqual(camera.connections, 2 + len(manifest["segments"]))

    def test_last_segment_is_shortened_to_end_time_and_low_disk_stops(self):
        calls = []

        def fake_record(opts):
            calls.append(opts.duration)
            return {"captured_at_utc": "x", "duration_seconds": opts.duration, "camera_frames_received": 1,
                    "repeated_output_frames": 0, "frames": 1}
        start = datetime(2026, 10, 27, 13, 0).astimezone()
        clock = [start]

        def now():
            return clock[0]

        def sleep(seconds):
            clock[0] += timedelta(seconds=seconds)
        with tempfile.TemporaryDirectory() as temporary, patch.object(record_session, "record", side_effect=lambda o: (
                clock.__setitem__(0, clock[0] + timedelta(seconds=o.duration)), fake_record(o))[1]):
            args = options(Path(temporary), segment=120, reconnect_delay=2)
            manifest = record_session.Session(args, start + timedelta(seconds=300)).run(sleep=sleep, now=now)
            self.assertEqual(calls, [120, 120, 56])
            self.assertEqual(manifest["recorded_seconds"], 296)
            with patch.object(record_session.Session, "free_gb", return_value=1.):
                args = options(Path(temporary), output_dir=Path(temporary) / "other", min_free_gb=3)
                manifest = record_session.Session(args, start + timedelta(hours=1)).run(sleep=sleep, now=now)
            self.assertEqual(manifest["status"], "stopped_low_disk")

    def test_until_rolls_to_tomorrow_when_past(self):
        now = datetime(2026, 10, 27, 19, 0).astimezone()
        self.assertEqual(record_session.parse_until("18:00", now).day, 28)
        self.assertEqual(record_session.parse_until("21:30", now).hour, 21)


if __name__ == "__main__":
    unittest.main()
