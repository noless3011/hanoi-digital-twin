"""Record a fixed-duration, unannotated clip from a named iHanoi camera.

The MP4 uses a constant frame rate and the camera's original dimensions.
Frames are sampled against elapsed time, so an irregular network arrival rate
does not speed up or slow down simulation playback. Session URLs stay in memory.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from threading import Condition, Event, Thread
import time

from ihanoi_source import stream_frames
from video_adapter import DEFAULT_IHANOI_CA, DEFAULT_IHANOI_FEEDS, resolve_ihanoi_feed, safe_message


class FrameMailbox:
    """Keep only the latest decoded frame while the recorder follows its clock."""

    def __init__(self):
        self.condition = Condition()
        self.frame = None
        self.sequence = 0
        self.arrived_at = None
        self.error = None
        self.finished = False

    def publish(self, frame):
        image = frame.to_ndarray(format="bgr24")
        with self.condition:
            self.frame = image
            self.sequence += 1
            self.arrived_at = time.monotonic()
            self.condition.notify_all()

    def receive(self, url, stopped, ca_file, resend_on_stall=True):
        try:
            stream_frames(url, stopped, self.publish, ca_file=ca_file, resend_on_stall=resend_on_stall)
        except Exception as exc:
            with self.condition:
                self.error = safe_message(exc)
        finally:
            with self.condition:
                self.finished = True
                self.condition.notify_all()

    def first(self, timeout):
        with self.condition:
            self.condition.wait_for(lambda: self.frame is not None or self.finished, timeout=timeout)
            if self.frame is None:
                raise RuntimeError(self.error or "No camera frame arrived before the startup timeout.")
            return self.frame

    def latest(self, stale_timeout):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            if self.finished:
                raise RuntimeError("The camera stopped before the requested clip was complete.")
            if time.monotonic() - self.arrived_at > stale_timeout:
                raise RuntimeError("Camera frames stopped arriving before the requested clip was complete.")
            return self.frame, self.sequence


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ihanoi", required=True, metavar="NAME", help="Exact camera key in the feed map")
    p.add_argument("--ihanoi-feeds", type=Path, default=DEFAULT_IHANOI_FEEDS)
    p.add_argument("--ihanoi-ca-file", type=Path, default=DEFAULT_IHANOI_CA)
    p.add_argument("--duration", type=float, default=15, help="Clip length in seconds, starting at the first frame")
    p.add_argument("--fps", type=float, default=25, help="MP4 sampling/playback rate (default: 25)")
    p.add_argument("--output", type=Path, required=True, help="Destination .mp4; metadata and preview use the same stem")
    p.add_argument("--startup-timeout", type=float, default=45)
    p.add_argument("--stale-timeout", type=float, default=5, help="Fail if no fresh camera frame arrives for this many seconds")
    p.add_argument("--overwrite", action="store_true")
    return p


def record(args):
    """Record real elapsed camera time, including explicitly reported repeats."""
    for name in ("duration", "fps", "startup_timeout", "stale_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and greater than zero.")
    output = args.output.resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("--output must end in .mp4.")
    metadata_path = output.with_suffix(".json")
    preview_path = output.with_suffix(".jpg")
    partial_path = output.with_name(output.stem + ".partial.mp4")
    destinations = (output, metadata_path, preview_path, partial_path)
    if not args.overwrite and any(path.exists() for path in destinations):
        raise FileExistsError("A recording, metadata, preview, or partial file already exists; choose a new output or use --overwrite.")
    url = resolve_ihanoi_feed(args.ihanoi, args.ihanoi_feeds)
    import cv2

    output.parent.mkdir(parents=True, exist_ok=True)
    stopped, mailbox = Event(), FrameMailbox()
    resend = getattr(args, "resend_on_stall", True)
    worker = Thread(target=mailbox.receive, args=(url, stopped, args.ihanoi_ca_file, resend), daemon=True)
    writer = None
    complete = False
    worker.start()
    try:
        first = mailbox.first(args.startup_timeout)
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(partial_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError("Could not create the MP4 recording.")
        frame_count = math.ceil(args.duration * args.fps)
        repeated = 0
        previous_sequence = None
        preview_indices = {0, frame_count // 2, frame_count - 1}
        preview_frames = []
        captured_at = datetime.now(timezone.utc).isoformat()
        started = time.monotonic()
        for index in range(frame_count):
            stopped.wait(max(0, started + index / args.fps - time.monotonic()))
            frame, sequence = mailbox.latest(args.stale_timeout)
            if frame.shape[:2] != (height, width):
                raise RuntimeError("The camera resolution changed during the recording.")
            repeated += sequence == previous_sequence
            previous_sequence = sequence
            writer.write(frame)
            if index in preview_indices:
                preview_frames.append(frame.copy())
        stopped.wait(max(0, started + args.duration - time.monotonic()))
        elapsed = time.monotonic() - started
        stopped.set()
        writer.release()
        writer = None
        # Inspect the container before publishing a finished source video.
        check = cv2.VideoCapture(str(partial_path))
        try:
            if not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT)) != frame_count:
                raise RuntimeError("The MP4 did not contain the expected number of frames.")
        finally:
            check.release()
        preview = cv2.hconcat(preview_frames)
        if not cv2.imwrite(str(preview_path), preview):
            raise RuntimeError("Could not create the camera preview.")
        metadata = {
            "schema_version": 1,
            "status": "complete",
            "source_type": "ihanoi",
            "feed_name": args.ihanoi,
            "captured_at_utc": captured_at,
            "requested_duration_seconds": args.duration,
            "capture_elapsed_seconds": round(elapsed, 6),
            "duration_seconds": frame_count / args.fps,
            "fps": args.fps,
            "width": width,
            "height": height,
            "frames": frame_count,
            "camera_frames_received": mailbox.sequence,
            "repeated_output_frames": repeated,
            "sampling": "latest decoded camera frame at each constant-rate output timestamp",
            "video_file": output.name,
            "preview_file": preview_path.name,
            "preview": "first, middle, and last output frames from left to right",
        }
        partial_path.replace(output)
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        complete = True
        return metadata
    finally:
        stopped.set()
        if writer is not None:
            writer.release()
        worker.join(timeout=3)
        if not complete:
            partial_path.unlink(missing_ok=True)


def main():
    args = parser().parse_args()
    try:
        result = record(args)
    except (Exception, KeyboardInterrupt) as exc:
        print("Recording failed: " + (safe_message(exc) or "Interrupted."))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
