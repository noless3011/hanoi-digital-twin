"""Record a named iHanoi camera for hours as a series of short clips.

Every segment opens a brand-new WebSocket connection, records --segment seconds,
and closes the connection before the next segment connects again. A failed
segment (stalled camera, dropped connection, expired URL) is logged and retried
with a fresh connection after a backoff, so one bad minute does not end the session.
The feed map is re-read for each segment: paste a new session URL into
feed/ihanoi.json and the next connection uses it without restarting.

Each clip is a normal record_video.py artifact (MP4 + JSON + JPEG preview).
session.json in the output folder lists every segment, failure and gap.
"""

from __future__ import annotations

import argparse
from argparse import Namespace
from datetime import datetime, timedelta
import json
import math
import os
from pathlib import Path
import shutil
import time

from record_video import record
from video_adapter import DEFAULT_IHANOI_CA, DEFAULT_IHANOI_FEEDS, safe_message

ROOT = Path(__file__).resolve().parents[1]


def parse_until(text, now=None):
    """HH:MM today in local time (tomorrow if already past)."""
    now = now or datetime.now().astimezone()
    try:
        hours, minutes = (int(part) for part in text.split(":"))
        end = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError("--until needs local time as HH:MM, for example 18:00") from None
    return end if end > now else end + timedelta(days=1)


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ihanoi", required=True, metavar="NAME", help="Exact camera key in the feed map")
    p.add_argument("--ihanoi-feeds", type=Path, default=DEFAULT_IHANOI_FEEDS)
    p.add_argument("--ihanoi-ca-file", type=Path, default=DEFAULT_IHANOI_CA)
    p.add_argument("--until", default="18:00", help="Local end time HH:MM (default 18:00)")
    p.add_argument("--hours", type=float, help="Record for this many hours instead of --until")
    p.add_argument("--segment", type=float, default=120, help="Seconds per clip/connection (default 120)")
    p.add_argument("--fps", type=float, default=25)
    p.add_argument("--output-dir", type=Path, help="Session folder; default data/<feed>-<date>-session")
    p.add_argument("--reconnect-delay", type=float, default=2,
                   help="Seconds between closing one connection and opening the next (default 2)")
    p.add_argument("--max-backoff", type=float, default=120, help="Longest wait after repeated failures")
    p.add_argument("--min-free-gb", type=float, default=3, help="Stop when the output drive has less free space")
    p.add_argument("--startup-timeout", type=float, default=45)
    p.add_argument("--stale-timeout", type=float, default=5)
    return p


class Session:
    def __init__(self, args, end):
        self.args, self.end = args, end
        stamp = datetime.now().strftime("%Y%m%d")
        self.folder = (args.output_dir or ROOT / "data" / f"{args.ihanoi}-{stamp}-session").resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.folder / "session.json"
        self.manifest = {"schema_version": 1, "feed_name": args.ihanoi, "segment_seconds": args.segment,
                         "fps": args.fps, "until_local": end.isoformat(timespec="seconds"),
                         "started_local": datetime.now().astimezone().isoformat(timespec="seconds"),
                         "status": "recording", "segments": [], "failures": []}
        if self.manifest_path.exists():  # resume an interrupted session in the same folder
            previous = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            self.manifest["segments"] = previous.get("segments", [])
            self.manifest["failures"] = previous.get("failures", [])
            self.manifest["started_local"] = previous.get("started_local", self.manifest["started_local"])
        self.index = 1 + max((s["index"] for s in self.manifest["segments"]), default=0)
        self.index = max(self.index, 1 + max((f["index"] for f in self.manifest["failures"]), default=0))

    def save(self):
        temporary = self.manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.manifest, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def free_gb(self):
        return shutil.disk_usage(self.folder).free / 1e9

    def run(self, sleep=time.sleep, now=None):
        now = now or (lambda: datetime.now().astimezone())
        args, failures_in_row = self.args, 0
        print(f"Recording {args.ihanoi} until {self.end:%H:%M} in {args.segment:g}s segments -> {self.folder}", flush=True)
        try:
            while True:
                remaining = (self.end - now()).total_seconds()
                if remaining < min(10, args.segment):
                    self.manifest["status"] = "complete"
                    break
                if self.free_gb() < args.min_free_gb:
                    self.manifest["status"] = "stopped_low_disk"
                    print(f"Stopping: less than {args.min_free_gb:g} GB free on the output drive.", flush=True)
                    break
                duration = min(args.segment, remaining)
                started = now()
                name = f"seg-{self.index:04d}-{started:%H%M%S}"
                options = Namespace(ihanoi=args.ihanoi, ihanoi_feeds=args.ihanoi_feeds,
                                    ihanoi_ca_file=args.ihanoi_ca_file, duration=duration, fps=args.fps,
                                    output=self.folder / f"{name}.mp4", startup_timeout=args.startup_timeout,
                                    stale_timeout=args.stale_timeout, overwrite=False,
                                    # A stalled camera ends the segment; the next one reconnects.
                                    resend_on_stall=False)
                try:
                    # record() opens its own connection and closes it before returning.
                    metadata = record(options)
                except KeyboardInterrupt:
                    raise
                except Exception as error:
                    failures_in_row += 1
                    message = safe_message(error) or type(error).__name__
                    self.manifest["failures"].append({"index": self.index, "started_local": started.isoformat(timespec="seconds"),
                                                      "ended_local": now().isoformat(timespec="seconds"), "error": message})
                    self.index += 1
                    self.save()
                    wait = min(args.max_backoff, 5 * 2 ** (failures_in_row - 1))
                    print(f"[{now():%H:%M:%S}] segment failed ({failures_in_row} in a row): {message}", flush=True)
                    if "Reopen this camera" in message and failures_in_row >= 3:
                        print("  The session URL has probably expired: capture a new one in iHanoi and paste it into "
                              f"{args.ihanoi_feeds}. The next attempt will read it.", flush=True)
                    print(f"  reconnecting in {wait:g}s", flush=True)
                    sleep(wait)
                    continue
                failures_in_row = 0
                self.manifest["segments"].append({
                    "index": self.index, "video_file": options.output.name,
                    "started_local": started.isoformat(timespec="seconds"),
                    "captured_at_utc": metadata["captured_at_utc"],
                    "duration_seconds": metadata["duration_seconds"],
                    "camera_frames_received": metadata["camera_frames_received"],
                    "repeated_output_frames": metadata["repeated_output_frames"]})
                self.index += 1
                self.save()
                total = sum(s["duration_seconds"] for s in self.manifest["segments"])
                print(f"[{now():%H:%M:%S}] saved {options.output.name} "
                      f"({metadata['repeated_output_frames']}/{metadata['frames']} repeated frames; "
                      f"{total / 60:.1f} min total, {self.free_gb():.1f} GB free)", flush=True)
                sleep(args.reconnect_delay)  # let the old socket finish closing before reconnecting
        except KeyboardInterrupt:
            self.manifest["status"] = "interrupted"
            print("Interrupted; the clip in progress was discarded, finished clips are kept.", flush=True)
        finally:
            self.manifest["ended_local"] = now().isoformat(timespec="seconds")
            self.manifest["recorded_seconds"] = sum(s["duration_seconds"] for s in self.manifest["segments"])
            self.save()
        print(f"Session {self.manifest['status']}: {len(self.manifest['segments'])} clips, "
              f"{len(self.manifest['failures'])} failed attempts. Manifest: {self.manifest_path}", flush=True)
        return self.manifest


def keep_awake(on):
    """Stop Windows from sleeping while recording (no effect elsewhere)."""
    if os.name != "nt":
        return
    import ctypes
    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | (ES_SYSTEM_REQUIRED if on else 0))


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ("segment", "fps", "reconnect_delay", "max_backoff", "startup_timeout", "stale_timeout"):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0 or (value == 0 and name in ("segment", "fps")):
            p.error(f"--{name.replace('_', '-')} must be a finite positive number")
    if args.hours is not None:
        if not math.isfinite(args.hours) or args.hours <= 0:
            p.error("--hours must be positive")
        end = datetime.now().astimezone() + timedelta(hours=args.hours)
    else:
        try:
            end = parse_until(args.until)
        except argparse.ArgumentTypeError as error:
            p.error(str(error))
    keep_awake(True)
    try:
        manifest = Session(args, end).run()
    finally:
        keep_awake(False)
    return 0 if manifest["segments"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
