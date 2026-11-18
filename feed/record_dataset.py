"""Record a dataset from one iHanoi camera until a chosen local time.

Pick the camera from feed/ihanoi.json (asked interactively if --ihanoi is not
given). Each clip is recorded on its own connection; after a clip the
connection is closed completely and the next one opens after the gap.
Defaults: 3-minute clips, 5-minute gaps, saved to data/<feed>-<date>/.

    .\\.venv\\Scripts\\python.exe feed\\record_dataset.py
    .\\.venv\\Scripts\\python.exe feed\\record_dataset.py --ihanoi auco --until 18:00
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys
import time

import record_session
from record_session import Session, keep_awake, parse_until
from video_adapter import DEFAULT_IHANOI_FEEDS

ROOT = Path(__file__).resolve().parents[1]


def load_feed_names(path):
    feeds = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(feeds, dict) or not feeds:
        raise SystemExit(f"No feeds found in {path}")
    return list(feeds)


def choose_feed(names, ask=input):
    print("Available feeds:")
    for number, name in enumerate(names, 1):
        print(f"  {number}. {name}")
    while True:
        answer = ask("Feed number or name: ").strip()
        if answer in names:
            return answer
        if answer.isdigit() and 1 <= int(answer) <= len(names):
            return names[int(answer) - 1]
        print("Not in the list, try again.")


def ask_until(ask=input):
    while True:
        answer = ask("Record until (local HH:MM): ").strip()
        try:
            return answer, parse_until(answer)
        except argparse.ArgumentTypeError as error:
            print(error)


def parser():
    p = record_session.parser()
    p.description = __doc__
    p.formatter_class = argparse.RawDescriptionHelpFormatter
    # Make the feed and end time optional so they can be asked interactively.
    for action in p._actions:
        if action.dest == "ihanoi":
            action.required = False
    p.set_defaults(until=None, segment=180, reconnect_delay=300)
    for action in p._actions:
        if action.dest == "segment":
            action.help = "Seconds per clip/connection (default 180 = 3 min)"
        elif action.dest == "reconnect_delay":
            action.help = "Seconds fully disconnected between clips (default 300 = 5 min)"
        elif action.dest == "until":
            action.help = "Local end time HH:MM (asked if omitted)"
        elif action.dest == "output_dir":
            action.help = "Dataset folder; default data/<feed>-<date>"
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.segment <= 0 or args.reconnect_delay < 0:
        p.error("--segment must be positive and --reconnect-delay must not be negative")
    names = load_feed_names(args.ihanoi_feeds or DEFAULT_IHANOI_FEEDS)
    if args.ihanoi is None:
        args.ihanoi = choose_feed(names)
    elif args.ihanoi not in names:
        p.error(f"Unknown feed {args.ihanoi!r}; choose from: {', '.join(names)}")
    if args.hours is not None:
        end = datetime.now().astimezone() + timedelta(hours=args.hours)
    elif args.until is None:
        args.until, end = ask_until()
    else:
        try:
            end = parse_until(args.until)
        except argparse.ArgumentTypeError as error:
            p.error(str(error))
    if args.output_dir is None:
        args.output_dir = ROOT / "data" / f"{args.ihanoi}-{datetime.now():%Y%m%d}"

    def sleep_until_end(seconds):
        # Do not wait past the end time during a gap.
        left = (end - datetime.now().astimezone()).total_seconds()
        time.sleep(max(0, min(seconds, left)))

    print(f"Clips of {args.segment / 60:g} min, {args.reconnect_delay / 60:g} min disconnected between clips.")
    keep_awake(True)
    try:
        manifest = Session(args, end).run(sleep=sleep_until_end)
    finally:
        keep_awake(False)
    return 0 if manifest["segments"] else 1


if __name__ == "__main__":
    sys.exit(main())
