"""Track every recorded clip of one camera and save endpoints + trajectories into one folder.

Runs export_vehicle_endpoints.py on each data/<feed>-<date>/seg-*.mp4 in turn, with the scene's
camera-lanes calibration. Resumable: a clip whose two output files already exist is skipped, and each clip
is written under <out>/.work and moved into place only when complete, so a crash never leaves a half file.

    .venv\\Scripts\\python.exe perception\\export_feed_tracks.py buoi-hht --out data\\buoi-hht-tracks

Output per clip: <feed>-<date>-seg-NNNN-HHMMSS.endpoints.json / .trajectories.json / .log
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("feed", help="Camera name, e.g. buoi-hht (data folders <feed>-YYYYMMDD...)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--calibration", type=Path,
                        help="Default: simulation/Assets/StreamingAssets/Traffic/<feed>.camera-lanes.json")
    args, extra = parser.parse_known_args(argv)   # anything else goes to export_vehicle_endpoints.py
    calibration = args.calibration or ROOT / "simulation/Assets/StreamingAssets/Traffic" / f"{args.feed}.camera-lanes.json"
    if not calibration.is_file():
        raise SystemExit(f"No calibration at {calibration}")

    clips = []
    for folder in sorted(p for p in args.data.iterdir() if p.is_dir() and re.match(rf"{re.escape(args.feed)}-\d{{8}}", p.name)):
        clips += [v for v in sorted(folder.glob("seg-*.mp4")) if not v.name.endswith(".partial.mp4")]
    work = args.out / ".work"
    work.mkdir(parents=True, exist_ok=True)
    todo = [v for v in clips if not all((args.out / f"{v.parent.name}-{v.stem}.{k}.json").exists()
                                        for k in ("endpoints", "trajectories"))]
    print(f"{len(clips)} clips, {len(clips) - len(todo)} already done, {len(todo)} to track", flush=True)

    started, failed = time.time(), []
    for n, video in enumerate(todo, 1):
        name = f"{video.parent.name}-{video.stem}"
        for f in work.glob(f"{name}.*"):
            f.unlink()
        cmd = [sys.executable, str(ROOT / "perception/export_vehicle_endpoints.py"), str(video), "--feed", args.feed,
               "--calibration", str(calibration), "--output", str(work / f"{name}.endpoints.json"),
               "--trajectories", str(work / f"{name}.trajectories.json"), *extra]
        with open(work / f"{name}.log", "w", encoding="utf-8") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        if code != 0:
            failed.append(name)
            print(f"[{n}/{len(todo)}] {name}: FAILED (exit {code}), see {work / (name + '.log')}", flush=True)
            continue
        for k in ("log", "endpoints.json", "trajectories.json"):   # trajectories last: it marks the clip done
            shutil.move(str(work / f"{name}.{k}"), str(args.out / f"{name}.{k}"))
        eta = (time.time() - started) / n * (len(todo) - n)
        print(f"[{n}/{len(todo)}] {name} done, about {eta / 60:.0f} min left", flush=True)
    print(f"Finished: {len(clips) - len(failed)} of {len(clips)} clips in {args.out}" + (f"; failed: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
