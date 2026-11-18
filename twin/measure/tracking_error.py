"""How good the video tracks are, measured before anything is fitted to them. Every later metric inherits
these errors, so this report comes first.

Without ground truth, three things can still be measured:

  position noise    raw ground point minus the Kalman-smoothed path (robust standard deviation), by
                    distance from the camera; and the wobble of vehicles standing still, which is pure noise
  fragmentation     tracks per vehicle: raw tracks vs stitched tracks vs vehicles crossing counting lines;
                    how long tracks last; how many start or end in the middle of the view
  layout coverage   share of tracks that do not fit any drawn route (off the drawn road, or against its
                    direction)

    .venv\\Scripts\\python.exe -m twin.measure.tracking_error --layout L.camera-lanes.json --video A.trajectories.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ..network import Network
from .tracks import load, match_routes, stitch


def _robust_sd(x):
    if len(x) < 5:
        return None
    q1, q3 = np.percentile(x, [25, 75])
    return float((q3 - q1) / 1.349)


def report(path, network, camera_xy=None):
    raw, meta = load(path)
    stitched = stitch(raw)
    match_routes(stitched, network)
    residual, distance, still = [], [], []
    for tr in raw:
        smooth = np.stack([np.interp(tr.raw_t, tr.t, tr.xy[:, 0]), np.interp(tr.raw_t, tr.t, tr.xy[:, 1])], axis=1)
        err = np.linalg.norm(tr.raw_xy - smooth, axis=1)
        residual.append(err)
        if camera_xy is not None:
            distance.append(np.linalg.norm(tr.raw_xy - camera_xy, axis=1))
        speed = np.interp(tr.raw_t, tr.t, tr.speed)
        stopped = speed < 0.2
        if stopped.sum() >= 25:
            pts = tr.raw_xy[stopped]
            still.append(float(np.sqrt(np.mean(np.sum((pts - pts.mean(axis=0)) ** 2, axis=1)))))
    residual = np.concatenate(residual) if residual else np.zeros(0)
    by_distance = {}
    if distance:
        distance = np.concatenate(distance)
        for lo in range(0, 80, 10):
            sel = (distance >= lo) & (distance < lo + 10)
            if sel.sum() >= 20:
                by_distance[f"{lo}-{lo + 10} m"] = round(_robust_sd(residual[sel]) or 0, 3)
    durations = np.array([tr.duration for tr in stitched])
    edges = network.view
    inner_start = inner_end = 0
    if edges is not None:
        from matplotlib.path import Path as MplPath
        shrunk = edges.mean(axis=0) + (edges - edges.mean(axis=0)) * 0.8
        inner = MplPath(shrunk)
        inner_start = int(sum(inner.contains_point(tr.xy[0]) for tr in stitched))
        inner_end = int(sum(inner.contains_point(tr.xy[-1]) for tr in stitched))
    from .metrics import flows
    crossings = flows(stitched, network, 1.0)
    unmatched = [tr for tr in stitched if tr.route < 0]
    return {
        "file": str(path),
        "position_noise_sd_m": round(_robust_sd(residual) or 0, 3),
        "position_noise_by_camera_distance_sd_m": by_distance,
        "standstill_wobble_rms_m": round(float(np.median(still)), 3) if still else None,
        "raw_tracks": len(raw), "stitched_tracks": len(stitched),
        "line_crossings": int(sum(crossings.values())),
        "track_duration_s_p25_p50_p75": [round(float(x), 2) for x in np.percentile(durations, [25, 50, 75])] if len(durations) else None,
        "tracks_starting_inside_view_share": round(inner_start / max(len(stitched), 1), 3),
        "tracks_ending_inside_view_share": round(inner_end / max(len(stitched), 1), 3),
        "off_layout_share": round(len(unmatched) / max(len(stitched), 1), 3),
        "note": "raw tracks per crossing above 1 means fragmentation; off-layout tracks drive where no route is drawn",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--video", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    network = Network.from_layout(args.layout)
    layout = json.loads(args.layout.read_text(encoding="utf-8-sig"))
    cam = layout.get("camera", {}).get("position")
    camera_xy = np.array([cam[0], cam[2]]) if cam else None
    reports = [report(p, network, camera_xy) for p in args.video]
    for r in reports:
        print(json.dumps(r, indent=1))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(reports, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
