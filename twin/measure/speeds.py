"""Distribution of each vehicle's average speed in video trajectories, per kind, as a figure.

Average speed of a vehicle = distance it travelled (along its smoothed track) / time it was tracked,
stops included. Tracks are smoothed and broken tracks stitched first (tracks.py); tracks shorter than
--min-duration seconds are left out, since a fragment that short gives a noisy average.

    .venv\\Scripts\\python.exe -m twin.measure.speeds --video runs\\buoi-hht\\*.trajectories.json --out runs\\twin\\average-speed.png
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np

from ..vehicles import KINDS
from .tracks import load, stitch

INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6a64", "#e4e3de", "#ffffff"
BAR = "#2a78d6"
BIN_MPS = 0.25


def average_speeds(paths, min_duration=2.0):
    """{kind: array of per-vehicle average speeds (m/s)} and counts of tracks kept and dropped."""
    out, kept, short = {}, 0, 0
    for path in paths:
        tracks, _ = load(path)
        for tr in stitch(tracks):
            if tr.duration < min_duration:
                short += 1
                continue
            distance = float(np.sum(np.linalg.norm(np.diff(tr.xy, axis=0), axis=1)))
            out.setdefault(tr.kind, []).append(distance / tr.duration)
            kept += 1
    return {k: np.array(v) for k, v in out.items()}, {"kept": kept, "too_short": short}


def plot(speeds, path, title, min_count=20):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kinds = [k for k in KINDS if len(speeds.get(k, ())) >= min_count]
    top = max(float(np.percentile(np.concatenate([speeds[k] for k in kinds]), 99.5)), 1.0)
    bins = np.arange(0, top + BIN_MPS, BIN_MPS)
    fig, axes = plt.subplots(len(kinds), 1, figsize=(9, 1.9 * len(kinds) + 1.0), sharex=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, kind in zip(axes[:, 0], kinds):
        v = speeds[kind]
        ax.hist(v, bins=bins, color=BAR, edgecolor=SURFACE, linewidth=0.8)
        med = float(np.median(v))
        q1, q3 = np.percentile(v, [25, 75])
        ax.axvline(med, color=INK, linewidth=1.5)
        ax.annotate(f"median {med:.1f} m/s ({med * 3.6:.0f} km/h)", (med, 1), xycoords=("data", "axes fraction"),
                    xytext=(5, -2), textcoords="offset points", va="top", fontsize=8, color=INK)
        ax.set_title(f"{kind}: {len(v)} vehicles, middle half {q1:.1f}–{q3:.1f} m/s", loc="left", fontsize=9,
                     color=INK)
        ax.set_ylabel("vehicles", fontsize=8, color=MUTED)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[-1, 0].set_xlabel("average speed while tracked, stops included (m/s)", fontsize=8, color=MUTED)
    axes[-1, 0].set_xlim(0, top)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", nargs="+", required=True, help="Trajectory files (globs allowed)")
    parser.add_argument("--min-duration", type=float, default=2.0, help="Shortest track kept (s)")
    parser.add_argument("--out", type=Path, required=True, help="Figure (.png); a .json summary is written beside it")
    args = parser.parse_args(argv)
    paths = sorted({p for item in args.video for p in (glob.glob(item) or [item])})
    speeds, counts = average_speeds(paths, args.min_duration)
    plot(speeds, args.out, f"Average speed per vehicle · {len(paths)} video segments · tracks ≥ {args.min_duration:g} s")
    summary = {"files": paths, **counts, "kinds": {
        k: {"vehicles": int(len(v)), "median_mps": round(float(np.median(v)), 2),
            "p25_mps": round(float(np.percentile(v, 25)), 2), "p75_mps": round(float(np.percentile(v, 75)), 2),
            "mean_mps": round(float(np.mean(v)), 2)} for k, v in speeds.items()}}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{counts['kept']} tracks kept, {counts['too_short']} shorter than {args.min_duration:g} s left out")
    for k, s in summary["kinds"].items():
        print(f"  {k:<11} {s['vehicles']:>5} vehicles  median {s['median_mps']:.2f} m/s  middle half {s['p25_mps']:.2f}-{s['p75_mps']:.2f}")
    print(f"Figure: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
