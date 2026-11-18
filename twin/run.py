"""Simulate a scene and write vehicle trajectories (for the Unity TrajectoryPlayer, or for measurement).

Demand comes from a video (its vehicles enter at the same times, on the same routes, at the speed and
sideways offset they were first seen at) or is random. The behaviour model: --params takes a fitted
parameter file, which names its model; otherwise --model names one (choice by default, priority, or
package.module:ClassName) and it runs with its default parameters; --model constant is the
constant-speed model:

    .venv\\Scripts\\python.exe -m twin.run --layout simulation\\Assets\\StreamingAssets\\Traffic\\buoi-hht.camera-lanes.json `
        --video runs\\buoi-hht\\buoi-hht-20261106-seg-0010-092146.trajectories.json `
        --out simulation\\Assets\\StreamingAssets\\Traffic\\simulation.trajectories.json

    .venv\\Scripts\\python.exe -m twin.run --layout L.camera-lanes.json --rate 150 --minutes 3 --speed motorcycle=6.5 car=6 `
        --out sim.trajectories.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .behaviour import ConstantSpeedModel, ParamSet, Scene, default_params, make_model, model_names
from .engine import Simulation
from .network import Network
from .scenarios import poisson
from .signals import SignalPlan
from .vehicles import KINDS


def parse_speeds(items, parser):
    speeds = {}
    for item in items:
        kind, _, value = item.partition("=")
        if kind not in KINDS or not value:
            parser.error(f"--speed {item}: expected KIND=M/S with KIND in {', '.join(KINDS)}")
        speeds[kind] = float(value)
    return speeds


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True, help="Camera lane layout JSON exported from Unity")
    parser.add_argument("--video", type=Path, help="Replay this video's demand")
    parser.add_argument("--rate", type=float, default=120, help="Random demand: vehicles per hour per route")
    parser.add_argument("--minutes", type=float, default=3, help="Random demand: duration")
    parser.add_argument("--model", default="choice",
                        help=f"constant, {', '.join(model_names())}, or package.module:ClassName")
    parser.add_argument("--params", type=Path, help="Fitted parameters (twin.calibrate.search writes fitted.params.json); "
                                                     "the file says which model they belong to")
    parser.add_argument("--speed", nargs="*", default=[], metavar="KIND=M/S",
                        help="Constant model: speed per kind for vehicles without a known entry speed")
    parser.add_argument("--signals", type=Path, help="Signal plan JSON (see twin/signals.py)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    network = Network.from_layout(args.layout)
    params = None
    if args.model == "constant":
        model = ConstantSpeedModel(parse_speeds(args.speed, parser))
    else:
        try:
            params = ParamSet.load(args.params) if args.params else default_params(args.model, Scene(network=network))
            model = make_model(params)
        except KeyError as error:
            parser.error(error.args[0])
    if args.video:
        from .measure.demand import from_tracks
        from .measure.tracks import load_matched
        tracks, meta = load_matched(args.video, network)
        arrivals, report = from_tracks(tracks, network, seed=args.seed)
        if params is not None and "turn_1" in params.values:      # fitted turning shares at the main entry
            from .measure.demand import apply_turning, turning_setup, turning_shifts
            turning = turning_setup(tracks, network, arrivals, seed=1)
            if turning:
                arrivals = apply_turning(arrivals, turning, turning_shifts(params.values, turning))
        duration = meta["duration_s"] + report["time_shift_s"]
        print(f"Video demand: {report['arrivals']} arrivals ({report['first_seen_inside']} tracks first seen inside "
              f"the view and {report['off_layout']} off the drawn roads were not replayed)")
    else:
        duration = args.minutes * 60
        arrivals = poisson(network, args.rate, duration, seed=args.seed)
    plan = SignalPlan.load(args.signals) if args.signals else None
    started = time.perf_counter()
    result = Simulation(network, model, arrivals, seed=args.seed, dt=args.dt, signals=plan).run(duration)
    wall = time.perf_counter() - started
    result.save(args.out, feed=network.name)
    summary = result.summary()
    print(json.dumps({k: v for k, v in summary.items() if k != "travel_time_median_s"}))
    print(f"Simulated {duration:.0f} s in {wall:.1f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
