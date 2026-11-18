"""Score model configurations against video on fresh seeds. This is where reported scores come from.

Each configuration replays every clip's own arrivals with each seed; the simulated vehicles are projected
into the camera image (measure/image.py from_result) and scored against the tracked video
(measure/image_score.py), the same measure the search uses. Contacts and jams come from the simulation.

The simulation side is not rendered and passed through the detector: that takes about 6 minutes per clip
on one GPU (render 1, RF-DETR + TrackTrack 5), so 5 configurations x 12 clips x 3 seeds would be about
17 hours, against a few minutes here. The cost is that the video side carries the detector's errors
(missed and merged vehicles, broken tracks, jitter) and the projected side does not, which puts a floor
under every score; comparisons between configurations share that floor (docs/twin.md section 4).

Configurations, as LABEL=SPEC:
  constant                 the constant-speed baseline
  model:NAME               a behaviour model at its default parameters for this scene (NAME as for --model)
  PATH.params.json         fitted parameters (the file names its model)

    .venv\\Scripts\\python.exe -m twin.calibrate.evaluate --layout simulation\\Assets\\StreamingAssets\\Traffic\\buoi-hht.camera-lanes.json `
        --video (ls runs\\buoi-hht\\buoi-hht-20261106-*.trajectories.json) --holdout (ls runs\\buoi-hht\\buoi-hht-20261108-*.trajectories.json) `
        --config constant=constant choice_default=model:choice choice_fitted=runs\\twin\\fit2\\fitted.params.json `
                 priority_start=model:priority priority_fitted=runs\\twin\\priority-sep\\fitted.params.json `
        --seeds 11 12 13 --out runs\\twin\\results

report.json holds, per configuration and per set (fit, held out): the mean over clips and seeds of the
score, its parts, the objective (score + contact and jam penalties), contacts, jams, and the spread between
seeds (mean over clips of the standard deviation between seeds).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..behaviour import ConstantSpeedModel, ParamSet, make_model
from ..measure import image, image_score
from ..network import Network
from . import search as S


def parse_configs(items, network, clips, fit, parser):
    """[(label, ParamSet or None for constant speed)]."""
    out = []
    for item in items:
        label, sep, spec = item.partition("=")
        if not sep or not label or not spec:
            parser.error(f"--config {item}: expected LABEL=SPEC")
        if spec == "constant":
            out.append((label, None))
        elif spec.startswith("model:"):
            try:
                out.append((label, S.starting_params(spec[6:], network, clips, fit)))
            except KeyError as error:
                parser.error(error.args[0])
        else:
            path = Path(spec)
            if not path.is_file():
                parser.error(f"--config {item}: no parameter file {path}")
            out.append((label, ParamSet.load(path)))
    return out


def _init(layout, clips, configs):
    S._init(layout, clips, image_score.DEFAULT_WEIGHTS)
    S._W["configs"] = configs


def _job(task):
    """One configuration on one clip with one seed. task: (configuration index, clip index, seed)."""
    c, k, seed = task
    params = S._W["configs"][c][1]
    clip = S._W["clips"][k]
    model = ConstantSpeedModel() if params is None else make_model(params)
    result, feats, done = S.simulate(clip, model, seed, None if params is None else params.values)
    s = image_score.score(clip["features"], feats, S._W["weights"])
    contact = result.contact_score()
    vehicles = max(len(result.vehicles), 1)
    return {"config": c, "clip": clip["name"], "seed": seed, "total": s["total"], "parts": s["parts"],
            "contact_share": contact["contact_pairs"] / vehicles, "jammed": 1.0 - done,
            "max_stopped_s": result.max_stuck}


def summarise(rows):
    """Means over clips and seeds; seed spread = mean over clips of the standard deviation between seeds."""
    parts = sorted({p for r in rows for p in r["parts"]})
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip"], []).append(r["total"])
    spread = [float(np.std(v, ddof=1)) for v in by_clip.values() if len(v) > 1]
    obj = [r["total"] + S.CONTACT_WEIGHT * r["contact_share"] + S.JAM_WEIGHT * r["jammed"] for r in rows]
    return {"jobs": len(rows), "clips": len(by_clip),
            "score": round(float(np.mean([r["total"] for r in rows])), 4),
            "objective": round(float(np.mean(obj)), 4),
            "parts": {p: round(float(np.mean([r["parts"][p] for r in rows if p in r["parts"]])), 4) for p in parts},
            "contact_share": round(float(np.mean([r["contact_share"] for r in rows])), 4),
            "jammed": round(float(np.mean([r["jammed"] for r in rows])), 3),
            "max_stopped_s": round(float(np.max([r["max_stopped_s"] for r in rows])), 1),
            "seed_spread": round(float(np.mean(spread)), 4) if spread else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--video", type=Path, nargs="+", required=True, help="Fit clips (their own set in the report)")
    p.add_argument("--holdout", type=Path, nargs="*", default=[], help="Held-out clips")
    p.add_argument("--config", nargs="+", required=True, metavar="LABEL=SPEC")
    p.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13])
    p.add_argument("--sampled-speeds", action="store_true",
                   help="Draw every desired speed (ignore the speeds vehicles were seen at)")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    network, camera = Network.from_layout(args.layout), image.Camera(args.layout)
    gates = image.gates(camera)
    clips = [S.prepare_clip(v, network, camera, gates, not args.sampled_speeds)
             for v in list(args.video) + list(args.holdout)]
    fit = list(range(len(args.video)))
    configs = parse_configs(args.config, network, clips, fit, p)     # also settles the shared turning setup
    sets = {"fit": {c["name"] for c in clips[:len(args.video)]}, "holdout": {c["name"] for c in clips[len(args.video):]}}
    tasks = [(c, k, s) for c in range(len(configs)) for k in range(len(clips)) for s in args.seeds]
    print(f"{len(configs)} configurations x {len(clips)} clips x {len(args.seeds)} seeds = {len(tasks)} simulations",
          flush=True)
    with Pool(args.workers, initializer=_init, initargs=(str(args.layout), clips, configs)) as pool:
        rows = pool.map(_job, tasks, chunksize=1)

    report = {"note": "score = simulation projected into the camera image vs tracked video (image_score)",
              "contact_weight": S.CONTACT_WEIGHT, "seeds": args.seeds,
              "observed_speeds": not args.sampled_speeds, "configs": {}}
    for c, (label, _) in enumerate(configs):
        report["configs"][label] = {name: summarise([r for r in rows if r["config"] == c and r["clip"] in names])
                                    for name, names in sets.items() if names}
    report["wall_s"] = round(time.perf_counter() - started, 1)
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    print(f"\n{'configuration':18s} {'fit score':>10s} {'held-out':>10s} {'contacts':>9s} {'seed spread':>12s}")
    for label, entry in report["configs"].items():
        f, h = entry.get("fit"), entry.get("holdout")
        last = h or f
        print(f"{label:18s} {f['score']:10.3f} {h['score'] if h else float('nan'):10.3f} "
              f"{last['contact_share']:9.2f} {last['seed_spread'] or float('nan'):12.4f}")
    print(f"Done in {report['wall_s']:.0f} s -> {args.out / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
