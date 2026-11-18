"""Fit a behaviour model's parameters to video, by CMA-ES on the image-space score.

--model names the behaviour model: one registered in twin/behaviour (choice, the default; priority) or
package.module:ClassName for a model kept elsewhere. The search knows nothing else about it: the model
gives its own parameter list for the scene (behaviour/base.py: Scene, default_params). The priority
model, for one, asks for a speed profile for each route carrying at least 5% of the fit clips' arrivals
and for the main entry's turning shares (measure/demand.py turning_setup).

For every candidate parameter set, each video clip's own arrivals are replayed, the simulated vehicles
are projected into the camera image, and the result is scored against the video's boxes
(measure/image_score.py). Reported scores come from twin.calibrate.evaluate: the same measure on fresh
seeds and on held-out clips. The objective is the mean score over clips plus penalties for jamming and for vehicles
touching (real traffic does not overlap). Within a generation every candidate runs with the same seeds,
so candidates differ only by their parameters; the seeds change from one generation to the next (--seeds
plus 1000 x the generation), so the search cannot settle on parameters that only suit one draw of the
drivers. Clips run in parallel on all cores.

Choosing the answer: each generation's value is a noisy estimate, so the lowest one seen is partly luck.
The best candidate of each of the last TOP_K generations (candidates are comparable only within a
generation, which shares its seeds) and the final distribution mean are re-scored on --select-seeds, which
neither the search nor the reported check (--check-seeds) uses, and the best of those is kept.

Vehicles from video keep the speed they were seen at (measure/demand.py seen_speed; the choice model's
v0_seen scales it); --sampled-speeds drops it so every desired speed is drawn, as before.

CMA-ES suits this: about a dozen continuous parameters, an objective with no gradient (each driver picks
one of a few actions), one simulation per clip per candidate taking seconds, and a population that fills
the machine's cores each generation.

  --truth FILE   recovery test: the "video" is a simulation under these known parameters (a different
                 seed). Parameters the search cannot find again cannot be fitted from video either.
  --holdout      clips the search never sees; the fitted parameters are scored on them at the end.

    .venv\\Scripts\\python.exe -m twin.calibrate.search --layout L.camera-lanes.json `
        --video runs\\buoi-hht\\buoi-hht-20261106-seg-0010-*.trajectories.json ... --holdout ... --out runs\\twin\\fit
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..behaviour import ConstantSpeedModel, ParamSet, Scene, default_params, make_model, model_names
from ..engine import Simulation
from ..measure import image, image_score
from ..measure.demand import apply_turning, from_tracks, turning_setup, turning_shifts
from ..measure.tracks import load_matched
from ..network import Network

CONTACT_WEIGHT = 0.5      # objective += this * (pairs of vehicles that touched / vehicles)
JAM_WEIGHT = 1.0          # objective += this * (share of the clip not simulated because the road jammed)
MAX_ACTIVE = 80           # vehicles on the road at once before a run counts as jammed
TRUTH_SEED = 101
SEED_STRIDE = 1000        # generation g runs seeds --seeds + SEED_STRIDE * g
TOP_K = 6                 # last generations whose best candidate is re-scored, besides the distribution mean
_W = {}                   # per-process state


# ------------------------------------------------------------------ clips

def prepare_clip(path, network, camera, gate_list, observed_speeds=True):
    """A video clip as the search needs it: its arrivals and the features of its boxes. observed_speeds=False
    drops each arrival's seen speed (Arrival.v_seen), so models draw every desired speed."""
    tracks, meta = load_matched(path, network)
    arrivals, report = from_tracks(tracks, network, seed=1)
    if not observed_speeds:
        for a in arrivals:
            a.v_seen = None
    boxes = image.from_video(path, camera, tracks)
    return {"name": Path(path).name.replace(".trajectories.json", ""), "arrivals": arrivals,
            "shift": report["time_shift_s"], "duration": meta["duration_s"],
            "turning": turning_setup(tracks, network, arrivals, seed=1),
            "features": image_score.features(boxes, gate_list)}


def shared_turning(clips, fit):
    """The turning setup of the clips to fit. Turning shifts mean the same routes in every clip: a clip
    whose main entry differs loses its setup and keeps its demand as read from the video."""
    turning = next((clips[k]["turning"] for k in fit if clips[k]["turning"]), None)
    for clip in clips:
        if clip["turning"] and (turning is None or clip["turning"]["routes"] != turning["routes"]):
            clip["turning"] = None
    return turning


def starting_params(model, network, clips, fit, path=None):
    """The parameters a search starts from: a file's (which names its own model), or the named model's
    defaults for this scene."""
    turning = shared_turning(clips, fit)
    if path:
        return ParamSet.load(path)
    counts = np.bincount([a.route for k in fit for a in clips[k]["arrivals"]], minlength=len(network.routes))
    scene = Scene(network=network, route_arrivals=counts, turns=len(turning["routes"]) - 1 if turning else 0)
    base = default_params(model, scene)
    print(f"{base.model} model: {len(base.names)} parameters"
          + (f"; turning shares over routes {turning['routes']} (prior {np.round(turning['prior'], 2).tolist()})"
             if turning and "turn_1" in base.values else ""))
    return base


def simulate(clip, model, seed, values=None):
    """Returns the Result, its image features, and the share of the clip that was simulated: a run is
    stopped once more than MAX_ACTIVE vehicles are on the road at once (a jam that would take minutes
    to compute and is a failed candidate anyway). values: the parameter dict; when it has turning
    shifts (turn_<k>), the clip's ambiguous arrivals are re-drawn under them."""
    if values is not None and "turn_1" in values and clip.get("turning"):
        arrivals = apply_turning(clip["arrivals"], clip["turning"], turning_shifts(values, clip["turning"]))
    else:
        arrivals = copy.deepcopy(clip["arrivals"])
    sim = Simulation(_W["network"], model, arrivals, seed=seed)
    end = int(round((clip["duration"] + clip["shift"]) / sim.dt))
    while sim.steps < end:
        sim.step()
        if sim.steps % 20 == 0 and int(sim.active.sum()) > MAX_ACTIVE:
            break
    result = sim.result()
    boxes = image.from_result(result, _W["camera"], time_shift=clip["shift"], duration=clip["duration"])
    return result, image_score.features(boxes, _W["gates"]), sim.steps / end


def _init(layout, clips, weights, base=None):
    _W["network"] = Network.from_layout(layout)
    _W["camera"] = image.Camera(layout)
    _W["gates"] = image.gates(_W["camera"])
    _W["clips"], _W["weights"], _W["base"] = clips, weights, base


def _evaluate(task):
    """One simulation of one clip. task: (values or None for the constant-speed model, clip index, seed)."""
    values, k, seed = task
    clip = _W["clips"][k]
    if values is None:
        result, feats, done = simulate(clip, ConstantSpeedModel(), seed)
    else:
        params = _W["base"].with_values(values)
        result, feats, done = simulate(clip, make_model(params), seed, params.values)
    s = image_score.score(clip["features"], feats, _W["weights"])
    contact = result.contact_score()
    vehicles = max(len(result.vehicles), 1)
    return {"total": s["total"], "parts": s["parts"], "contact_pairs": contact["contact_pairs"],
            "contact_share": contact["contact_pairs"] / vehicles, "max_stopped_s": result.max_stuck,
            "jammed": 1.0 - done,
            "exited_share": sum(v["t_exit"] is not None for v in result.vehicles) / vehicles}


def _set_contact_weight(value):
    global CONTACT_WEIGHT
    CONTACT_WEIGHT = float(value)


def objective(rows):
    return float(np.mean([r["total"] + CONTACT_WEIGHT * r["contact_share"] + JAM_WEIGHT * r["jammed"] for r in rows]))


def summarise(rows):
    parts = sorted({k for r in rows for k in r["parts"]})
    return {"objective": round(objective(rows), 4), "score": round(float(np.mean([r["total"] for r in rows])), 4),
            "parts": {p: round(float(np.mean([r["parts"][p] for r in rows if p in r["parts"]])), 4) for p in parts},
            "contact_share": round(float(np.mean([r["contact_share"] for r in rows])), 4),
            "exited_share": round(float(np.mean([r["exited_share"] for r in rows])), 3),
            "jammed": round(float(np.mean([r["jammed"] for r in rows])), 3),
            "max_stopped_s": round(float(np.max([r["max_stopped_s"] for r in rows])), 1)}


# ------------------------------------------------------------------ the search

def generation_seeds(seeds, g, rotate=True):
    return [s + SEED_STRIDE * g for s in seeds] if rotate else list(seeds)


def search(pool, base, names, clip_indices, *, generations, popsize, seeds, sigma=0.2, seed=1, rotate=True,
           top_k=TOP_K, log=print):
    """Returns (the best candidate of each of the last top_k generations, as (value, generation, ParamSet);
    the final distribution mean; history). With rotate, each generation runs its own seeds
    (generation_seeds), so values from different generations are estimates under different draws and only
    candidates of the same generation can be ranked against each other."""
    from cmaes import CMA
    n = len(names)
    opt = CMA(mean=np.array(base.to_unit(names)), sigma=sigma, bounds=np.tile([0.0, 1.0], (n, 1)),
              population_size=popsize, seed=seed)
    history, top, best, started = [], [], (np.inf, None), time.perf_counter()
    for g in range(generations):
        xs = [opt.ask() for _ in range(popsize)]
        candidates = [base.from_unit(names, x) for x in xs]
        gen_seeds = generation_seeds(seeds, g, rotate)
        tasks = [(c.values, k, s) for c in candidates for k in clip_indices for s in gen_seeds]
        rows = pool.map(_evaluate, tasks, chunksize=1)
        per = len(clip_indices) * len(gen_seeds)
        values = [objective(rows[i * per:(i + 1) * per]) for i in range(popsize)]
        opt.tell(list(zip(xs, values)))
        i = int(np.argmin(values))
        if values[i] < best[0]:
            best = (values[i], candidates[i])
        top = (top + [(values[i], g, candidates[i])])[-top_k:]
        history.append({"generation": g, "best": round(values[i], 4), "median": round(float(np.median(values)), 4),
                        "best_so_far": round(best[0], 4)})
        log(f"gen {g:3d}  best {values[i]:.4f}  median {np.median(values):.4f}  best so far {best[0]:.4f}  "
            f"jammed {sum(r['jammed'] > 0 for r in rows)}/{len(rows)}  {time.perf_counter() - started:.0f} s")
    mean = base.from_unit(names, opt.mean)
    return top, mean, history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layout", type=Path, required=True)
    parser.add_argument("--video", type=Path, nargs="+", required=True, help="Clips to fit on")
    parser.add_argument("--holdout", type=Path, nargs="*", default=[], help="Clips to check on")
    parser.add_argument("--model", default="choice",
                        help=f"Behaviour model: {', '.join(model_names())}, or package.module:ClassName")
    parser.add_argument("--params", type=Path, help="Starting parameters (default: the model's defaults); "
                                                     "the file says which model they belong to")
    parser.add_argument("--names", nargs="*", help="Parameters to search (default: all of the model's)")
    parser.add_argument("--contact-weight", type=float, default=0.5,
                        help="Penalty per touching pair of vehicles per vehicle, added to the score")
    parser.add_argument("--sigma", type=float, default=0.2, help="Initial search width, as a share of each range")
    parser.add_argument("--truth", type=Path, help="Recovery test: fit to a simulation under these parameters")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--popsize", type=int, default=16)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2],
                        help="Seeds of each candidate's simulations in generation 0 (later: + 1000 x generation)")
    parser.add_argument("--cma-seed", type=int, default=1, help="Seed of CMA-ES's own sampling (repeat a fit with another "
                                                                     "to see how much the answer depends on it)")
    parser.add_argument("--fixed-seeds", action="store_true", help="Run every generation on --seeds unchanged")
    parser.add_argument("--select-seeds", type=int, nargs="+", default=[21, 22],
                        help="Seeds on which the best candidates and the mean are re-scored to choose the answer")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Last generations whose best candidate is re-scored, besides the mean")
    parser.add_argument("--sampled-speeds", action="store_true",
                        help="Draw every desired speed (ignore the speeds vehicles were seen at)")
    parser.add_argument("--check-seeds", type=int, nargs="+", default=[11, 12, 13], help="Seeds for the final scores")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    _set_contact_weight(args.contact_weight)
    args.out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    network, camera = Network.from_layout(args.layout), image.Camera(args.layout)
    gate_list = image.gates(camera)
    clips = [prepare_clip(p, network, camera, gate_list, not args.sampled_speeds)
             for p in list(args.video) + list(args.holdout)]
    fit = list(range(len(args.video)))
    hold = list(range(len(args.video), len(clips)))
    try:
        base = starting_params(args.model, network, clips, fit, args.params)
    except KeyError as error:
        parser.error(error.args[0])
    names = tuple(args.names or base.names)
    if not names:
        parser.error(f"The {base.model} model has no parameters to search")
    for name in names:
        if name not in base.by_name:
            parser.error(f"Unknown parameter {name}")
    print(f"{len(fit)} clips to fit, {len(hold)} held out, {len(gate_list)} counting lines; "
          f"searching {len(names)} parameters: {', '.join(names)}")

    weights = image_score.DEFAULT_WEIGHTS
    truth = None
    if args.truth:
        truth = ParamSet.load(args.truth)
        _init(str(args.layout), clips, weights, base)
        for clip in clips:
            clip["features"] = simulate(clip, make_model(truth), TRUTH_SEED, truth.values)[1]
            clip["name"] += " (simulated truth)"
    with Pool(args.workers, initializer=_init, initargs=(str(args.layout), clips, weights, base)) as pool:
        def check(values, indices, seeds=args.check_seeds):
            rows = pool.map(_evaluate, [(values, k, s) for k in indices for s in seeds], chunksize=1)
            return summarise(rows)

        report = {"model": base.model, "contact_weight": CONTACT_WEIGHT, "names": names,
                  "seeds": {"cma": args.cma_seed, "search": args.seeds, "rotate": not args.fixed_seeds, "select": args.select_seeds,
                            "check": args.check_seeds},
                  "observed_speeds": not args.sampled_speeds,
                  "fit_clips": [clips[k]["name"] for k in fit], "holdout_clips": [clips[k]["name"] for k in hold],
                  "before": {"fit": check(base.values, fit)}, "constant_speed": {"fit": check(None, fit)}}
        print("constant speed :", json.dumps(report["constant_speed"]["fit"]))
        print("start          :", json.dumps(report["before"]["fit"]))
        top, mean, history = search(pool, base, names, fit, generations=args.generations, popsize=args.popsize,
                                    seeds=args.seeds, sigma=args.sigma, seed=args.cma_seed, rotate=not args.fixed_seeds,
                                    top_k=args.top_k)
        # Re-score the lowest candidates and the mean on seeds nothing else uses; keep the best of them.
        shortlist = {"mean": mean, **{f"gen{g}": c for _, g, c in top}}
        selection = {label: check(c.values, fit, args.select_seeds)["objective"] for label, c in shortlist.items()}
        chosen = min(selection, key=selection.get)
        fitted = shortlist[chosen]
        print("selection      :", json.dumps(selection), "->", chosen)
        report["selection"] = {"search_values": {f"gen{g}": round(v, 4) for v, g, _ in top},
                               "select_objective": selection, "taken": chosen}
        report["after"] = {"fit": check(fitted.values, fit), "taken": chosen}
        print("fitted         :", json.dumps(report["after"]["fit"]))
        if hold:
            report["constant_speed"]["holdout"] = check(None, hold)
            report["before"]["holdout"] = check(base.values, hold)
            report["after"]["holdout"] = check(fitted.values, hold)
            print("holdout: constant", report["constant_speed"]["holdout"]["objective"], " start",
                  report["before"]["holdout"]["objective"], " fitted", report["after"]["holdout"]["objective"])

    report["parameters"] = {}
    for name in names:
        row = {"start": round(base.values[name], 4), "fitted": round(fitted.values[name], 4),
               "bounds": [base.by_name[name].lo, base.by_name[name].hi]}
        if truth is not None:
            row["truth"] = round(truth.values[name], 4)
            row["error_share_of_range"] = round(abs(fitted.to_unit([name])[0] - truth.to_unit([name])[0]), 3)
        report["parameters"][name] = row
        print(f"  {name:10s}", json.dumps(row))
    report["history"] = history
    report["wall_s"] = round(time.perf_counter() - started, 1)
    fitted.save(args.out / "fitted.params.json")
    (args.out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"Done in {report['wall_s']:.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
