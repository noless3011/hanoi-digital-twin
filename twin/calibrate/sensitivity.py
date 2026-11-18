"""Global sensitivity of the image-space score to each behaviour parameter (Morris elementary effects).

Each trajectory starts at a random point of the parameter space (every parameter in 0..1 on its own log or
linear scale, the same space the CMA-ES search uses) and moves one parameter at a time, in random order, by
DELTA (2/3 of its range). Every point is simulated on every clip with the trajectory's seed (so the changes
within a trajectory come from the parameters, not the draws) and scored as in the search. For each
parameter:

  effect   mean |change of the score| for one step (the Morris mu*), in score units
  sigma    spread of the signed changes: large against effect means the parameter's effect depends on
           where the others are (interactions) or is not monotonic
  the same for each score part, for touching pairs per vehicle and for the jammed share.

Noise floor: each trajectory's start is also simulated with a second seed; the mean |change| between the two
seeds is what a step with no effect at all would show. Effects near it are not distinguishable from noise.

    python -m twin.calibrate.sensitivity --layout L --video clips... --model choice --trajectories 16 --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from ..behaviour import model_names
from ..measure import image, image_score
from ..network import Network
from . import search as S

LEVELS = 4
DELTA = (LEVELS / 2) / (LEVELS - 1)     # 2/3
DEFAULT_WORKERS = max(1, (os.cpu_count() or 2) - 6)   # leave cores free so the machine stays usable


def low_priority():
    """Run this process below normal priority, so a long analysis does not make the computer sluggish."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(10)
    except Exception:
        pass


def _worker_init(*args):
    low_priority()
    S._init(*args)


def _run(item):
    index, task = item
    return index, S._evaluate(task)


def trajectories(k, r, rng):
    """r Morris trajectories in [0, 1]^k: (r, k + 1, k) points and (r, k) the parameter moved at each step."""
    grid = np.arange(LEVELS) / (LEVELS - 1)
    points, order = [], []
    for _ in range(r):
        x = rng.choice(grid, size=k)
        perm = rng.permutation(k)
        traj = [x.copy()]
        for i in perm:
            x = x.copy()
            x[i] = x[i] + DELTA if x[i] + DELTA <= 1 + 1e-9 else x[i] - DELTA
            traj.append(x)
        points.append(traj)
        order.append(perm)
    return np.array(points), np.array(order)


def mean_rows(rows):
    parts = sorted({p for r in rows for p in r["parts"]})
    return {"score": float(np.mean([r["total"] for r in rows])),
            "parts": {p: float(np.mean([r["parts"][p] for r in rows if p in r["parts"]])) for p in parts},
            "contact_share": float(np.mean([r["contact_share"] for r in rows])),
            "jammed": float(np.mean([r["jammed"] for r in rows]))}


def flat(m):
    out = {"score": m["score"], "contact_share": m["contact_share"], "jammed": m["jammed"]}
    out.update({"part:" + p: v for p, v in m["parts"].items()})
    return out


def _stats(v):
    if not v:
        return {"effect": None, "mean": None, "sigma": None, "n": 0}
    return {"effect": round(float(np.mean(np.abs(v))), 4), "mean": round(float(np.mean(v)), 4),
            "sigma": round(float(np.std(v, ddof=1)), 4) if len(v) > 1 else None, "n": len(v)}


def analyse(names, points, order, values, noise_pairs):
    """values[t][j]: flat metrics of trajectory t, point j. A score part is missing at a point when one
    side had no data for it (no crossings of any line, say); a step counts for a part only when the part
    is there at both of its ends, and `n` says how many steps did."""
    metrics = sorted({m for traj in values for point in traj for m in point})
    out = {}
    for i, name in enumerate(names):
        rows = {m: [] for m in metrics}
        for t in range(len(order)):
            j = int(np.flatnonzero(order[t] == i)[0])
            sign = np.sign(points[t][j + 1][i] - points[t][j][i])
            for m in metrics:
                if m in values[t][j] and m in values[t][j + 1]:
                    rows[m].append(sign * (values[t][j + 1][m] - values[t][j][m]))
        out[name] = {m: _stats(v) for m, v in rows.items()}
    noise = {m: _stats([abs(a[m] - b[m]) for a, b in noise_pairs if m in a and m in b])["effect"] for m in metrics}
    return out, noise


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--layout", type=Path, required=True)
    p.add_argument("--video", type=Path, nargs="+", required=True, help="Clips every point is scored on")
    p.add_argument("--fit-days", nargs="*", default=["20261106"],
                   help="Clips whose name contains one of these describe the scene to the model "
                        "(the priority model's busy routes and turning)")
    p.add_argument("--model", default="choice",
                   help=f"Behaviour model: {', '.join(model_names())}, or package.module:ClassName")
    p.add_argument("--params", type=Path, help="A parameter file to take the model and its parameter list from")
    p.add_argument("--trajectories", type=int, default=16)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    network, camera = Network.from_layout(args.layout), image.Camera(args.layout)
    gates = image.gates(camera)
    clips = [S.prepare_clip(v, network, camera, gates) for v in args.video]
    fit = [k for k, c in enumerate(clips) if any(d in c["name"] for d in args.fit_days)] or list(range(len(clips)))
    try:
        base = S.starting_params(args.model, network, clips, fit, args.params)
    except KeyError as error:
        p.error(error.args[0])
    names = base.names
    rng = np.random.default_rng(args.seed)
    points, order = trajectories(len(names), args.trajectories, rng)
    seeds = 1000 + np.arange(args.trajectories)

    # Every task: (values, clip, seed). Points first, then the noise checks (trajectory starts, other seed).
    jobs = []
    for t in range(args.trajectories):
        for j in range(len(names) + 1):
            jobs.append(("pt", t, j, base.from_unit(names, points[t][j]).values, int(seeds[t])))
        jobs.append(("noise", t, 0, base.from_unit(names, points[t][0]).values, int(seeds[t]) + 500))
    tasks = [(values, k, seed) for (_, _, _, values, seed) in jobs for k in range(len(clips))]
    print(f"{base.model}: {len(names)} parameters, {args.trajectories} trajectories, {len(jobs)} points x "
          f"{len(clips)} clips = {len(tasks)} simulations", flush=True)

    # Every finished simulation is appended to a checkpoint; a rerun with the same arguments resumes.
    checkpoint = args.out / "evaluations.jsonl"
    rows = [None] * len(tasks)
    if checkpoint.exists():
        for line in checkpoint.read_text(encoding="utf-8").splitlines():
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue      # a line cut off by a crash
            if rec.get("n_tasks") == len(tasks) and rec.get("sampling_seed") == args.seed:
                rows[rec["index"]] = rec["row"]
    todo = [(q, t) for q, t in enumerate(tasks) if rows[q] is None]
    if len(todo) < len(tasks):
        print(f"resuming: {len(tasks) - len(todo)} simulations already done", flush=True)

    low_priority()
    started = time.perf_counter()
    with Pool(args.workers or DEFAULT_WORKERS, initializer=_worker_init,
              initargs=(str(args.layout), clips, image_score.DEFAULT_WEIGHTS, base)) as pool, \
            open(checkpoint, "a", encoding="utf-8") as log:
        for n, (q, row) in enumerate(pool.imap_unordered(_run, todo, chunksize=1), 1):
            rows[q] = row
            log.write(json.dumps({"index": q, "n_tasks": len(tasks), "sampling_seed": args.seed, "row": row}) + "\n")
            log.flush()
            if n % 100 == 0:
                el = time.perf_counter() - started
                print(f"  {len(tasks) - len(todo) + n}/{len(tasks)}  {el:.0f} s, about "
                      f"{el / n * (len(todo) - n):.0f} s left", flush=True)

    per = len(clips)
    values = [[None] * (len(names) + 1) for _ in range(args.trajectories)]
    noise_pairs = []
    for q, (kind, t, j, _, _) in enumerate(jobs):
        m = flat(mean_rows(rows[q * per:(q + 1) * per]))
        if kind == "pt":
            values[t][j] = m
        else:
            noise_pairs.append((values[t][0], m))
    effects, noise = analyse(names, points, order, values, noise_pairs)
    base_scores = [v[0]["score"] for v in values]
    report = {"model": base.model, "names": list(names), "delta": DELTA, "trajectories": args.trajectories,
              "clips": [c["name"] for c in clips], "noise_floor": noise, "effects": effects,
              "bounds": {s.name: [s.lo, s.hi, "log" if s.log else "linear"] for s in base.specs},
              "score_at_starts": {"min": round(min(base_scores), 4), "median": round(float(np.median(base_scores)), 4),
                                  "max": round(max(base_scores), 4)},
              "wall_s": round(time.perf_counter() - started, 1)}
    (args.out / "sensitivity.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    ranked = sorted(names, key=lambda n: -effects[n]["score"]["effect"])
    print(f"noise floor (score): {noise['score']:.4f}")
    for n in ranked:
        e = effects[n]["score"]
        print(f"  {n:12s} effect {e['effect']:.4f}  mean {e['mean']:+.4f}  sigma {e['sigma']:.4f}  "
              f"contacts {effects[n]['contact_share']['effect']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
