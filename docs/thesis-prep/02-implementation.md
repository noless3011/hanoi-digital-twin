# 2. Implementation (for Chapter 4, "Triển khai")

## 2.1 Technology

| Layer | Technology | Why |
| --- | --- | --- |
| Video capture | Python, OpenCV, iHanoi public camera streams (`feed/`) | The cameras are public; clips are recorded as 3-minute segments at 25 fps |
| Detection | RF-DETR Nano, 7 classes (car, bus, van, others, person, motorcycle, traffic light), fine-tuned on UA-DETRAC, COCO, BDD100K and labelled Hanoi frames (`perception/`) | Real-time transformer detector; small enough for a desktop GPU |
| Tracking | TrackTrack (Ultralytics) | Multi-object tracker with IDs |
| Simulation, measurement, calibration | Python 3.12, NumPy, SciPy (`twin/`) | Vectorised over all vehicles; one 3-minute clip in 1–3 s; runs in a process pool |
| Optimiser | `cmaes` library | Standard CMA-ES with bounds |
| Scene authoring, playback, rendering | Unity 6 (URP), C# (`simulation/`) | Lane editor, camera matching, playback of simulated trajectories, rendering a simulation to video for viewing |
| Reports | Matplotlib, ReportLab, LaTeX | |

## 2.2 Code organisation

| Module | Lines | Responsibility |
| --- | --- | --- |
| `twin/network.py` | 504 | Routes from the exported lane layout; projection of points into route coordinates; crossing and merge points |
| `twin/engine.py` | 315 | The simulation loop (spawn, view, decide, move, record, contacts) |
| `twin/behaviour/base.py` | 146 | The contract: `View`, `Decision`, `BehaviourModel`, `Scene` |
| `twin/behaviour/registry.py` | 60 | Finds a model by name or `module:Class` path |
| `twin/behaviour/params.py` | 139 | `Spec` (name, default, bounds, scale) and `ParamSet` (values, unit-cube mapping, JSON) |
| `twin/behaviour/choice.py` | 237 | Choice model |
| `twin/behaviour/priority.py` | 274 | Priority model |
| `twin/behaviour/constant_speed.py` | 42 | Baseline |
| `twin/measure/tracks.py` | 270 | Loading video tracks, Kalman/RTS smoothing, stitching, route matching |
| `twin/measure/demand.py` | 131 | Arrivals from video; turning shares |
| `twin/measure/image.py` | 243 | Camera model, projection of simulated vehicles to boxes, road-scale map, counting lines |
| `twin/measure/image_score.py` | 260 | The six features and the distance |
| `twin/measure/tracking_error.py` | 106 | Tracking quality report |
| `twin/calibrate/search.py` | 275 | CMA-ES search, recovery test, held-out check |
| `twin/calibrate/sensitivity.py` | 220 | Morris analysis with checkpoint and resume |
| `twin/calibrate/evaluate.py` | 149 | Reported scores: every configuration on every clip with fresh seeds, scored by projection; parallel |
| `twin/run.py` | 97 | Simulate one scene and write trajectories for Unity |
| `twin/tests/` | ~640 | 49 tests (unit, model properties, end-to-end calibration) |
| `simulation/Assets/Scripts/` | ~4,300 (C#) | Lane editor, camera lane export, trajectory player, video renderer |
| `perception/`, `feed/` | ~12,000 | Capture, detection, tracking, export, detector training and fine-tuning |

## 2.3 Data flow and formats

| Artefact | Format | Produced by | Used by |
| --- | --- | --- | --- |
| Video clip | MP4, 384 × 216, 25 fps, 3 min | `feed/record_dataset.py` | perception |
| Trajectories | `*.trajectories.json`: per frame, boxes with track ID, class, confidence | `perception/export_feed_tracks.py` | `twin.measure` |
| Lane layout + camera | `*.camera-lanes.json`: lanes (polylines, width, connections), camera intrinsics/extrinsics, `world_to_clip` matrix | Unity Lane Editor (`CameraLaneExport.cs`) | `twin.network`, `twin.measure.image` |
| Parameters | `fitted.params.json`: model name, values, parameter specs | `twin.calibrate.search` | `twin.run`, next search |
| Search report | `report.json`: start/fitted/constant-speed scores for fit and held-out clips, per-parameter values, history per generation, wall time | `twin.calibrate.search` | `docs/thesis-prep/build.py` |
| Sensitivity | `sensitivity.json`, `evaluations.jsonl` (checkpoint) | `twin.calibrate.sensitivity` | report builder |
| Simulated trajectories | `simulation.trajectories.json` | `twin.run` | Unity `TrajectoryPlayer` |
| Evaluation | `runs/twin/results/report.json`: per configuration and set, mean score, parts, objective, contacts, jams, seed spread | `twin.calibrate.evaluate` | report builder |

## 2.4 The behaviour-model interface (extensibility)

The engine, the search and the sensitivity analysis do not know any model by name. A model is a class
with a parameter list and two methods:

```python
class MyModel(BehaviourModel):
    SPECS = (Spec("v0", 8.0, 3.0, 14.0, True, 1, "desired speed (m/s)"), ...)
    def sample_driver(self, kind, arrival, rng, right, left, width): ...   # entry speed, offset, driver traits
    def decide(self, view): ...                                            # Decision(acc, vd) for all vehicles
```

It is used with `--model package.module:MyModel` (no registration), or by a short name after
`@register("mine")`. A fitted parameter file records the model it belongs to. A model whose parameter
list depends on the scene (the priority model's speed profiles) overrides
`default_params(cls, scene)`. `twin/examples/follower_model.py` is a 60-line working example used by
the tests. Requirements on a model for it to be fittable: few, continuous, bounded parameters; a fixed
number of random draws per driver independent of the parameters; a 3-minute clip in seconds.

Point for the thesis: the calibration method is not tied to a particular driver model. The three
models compared in Chapter 5 differ in kind (no interaction; cost minimisation; rule-based with an
order), and the same search, score and sensitivity analysis apply to all of them unchanged.

## 2.5 Unity side

- **Lane editor** (`LaneEditor.cs`, `LaneEditorInspector.cs`): lanes are drawn over the road mesh in the
  scene that matches the camera view; connections at junctions; export with the camera
  (`CameraLaneExport.cs`) into `StreamingAssets/Traffic/<scene>.camera-lanes.json`.
- **Camera matching**: the Unity camera is aligned with the real camera so the same layout serves for
  projecting video boxes to the road and simulated vehicles to the image.
- **Playback** (`Replay/TrajectoryPlayer.cs`): plays any trajectories file, simulated or from video.
- **Rendering** (`Editor/TrajectoryVideoRenderer.cs`): renders a simulation to video from the feed
  camera at its resolution, for viewing and for the qualitative comparison (`videos/`). Simulations are
  not passed through the detector for scoring: rendering and detecting a 3-minute clip takes about
  6 minutes on one GPU (GTX 1650: render 1, RF-DETR + TrackTrack 5), so they are projected instead
  (method 1.6).

## 2.6 Testing

49 automated tests (`python -m unittest discover -s twin/tests -t .`, about 55 s):

| Group | Examples |
| --- | --- |
| Network and engine | projection round trip; crossing and merge points; constant-speed motion; determinism for a seed; trajectory export round trip |
| Measurement | Kalman recovers speed; stitching joins a broken track; Wasserstein distance |
| Choice model | free speed; following; passing; stopping; head-on and crossing pairs without contact; parameters change behaviour |
| Priority model | a four-way junction without signals empties at 800 veh/h per route; the most senior vehicle takes the same time alone and in a crowd; speed follows the route profile; parameter files remember their model |
| Registry | models by name and by path; a plug-in model fits, runs and changes with its parameters |
| Calibration (end to end) | on a simulated 90 s clip, the search recovers a known `v0` within 10% of its range; the sensitivity analysis reports every parameter and resumes from its checkpoint |

## 2.7 Performance

| Item | Value |
| --- | --- |
| One 3-minute clip, about 220 vehicles, choice model | 2–3 s |
| One 3-minute clip, priority model | 1.5 s |
| Scoring one clip | about 0.1 s |
| Choice fit, 6 parameters, 25 generations × 12, 6 clips (+ checks on 6 held-out) | 18 min |
| Priority fit, 25 parameters, 80 generations × 20, 6 clips | 44 min |
| Morris analysis, choice (16 trajectories × 12 clips) | 77 min |
| Evaluation of 5 configurations × 12 clips × 3 seeds (180 simulations, projection) | about 3 min |
| For comparison, not used: render + RF-DETR/tracking of one 3-minute clip (GTX 1650) | about 6 min (17 h for the evaluation) |

Table `tables/compute-cost.tex` has the per-run numbers.
