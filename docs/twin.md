# Twin: simulation engine, behaviour models, measurement

Code is in [`twin/`](../twin). The behaviour model was rebuilt from the simplest possible driver
(constant speed) to the choice model (section 2), which is fitted to video in the camera image
(sections 4 and 5). The previous interaction model (IDM, sideways offset choice, crossing points,
impatience), its structural checks, direct parameter measurement and CMA-ES calibration are kept in git under the tag
`archive/snapshots-20261116`, folder `backup/interaction-model-20261114/`; the older Unity C# system
under the same tag in `backup/legacy-20261111/`.

## 1. Engine and behaviour

Pure Python/NumPy, no Unity, deterministic for a seed. Unity only plays trajectories back
(`simulation/Assets/Scripts/Replay/TrajectoryPlayer.cs`).

| Module | Owns |
| --- | --- |
| `twin/network.py` | Roads: routes in (s, d) coordinates (s along the route, d sideways), projection, crossing and merge points |
| `twin/vehicles.py` | Vehicle kinds and body sizes |
| `twin/engine.py` | The world: vehicle bodies and kinematics, spawning, perception geometry, integration, exits, recording, contact scoring |
| `twin/behaviour/base.py` | The contract: each step the engine passes a `View` (every vehicle's pose, every nearby vehicle seen in its route coordinates, signal states) and the model returns a `Decision` (acceleration and sideways speed per vehicle) |
| `twin/behaviour/choice.py`, `params.py` | The choice model and its parameters (section 2) |
| `twin/behaviour/priority.py` | The priority model: cannot deadlock, built to be fitted (section 2b) |
| `twin/behaviour/constant_speed.py` | The baseline: one speed, one offset, ignores everyone |
| `twin/behaviour/registry.py` | How the tools find a model by name (section 2c) |
| `twin/examples/follower_model.py` | A forty-line model kept outside `twin/behaviour`, as a template |

Routes come from the lanes drawn in Unity's Lane Editor (exported `*.camera-lanes.json`): every lane
sequence from an entry node to an exit node. There are no lanes inside a route; `d` is one continuous
width.

The engine enforces no driving rules. Every arrival enters at its time, at the speed and offset the model
gives, whether or not there is room. Vehicles then move exactly as the model decides and can pass through
each other. Every overlap of two bodies (each shrunk 10%, so a graze does not count) is recorded in
`Result.contacts` per pair of vehicles: first time, number of separate contacts, time overlapping and
deepest overlap. `Result.contact_score()` (also in `summary()`) gives `contact_pairs`, `contact_events`,
`contact_time_s` and `max_contact_depth_m`. Each recorded sample is the vehicle's state at that time.

    Simulation(network, ChoiceModel(ParamSet()), arrivals, seed=1).run(180)

A model instance holds one simulation's drivers; create a new one per `Simulation`.

## 2. Choice model (`twin/behaviour/choice.py`)

Each step every driver compares 30 actions (6 accelerations x 5 sideways speeds), predicts `tau` seconds
ahead for each (others keep their velocity), and takes the cheapest:

| Cost | What it is | Parameters |
| --- | --- | --- |
| progress | `((speed reached - v0) / v0)^2`, weighted more the longer the driver has waited | `v0`, `v0_spread`, `v0_seen`, `a`, `T_imp` |
| risk | closeness to each other vehicle at the predicted positions: 0 when the gap ahead is above `s0 + T * speed` or the sideways gap above `clat`, rising as the gap closes and the bodies line up; much more when they would overlap | `T`, `s0`, `clat`, `w_risk`, `tau`, `b` |
| open road | the share of desired speed that slower or oncoming vehicles in line ahead do not allow, at the sideways position the action leads to | (none of its own) |
| keep clear | coming to rest in the way of a vehicle waiting to cross (the 4 m of road ahead of it); whoever has waited longer counts for more | (none of its own) |
| position | distance from the driver's own position across the road (where it entered, as a share of the width); leaving the road | `w_pos`, `vlat` |
| effort | hard acceleration, braking, sideways movement | fixed |
| stop line | passing a red light, or an amber one it could stop for | (with signals) |

There are no separate rules: following and giving way come from risk; filtering past slower vehicles
and easing aside for oncoming ones from risk and open road. With nobody near, a driver holds `v0` and its
position, which is the constant-speed model (`twin/behaviour/constant_speed.py`, kept as the baseline).

Parameters (`twin/behaviour/params.py`) are declared once with a default, bounds, a log or linear scale
and a stage, so a search sees a vector in 0..1. Values are the motorcycle's; other kinds are the
motorcycle's times fixed ratios (`RATIOS`). A vehicle replayed from video keeps its own pace: its desired
speed is `v0_seen` times the speed it was seen at when not held up (85th percentile of its smoothed
speed, `Arrival.v_seen`, from `measure/demand.py`; none for tracks under 1 s or never above 2 m/s).
Otherwise the desired speed is drawn around `v0` with `v0_spread`. Desired speed and caution are drawn
once at entry and the draws do not depend on the parameters, so runs with the same seed differ only by
them.

`python -m unittest twin.tests.test_choice` checks: free speed, following, passing, stopping, head-on
and crossing pairs without contact.

Known limit: gridlock. A four-way junction with no signals carries 250 vehicles per hour per route and
locks up at 400 (a ring of drivers each waiting for the next). Replayed video demand at Bưởi normally
runs freely (longest stop 2-7 s), but in one clip of twelve (8 Nov, segment 0032) a bus and a van on two
left-turn routes that run through each other meet nose to nose with traffic packed round them, and
stand for a minute or more under some parameter values. Routes are single lines with a width; two
drivers on opposed routes have no agreed side to pass on.

## 2b. Priority model (`twin/behaviour/priority.py`)

A deliberately simple driver, made for two things only: it cannot deadlock, and its parameters are ones
a video can pin down. It is not meant to look right vehicle by vehicle.

- Every vehicle has a seniority (the order it entered in). A driver reacts only to vehicles senior to
  it: it follows them and waits for them to cross, at Gipps' safe speed (the speed it can still stop
  from). It moves aside for a senior vehicle coming the other way. It ignores everyone junior.
- So the most senior vehicle on the road reacts to nobody and always drives on; then the next. No ring
  of drivers waiting for each other can form, whatever the parameters. Where two vehicles would lock,
  the senior one drives through and the engine records the contact.
- A junior vehicle ahead in a senior one's line, going the same way, swaps seniority with it, so the one
  in front leads. Seniority stays a strict order.
- Wanted speed depends on place: each route carrying at least 5% of the arrivals has a speed profile
  (4 values along its length), times the driver's own factor.

Parameters (25 at Bưởi): `v_r<route>_k<knot>` and `v_free` (speeds), `keep` (share of its observed entry
speed a driver keeps), `spread`, `large` (bigger vehicles relative to two-wheelers), `T`, `s0`, `clat`
(gaps to senior vehicles), and `turn_1`, `turn_2`. The last two are demand, not behaviour: they shift
the main entry's turning shares, re-drawing the route of arrivals whose track ended before the routes
split (`measure/demand.py`, `turning_setup` and `apply_turning`).

`python -m unittest twin.tests.test_priority` checks, among others, that a four-way junction without
signals empties at 800 vehicles per hour per route (the choice model locks at 400) and that the most
senior vehicle takes the same time alone and in a crowd.

## 2c. Adding a model

The search, the sensitivity analysis and `twin.run` do not name a model class. They ask
`twin/behaviour/registry.py` for the model called `--model NAME`, ask that model for its parameter list,
and treat it as a black box from there. A new model is one file:

    from twin.behaviour import BehaviourModel, Decision, Spec

    class MyModel(BehaviourModel):
        SPECS = (Spec("v0", 8.0, 3.0, 14.0, True, 1, "desired speed (m/s)"), ...)   # name, default, low, high, log, stage, meaning

        def sample_driver(self, kind, arrival, rng, right, left, width): ...   # speed and offset at entry
        def decide(self, view): ...                                            # Decision(acc, vd); parameters are self.params.values

Two ways to make it known:

| | How | `--model` |
| --- | --- | --- |
| by path | nothing: the module only has to be importable from the project root | `package.module:MyModel` |
| by name | `@register("mine")` on the class, and import its module in `twin/behaviour/__init__.py` | `mine` |

`twin/examples/follower_model.py` is a working example of the first (`--model
twin.examples.follower_model:FollowerModel`). A fitted `fitted.params.json` stores the model's name or
path with its parameter list, so `twin.run --params FILE` needs no `--model`.

What a model must do to be fitted:

- **Few, continuous, bounded parameters.** CMA-ES handles dozens. It cannot choose between rule
  variants or train a network's weights.
- **The same random draws whatever the parameters.** Draw a fixed number of values from `rng` per driver
  in `sample_driver` and nowhere else; then two runs with one seed differ only by their parameters.
- **A clip in seconds.** One generation is candidates x clips x seeds simulations.

Optional:

- **A parameter list that depends on the place.** Override `default_params(cls, scene)`; `scene`
  (`base.Scene`) carries the network, the arrivals per route in the fit clips and the number of turning
  shares available. The priority model builds its speed profiles from it.
- **Turning shares.** Add `params.turning_specs(scene.turns)` to the list: the search then fits the
  `turn_<k>` shifts with the model's own parameters and re-draws the demand under them. The model
  never reads them.
- **Values per vehicle kind.** Pass `ratios={kind: {name: ratio}}` to `ParamSet`; `params.of(kind)`
  gives that kind's values.

The constant-speed baseline is not in the registry: it has nothing to fit, and the search runs it only
as the reference row.

## 3. Measurement (`twin/measure/`)

Same code for video and simulation.

- `tracks.py`: loads `*.trajectories.json` (metres on the road via `world_position`), drops boxes cut by
  the frame edge, Kalman/RTS-smooths each track, stitches broken tracks, and matches tracks to routes.
- `tracking_error.py`: position noise (by distance from the camera), standstill wobble, fragmentation,
  share of tracks off the drawn layout. Every later number inherits these errors.
- `demand.py`: the video's arrivals (time, kind, route, entry offset and speed) for replay.
- `speeds.py`: a figure of each vehicle's average speed in the video, per kind.
- `image.py`, `image_score.py`: the comparison in the camera image (section 4).

```powershell
$L = "simulation\Assets\StreamingAssets\Traffic\buoi-hht.camera-lanes.json"
.\.venv\Scripts\python.exe -m twin.measure.tracking_error --layout $L --video runs\buoi-hht\*.trajectories.json
.\.venv\Scripts\python.exe -m unittest discover -s twin/tests -t . -p "test_*.py"
```

## 4. Image-space score (`twin/measure/image.py`, `image_score.py`)

Video and simulation are compared as boxes in the camera image. Video: the tracker's boxes of tracks
that follow a drawn route. Simulation: each vehicle's box projected through the layout's camera
(`camera.world_to_clip`, the matrix Unity renders with). The search, the sensitivity analysis and the
reported results (section 5b) all use this projection.

A vehicle's position is the bottom centre of its box. Image distances are divided by a fixed map of the picture's scale (pixels per metre of road at each
point, from the camera), which removes perspective the same way on both sides; vehicles where the road
is under 2.5 px/m are left out. Nothing follows a vehicle for more than 0.6 s.

| Part | Compares | Weight |
| --- | --- | --- |
| `occupancy_map` | where vehicles are over the clip (16 px cells, blurred) | 1 |
| `occupancy_time` | how many are in each of 6 x 4 zones in each 10 s | 1 |
| `motion` | image velocity over 0.6 s / scale, per zone and group (two-wheelers, larger) | 1 |
| `spacing` | distance to the nearest other vehicle / scale, per group | 1 |
| `crossings` | cumulative crossings of one counting line per drawn lane | 0.5 |
| `headways` | time between successive crossings of each line | 0.5 |

Each part is 0 (the same) to 1; the total is their weighted mean. Scoring one clip takes about 0.1 s.

Crossings and headways count half because the tracker holds a vehicle across a line less reliably than
it detects it.

Why the simulation is not passed through the detector. Rendering a simulation in Unity and tracking the
render with RF-DETR + TrackTrack, as the video is, would put the detector's errors (missed and merged
vehicles, box jitter, broken tracks) on both sides. It takes about 6 minutes per 3-minute clip on one
GPU (GTX 1650: render 1, detection 5), so a search of hundreds of candidates is out of reach and even
the final evaluation (5 configurations x 12 clips x 3 seeds) would take about 17 hours, against about
3 minutes by projection. That path was tried and removed (17 Nov 2026; the code is in git before that
date: `twin/calibrate/evaluate.py`, `feed/emulate_camera.py`, `TrajectoryVideoRenderer -jobs`).

What projection costs, from the one check that was made: the constant-speed run of segment 0010, rendered
and tracked, scored 0.20 against its own projection (motion 0.11, occupancy 0.19-0.20, spacing 0.21,
crossings 0.27, headways 0.30), 0.26 with the real camera's irregular frame holds; the detector found
74-85% of the projected vehicles in every distance band. So the detector alone accounts for a distance of
the same size as the differences between models, every score has that floor, and the parts the detector
changes most (spacing: it merges overlapping vehicles) favour models that keep vehicles apart. Comparisons
between configurations share the floor; absolute scores do not mean "how far the behaviour is off".

## 5. Fitting (`python -m twin.calibrate.search`)

CMA-ES (`cmaes`) over a model's parameters in their 0..1 scales (`--model`, section 2c; the choice model by default). Each candidate replays every
clip's arrivals; the objective is the mean image score plus 0.5 x the share of vehicles that touched
another. Clips run in parallel. Within a generation every candidate runs the same seeds (`--seeds`, two
by default), so candidates differ only by their parameters; the next generation adds 1000 to them, so the
search cannot settle on parameters that only suit one draw of the drivers (`--fixed-seeds` keeps them).
The answer is chosen among the best candidate of each of the last six generations and the final
distribution mean, re-scored on `--select-seeds` (21 22), which neither the search nor the reported
check uses. `--sampled-speeds` (also on the evaluation) ignores the speeds vehicles were seen at.

```powershell
.\.venv\Scripts\python.exe -m twin.calibrate.search --layout $L `
    --video (ls runs\buoi-hht\buoi-hht-20261106-*.trajectories.json) `
    --holdout (ls runs\buoi-hht\buoi-hht-20261108-*.trajectories.json) `
    --names v0 T clat vlat w_pos w_risk --generations 25 --popsize 12 --out runs\twin\fit2
```

`--truth FILE` runs the recovery test (the "video" is a simulation under known parameters), `--names`
limits the search to some parameters. It writes `fitted.params.json` and `report.json`.

### Results (15 Nov 2026, Bưởi, 12 clips of 3 minutes)

Score = image distance, objective = score + penalties; lower is closer. Two seeds per clip.

| | Fit clips (six, 6 Nov) | Held-out clips (six, 8 Nov) |
| --- | --- | --- |
| constant speed | score 0.333, objective 0.986 (1.3 touching pairs per vehicle) | 0.315, 0.887 |
| choice model, defaults | 0.247, 0.296 | 0.255, 0.312 |
| choice model, fitted (`runs/twin/fit2`) | 0.232, 0.282 | 0.250, 0.308 |

Fitted: `v0` 8.2 m/s, `T` 1.9 s, `clat` 0.68 m, `vlat` 0.59 m/s, `w_pos` 0.02, `w_risk` 2.4 (from 8.0, 0.8,
0.3, 0.8, 0.05, 3.0). What these numbers do and do not show:

- The model's structure does the work, not the fitting. Moving from constant speed to the choice model
  with default parameters takes the score from 0.33 to 0.25 and removes nine tenths of the contacts.
  Fitting six parameters adds 0.015 on the clips it saw and 0.005 on the ones it did not.
- Noise floor: two runs with the same parameters and different seeds score 0.07-0.10 against each other
  on one clip. A difference smaller than that on a single clip means nothing.
- Recovery test (`runs/twin/recovery`, run before the keep-clear cost was added): fitting to a simulation under known parameters brought the
  score from 0.24 to 0.08, the noise floor, and found `v0`, `T`, `clat`, `vlat`, `w_pos`, `w_risk` to
  within 10% of their ranges. `v0_spread`, `a`, `b`, `s0`, `tau`, `T_imp` were not found (17-40% off):
  this scene does not show them, so they stay at their defaults.
- One-at-a-time scan over all 12 clips (`runs/twin/scan.log`): real video pins down only `v0` (best
  between 6.5 and 8 m/s; 0.06-0.07 worse at 5 and at 11.5). Every other parameter is flat to within 0.01
  except at extremes (`clat` 0.1 m, `vlat` 0.4 m/s, `w_pos` 0.5). The fitted `T` and `clat` are therefore
  not measurements.
- A first fit on three clips (`runs/twin/fit`) made the held-out clips worse (0.343 to 0.450): it had
  pushed `T` and `clat` to their bounds, which gridlocked one held-out clip. Fit on more clips than
  seems necessary.
- Through the real detector the gain is not confirmed. Segment 0010 simulated with the fitted
  parameters, rendered in Unity, given the camera's frame holds and tracked by RF-DETR scores 0.31
  against the video; the constant-speed run through the same path scores 0.28. By projection the same
  two score 0.26 and 0.33. The difference between the paths: the detector does not see vehicles that
  overlap, which is most of what the choice model fixes (spacing 0.32 by projection, 0.10 through the
  detector, for constant speed). One clip, and both differences are inside the noise floor.

### Priority model (15 Nov 2026, `runs/twin/priority`)

`--model priority`, 25 parameters, 80 generations of 20, same clips, score and seeds as above.

| | Fit clips (six, 6 Nov) | Held-out clips (six, 8 Nov) |
| --- | --- | --- |
| start (flat 7 m/s profiles, video's turning shares) | score 0.319, objective 0.445 | 0.297, 0.426 |
| fitted | 0.220, 0.293 | 0.245, 0.331 |
| choice model, fitted (from the table above) | 0.232, 0.282 | 0.250, 0.308 |

- The search works here: the objective falls by a third on the clips it saw and by a fifth on the day it
  did not, and was still falling slowly at generation 80 (0.354 at generation 0, 0.285 at the end).
- No deadlock: the longest any vehicle stood, in 24 checking runs at the fitted values, was 0.0 s. In
  9,600 simulations during the search none jammed after generation 0.
- Where the gain comes from. Line crossings 0.34 to 0.23 and headways 0.36 to 0.25: `turn_1` went to
  +2.2, moving most ambiguous arrivals from the first route to the second, which is the demand fault
  listed under Known limits. Spacing 0.30 to 0.07 and touching pairs per vehicle 0.25 to 0.14: `keep`
  went to 0.08, so drivers do not keep the speed they were first seen at (a vehicle first seen slow no
  longer crawls and gets run into).
- What got worse: motion, 0.28 to 0.32 on the fit clips and 0.30 to 0.39 on the held-out ones. The
  speed profiles ended at values no driver holds (1.9 m/s at the end of route 1 and a third of the way
  along route 3, 12.9 m/s before it): the search used them to put vehicles in the right places at the
  right times, at the cost of their speeds. Do not read the profiles as measured speeds.
- Against the fitted choice model it is level: slightly better by score, slightly worse by objective,
  because it has more contacts (0.14-0.17 touching pairs per vehicle against 0.10-0.12). That is the
  price of never waiting for a junior vehicle.
What holds the score at about 0.25 is not behaviour: the demand's route split (see Known limits), the
detector gap (section 4), and, in the runs above, that the constant-speed baseline was given each
vehicle's own observed speed while the choice model drew desired speeds from one distribution. Giving the
choice model the observed speeds too (`v0_seen`, section 2, 17 Nov) brought its held-out score at default
parameters from 0.262 to 0.243 (motion 0.378 -> 0.273, now below constant speed's 0.283; spacing 0.091 ->
0.115); the fitted parameters of `fit2` were searched with drawn speeds and score 0.249 with observed
ones (`runs/twin/speeds-sampled`, `runs/twin/speeds-observed`). A new fit with observed speeds
(`runs/twin/fit3`: `v0_seen T clat vlat w_pos w_risk`, 25 x 12, two rotating seeds, 48 min) scores 0.229
on the fit clips and 0.237 held out (objective 0.297; longest stop 5.5 s, was 30.5 s at defaults):
spacing improves (0.115 -> 0.070), motion gives some back (0.273 -> 0.307). `v0_seen` 1.19; `T` (2.18 s)
and `clat` (1.02 m) again go far from their defaults along flat directions, so they are not measurements.
Adding `a` to the search (`runs/twin/fit4`, 43 min) gives 0.233 held out, objective 0.283, contacts 0.10
per vehicle, and motion back to 0.283 (constant speed's level); `a` goes to 3.6 m/s2, near its bound of
4, and the longest held-out stop is 26 s.


## 5b. Evaluation (`python -m twin.calibrate.evaluate`)

Every reported score. Each configuration (constant speed, a model's defaults, or a fitted parameter file)
replays each clip with each seed and is scored by projection against the video; contacts and jams are
counted in the simulation. Seeds 11-13 are not the search's, so this is a fresh check. About 3 minutes
for 5 configurations x 12 clips x 3 seeds on all cores.

```powershell
.\.venv\Scripts\python.exe -m twin.calibrate.evaluate --layout $L `
    --video (ls runs\buoi-hht\buoi-hht-20261106-*.trajectories.json) --holdout (ls runs\buoi-hht\buoi-hht-20261108-*.trajectories.json) `
    --config constant=constant choice_default=model:choice choice_fitted=runs\twin\fit2\fitted.params.json `
             priority_start=model:priority priority_fitted=runs\twin\priority-sep\fitted.params.json `
    --seeds 11 12 13 --out runs\twin\results
```

Results (17 Nov 2026, `runs/twin/results/report.json`; means over 6 clips x 3 seeds; seed spread = mean
over clips of the standard deviation between seeds):

| | Fit score (6 Nov) | Held-out score (8 Nov) | Held-out contacts per vehicle | Seed spread |
| --- | --- | --- | --- | --- |
| constant speed | 0.333 | 0.315 | 1.14 | 0 (deterministic) |
| choice model, defaults | 0.248 | 0.262 | 0.12 | 0.017 |
| choice model, fitted (`fit2`) | 0.233 | 0.260 | 0.12 | 0.019 |
| priority model, start | 0.323 | 0.303 | 0.16 | 0.006 |
| priority model, fitted (`priority-sep`) | 0.252 | 0.271 | 0.08 | 0.004 |

They differ slightly from the searches' own end-of-run checks (section 5; e.g. choice defaults held out
0.255 there, 0.262 here, same seeds) because the code changed after those searches ran on 15 Nov; these are
the current code's numbers. Fitting the choice model gains 0.015 on the fit clips but only 0.002 on the
held-out ones, well inside the seed spread; fitting the priority model gains 0.07 and 0.03.

## 6. Run and play back in Unity

```powershell
.\.venv\Scripts\python.exe -m twin.run --layout $L --video runs\buoi-hht\buoi-hht-20261106-seg-0010-092146.trajectories.json `
    --params runs\twin\fit2\fitted.params.json --out simulation\Assets\StreamingAssets\Traffic\simulation.trajectories.json
.\.venv\Scripts\python.exe -m twin.run --layout $L --rate 150 --minutes 3 --out sim.trajectories.json
.\.venv\Scripts\python.exe -m twin.run --layout $L --model constant --speed motorcycle=6.5 car=6 --out sim.trajectories.json
```

Open the scene; its `TrajectoryPlayer` plays `Traffic/simulation.trajectories.json`. It also plays video
trajectory files.

## Known limits

- The Bưởi layout does not cover every path vehicles use: about 40% of stitched tracks drive off the
  drawn roads or against their direction (contraflow). Widen or add lanes in the Lane Editor; such tracks
  are left out of demand and metrics until then.
- Signal states are not read from video yet; signal plans are given as JSON (`twin/signals.py`). The
  choice model stops for them; the constant-speed model ignores them.
- Demand sends too many vehicles down the first route of the main entry: in segment 0010, 111 of that
  entry's 172 arrivals, where the video's counting lines show about half. Tracks that end before the
  routes split are given a route by the shares of longer tracks, which are skewed. No behaviour
  parameter can make up for this, so it sets a floor under the score.
