# 1. Method (for Chapter 3, "Đề xuất")

Source material for the method chapter. Everything here describes the code as it is in `twin/` at the
end of November 2026. Notation is used the same way in `algorithms.tex`.

The current outline of `thesis/Chuong/3_De_xuat.tex` still describes the old C# model ("mô hình hành
vi tham số hoá theo phân phối", red-light pass probability, lane flexibility). A matching outline for
what was built:

| Section | Content | Material below |
| --- | --- | --- |
| 3.1 Solution overview | the loop: video → demand and features; simulation → camera → features; compare → update | 1.1, `figures/architecture.pdf` |
| 3.2 Problem statement | calibration as stochastic optimisation | 1.2 |
| 3.3 Simulation engine and behaviour contract | engine/model split, route coordinates | 1.3 |
| 3.4 Behaviour models | choice model, priority model, constant-speed baseline | 1.4 |
| 3.5 Reproducing the input conditions | demand from video | 1.5 |
| 3.6 Image-space distance | the six parts; projecting the simulation instead of passing it through the detector | 1.6 |
| 3.7 Objective function | score + penalties | 1.7 |
| 3.8 Parameter search | CMA-ES, common random numbers | 1.8 |
| 3.9 Sensitivity and identifiability | Morris, recovery test, one-at-a-time scan | 1.9 |

---

## 1.1 Solution overview

The goal is a simulation of one intersection whose traffic looks like the traffic a public camera
records. Four things are needed: (1) a simulator whose driver behaviour is controlled by a small set
of parameters, (2) the same input as the video (which vehicles arrive, when, where), (3) a way to
measure how far a simulated clip is from a recorded one, and (4) a search that changes the parameters
to reduce that distance.

The loop (`figures/architecture.pdf`):

1. **Perception.** The camera video goes through RF-DETR (object detector, 7 classes) and TrackTrack
   (multi-object tracker). The result is a box per vehicle per frame, with a track ID.
2. **Demand.** Each track is projected to the road plane (metres), smoothed, stitched where the tracker
   broke it, and matched to a route of the drawn road layout. Tracks that start at a route's entrance
   become *arrivals*: time, vehicle kind, route, entry offset, entry speed.
3. **Simulation.** The engine replays exactly these arrivals. A behaviour model decides every driver's
   acceleration and sideways speed at every step.
4. **Same camera.** Each simulated vehicle's 3D box is projected into the image through the camera
   matched to the real one (the matrix Unity renders with), giving boxes in the same picture as the
   video's.
5. **Comparison.** The tracked video's boxes and the projected simulation's boxes are compared by six
   statistics in the image (where vehicles are, when, how they move, how close they are, how they cross
   counting lines).
6. **Search.** CMA-ES proposes parameter sets and keeps those that bring the distance down. The final
   configurations are re-scored on fresh seeds and on clips from another day.

**Key design decision: compare in the image, not on the road.** The first version compared quantities
in metres (travel time per route, flow, sideways clearance, gaps at crossing points). That needs every
video track to be followed through the whole view; with this camera a vehicle is tracked for a median
3.4 s and the tracker breaks one vehicle into 1.7 tracks on average (table `dataset`). Statistics that
follow a vehicle for at most 0.6 s do not depend on track continuity. Comparing boxes in the image
also means errors of the road-plane projection do not enter the score.

**Key design decision: project the simulation into the image instead of passing it through the
detector.** The detector is not a perfect observer: it misses small and far vehicles, merges vehicles
that overlap, jitters boxes and breaks tracks. Rendering the simulation in Unity and running it through
the same detector and tracker would put these errors on both sides of the comparison. It costs about 6
minutes per 3-minute clip on one GPU (render 1, detection 5): a search needs hundreds to thousands of
simulations, and even the final evaluation alone would take about 17 hours. Projection takes about 0.1 s
per clip, so it is used everywhere, and the detector's errors stay on the video side only (1.6).

**Key design decision: replay the video's own arrivals.** Each simulated clip has the same vehicles
entering at the same times on the same routes as the video. Differences in demand then cannot be
mistaken for differences in behaviour, and time-aligned statistics (occupancy over time, cumulative
crossings) become possible.

## 1.2 Problem statement

Let $\mathcal{C}$ be a set of video clips. For each clip $c$, $\mathcal{D}_c$ is its demand (the
arrivals) and $F^V_c$ its image features. A behaviour model $\mathcal{M}$ has parameters
$\theta \in \Theta = \prod_{k=1}^{n} [l_k, h_k]$. Simulating clip $c$ with parameters $\theta$ and random
seed $s$ gives features $F^S_c(\theta, s)$ and safety counts. Calibration is

$$
\hat\theta = \arg\min_{\theta \in \Theta} \; J(\theta), \qquad
J(\theta) = \frac{1}{|\mathcal{C}|\,|\mathcal{S}|} \sum_{c \in \mathcal{C}} \sum_{s \in \mathcal{S}}
\Big[ D\big(F^V_c, F^S_c(\theta, s)\big) + \alpha\, \frac{P_c(\theta, s)}{N_c} + \beta\, U_c(\theta, s) \Big]
$$

- $D \in [0, 1]$: image-space distance (1.6).
- $P_c / N_c$: pairs of vehicles whose bodies touched, per vehicle. Real traffic does not overlap, so a
  model that lets vehicles drive through each other is penalised. $\alpha = 0.5$.
- $U_c$: share of the clip not simulated because the road jammed (more than 80 vehicles on it at once).
  $\beta = 1$.
- $\mathcal{S}$: a fixed set of seeds, the same for every candidate (common random numbers, 1.8).

$J$ has no gradient (drivers choose among discrete actions, contacts are counts), is noisy (two seeds
give different traffic), and costs seconds per clip. These three properties decide the search method.

## 1.3 Simulation engine and the behaviour contract

The engine (`twin/engine.py`) is written in Python/NumPy and is deterministic for a seed. Unity is not
in the loop; it only draws the layout and plays results back.

**Road network.** Lanes drawn in Unity's lane editor are exported with the camera
(`*.camera-lanes.json`). Every sequence of lanes from an entry to an exit is a *route*: a polyline with
a width. A vehicle on route $r$ is at $(s, d)$: $s$ along the route, $d$ sideways. There are no lanes
inside a route: $d$ is continuous, which is what mixed motorcycle traffic needs.

**Split of responsibility.** The engine owns the world: bodies (length, width by kind), kinematics,
spawning, perception geometry, integration, exits, recording and contact detection. It enforces no
driving rule; vehicles move exactly as the model says, through each other if the model lets them, and
every overlap is recorded. The model owns the drivers: their parameters, their memory, every decision.

**One step** (Algorithm `alg:step`), $\Delta t = 0.1$ s:

1. due arrivals enter at the start of their route at the speed and offset the model gives;
2. the engine builds a *View*: every vehicle's pose, and every nearby pair $(i, j)$ with $j$
   projected into $i$'s route coordinates (gap along the route, sideways offset, extents, speed along
   $i$'s route, heading agreement), plus signal states;
3. the model returns an acceleration $a_i$ and a sideways speed $\dot d_i$ per vehicle;
4. the engine integrates: $v_i' = \max(v_i + a_i \Delta t, 0)$, $s_i \mathrel{+}= \tfrac12(v_i + v_i')\Delta t$,
   $d_i \mathrel{+}= \dot d_i \Delta t$;
5. it records samples, detects contacts (oriented boxes shrunk by 10%, separating-axis test) and
   removes vehicles that left.

Every array operation covers all vehicles at once, so a 3-minute clip with about 220 vehicles runs in
1–3 s.

## 1.4 Behaviour models

Three models implement the contract. They are deliberately different in kind, which is what the
comparison in Chapter 5 is about.

### Constant-speed model (baseline)

Each vehicle keeps the speed and sideways offset it was first seen at and ignores everyone. It has no
parameter to fit. It is the reference: anything a model adds must beat it.

### Choice model (`twin/behaviour/choice.py`, Algorithm `alg:choice`)

A driver chooses, every step, the cheapest of 30 actions (6 accelerations × 5 sideways speeds). For each
action it predicts its own path over a horizon $\tau$ at 5 time points, assumes every other vehicle keeps
its velocity, and adds costs:

| Cost | Definition | Parameters |
| --- | --- | --- |
| progress | $\big(\frac{v(\tau) - v_0}{v_0}\big)^2$, growing with the time the driver has waited ($T_\mathrm{imp}$) | $v_0$, $v_{0,\mathrm{spread}}$, $a$, $T_\mathrm{imp}$ |
| risk | for each nearby vehicle, $\max_m [p_\mathrm{long}^2 \, p_\mathrm{side}]$ plus a large term if the bodies would overlap; $p_\mathrm{long} = \mathrm{clip}(1 - g / (s_0 + T v), 0, 1)$, $p_\mathrm{side} = \mathrm{clip}\big((c_\mathrm{lat} - \text{side gap}) / (c_\mathrm{lat} + \text{narrower width}), 0, 1\big)$; weighted by $w_\mathrm{risk}$ times the driver's caution | $T$, $s_0$, $c_\mathrm{lat}$, $w_\mathrm{risk}$, $\tau$, $b$ |
| open road | the share of $v_0$ that slower or oncoming vehicles in line ahead do not allow, at the offset the action leads to | — |
| keep clear | coming to rest in the road just ahead of a vehicle that is waiting to cross; whoever waited longer weighs more | — |
| position | $w_\mathrm{pos}(d - d^\ast)^2$ ($d^\ast$: the driver's own share of the road width), plus leaving the road | $w_\mathrm{pos}$, $v_\mathrm{lat}$ |
| effort | hard acceleration, braking, sideways movement | fixed |
| stop line | running a red light (with signals) | — |

Following, giving way, filtering past slower vehicles and moving aside for oncoming traffic all come
from these costs; there are no separate rules. With nobody near, a driver holds $v_0$ and its position,
which is the constant-speed model. Parameters are given for the motorcycle; other kinds use fixed
ratios (table `params-choice`; e.g. a car's $T$ is 1.4 × the motorcycle's).

**Known limit: gridlock.** In a junction without signals, a ring of drivers each waiting for the next
can form. A four-way junction locks at 400 vehicles/h per route. In replayed video demand this happened
in one clip of twelve under some parameter values.

### Priority model (`twin/behaviour/priority.py`, Algorithm `alg:priority`)

Built for two properties: it cannot deadlock, and its parameters are things a video can show.

- Every vehicle has a *seniority*: the order it entered in. A driver reacts only to vehicles senior to
  it. It follows them and waits for them at Gipps' safe speed,
  $v^\mathrm{safe} = -bT + \sqrt{(bT)^2 + v_j^2 + 2b \max(g - s_0, 0)}$. It ignores everyone junior.
- **No-deadlock property.** The most senior vehicle on the road reacts to nobody, so it always moves
  and eventually leaves; then the next is most senior. A cycle of waiting drivers cannot form, whatever
  the parameters. The cost: where two vehicles would lock, the senior one drives through the junior
  one, and the engine records a contact. (Proof sketch for the thesis: the "waits for" relation is a
  subset of the strict seniority order, which is acyclic.)
- A junior vehicle directly ahead of a senior one in the same direction swaps seniority with it, so the
  vehicle in front leads. Seniority remains a strict total order.
- Three sideways rules keep vehicles apart without breaking the order, because each changes only what
  the junior driver does: *separate* (a junior body touching a senior one brakes if it is ahead and in
  line, and always steers away), *enter* (an arriving vehicle takes the free place across the road
  nearest where it was seen), *look aside* (no drift towards someone beside you).
- Wanted speed depends on place: each route carrying at least 5% of arrivals has a speed profile of 4
  values along its length (linear between them), multiplied by the driver's own factor. The factor
  blends the speed the vehicle was first seen at with a random draw: $f = (v_\mathrm{seen}/\bar v)^{k}
  \, \epsilon^{1-k}$, $k$ = `keep`.

Parameters (25 at Bưởi, table `params-priority`): 16 profile speeds + `v_free`, `keep`, `spread`,
`large`, $T$, $s_0$, $c_\mathrm{lat}$, and two turning-share shifts `turn_1`, `turn_2` (demand, see 1.5).

### Randomness

A driver's random quantities (desired speed, caution, entry position) are drawn once, at entry, with a
fixed number of draws whatever the parameters. Two runs with the same seed and demand therefore differ
only by their parameters, which is what makes common random numbers work (1.8).

## 1.5 Reproducing the input conditions (demand)

Algorithm `alg:demand`, `twin/measure/demand.py`.

1. **Projection to the road plane.** The bottom centre of each box is projected to the road through the
   camera exported from Unity (the same camera the video's layout was drawn under).
2. **Smoothing.** Each track is Kalman-filtered and RTS-smoothed (constant-velocity model). This gives
   speeds and removes jitter: position noise is 0.12 m near the camera and 0.46 m at 70–80 m
   (`figures/*/tracking-noise.pdf`).
3. **Stitching.** Tracks broken by the tracker are joined when the end of one and the start of another
   agree in time, place and heading. On the 12 clips: 1,159 raw tracks became 675 for clip 0010.
4. **Route matching.** Each track is matched to the routes it fits (distance to the route line and
   direction). 43% of stitched tracks fit no drawn route (they drive off the drawn road or against its
   direction); they are left out (a known limit).
5. **Arrivals.** A track first seen within 15 m of the visible start of its route is an arrival; its
   entry time is extrapolated back to the route start. Tracks first seen further in are
   re-acquisitions: counted, not replayed.
6. **Ambiguous routes.** A track that disappears before its route splits has several candidate routes.
   It gets one drawn from the turning shares of tracks from the same entry that were seen past the
   split (seeded, so the same video always gives the same demand). Because longer tracks are not a
   fair sample, these shares are skewed; the priority model therefore fits a log-odds shift
   `turn_k` for each route of the main entry. They are parameters of the demand, not of the drivers.

## 1.6 Image-space distance

Algorithm `alg:score`, `twin/measure/image.py` and `image_score.py`.

**Same geometry on both sides.** Video: the tracker's boxes of tracks that follow a drawn route.
Simulation: each vehicle's 3D box (length, width, height by kind) projected through the camera matrix
Unity renders with. Position = bottom centre of the box. Distances in the image are divided by a fixed
map of the road scale (pixels per metre at each point), which removes perspective identically on both
sides. Vehicles where the road scale is below 2.5 px/m (far away) are dropped.

**Six parts**, each a distance from 0 (identical) to 1 (table `score-parts`):

| Part | What it compares | Distance |
| --- | --- | --- |
| occupancy map | mean vehicles per frame in 16 px cells, Gaussian-blurred | Bray–Curtis |
| occupancy over time | vehicles per zone (6 × 4 grid) per 10 s | Bray–Curtis |
| motion | image velocity over 0.6 s / road scale (≈ m/s), per zone and vehicle group | Wasserstein-1 of x and y parts / video mean speed |
| spacing | distance to the nearest other vehicle / road scale (≈ m), per group | Wasserstein-1 / video mean |
| crossings | cumulative count of crossings of one counting line per drawn lane | area between the curves / area under the upper one |
| headways | time between successive crossings of a line | Wasserstein-1 / video mean |

Definitions for the thesis:

- Bray–Curtis: $\mathrm{BC}(a, b) = \sum_i |a_i - b_i| \,/\, \sum_i (a_i + b_i)$.
- Wasserstein-1 between 1-D samples with empirical CDFs $F_a$, $F_b$:
  $W_1(a, b) = \int |F_a(x) - F_b(x)|\,dx$.
- Crossings: with $N^V(t)$, $N^S(t)$ the cumulative counts at 1 s steps,
  $D = \sum_t |N^V(t) - N^S(t)| \,/\, \sum_t \max(N^V(t), N^S(t))$.
- Parts that pool several keys (zones, groups, lines) take the mean weighted by the video's sample
  count; a key the video has and the simulation lacks counts as 1.
- Total: $D = \sum_p w_p D_p / \sum_p w_p$ over parts both sides have data for; $w = 1$ except
  crossings and headways, $w = 0.5$, because they need a track to hold across a line, which the tracker
  does less reliably than detecting vehicles frame by frame.

**Projection, not the detector, on the simulation side.** The simulation's boxes are computed from
the simulation through the camera matrix, not detected in a render: rendering and detecting takes about 6
minutes per clip on one GPU, against 0.1 s for projection (1.1). What that leaves out was measured once:
one simulated clip, rendered and passed through RF-DETR and the tracker, scored 0.20 against its own
projection (0.26 with the real camera's frame timing), with 74–85% of vehicles detected at every distance.
So the scores have a floor of that size from the detector alone, and the parts the detector changes most
(spacing, because it merges overlapping vehicles) favour models that keep vehicles apart. All
configurations are scored the same way, so comparisons between them share this floor; an absolute score
is not a measure of how far the behaviour is off.

## 1.7 Objective function

$J$ as in 1.2. Why the penalties: the image score alone does not see overlap (two vehicles drawn on
top of each other look almost like one), so a model could improve the score by letting vehicles pass
through each other. The contact term makes that cost. The jam term makes a run that grid-locks count
as a failure instead of taking minutes. Contact weight 3.0 instead of 0.5 was also tried (runs
`priority-cw3`, `priority-sep`).

## 1.8 Parameter search: CMA-ES

Algorithm `alg:cmaes`, `twin/calibrate/search.py`, library `cmaes`.

- **Why CMA-ES.** A dozen to 25 continuous, bounded parameters; an objective with no gradient and with
  noise; one evaluation costing seconds per clip; and a population per generation that fills the
  machine's cores. CMA-ES (Hansen, 2001/2016) adapts a full covariance, so it handles parameters that
  act together (e.g. $T$ and $s_0$), and it is invariant to monotone transforms of $J$.
- **Search space.** Every parameter is mapped to $[0, 1]$, log scale for positive quantities
  spanning a decade or more ($v_0$, $T$, $w_\mathrm{risk}$, ...), linear otherwise. Start: the model's
  defaults; $\sigma_0 = 0.2$ of each range.
- **Common random numbers.** Every candidate in a run is simulated with the same seeds. Since the
  random draws do not depend on the parameters (1.4), the difference between two candidates is caused by
  their parameters, not by different traffic. This removes most of the noise from comparisons between
  candidates; it does not remove the bias of fitting to one particular draw, which is why the final
  choice is re-checked on fresh seeds.
- **Final choice.** Both the best candidate seen and the final distribution mean are re-scored on 3 new
  seeds; the better is kept.
- **Evaluation** (`twin/calibrate/evaluate.py`). The chosen parameters, the defaults and the
  constant-speed baseline are scored on the fit clips and on held-out clips the search never saw, with
  3 seeds the search did not use. These are the reported numbers.
- **Parallelism.** Each generation's (candidate, clip, seed) simulations run in a process pool.

## 1.9 Sensitivity and identifiability

Three complementary checks of whether the video can determine the parameters:

1. **Recovery test (synthetic truth).** Replace the video with a simulation under known parameters
   $\theta^\dagger$ (different seed), fit, and compare $\hat\theta$ with $\theta^\dagger$. A parameter
   the search cannot find in this ideal case cannot be fitted from real video either.
2. **One-at-a-time scan.** Move one parameter over 3–5 values with the others at defaults, on all
   12 clips with 2 seeds. Shows the shape of $J$ along each axis.
3. **Morris elementary effects** (Algorithm `alg:morris`). $r = 16$ (choice) or 12 (priority)
   trajectories through a 4-level grid; each step changes one parameter by $\Delta = 2/3$ of its range.
   For parameter $k$: $\mu^\ast_k = \mathrm{mean}\,|EE_k|$ (importance) and $\sigma_k$ (interactions or
   non-linearity). **Noise floor:** each trajectory's start is simulated again with another seed; the
   mean change is what a step with no effect would show. Effects at or below it are indistinguishable
   from noise.
