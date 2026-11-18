# 3. Experiments (for Chapter 5, "Thực nghiệm")

Each section below matches a section already in `thesis/Chuong/5_Thuc_nghiem.tex`. For each: what was
run, the numbers (with the table or figure file), how to read them, and what is still missing.

Figures: `figures/vi/*.pdf` (Vietnamese labels) or `figures/en/*.pdf`. Tables: `tables/*.tex`
(paste inside a `table` environment and add a caption) and `tables/*.csv`.

**Units and conventions used throughout.** "Score" is the image-space distance $D \in [0, 1]$ between the
tracked video and the simulation projected into the camera image (method 1.6), lower is closer to the
video. The search, the sensitivity analysis and the evaluation all use this same score. "Objective" is $D + 0.5 \times$ (touching pairs per vehicle) $+$ jammed share,
recomputed at weight 0.5 for every row so rows are comparable. "Contacts" is touching pairs of
vehicles per vehicle. Every result is a mean over 6 clips × 3 seeds (seeds 11, 12, 13), never seen by
the search.

---

## 5.1 Evaluation measures ("Các tham số đánh giá")

- Image-space score and its six parts (table `score-parts`, method 1.6).
- Contacts per vehicle (safety; real vehicles do not overlap).
- Jammed share; longest standstill of any vehicle (deadlock check).
- For identifiability: error of a recovered parameter as a share of its range; Morris $\mu^\ast$,
  $\sigma$ against the noise floor.
- Cost: wall time per run.

## 5.2 Experimental method ("Phương pháp thí nghiệm")

**Site.** Bưởi – Hoàng Hoa Thám, Hanoi, iHanoi public camera, 384 × 216 px at 25 fps. Mixed traffic,
mostly motorcycles; the main entry splits three ways.

**Dataset** (table `dataset`). 12 clips of 3 minutes: 6 recorded on 6 November 2026 (fit set) and 6 on
8 November 2026 (held-out set, a different day). In total 13,369 raw tracks, 7,671 after stitching
(the tracker breaks a vehicle's path into 1.7 pieces on average), 2,739 counting-line crossings.

**Tracking quality** (`figures/*/tracking-noise.pdf`, table `dataset`). Measured before anything is
fitted, because every later number inherits it:

- position noise (raw minus smoothed path): 0.22 m on average; 0.13 m at 0–10 m from the camera
  rising to 0.46 m at 70–80 m;
- standstill wobble: 0.20 m RMS;
- median track duration 3.3–3.8 s; 37% of tracks start inside the view (re-acquisitions);
- 42–49% of stitched tracks fit no drawn route (off the drawn road or contraflow) and are left out.

**Procedure.**

1. Demand from each video (method 1.5); each clip is replayed with exactly its own arrivals.
2. Baselines: constant speed; each model with default parameters.
3. Fit on the 6 clips of 6 November (CMA-ES; 1 seed during the search, common random numbers).
4. Re-score the chosen parameters on fresh seeds on the fit clips and on the 6 held-out clips.
5. Identifiability: recovery test, one-at-a-time scan, Morris analysis.

## 5.3 Check on synthetic data ("Kiểm chứng trên dữ liệu tổng hợp")

This tests the search, so it uses the search's internal score on both sides (truth and candidates are
both simulations).

**What.** The video features of 3 clips were replaced by a simulation under known choice-model
parameters (different seed). All 12 parameters were then fitted from defaults (30 generations).

**Results** (table `recovery`, `figures/*/recovery.pdf`).

- The score fell from 0.240 to 0.079, which is the noise floor (two seeds of the same parameters score
  about 0.07–0.10 against each other on one clip). The search reaches what is reachable.
- Found within 10% of their range: `vlat` (0%), `w_pos` (0%), `w_risk` (1%), `v0` (8%), `clat` (8%),
  `T` (9%).
- Not found (17–40% off): `tau`, `T_imp`, `a`, `b`, `v0_spread`, `s0`.

**Reading.** In this scene, with this measurement, only six of twelve parameters leave a trace that the
image statistics can pick up. The others do not change what the camera sees enough; they are kept at
their defaults in the real fit. This is a property of the data and the measure, not of the optimiser.

**Caveat to state in the thesis.** This run was made before the "keep clear" cost was added to the
choice model. Re-running it on the current model (`python -m twin.calibrate.search --truth ...`) is
about 25 minutes and would make the section exact.

## 5.4 Calibration on real data ("Hiệu chỉnh trên dữ liệu thực")

**What.** Each model was fitted on the 6 clips of 6 November (choice model: 6 parameters, 25 generations
of 12; priority model: 25 parameters, 80 generations of 20, contact weight 3). The resulting
configurations are then scored on all 12 clips with 3 seeds the search did not use
(`python -m twin.calibrate.evaluate`, `runs/twin/results/report.json`, 17 Nov 2026):

| Configuration | Parameters | Source |
| --- | --- | --- |
| Constant speed | none | baseline |
| Choice, defaults | `params-choice` defaults | |
| Choice, fitted | `fitted-choice` | `runs/twin/fit2` |
| Priority, start | flat 7 m/s profiles, video's turning shares | |
| Priority, fitted | `fitted-priority` | `runs/twin/priority-sep` |

**Results** (table `results-main`, `results-parts-holdout`; figures `score-by-model`, `contacts-by-model`,
`score-parts`). Means over 6 clips × 3 seeds. Seed spread = mean over clips of the standard deviation of
the score between seeds (held out).

| Configuration | Fit score | Held-out score | Held-out objective | Held-out contacts | Seed spread |
| --- | --- | --- | --- | --- | --- |
| Constant speed | 0.333 | 0.315 | 0.887 | 1.14 | 0 (deterministic) |
| Choice, defaults | 0.248 | 0.262 | 0.320 | 0.12 | 0.017 |
| Choice, fitted | 0.233 | 0.260 | 0.318 | 0.12 | 0.019 |
| Priority, start | 0.323 | 0.303 | 0.385 | 0.16 | 0.006 |
| Priority, fitted | 0.252 | 0.271 | 0.311 | 0.08 | 0.004 |

How to read them:

1. *Structure, not fitting, makes most of the difference for the choice model.* Constant speed → choice
   with defaults: −0.085 on the fit clips, −0.053 held out (−17%). Defaults → fitted: −0.015 on the fit
   clips but only −0.002 held out, far inside the seed spread (0.019). Fitting the choice model did not
   improve it on another day.
2. *Fitting helps the priority model.* Start → fitted: −0.071 on the fit clips, −0.032 held out (−11%).
   Its start values carry no information, so this was expected. Fitted, it is still slightly behind the
   choice model on held-out score (0.271 against 0.260) but has the fewest contacts (0.08 per vehicle).
3. *Generalisation.* Both fitted models score 0.02–0.03 worse on the held-out day than on the fit day,
   while constant speed scores 0.02 *better* on it; part of the gap is the day, not overfitting. The
   gains from fitting shrink on the held-out day (choice: 0.015 → 0.002; priority: 0.071 → 0.032).
4. *Which parts move* (held out). Spacing 0.31 → 0.09–0.10 and occupancy map 0.28 → 0.19–0.24: removing
   overlaps is what the interacting models do best. Crossings and headways improve only with the fitted
   priority model (0.32 → 0.26, 0.34 → 0.25), from its turning shifts that correct the demand. Motion gets
   *worse* with every interacting model (0.28 → 0.35–0.42): the constant-speed baseline is given each
   vehicle's own observed entry speed, the models draw desired speeds from one distribution (threats).
5. *Contacts.* From constant speed to either interacting model, touching pairs per vehicle drop by about
   90% (1.14 → 0.08–0.12 held out). No configuration jammed; the choice model has one held-out clip where
   a vehicle stood for about 70 s (the nose-to-nose failure of 5.7).
6. *Projection floor.* These scores compare a projected simulation with a detected video (method 1.6).
   Differences between configurations are meaningful; the absolute level (about 0.25) includes the
   detector's errors on the video side and is not "how far the behaviour is off".

**Statements that do not depend on the evaluation** (they are about the parameters, not about scores):

- *The fitted speed profiles are not measurements.* The priority model's profiles end at values no
  driver holds (1.8–2.6 m/s at the end of some routes, 11–12 m/s elsewhere) and `keep` goes to 0.07
  (drivers forget the speed they were first seen at). The search uses speed to put vehicles in the right
  place at the right time (table `fitted-priority`).
- *The choice model's fitted `T` (1.9 s) and `clat` (0.68 m) are not measurements either*: the search's
  objective is flat along them (5.6). `v0` 8.2 m/s is supported (best between 6.5 and 8 m/s).
- *The turning shifts correct the demand*: `turn_1` → +1.8, `turn_2` → −2.1 move ambiguous arrivals off
  the main entry's first route, which the demand over-assigns (threats to validity).
- *Fit on enough clips*: a first choice-model fit on only 3 clips pushed `T` and `clat` to their bounds,
  which jammed one held-out clip.

**Convergence** (`figures/*/convergence.pdf`, the search's internal objective). Choice: the best
objective falls from 0.293 to 0.278 in 25 generations, most of it in the first 3. Priority: 0.627 →
0.373 (at contact weight 3) over 80 generations and still falling slowly at the end.

## 5.5 Optimiser and objective configuration ("So sánh thuật toán tối ưu và cấu hình hàm mục tiêu")

**What exists.**

- *Contact weight* (objective configuration): the priority model was fitted at weight 0.5 (v1) and at
  3.0 (`runs/twin/priority-cw3`, v1 rules; `priority-sep`, final). At weight 3 the v1 model's contacts
  fell from 0.17 to 0.15 per vehicle on held-out clips while its score rose from 0.245 to 0.279. That is
  the expected trade-off: the image score does not see overlap, so penalising overlap costs some score.
- *Number of fit clips*: 3 clips overfit, 6 did not (5.4, point 3).

**What is missing.** No comparison of optimisers exists for the current system. (The CMA-ES vs Optuna
comparison made for the old Unity model, in the git tag `archive/snapshots-20261116`, should not be reused.)
Two options:

- Run random search with the same budget (25 × 12 evaluations, choice model, same 6 parameters) and
  compare the curves with `figures/*/convergence.pdf`. About 20 minutes. This would show whether
  CMA-ES's adaptation helps on this objective.
- Or drop the optimiser half of this section and keep the objective-configuration half.

## 5.6 Stability and computational cost ("Độ ổn định và chi phí tính toán")

The scan and the Morris analysis describe the landscape of the search's objective: which parameters it
can move. The seed spread of the reported score comes from the evaluation (`seed_spread` in
`runs/twin/results/report.json`): 0.017–0.019 for the choice model, 0.004–0.006 for the priority model,
per clip on the held-out day.

**Noise floor.** The same parameters with two different seeds score 0.07–0.10 apart on one clip.
Averaged over 12 clips, the Morris noise floor is 0.015 (choice) and 0.007 (priority) in score units.
Any single-clip difference under 0.1 means nothing; differences between 12-clip means above 0.015 are
real.

**One-at-a-time scan** (`figures/*/scan-one-at-a-time.pdf`; 12 clips, 2 seeds). Only `v0` has a clear
minimum (6.5–8 m/s; 0.06–0.07 worse at 5 and at 11.5 m/s). Everything else is flat within about 0.01,
except at extreme values (`clat` 0.1 m, `vlat` 0.4 m/s, `w_pos` 0.5).

**Morris analysis** (tables `sensitivity-choice`, `sensitivity-priority`; existing figures
`figures/*/sensitivity-*.png`).

- Choice model (16 trajectories × 12 clips, 77 min): 11 of 12 parameters have an effect above the noise
  floor of 0.015; `v0` has the largest ($\mu^\ast$ 0.091), then `a` (0.060), `w_risk` (0.044), `w_pos`
  (0.039), `clat` (0.038). $\sigma \approx \mu^\ast$ for most parameters: effects depend on where the
  other parameters are (interactions). On contacts, `clat` and `w_risk` dominate.
- Priority model (12 trajectories × 12 clips, 24 min): 19 of 25 above the noise floor (0.007); the
  largest is one speed knot of the main route (`v_r0_k1`, 0.098), then `v_r0_k0`, `keep`, `turn_1`,
  `large`. On contacts, `s0`, `clat`, `T` dominate.
- Reconciling with the scan: Morris moves parameters over wide steps from random starting points, so it
  finds effects that are there *somewhere* in the space; the scan shows that near the defaults, most of
  them are flat. Both are true; the second is what limits calibration.

**Stability of the fit itself.** Not measured yet: each fit was run once. To fill this, repeat the
choice fit with 3 different `--seed` values for CMA-ES (about 1 hour) and report the spread of the
fitted values and of the held-out score.

**Cost** (table `compute-cost`). Choice fit 18 min, priority fit 44 min, choice Morris 77 min, on one
desktop machine using all cores (simulation dominates; scoring is about 0.1 s per clip). The evaluation
(180 simulations) takes about 3 minutes. Scoring through the detector instead would cost about 6 minutes
per simulated clip on one GPU (render 1, detection 5), 17 hours for the evaluation alone, which is why
the simulation is projected (method 1.6).

## 5.7 Qualitative assessment ("Đánh giá định tính các hiện tượng đặc trưng")

**Material.** `videos/` holds renders of every model version replaying the same clip (6 Nov, segment
0010) from the camera's viewpoint, and `three-models-<scene>.mp4` puts constant speed, the fitted choice
model and the fitted priority model side by side with a live count of touching pairs. Take stills for
the thesis from these.

**Contacts over time** (`figures/*/contacts-over-time.pdf`): constant speed accumulates 237 touching
pairs over 193 s, the choice model 24, the priority model 18.

**Phenomena to describe** (what to look for in the videos):

- motorcycles filtering between slower vehicles and spreading across the road width (choice model;
  from the risk and open-road costs);
- vehicles moving aside for oncoming traffic on shared approaches;
- following at a time gap, queues forming and discharging;
- the choice model's failure mode: two large vehicles nose to nose on crossing left turns, standing
  for a minute (one clip of twelve, under some parameter values);
- the priority model's failure mode: where two vehicles would lock, the senior one drives through the
  junior one (a recorded contact) instead of both stopping.

---

## Threats to validity (for 5.x or 6.x)

- **One site, two days.** Results are for one intersection and one camera. The Cầu Giấy layout exists
  and has simulations in `videos/three-models/caugiay-tranduykien/`, but no fit there.
- **Demand quality.** 42–49% of tracks fit no drawn route; the route split of tracks that end before
  the split is skewed (in segment 0010, 111 of 172 arrivals of the main entry go to its first route,
  where the counting lines show about half). This puts a floor under the score that no behaviour
  parameter can remove; the priority model's turning parameters partly compensate.
- **The simulation is projected, the video is detected.** The video's boxes carry the detector's
  errors (misses, merged overlapping vehicles, jitter, broken tracks); the projected simulation's do not.
  One check (method 1.6) put this gap at 0.20–0.26, as large as the differences between models, and
  one clip scored through the detector ranked the fitted choice model behind constant speed (0.31
  against 0.28), the reverse of projection (`docs/twin.md` section 5). One clip is inside the noise
  floor, so this proves nothing either way, but the ranking through a real camera is not established.
- **Baseline advantage.** The constant-speed baseline is given each vehicle's own observed entry speed,
  while the choice model draws desired speeds from one distribution; this favours the baseline on
  *motion*.
- **No signals from video.** Signal states are not read from the video.
