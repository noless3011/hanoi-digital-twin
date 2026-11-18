# 4. Conclusion notes (for Chapter 6, "Kết luận") and the abstract

## Contributions (what the project delivers)

1. **A calibration loop from public-camera video to a traffic simulation**, end to end: capture,
   detection and tracking, demand extraction, simulation, projection of the simulation into the same
   camera image, comparison, parameter search. Every step is automated and reproducible from the command line.
2. **An image-space distance between a simulation and a video** that does not need long, unbroken
   tracks and is fast enough (0.1 s per clip) to sit inside a search. Six statistics compared in the camera image after removing perspective with a road-scale
   map, robust to the tracker breaking a vehicle into 1.7 tracks on average.
3. **Two behaviour models for mixed, lane-free traffic**, behind one interface:
   a choice model (each driver minimises a predicted cost over 30 actions; following, giving way,
   filtering and moving aside all come from one cost) and a priority model (drivers react only to
   vehicles that entered before them; provably free of deadlock).
4. **A model-independent calibration and identifiability toolkit**: CMA-ES with common random numbers,
   held-out evaluation on another day, a recovery test with known parameters, a one-at-a-time scan and a
   Morris analysis with a measured noise floor. Any model with a few continuous parameters plugs in.
5. **An honest measurement of what video can and cannot pin down** (the main scientific result, below).

## Main findings (for the conclusion and the abstract)

- Contacts, counted in the simulation itself: from constant speed to either interacting model, touching
  pairs per vehicle drop by about 90% (held out: 1.14 → 0.08–0.12).
- Distance to the video on the held-out day (`runs/twin/results/report.json`): constant speed 0.315,
  choice model 0.260 (−17%), priority model fitted 0.271 (−14%).
- The model's *structure* explains most of the improvement. For the choice model, fitting gains 0.015
  on the fit clips and 0.002 on the held-out day, inside the seed spread: the defaults were already as
  good. For the priority model, whose start values carry no information, fitting gains 0.071 and 0.032.
- What improves is where vehicles are and how close they are (spacing, occupancy); motion gets worse
  than the constant-speed baseline, which is given each vehicle's observed entry speed.
- From this camera, only some parameters are **identifiable**: in the recovery test 6 of 12; on real
  video essentially only the desired speed `v0` has a clear optimum (6.5–8 m/s). Fitted values of the
  others are not measurements of driver behaviour.
- Part of the remaining distance comes from **input**, not behaviour: the demand's route split and roads
  not drawn in the layout.

## Limitations

- One intersection, two days, 36 minutes of video.
- 42–49% of tracks drive where no route is drawn (contraflow, off-road paths) and are not replayed.
- Route of tracks that end before a split is guessed from skewed shares.
- Signal states are not read from the video.
- The simulation is projected into the image, the video is detected: the detector's errors are on one
  side only. Passing simulations through the detector costs about 6 minutes per clip on one GPU (17 hours
  for the evaluation), more than the project could spend. One check put the gap at 0.20–0.26, so absolute
  scores include it, and on one clip the detector path ranked constant speed ahead of the fitted choice
  model (inside the noise floor, but unresolved).
- The choice model can gridlock in rare configurations; the priority model avoids it by allowing
  contacts.
- Each fit was run once (stability of the search not measured).

## Future work

- Fit and validate at a second intersection (Cầu Giấy layout already exists).
- Draw the missing paths in the layout, or let routes be learnt from tracks.
- Estimate turning shares from counting lines instead of from long tracks.
- Read signal states from the video (the detector already has a traffic-light class).
- Score the final configurations through the detector (render + RF-DETR, about 6 min per clip; 5 h for
  one seed of the evaluation), or make that path faster (smaller renders, frame skipping, a lighter
  detector), or learn a correction from projection to detector so that a search can use it.
- Add identifiable structure instead of more parameters: e.g. desired-speed distributions per route
  measured directly from video, leaving the search only what video cannot measure.
- A neural behaviour model trained on tracks could plug into the same interface for comparison (a first
  study on synthetic data was made and later removed from the repository).
