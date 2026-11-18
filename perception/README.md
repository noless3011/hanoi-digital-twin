# Perception: traffic detection and tracking

Everything here uses the seven-class RF-DETR checkpoint in
`models/rfdetr_nano_traffic7_bundle/` (car, bus, van, others, person,
motorcycle, traffic light) with the Ultralytics TrackTrack tracker. It turns
camera video into tracked vehicles and per-frame trajectories, which feed
[`twin/`](../docs/twin.md) measurement and calibration. Training the detector is in
[`training/`](training/README.md).
Dependencies are in `requirements.txt` and `rfdetr-requirements.txt`; the root
`setup.ps1` installs them into the shared root `.venv`.

| File | Purpose |
| --- | --- |
| `run_rfdetr.ps1` / `run_rfdetr.py` | Live tracking preview of the local video feed |
| `export_vehicle_endpoints.py` | Recorded video → vehicle endpoints and per-frame trajectories |
| `export_feed_tracks.py` | The same for every recorded clip of one camera, into one folder; resumable (`… buoi-hht --out data\buoi-hht-tracks`) |
| `local_feed.py` | Reads the MJPEG feed served by `feed/video_adapter.py` |
| `vehicle_tracking.py`, `tracktrack.yaml` | TrackTrack wrapper and its settings |
| `traffic_detections.py` | Motorcycle-rider filter and `person` → `pedestrian` labels |
| `tracking_visualization.py` | Boxes, IDs, trails and the diagnostics panel |
| `vehicle_endpoints.py` | Endpoint/trajectory collection and optional calibration |
| `models/` | RF-DETR checkpoints (weights are not in git) |
| `calibrations/` | Camera calibrations for the trajectory export |

## Live tracking

From the project root, start a feed in one terminal:

```powershell
.\feed\run.ps1 --ihanoi 'yenphu-hangthan'
.\feed\run.ps1 --source 'https://www.youtube.com/watch?v=VIDEO_ID'
```

Then start tracking in another:

```powershell
.\perception\run_rfdetr.ps1
```

It reads `http://127.0.0.1:8766/stream.mjpg` (pass a matching `--port` to both
launchers to change it) and opens an OpenCV window. Press **Q**, **Escape**, or
close the window to stop. CUDA is used when available, in FP32 (GTX 1650).
While the adapter is still resolving a source, the preview waits for frames.

```powershell
.\perception\run_rfdetr.ps1 --headless --max-frames 30           # short test
.\perception\run_rfdetr.ps1 --model 'D:\models\checkpoint_best_total.pth' --conf 0.25
.\perception\run_rfdetr.ps1 --output D:\videos\tracked.mp4       # annotated MP4
.\perception\run_rfdetr.ps1 --trail-length 60 --track-buffer 30
.\perception\run_rfdetr.ps1 --no-tracking                        # detections only
.\perception\run_rfdetr.ps1 --help
```

Each run saves `preview.jpg`, `latest.jpg` and `summary.json` under
`runs/rfdetr/`.

## Export endpoints and trajectories from a recording

```powershell
.\.venv\Scripts\python.exe perception\export_vehicle_endpoints.py data\FEED-DATE\seg-0001-HHMMSS.mp4 `
    --feed auco --output out\endpoints.json --trajectories out\trajectories.json `
    --calibration calibration.json --annotated out\check.mp4
```

Every frame of the video is processed (nothing is skipped as in the live
preview). Only vehicle classes (car, bus, van, motorcycle) are tracked. The
endpoint file stores each vehicle's first and last observed ground point with
video timestamps and a termination reason. `--trajectories` also writes every
real per-frame observation (frame, time, normalized box, bottom-centre ground
point, confidence, and Unity `world_position` when `--calibration` is given),
with the same track IDs and `--min-hits` filtering. Frames where the tracker only
predicted a position (LOST) are left out, so gaps show up as missing frame
numbers. iHanoi recordings repeat the last camera frame when the feed stalls
(see `repeated_output_frames` in the clip's JSON), so derive speeds over time
windows or after smoothing, not from consecutive frames.

## Read the tracking visualization

Each observed object has a persistent `#ID`, a consistent color, and a trail
of its recent observed box bottom-center positions. Watch a moving object over
several frames: its ID should remain the same and its trail should grow behind it.
The diagnostics panel shows observed tracks, temporarily lost tracks, newly
created IDs, recovered IDs, and the number of IDs created during this run.

| State | Meaning |
| --- | --- |
| `NEW` | A new candidate identity has appeared. |
| `TRACKED` | The tracker associated this object with an existing identity. |
| `RECOVERED` | A previously missing identity was associated again. |
| `LOST` | No current detection matched this identity; a dimmed dashed box shows its last observed position. |

Lost boxes and trails are remembered observations, not predictions. An
unassigned detection is shown in gray without an ID. An object that repeatedly
receives new IDs, or two objects that exchange IDs, indicates an association
problem worth inspecting in the recorded video.

ReID appearance matching and camera motion compensation are off (fixed
cameras). `--trail-length` defaults to 40 positions; `--track-buffer` to 30
processed frames, which are not necessarily consecutive source frames when
inference falls behind. `--conf` sets the display/count threshold; the tracker
also receives lower-confidence candidates, so a track can stay visible below it.
Inference and tracking times are reported separately, and the panel shows how
many older feed frames the reader skipped.

The checkpoint's `person` class covers both pedestrians and riders. Person
boxes sitting on a confident motorcycle detection are dropped before tracking,
display and counting; the remaining ones are labeled `pedestrian`. This
heuristic needs the motorcycle to be detected, so missed bikes can leave riders
labeled as pedestrians. RF-DETR's background/no-object predictions are also
removed. Traffic lights are detected as objects; their colour is not classified.

Detection counts are repeated per-frame observations, and created IDs are not a
ground-truth count of unique objects. These checks do not measure tracking
accuracy (IDF1/HOTA); that needs annotated evaluation data.

## Tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s perception/tests -p 'test_*.py'
```
