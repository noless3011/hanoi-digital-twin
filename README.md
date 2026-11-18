# Digital twin, traffic simulation, and video inference

The traffic simulation is a light Python engine ([`twin/`](twin), design in
[docs/twin.md](docs/twin.md)): deterministic for a seed, no rendering. Driver
behaviour is a separate model the engine asks each step: the choice model, in which
each driver compares a few actions and takes the cheapest (a constant-speed model is
kept as the baseline). The engine does not prevent overlaps; it scores every
contact between vehicles. Unity only plays trajectories back, and can render them
to a video from the feed camera. Recorded videos are tracked by RF-DETR and
exported as per-frame trajectories ([perception/README.md](perception/README.md));
`twin/measure` extracts the demand (arrivals) to replay and scores a simulation
against the video in the camera image; `twin/calibrate` fits the model's
parameters to that score.

Replaced code (the previous interaction model, and the older Unity C# system with its
decision detection, Optuna calibration and C# validation) is kept in git under the tag
`archive/snapshots-20261116`: `git show archive/snapshots-20261116:backup/<path>`.

| Folder | Purpose |
| --- | --- |
| `twin/` | Simulation engine (`engine.py`), driver behaviour models (`behaviour/`), measurement and the image-space score (`measure/`), fitting (`calibrate/`). |
| `simulation/` | Unity scenes, lane authoring (exports the road layout), camera tools, and the trajectory player. |
| `feed/` | Video adapters and recorders for iHanoi cameras, files, YouTube, and browser publishers. |
| `perception/` | RF-DETR detection and tracking, endpoint/trajectory export, model weights, and detector training (`training/`). |
| `runs/` | Generated outputs: trajectories, reports, calibration runs (not in git). |
| `data/` | Recorded camera datasets (`feed/record_dataset.py`, not in git). |
| `docs/` | Design notes (`twin.md`), thesis preparation material (`thesis-prep/`), and report tools (`tools/`). |

## Simulate and play back

```powershell
$L = "simulation\Assets\StreamingAssets\Traffic\buoi-hht.camera-lanes.json"
.\.venv\Scripts\python.exe -m twin.run --layout $L --video runs\buoi-hht\buoi-hht-20261106-seg-0010-092146.trajectories.json `
    --params runs\twin\fit2\fitted.params.json --out simulation\Assets\StreamingAssets\Traffic\simulation.trajectories.json
```

`--params` is optional (the model's defaults are used without it); `python -m
twin.calibrate.search` writes that file. Open `simulation` with Unity **6000.6.4f1**, open a scene in `Assets/Scenes/`
(`buoi-hht`, `caugiay-tranduykien`, `yenphu-hangthan`) and enter Play mode: the
`TrajectoryPlayer` plays `StreamingAssets/Traffic/simulation.trajectories.json`
(or any video trajectories file). Export a scene's road layout with **Export camera
lane layout** on its LaneEditor; `twin` builds routes from that file. Measurement
and the full workflow are in [docs/twin.md](docs/twin.md).

The Main Camera also draws projected vehicle bounding boxes with C# TrackTrack
IDs and trails. No detection model is used in Unity.

## Track a camera, YouTube video, or another source

Run `.\setup.ps1` once to install the shared Python environment for video tools.
Use `setup.ps1 -Cpu` for CPU inference dependencies.

For an iHanoi traffic camera, put your captured URLs in `feed/ihanoi.json` as
`{"feed name": "https://rec01ihanoi.vtscloud.vn/evup/CURRENT_SESSION_PATH"}`.
See [feed/README.md](feed/README.md) for the JSON format, recording, and options.

Start the adapter in one terminal and tracking in another:

```powershell
.\feed\run.ps1 --ihanoi 'feed name'
.\feed\run.ps1 --source 'https://www.youtube.com/watch?v=1EiC9bvVGnk'
```

```powershell
.\perception\run_rfdetr.ps1
```

The adapter resolves and plays the source, then publishes frames at
`http://127.0.0.1:8766/stream.mjpg`. It does not load a model. The tracker reads
that feed, runs the seven-class RF-DETR model (car, bus, van, others, person,
motorcycle, traffic light) with TrackTrack, and opens an annotated window with
persistent IDs, trails and a diagnostics panel. Press **Q**, **Escape**, or close
the window to stop it; Ctrl+C stops the adapter. Run one adapter per port. See
[perception/README.md](perception/README.md) for tracker options.

Other sources and options, from the project root:

```powershell
# A file, optionally repeated, or a local camera.
.\feed\run.ps1 --source 'D:\videos\traffic.mp4' --loop
.\feed\run.ps1 --source 0

# Lower YouTube resolution or increase its playback reserve.
.\feed\run.ps1 --source 'https://www.youtube.com/watch?v=VIDEO_ID' --max-height 480 --buffer-seconds 8

# Select a different video port in both processes.
.\feed\run.ps1 --source 'D:\videos\traffic.mp4' --port 9000
.\perception\run_rfdetr.ps1 --port 9000

.\feed\run.ps1 --help
.\perception\run_rfdetr.ps1 --help
```

User-supplied relative source, output, and other file paths resolve from the
terminal's current directory. If local PowerShell scripts are disabled, run
`.venv\Scripts\python.exe -u feed\video_adapter.py` or
`.venv\Scripts\python.exe -u perception\run_rfdetr.py` directly.

A single YouTube video's page is resolved in memory; playlists and audio are
excluded. Restricted videos can use an explicit `--cookies PATH` file.
`--js-runtime PATH` selects a Node.js executable. Media links are refreshed on
live-source reconnections. Extraction depends on YouTube accepting the request
and the installed yt-dlp version.

Finite videos finish after their last frame unless `--loop` is supplied. The
adapter keeps its HTTP status endpoint available after completion. The runner
receives the end state and exits. Source playback is paced independently of
inference; a slow consumer skips older frames instead of building up latency.

## Local video interface

The adapter binds to loopback. Browser publishing accepts local HTTP/HTTPS
origins; a compatible browser publisher is separate from the Unity runtime.

| Route | Purpose |
| --- | --- |
| `GET /stream.mjpg` | Multipart JPEG feed. Each part has `Content-Length`; the response includes `X-Stream-Fps`. |
| `GET /status` | JSON source mode, state, FPS, frame count, and any source error. |
| `POST /frame` | In twin mode, publish one raw `image/jpeg` frame, limited to 8 MiB. |
| `POST /stop` | In twin mode, stop publishing and close existing video consumers. |

In twin mode, `/status` includes a `session` token. Pass it as
`X-Stream-Session` on both POST routes. Stopping rotates the token, so an old
upload cannot restart a stopped feed. Get a fresh status before restarting.
A browser publisher must handle this token. A GET stream waits for the first
frame and ends when its source finishes, stops, or fails. The adapter bounds
images to 3840 × 2160 pixels; `--fps` selects the advertised rate.

## Artifacts and checks

Each tracking run saves an early preview, a final `latest.jpg`, and a JSON
summary under `runs/rfdetr/`. Full video recording, including tracking
diagnostics, is opt-in through `--output`. Created track IDs can fragment or
switch, so they are not a ground-truth count of unique objects.

From the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s feed/tests -p 'test_*.py'
.\.venv\Scripts\python.exe -m unittest discover -s perception/tests -p 'test_*.py'
.\.venv\Scripts\python.exe -m unittest discover -s twin/tests -t . -p 'test_*.py'
```

Run Unity EditMode tests in its Test Runner.
