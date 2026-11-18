# Video feed adapter

`video_adapter.py` takes a YouTube video or stream, an iHanoi traffic camera, a video file, a camera, or
frames from the twin's browser and publishes a local HTTP MJPEG feed. It does
not load an inference or behaviour model. Dependencies are listed in
`requirements.txt`; the root `setup.ps1` installs them into the shared root
`.venv`.

From the project root:

```powershell
.\feed\run.ps1 --source 'https://www.youtube.com/watch?v=VIDEO_ID'
.\feed\run.ps1 --source 'D:\videos\traffic.mp4' --loop
.\feed\run.ps1 --source 0
```

Run one source at a time on each port. The default feed is
`http://127.0.0.1:8766/stream.mjpg`; use `--port PORT` to change it. Run
`.\feed\run.ps1 --help` for resolution, buffering, playback, and other options.
User-supplied relative file paths resolve from the terminal's current directory.
Ctrl+C stops the adapter.

Record an unannotated, fixed-length camera clip for offline tracking:

```powershell
.\.venv\Scripts\python.exe feed\record_video.py --ihanoi 'yenphu-hangthan' --duration 15 --output runs\yenphu-hangthan-test\source.mp4
```

Recording begins at the first decoded camera frame and keeps the original
resolution. The default output rate is 25 FPS; `--fps` can change it. The
recorder samples the latest frame against elapsed time, repeating frames when
necessary so network timing does not change playback speed. It writes a JSON
sidecar containing timing, dimensions, frame counts, and the number of repeated
frames, plus a JPEG preview with the first, middle, and last frames. These
artifacts contain the feed name and never its captured session URL. A camera
that stops delivering frames for five seconds fails the recording instead of
producing a frozen clip. Existing artifacts are protected unless `--overwrite`
is supplied. Startup and stale-frame limits can be adjusted with
`--startup-timeout` and `--stale-timeout`.

### Record for hours

```powershell
.\.venv\Scripts\python.exe feed\record_session.py --ihanoi 'yenphu-hangthan' --until 18:00
```

`record_session.py` records back-to-back clips (`--segment`, default 120 s)
until `--until` (local HH:MM) or for `--hours`. Each clip opens a new WebSocket
connection and closes it when the clip ends; a stalled camera fails that clip
and the next one reconnects, instead of resending `mobile:` on the old socket.
Failed clips are retried with backoff (5 s doubling to `--max-backoff`), and
`feed/ihanoi.json` is re-read before every connection, so a fresh session URL
can be pasted in while it runs. Clips and `session.json` (segments, failures,
times) go to `data/<feed>-<date>-session/` or `--output-dir`; rerunning
with the same folder resumes numbering. Recording stops below `--min-free-gb`
(default 3); 1080p clips are about 75 MB per minute. Windows is kept awake
while it runs. Ctrl+C discards only the clip in progress.

### Record a dataset with gaps

```powershell
.\.venv\Scripts\python.exe feed\record_dataset.py
.\.venv\Scripts\python.exe feed\record_dataset.py --ihanoi auco --until 18:00
```

`record_dataset.py` asks for the feed (from `feed/ihanoi.json`) and the end
time if they are not given, then records 3-minute clips with a full disconnect
and a 5-minute gap between them (`--segment`, `--reconnect-delay` in seconds).
Output goes to `data/<feed>-<date>/` (or `--output-dir`) in the same format as
`record_session.py`.

For an iHanoi traffic camera, edit `feed/ihanoi.json` with a JSON object mapping
your chosen names to the URLs captured from the app:

```json
{
  "feed name": "https://rec01ihanoi.vtscloud.vn/evup/PASTE_CURRENT_SESSION_PATH",
  "another camera": "wss://rec01ihanoi.vtscloud.vn/evup/ANOTHER_CURRENT_SESSION_PATH"
}
```

Then run:

```powershell
.\feed\run.ps1 --ihanoi 'feed name'
```

Names match JSON keys exactly, including spaces and letter case. The default
JSON path is next to the adapter, so it works from other working directories.
Use `--ihanoi-feeds PATH` to select another map. The adapter sends the required
`mobile:` text command, decodes the observed HEVC/H.265 stream, and publishes
JPEG frames at the same `http://127.0.0.1:8766/stream.mjpg` endpoint. `/status`
reports `mode: "ihanoi"` and the usual waiting/streaming/finished/error states.
The existing YOLO and RF-DETR launchers read it without additional options.

`--port`, `--max-height`, and `--quality` work for iHanoi too. Frames are
published as they arrive; `--fps` changes the advertised rate (default 25).
The HLS buffer and file-loop options apply to other sources. `--ihanoi` and
`--source` select different source modes and cannot be combined.

The file is initially empty for you to populate. Missing names, invalid JSON,
and invalid URLs produce a clear error before the local server starts. Session
URLs are held in memory and omitted from status/logs. When frames stop for
20 seconds, the adapter resends `mobile:` on the existing connection and
starts a fresh decoder. If the server closes a stream that delivered frames,
the adapter reconnects to the same URL after one second and sends `mobile:`
again. Ctrl+C cancels reception and reconnect waits. If restarting produces
no frames for another 20 seconds, or a new connection fails, the adapter
reports an error. For an expired URL, capture its new URL in iHanoi, update
the JSON, and restart the adapter. TLS verification
stays enabled, with a bundled public GlobalSign CA chain for VTSCloud. A custom
additional CA bundle can be supplied with `--ihanoi-ca-file PATH`.

For an existing setup, install the new decoding dependency into the shared
environment once:

```powershell
.\.venv\Scripts\python.exe -m pip install -r feed\requirements.txt
```

To test the selected source with the seven-class RF-DETR model, start
`.\perception\run_rfdetr.ps1` from the project root in another terminal. It reads
the adapter's local feed and loads `perception/models/rfdetr_nano_traffic7_bundle/`
by default. Stop the current adapter before selecting another
source on the same port; a second adapter reports a port conflict.

For browser-rendered video:

```powershell
.\feed\run.ps1 --source twin
.\twin\run.ps1
```

In the twin, click **Stream to local port**, matching the adapter's port.
Keep the page visible while streaming. Click **Stop streaming** to end the
feed. Start `.\perception\run_rfdetr.ps1` in another terminal to run tracking.

For moving twin actors, first run `.\behaviour.bat 8765`, then connect the web UI
to `ws://127.0.0.1:8765` and click **Start**. Behaviour observations and controls on port 8765
and video frames on port 8766 have separate roles.

See [the project README](../README.md) for source limitations, recording, and
the local video endpoint contract. From the project root, run adapter tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s feed/tests -p 'test_*.py'
```
