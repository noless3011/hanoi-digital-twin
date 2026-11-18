"""Publish YouTube, iHanoi, video sources, or digital-twin frames on a loopback MJPEG port."""

from __future__ import annotations

import argparse
import errno
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import math
import os
from pathlib import Path
from queue import Empty, Full, Queue
import re
import select
import shutil
import socket
import sys
from threading import Condition, Event, Thread
import time
from urllib.parse import parse_qs, urlparse
from uuid import uuid4

# Failed FFmpeg opens can otherwise print temporary signed media URLs.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = "https://www.youtube.com/watch?v=1EiC9bvVGnk"
DEFAULT_IHANOI_FEEDS = Path(__file__).resolve().with_name("ihanoi.json")
DEFAULT_IHANOI_CA = Path(__file__).resolve().parent / "certs" / "vtscloud-ca.pem"
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_IMAGE_WIDTH = 3840
MAX_IMAGE_HEIGHT = 2160
TERMINAL_STATES = {"finished", "stopped", "error"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sources = p.add_mutually_exclusive_group()
    sources.add_argument("--source", default=DEFAULT_SOURCE,
                         help="YouTube URL, video path, stream URL, camera index, or twin")
    sources.add_argument("--ihanoi", metavar="NAME", help="Camera name from the iHanoi JSON URL map")
    p.add_argument("--ihanoi-feeds", type=Path, default=DEFAULT_IHANOI_FEEDS,
                   help="iHanoi name-to-URL JSON file (default: feed/ihanoi.json)")
    p.add_argument("--ihanoi-ca-file", type=Path, default=DEFAULT_IHANOI_CA,
                   help="Additional trusted CA PEM for iHanoi TLS")
    p.add_argument("--host", default="127.0.0.1", help="Loopback address only")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--max-height", type=int, default=720, help="Preferred maximum YouTube/video height")
    p.add_argument("--buffer-seconds", type=float, default=5,
                   help="Live/HLS reserve; 0 disables prebuffering")
    p.add_argument("--fps", type=float, help="Playback/advertised FPS; defaults to source FPS, 25 for iHanoi, or 15 for twin")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality, from 1 to 100")
    p.add_argument("--loop", action="store_true", help="Replay finite videos when they end")
    p.add_argument("--cookies", type=Path, help="Explicit Netscape cookies file for YouTube")
    p.add_argument("--js-runtime", type=Path, help="Path to Node.js for yt-dlp YouTube support")
    return p


def safe_message(value: object) -> str:
    message = re.sub(r"(?:https?|wss?|rtsps?|rtmps?|srt|udp|tcp)://[^\s\"'<>]+", "[URL]", str(value), flags=re.I)
    return re.sub(r"\b(?:cookies?|authorization)\s*[:=]\s*[^\r\n]+", "[REDACTED]", message, flags=re.I)


def resolve_ihanoi_feed(name: str, path: Path) -> str:
    """Read the user's mapping without persisting or logging its session URLs."""
    if not isinstance(name, str) or not name.strip():
        raise ValueError("--ihanoi requires a nonempty feed name.")
    path = Path(path)
    try:
        feeds = json.loads(path.read_text(encoding="utf-8-sig"))
    except FileNotFoundError:
        raise FileNotFoundError(f"iHanoi feed map not found: {path}. Create a JSON object mapping feed names to URLs.") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ValueError(f"iHanoi feed map is not valid UTF-8 JSON: {path}.") from None
    if not isinstance(feeds, dict) or any(not isinstance(key, str) or not key.strip()
                                          or not isinstance(value, str) or not value.strip()
                                          for key, value in feeds.items()):
        raise ValueError("The iHanoi feed map must be a JSON object with nonempty feed names and URL strings.")
    if name not in feeds:
        raise ValueError(f"iHanoi feed {name!r} is not defined in {path}. Add its name and captured URL to the JSON map.")
    try:
        from ihanoi_source import websocket_url
        return websocket_url(feeds[name])
    except ValueError:
        raise ValueError(f"iHanoi feed {name!r} has an invalid URL. Use the captured HTTP(S) or WS(S) URL.") from None


class ExtractorLogger:
    def debug(self, message: str) -> None:
        pass

    def warning(self, message: str) -> None:
        print(f"YouTube: {safe_message(message)}", file=sys.stderr)

    def error(self, message: str) -> None:
        pass


def youtube_url(source: str) -> str | None:
    parsed = urlparse(source)
    host = (parsed.hostname or "").lower()
    if host not in {"youtube.com", "www.youtube.com", "m.youtube.com", "youtu.be", "www.youtu.be"}:
        return None
    video_id = (parsed.path.strip("/").split("/")[-1] if host.endswith("youtu.be")
                else parse_qs(parsed.query).get("v", [None])[0])
    if not video_id and parsed.path.startswith(("/live/", "/shorts/", "/embed/")):
        video_id = parsed.path.rstrip("/").split("/")[-1]
    if not video_id or not re.fullmatch(r"[A-Za-z0-9_-]+", video_id):
        raise ValueError("Supply one YouTube video URL, including its video ID.")
    return f"https://www.youtube.com/watch?v={video_id}"


def node_path(explicit: Path | None) -> str | None:
    if explicit:
        if not explicit.is_file():
            raise FileNotFoundError(f"JavaScript runtime does not exist: {explicit}")
        return str(explicit.resolve())
    candidates = [PROJECT / ".venv" / "tools" / "node" / "node.exe",
                  PROJECT / ".venv" / "Scripts" / "node.exe"]
    candidates.extend((PROJECT / ".venv" / "tools").glob("node-*/node.exe"))
    return next((str(path) for path in candidates if path.is_file()), None) or shutil.which("node")


def resolve_source(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Resolve URLs only in memory; never download full videos or persist tokens."""
    url = youtube_url(args.source)
    if not url:
        source = int(args.source) if args.source.isdecimal() else args.source
        parsed = urlparse(args.source)
        finite_suffixes = {".mp4", ".m4v", ".webm", ".mkv", ".mov", ".avi", ".mpeg", ".mpg"}
        live = isinstance(source, int) or (bool(parsed.scheme) and len(parsed.scheme) > 1
                                         and Path(parsed.path).suffix.lower() not in finite_suffixes)
        public_source = re.sub(r"\?.*$", "", args.source) if "://" in args.source else args.source
        return [{"url": source}], {"source": public_source, "is_live": live}

    import yt_dlp

    options = {
        "noplaylist": True, "quiet": True, "no_warnings": False,
        "logger": ExtractorLogger(), "socket_timeout": 20, "retries": 2, "cachedir": False,
        "format": f"bestvideo[height<={args.max_height}]/best[height<={args.max_height}]/bestvideo/best",
    }
    runtime = node_path(args.js_runtime)
    if runtime:
        options["js_runtimes"] = {"node": {"path": runtime}}
    if args.cookies:
        if not args.cookies.is_file():
            raise FileNotFoundError(f"Cookie file does not exist: {args.cookies}")
        options["cookiefile"] = str(args.cookies.resolve())
    print("Resolving YouTube video (no full-video download)...", flush=True)
    try:
        with yt_dlp.YoutubeDL(options) as downloader:
            info = downloader.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            f"YouTube extraction failed: {safe_message(exc)}. "
            'Update yt-dlp with .venv\\Scripts\\python -m pip install -U "yt-dlp[default]". '
            "Check Node.js, or use a local video or explicit --cookies PATH."
        ) from None
    if not info:
        raise RuntimeError("YouTube did not return video information.")
    formats = [fmt for fmt in info.get("formats", [])
               if fmt.get("url") and fmt.get("vcodec") not in {None, "none"}
               and not fmt.get("has_drm")
               and fmt.get("protocol") in {"http", "https", "m3u8", "m3u8_native"}]

    def rank(fmt: dict) -> tuple:
        height = fmt.get("height") or 0
        return (height <= args.max_height,
                str(fmt.get("vcodec", "")).startswith(("avc1", "h264")),
                fmt.get("protocol") in {"http", "https"},
                height if height <= args.max_height else -height,
                fmt.get("fps") or 0, fmt.get("tbr") or 0)

    formats.sort(key=rank, reverse=True)
    if not formats and info.get("url"):
        formats = [info]
    candidates, seen = [], set()
    for fmt in formats:
        if fmt["url"] not in seen:
            candidates.append(fmt)
            seen.add(fmt["url"])
        if len(candidates) == 3:
            break
    if not candidates:
        raise RuntimeError("YouTube returned no playable video formats. Update yt-dlp or use a local video.")
    return candidates, {"source": url, "video_id": info.get("id"),
                        "title": info.get("title"), "is_live": bool(info.get("is_live"))}


def open_capture(cv2, candidates: list[dict]):
    for fmt in candidates:
        source = fmt["url"]
        capture = (cv2.VideoCapture(source) if isinstance(source, int)
                   else cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                       cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 20_000,
                       cv2.CAP_PROP_READ_TIMEOUT_MSEC, 20_000,
                   ]))
        try:
            if capture.isOpened():
                ok, first_frame = capture.read()
                if ok and first_frame is not None:
                    return capture, first_frame, fmt
        except Exception:
            capture.release()
            raise
        capture.release()
    raise RuntimeError("Could not read the video feed. Re-run to resolve a fresh URL or use --source PATH_TO_VIDEO.")


class LatestFrameReader:
    """Bound an HLS reserve by memory and retain one latest paced output frame."""

    def __init__(self, capture, fps: float, buffer_seconds: float, first_frame):
        self.capture = capture
        self.interval = 1 / fps
        memory_capacity = max(1, (512 * 1024 * 1024) // max(1, first_frame.nbytes))
        self.capacity = max(1, min(memory_capacity, math.ceil(min(buffer_seconds + 1, memory_capacity / fps) * fps)))
        self.reserve = min(self.capacity, math.ceil(min(buffer_seconds, self.capacity / fps) * fps))
        self.buffer = Queue(maxsize=self.capacity)
        self.buffer.put_nowait(first_frame)
        self.frames = Queue(maxsize=1)
        self.stopped = Event()
        self.producer_done = Event()
        self.playback_done = Event()
        self.error = None
        self.dropped = 0
        self.thread = Thread(target=self._capture_frames, daemon=True)
        self.playback_thread = Thread(target=self._play_frames, daemon=True)
        self.thread.start()
        self.playback_thread.start()

    def _capture_frames(self) -> None:
        try:
            while not self.stopped.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    break
                while not self.stopped.is_set():
                    try:
                        self.buffer.put(frame, timeout=0.05)
                        break
                    except Full:
                        pass
        except Exception as exc:
            self.error = safe_message(exc)
        finally:
            self.producer_done.set()
            # Only the capture thread releases its handle, never during a read.
            self.capture.release()

    def _prefill(self) -> None:
        while self.buffer.qsize() < self.reserve and not self.producer_done.is_set():
            if self.stopped.wait(0.025):
                break

    def _play_frames(self) -> None:
        try:
            self._prefill()
            deadline = time.monotonic()
            while not self.stopped.is_set():
                try:
                    frame = self.buffer.get(timeout=0.05)
                except Empty:
                    if self.producer_done.is_set():
                        break
                    self._prefill()
                    deadline = time.monotonic()
                    continue
                try:
                    self.frames.put_nowait(frame)
                except Full:
                    try:
                        self.frames.get_nowait()
                        self.dropped += 1
                    except Empty:
                        pass
                    self.frames.put_nowait(frame)
                deadline = max(deadline + self.interval, time.monotonic())
                if self.stopped.wait(max(0, deadline - time.monotonic())):
                    break
        finally:
            self.playback_done.set()

    def read(self, timeout: float = 0.1):
        try:
            return True, self.frames.get(timeout=timeout)
        except Empty:
            return (False if self.playback_done.is_set() else None), None

    def close(self) -> None:
        self.stopped.set()
        self.playback_thread.join(timeout=1)
        self.thread.join(timeout=1)


class FrameStore:
    """One JPEG shared by all clients, with atomic twin upload/stop sessions."""

    def __init__(self, mode: str, fps: float = 15):
        self.condition = Condition()
        self.mode = mode
        self.fps = fps
        self.state = "waiting"
        self.frames = 0
        self.error = None
        self.jpeg = None
        self.sequence = 0
        self.generation = 0
        self.session = uuid4().hex
        self.closed = False

    def status(self) -> dict:
        with self.condition:
            result = {"mode": self.mode, "state": self.state, "fps": self.fps,
                      "frames": self.frames, "error": self.error, "stream": "/stream.mjpg"}
            if self.mode == "twin":
                result["session"] = self.session
            return result

    def publish(self, jpeg: bytes, session: str | None = None) -> bool:
        with self.condition:
            if self.closed or (self.mode == "twin" and session != self.session):
                return False
            if self.state in TERMINAL_STATES:
                self.frames = 0
            self.jpeg = jpeg
            self.sequence += 1
            self.frames += 1
            self.state = "streaming"
            self.error = None
            self.condition.notify_all()
            return True

    def stop(self, session: str | None) -> bool:
        with self.condition:
            if self.closed or session != self.session:
                return False
            self.session = uuid4().hex
            self.generation += 1
            self.jpeg = None
            self.state = "stopped"
            self.condition.notify_all()
            return True

    def set_state(self, state: str, error: object = None) -> None:
        with self.condition:
            self.state = state
            self.error = safe_message(error) if error is not None else None
            self.condition.notify_all()

    def close(self) -> None:
        with self.condition:
            self.closed = True
            self.generation += 1
            self.state = "stopped"
            self.condition.notify_all()

    def wait_frame(self, sequence: int, generation: int):
        with self.condition:
            self.condition.wait_for(lambda: self.closed or self.generation != generation
                                    or self.sequence > sequence or self.state in TERMINAL_STATES,
                                    timeout=1)
            if self.closed or self.generation != generation:
                return None, self.sequence
            if self.sequence > sequence and self.jpeg is not None:
                return self.jpeg, self.sequence
            return (None if self.state in TERMINAL_STATES else b""), self.sequence


def loopback_host(value: str) -> str:
    if value.lower() == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(value)
    except ValueError:
        raise ValueError("--host must be localhost or a numeric loopback address.") from None
    if not address.is_loopback:
        raise ValueError("The adapter can listen only on a loopback address.")
    return str(address)


def local_origin(value: str) -> bool:
    try:
        parsed = urlparse(value)
        # Accessing .port also rejects malformed/oversized port numbers.
        parsed.port
        return (parsed.scheme in {"http", "https"}
                and parsed.hostname in {"localhost", "127.0.0.1", "::1"}
                and parsed.username is None and parsed.password is None
                and not parsed.path and not parsed.query and not parsed.fragment)
    except ValueError:
        return False


def jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Inspect SOF dimensions before OpenCV can allocate an oversized image."""
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError("The request body must contain one complete JPEG image.")
    position = 2
    while position < len(payload):
        if payload[position] != 0xFF:
            raise ValueError("Invalid JPEG image header.")
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            break
        marker = payload[position]
        position += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(payload):
            break
        length = int.from_bytes(payload[position:position + 2], "big")
        if length < 2 or position + length > len(payload):
            raise ValueError("Invalid JPEG segment length.")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length < 8:
                raise ValueError("Invalid JPEG image dimensions.")
            height = int.from_bytes(payload[position + 3:position + 5], "big")
            width = int.from_bytes(payload[position + 5:position + 7], "big")
            if not (0 < width <= MAX_IMAGE_WIDTH and 0 < height <= MAX_IMAGE_HEIGHT):
                raise ValueError(f"JPEG dimensions cannot exceed {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}.")
            return width, height
        if marker == 0xDA:
            break
        position += length
    raise ValueError("JPEG image dimensions were not found.")


def validate_jpeg(payload: bytes) -> None:
    width, height = jpeg_dimensions(payload)
    import cv2
    import numpy as np

    decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (height, width):
        raise ValueError("The request body is not a decodable JPEG image.")


class AdapterServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False
    # Windows SO_REUSEADDR permits a second adapter to bind the live port,
    # making the source received by clients unpredictable. POSIX reuse still
    # permits quick restarts while refusing a second active listener.
    allow_reuse_address = os.name != "nt"

    def __init__(self, address, store: FrameStore):
        self.store = store
        super().__init__(address, AdapterHandler)

    def server_bind(self):
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


class IPv6AdapterServer(AdapterServer):
    address_family = socket.AF_INET6


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "VideoAdapter/1"

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, format, *args):
        # No incoming URLs, headers, tokens, or upload bodies are logged.
        pass

    def _origin_allowed(self) -> bool:
        origin = self.headers.get("Origin")
        if origin is not None and not local_origin(origin):
            self._json(403, {"error": "Only localhost browser origins are allowed."}, cors=False)
            return False
        return True

    def _cors(self) -> None:
        origin = self.headers.get("Origin")
        if origin and local_origin(origin):
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
            self.send_header("Access-Control-Expose-Headers", "X-Stream-Fps")

    def _json(self, code: int, value: dict, cors: bool = True) -> None:
        payload = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.close_connection = True
        if cors:
            self._cors()
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(payload)

    def do_OPTIONS(self) -> None:
        if not self._origin_allowed():
            return
        if urlparse(self.path).path not in {"/frame", "/stop", "/status", "/stream.mjpg"}:
            self._json(404, {"error": "Unknown endpoint."})
            return
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, HEAD, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Stream-Session")
        if self.headers.get("Access-Control-Request-Private-Network") == "true":
            self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Content-Length", "0")
        self.send_header("Connection", "close")
        self.close_connection = True
        self.end_headers()

    def do_HEAD(self) -> None:
        self.do_GET()

    def do_GET(self) -> None:
        if not self._origin_allowed():
            return
        path = urlparse(self.path).path
        if path == "/status":
            self._json(200, self.server.store.status())
        elif path == "/stream.mjpg":
            self._stream()
        else:
            self._json(404, {"error": "Use /status or /stream.mjpg."})

    def _stream(self) -> None:
        store = self.server.store
        with store.condition:
            generation, fps = store.generation, store.fps
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Stream-Fps", f"{fps:g}")
        self.send_header("Connection", "close")
        self._cors()
        self.close_connection = True
        self.end_headers()
        if self.command == "HEAD":
            return
        sequence = 0
        try:
            while True:
                jpeg, sequence = store.wait_frame(sequence, generation)
                if jpeg is None:
                    break
                if not jpeg:
                    # Detect disconnected readers even when no producer is active.
                    readable, _, _ = select.select([self.connection], [], [], 0)
                    if readable and not self.connection.recv(1, socket.MSG_PEEK):
                        break
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                                 + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
            self.wfile.write(b"--frame--\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionError, TimeoutError, OSError):
            pass

    def do_POST(self) -> None:
        if not self._origin_allowed():
            return
        store = self.server.store
        path = urlparse(self.path).path
        if path not in {"/frame", "/stop"}:
            self._json(404, {"error": "Unknown endpoint."})
            return
        if store.mode != "twin":
            self._json(409, {"error": "Frame uploads and stop are available only with --source twin."})
            return
        session = self.headers.get("X-Stream-Session")
        with store.condition:
            valid_session = session == store.session and not store.closed
        if not valid_session:
            self._json(409, {"error": "Stream session changed; read /status before starting again."})
            return
        if path == "/stop":
            if self.headers.get("Transfer-Encoding") or self.headers.get("Content-Length", "0") != "0":
                self._json(400, {"error": "/stop takes no request body."})
                return
            if not store.stop(session):
                self._json(409, {"error": "Stream session changed."})
                return
            self._json(200, store.status())
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() != "image/jpeg":
            self._json(415, {"error": "Use Content-Type: image/jpeg."})
            return
        if self.headers.get("Transfer-Encoding"):
            self._json(400, {"error": "Send a JPEG with Content-Length; chunked uploads are unsupported."})
            return
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self._json(411, {"error": "A valid Content-Length is required."})
            return
        if not 0 < length <= MAX_JPEG_BYTES:
            self._json(413, {"error": "JPEG body must be between 1 byte and 8 MiB."})
            return
        try:
            payload = self.rfile.read(length)
            if len(payload) != length:
                raise ValueError("Incomplete JPEG request body.")
            validate_jpeg(payload)
        except ValueError as exc:
            self._json(400, {"error": safe_message(exc)})
            return
        except (TimeoutError, OSError):
            self._json(408, {"error": "Timed out reading the JPEG request body."})
            return
        except Exception as exc:
            self._json(503, {"error": f"JPEG decoder unavailable: {safe_message(exc)}"})
            return
        if not store.publish(payload, session):
            self._json(409, {"error": "Stream session changed during upload."})
            return
        self._json(200, store.status())


def create_server(host: str, port: int, store: FrameStore) -> AdapterServer:
    host = loopback_host(host)
    server_type = IPv6AdapterServer if ":" in host else AdapterServer
    try:
        return server_type((host, port), store)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE and getattr(exc, "winerror", None) != 10048:
            raise
        raise OSError(exc.errno, (
            f"Local feed port {port} on {host} is already in use. "
            "Stop the existing feed terminal with Ctrl+C before starting a new source, "
            "or choose another --port and use the same --port for the detector."
        )) from exc


def source_fps(capture, cv2, chosen: dict, override: float | None) -> float:
    if override is not None:
        return override
    candidates = [capture.get(cv2.CAP_PROP_FPS), chosen.get("fps"), 30]
    for value in candidates:
        try:
            fps = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fps) and 0 < fps <= 240:
            return fps
    return 30


def publish_video_frame(cv2, frame, args: argparse.Namespace, store: FrameStore) -> None:
    height, width = frame.shape[:2]
    scale = min(1, args.max_height / height, MAX_IMAGE_HEIGHT / height, MAX_IMAGE_WIDTH / width)
    if scale < 1:
        frame = cv2.resize(frame, (max(1, round(width * scale)), max(1, round(height * scale))))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
    if not ok:
        raise RuntimeError("Could not encode a video frame as JPEG.")
    store.publish(jpeg.tobytes())


def video_worker(args: argparse.Namespace, store: FrameStore, stopped: Event) -> None:
    capture, reader = None, None
    reconnections = 0
    try:
        import cv2

        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        while not stopped.is_set():
            candidates, metadata = resolve_source(args)
            capture, frame, chosen = open_capture(cv2, candidates)
            fps = source_fps(capture, cv2, chosen, args.fps)
            with store.condition:
                store.fps = fps
            live = metadata.get("is_live", False)
            if live:
                reader = LatestFrameReader(capture, fps, args.buffer_seconds, frame)
                capture = None  # The reader owns and releases this capture.
                last_received = time.monotonic()
                while not stopped.is_set():
                    ok, frame = reader.read()
                    if ok:
                        publish_video_frame(cv2, frame, args, store)
                        last_received = time.monotonic()
                    elif ok is False or time.monotonic() - last_received >= max(22, args.buffer_seconds + 22):
                        break
                reader.close()
                reader = None
                if stopped.is_set():
                    return
                if reconnections >= 3:
                    raise RuntimeError("The live feed stopped delivering frames after three reconnections.")
                reconnections += 1
                store.set_state("waiting")
                print(f"Live feed interrupted; reconnecting ({reconnections}/3)...", flush=True)
                if stopped.wait(1):
                    return
                continue
            deadline = time.monotonic()
            while not stopped.is_set():
                publish_video_frame(cv2, frame, args, store)
                deadline = max(deadline + 1 / fps, time.monotonic())
                if stopped.wait(max(0, deadline - time.monotonic())):
                    return
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
            capture.release()
            capture = None
            if not args.loop:
                store.set_state("finished")
                return
    except Exception as exc:
        if not stopped.is_set():
            store.set_state("error", exc)
            print(f"Video adapter: {safe_message(exc)}", file=sys.stderr, flush=True)
    finally:
        if reader is not None:
            reader.close()
        elif capture is not None:
            capture.release()


def ihanoi_worker(args: argparse.Namespace, store: FrameStore, stopped: Event, *, url: str | None = None) -> None:
    try:
        import cv2
        from ihanoi_source import stream_frames

        if hasattr(cv2, "setLogLevel"):
            cv2.setLogLevel(0)
        url = url or resolve_ihanoi_feed(args.ihanoi, args.ihanoi_feeds)
        with store.condition:
            store.fps = args.fps or 25
        stream_frames(url, stopped,
                      lambda frame: publish_video_frame(cv2, frame.to_ndarray(format="bgr24"), args, store),
                      ca_file=args.ihanoi_ca_file)
        if not stopped.is_set():
            store.set_state("finished")
    except Exception as exc:
        if not stopped.is_set():
            store.set_state("error", exc)
            print(f"iHanoi adapter: {safe_message(exc)}", file=sys.stderr, flush=True)
    finally:
        if stopped.is_set():
            store.set_state("stopped")


def run(args: argparse.Namespace) -> int:
    mode = "ihanoi" if args.ihanoi is not None else "twin" if args.source == "twin" else "video"
    ihanoi_url = resolve_ihanoi_feed(args.ihanoi, args.ihanoi_feeds) if mode == "ihanoi" else None
    store = FrameStore(mode, args.fps or {"twin": 15, "ihanoi": 25, "video": 30}[mode])
    server = create_server(args.host, args.port, store)
    stopped = Event()
    http_thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.2}, daemon=True)
    http_thread.start()
    worker = None
    host = f"[{args.host}]" if ":" in args.host else args.host
    print(f"Adapter ({mode}): http://{host}:{server.server_port}/stream.mjpg", flush=True)
    if mode == "twin":
        print("Ready for the digital twin's Stream to local port button.", flush=True)
    elif mode == "ihanoi":
        worker = Thread(target=ihanoi_worker, args=(args, store, stopped), kwargs={"url": ihanoi_url}, daemon=True)
        worker.start()
    else:
        worker = Thread(target=video_worker, args=(args, store, stopped), daemon=True)
        worker.start()
    try:
        while not stopped.wait(0.5):
            pass
    except KeyboardInterrupt:
        print("Stopping video adapter...", flush=True)
    finally:
        stopped.set()
        store.close()
        server.shutdown()
        server.server_close()
        http_thread.join(timeout=1)
        if worker:
            worker.join(timeout=3)
    return 0


def main() -> int:
    p = parser()
    args = p.parse_args()
    try:
        args.host = loopback_host(args.host)
    except ValueError as exc:
        p.error(str(exc))
    if not 1 <= args.port <= 65535:
        p.error("--port must be between 1 and 65535")
    if not 1 <= args.max_height <= MAX_IMAGE_HEIGHT:
        p.error(f"--max-height must be between 1 and {MAX_IMAGE_HEIGHT}")
    if not math.isfinite(args.buffer_seconds) or args.buffer_seconds < 0:
        p.error("--buffer-seconds must be finite and nonnegative")
    if args.fps is not None and (not math.isfinite(args.fps) or not 0 < args.fps <= 240):
        p.error("--fps must be finite, positive, and at most 240")
    if not 1 <= args.quality <= 100:
        p.error("--quality must be between 1 and 100")
    if not args.source.strip():
        p.error("--source cannot be empty")
    if args.ihanoi is not None and not args.ihanoi.strip():
        p.error("--ihanoi cannot be empty")
    try:
        return run(args)
    except Exception as exc:
        print(f"Video adapter: {safe_message(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
