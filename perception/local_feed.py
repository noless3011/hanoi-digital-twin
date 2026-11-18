"""Read the local MJPEG video adapter feed (feed/video_adapter.py).

LatestFrameReader drains the stream in a background thread and keeps only the
newest decoded frame, so slow inference skips frames instead of lagging.
"""

from __future__ import annotations

from email.message import Message
import ipaddress
import json
import math
from queue import Empty, Full, Queue
import re
import socket
from threading import Event, Lock, Thread
from urllib.error import HTTPError
from urllib.parse import urlparse, urlunparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

DEFAULT_PORT = 8766
MAX_JPEG_BYTES = 8 * 1024 * 1024
MAX_IMAGE_WIDTH = 3840
MAX_IMAGE_HEIGHT = 2160
TERMINAL_STATES = {"finished", "stopped", "error"}


def local_endpoint(source: str | None, port: int = DEFAULT_PORT) -> str:
    if not 1 <= port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    endpoint = source or f"http://127.0.0.1:{port}/stream.mjpg"
    parsed = urlparse(endpoint)
    try:
        loopback = parsed.hostname == "localhost" or ipaddress.ip_address(parsed.hostname or "").is_loopback
        endpoint_port = parsed.port
    except ValueError:
        loopback, endpoint_port = False, None
    if (parsed.scheme not in {"http", "https"} or not loopback or parsed.username
            or parsed.password or parsed.fragment or endpoint_port == 0):
        raise ValueError("--source must be a loopback HTTP(S) MJPEG endpoint. Route YouTube, files, cameras, and remote streams through feed/video_adapter.py.")
    return endpoint


def safe_message(value: object) -> str:
    """Avoid persisting request tokens if a server returns an unexpected error."""
    return re.sub(r"https?://[^\s\"'<>]+", "[URL]", str(value))


class LocalRedirectHandler(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        local_endpoint(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def stream_boundary(content_type: str) -> bytes:
    message = Message()
    message["content-type"] = content_type
    boundary = message.get_param("boundary")
    if message.get_content_type() != "multipart/x-mixed-replace" or not boundary:
        raise ValueError("Local feed must use multipart/x-mixed-replace with a boundary.")
    try:
        encoded = boundary.encode("ascii")
    except UnicodeEncodeError:
        raise ValueError("Invalid MJPEG boundary.") from None
    if len(encoded) > 200 or b"\r" in encoded or b"\n" in encoded:
        raise ValueError("Invalid MJPEG boundary.")
    return b"--" + encoded


def _read_exact(stream, count: int) -> bytes:
    chunks = []
    while count:
        chunk = stream.read(count)
        if not chunk:
            raise EOFError("Local feed ended during a JPEG frame.")
        chunks.append(chunk)
        count -= len(chunk)
    return b"".join(chunks)


def iter_jpeg_frames(stream, boundary: bytes):
    """Parse bounded complete MJPEG parts; never pass partial JPEGs to OpenCV."""
    scanned = 0
    while True:
        line = stream.readline(16 * 1024 + 1)
        if not line:
            return
        scanned += len(line)
        if scanned > 64 * 1024:
            raise ValueError("MJPEG boundary was not found within 64 KiB.")
        marker = line.rstrip(b"\r\n")
        if marker == boundary + b"--":
            return
        if marker != boundary:
            continue
        scanned = 0
        headers, header_bytes = {}, 0
        while True:
            line = stream.readline(16 * 1024 + 1)
            if not line:
                raise EOFError("Local feed ended during MJPEG headers.")
            header_bytes += len(line)
            if header_bytes > 16 * 1024:
                raise ValueError("MJPEG part headers exceed 16 KiB.")
            if line in {b"\r\n", b"\n"}:
                break
            key, separator, value = line.partition(b":")
            if not separator:
                raise ValueError("Malformed MJPEG part header.")
            key = key.strip().lower()
            if key in headers:
                raise ValueError("Duplicate MJPEG part header.")
            headers[key] = value.strip()
        length = headers.get(b"content-length", b"")
        if not length.isdigit() or not 0 < int(length) <= MAX_JPEG_BYTES:
            raise ValueError("MJPEG Content-Length must be between 1 byte and 8 MiB.")
        if headers.get(b"content-type", b"").split(b";", 1)[0].lower() != b"image/jpeg":
            raise ValueError("Local feed parts must contain image/jpeg.")
        yield _read_exact(stream, int(length))


def jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    """Check JPEG frame dimensions before a decoder can allocate its pixels."""
    if not 0 < len(payload) <= MAX_JPEG_BYTES:
        raise ValueError("Local feed JPEG must be between 1 byte and 8 MiB.")
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise ValueError("The local feed must send one complete JPEG image.")
    position = 2
    dimensions = None
    while position < len(payload):
        if payload[position] != 0xFF:
            raise ValueError("Invalid local feed JPEG header.")
        while position < len(payload) and payload[position] == 0xFF:
            position += 1
        if position >= len(payload):
            break
        marker = payload[position]
        position += 1
        if marker == 0xD9:
            break
        if marker in {0xD8, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if position + 2 > len(payload):
            raise ValueError("Incomplete local feed JPEG segment header.")
        length = int.from_bytes(payload[position:position + 2], "big")
        if length < 2 or position + length > len(payload):
            raise ValueError("Invalid local feed JPEG segment length.")
        if marker in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
            if length < 8:
                raise ValueError("Invalid local feed JPEG dimensions.")
            height = int.from_bytes(payload[position + 3:position + 5], "big")
            width = int.from_bytes(payload[position + 5:position + 7], "big")
            if not (0 < width <= MAX_IMAGE_WIDTH and 0 < height <= MAX_IMAGE_HEIGHT):
                raise ValueError(f"Local feed JPEG dimensions must be between 1x1 and {MAX_IMAGE_WIDTH}x{MAX_IMAGE_HEIGHT}.")
            if dimensions is not None:
                raise ValueError("The local feed JPEG contains multiple image frame headers.")
            dimensions = width, height
        if marker == 0xDA:
            break
        position += length
    if dimensions is None:
        raise ValueError("Local feed JPEG dimensions were not found.")
    return dimensions


def decode_jpeg(payload: bytes, cv2, np):
    width, height = jpeg_dimensions(payload)
    try:
        frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        raise ValueError("The local feed JPEG could not be decoded.") from None
    if frame is None or frame.shape[:2] != (height, width):
        raise ValueError("The local feed JPEG could not be decoded with its declared dimensions.")
    return frame


def valid_fps(value: object, default: float = 30.0) -> float:
    try:
        fps = float(value)
    except (TypeError, ValueError):
        return default
    return fps if math.isfinite(fps) and 0 < fps <= 240 else default


class LatestFrameReader:
    """Drain the local feed continuously, retaining only one decoded frame."""

    def __init__(self, endpoint: str, decoder, *, opener=None, require_status: bool = True,
                 connect_timeout: float = 1.0, poll_interval: float = 0.5,
                 retry_delay: float = 0.25, max_retries: int = 3):
        self.endpoint = local_endpoint(endpoint)
        parsed = urlparse(endpoint)
        self.status_url = urlunparse(parsed._replace(path="/status", params="", query="", fragment=""))
        self.decoder = decoder
        self.open = opener or build_opener(ProxyHandler({}), LocalRedirectHandler()).open
        self.require_status = require_status
        self.connect_timeout = connect_timeout
        self.poll_interval = poll_interval
        self.retry_delay = retry_delay
        self.max_retries = max_retries
        self.frames = Queue(maxsize=1)
        self.stopped, self.done = Event(), Event()
        self.lock = Lock()
        self.response = self.stream_socket = self.stream_raw = None
        self.dropped = self.produced = self.reconnections = 0
        self.source_fps = 30.0
        self.terminal_state = None
        self.error = None
        self.adapter_status = None
        self.status_supported = require_status
        self.thread = Thread(target=self._run, name="local-video-reader", daemon=True)
        self.status_thread = Thread(target=self._monitor_status, name="local-video-status", daemon=True)
        self.thread.start()
        self.status_thread.start()

    def _publish(self, frame) -> None:
        while not self.stopped.is_set():
            try:
                self.frames.put_nowait(frame)
                self.produced += 1
                return
            except Full:
                try:
                    self.frames.get_nowait()
                    self.dropped += 1
                except Empty:
                    pass

    def _fetch_status(self) -> dict | None:
        try:
            request = Request(self.status_url, headers={"Accept": "application/json"})
            with self.open(request, timeout=min(self.connect_timeout, 0.5)) as response:
                raw = response.read(64 * 1024 + 1)
            if len(raw) > 64 * 1024:
                return None
            status = json.loads(raw)
            if not isinstance(status, dict) or status.get("state") not in {"waiting", "streaming", *TERMINAL_STATES}:
                return None
            self.status_supported = True
            self.adapter_status = status
            self.source_fps = valid_fps(status.get("fps"), self.source_fps)
            return status
        except HTTPError as exc:
            if exc.code in {404, 405} and not self.require_status:
                self.status_supported = False
        except (OSError, ValueError):
            pass
        return None

    def _accept_terminal(self, status: dict | None) -> bool:
        if not status or status.get("state") not in TERMINAL_STATES:
            return False
        self.terminal_state = status["state"]
        if self.terminal_state == "error":
            self.error = safe_message(status.get("error") or "The video adapter failed.")
        return True

    def _shutdown_stream(self) -> None:
        # close() may block on a BufferedReader lock while it is reading. Wake the
        # worker through the socket first, then let it close its own response.
        with self.lock:
            stream_socket = self.stream_socket
            stream_raw = self.stream_raw
        if stream_socket is not None:
            try:
                stream_socket.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        if stream_raw is not None:
            # Windows shutdown does not cancel an already pending recv. Closing
            # SocketIO releases the native handle without the buffered read lock.
            try:
                stream_raw.close()
            except (OSError, ValueError):
                pass

    def _monitor_status(self) -> None:
        while not self.stopped.is_set() and not self.done.is_set():
            status = self._fetch_status()
            # A finished source still serves its final JPEG. Let the multipart
            # reader reach EOF so this frame remains available to the consumer.
            if status and status["state"] != "finished" and self._accept_terminal(status):
                self._shutdown_stream()
                return
            if self.stopped.wait(self.poll_interval):
                return

    def _run(self) -> None:
        failures = 0
        try:
            while not self.stopped.is_set() and self.terminal_state is None:
                response = None
                try:
                    request = Request(self.endpoint, headers={"Accept": "multipart/x-mixed-replace"})
                    response = self.open(request, timeout=self.connect_timeout)
                    boundary = stream_boundary(response.headers.get("Content-Type", ""))
                    self.source_fps = valid_fps(response.headers.get("X-Stream-Fps"), self.source_fps)
                    # urllib's HTTPResponse wraps SocketIO. Blocking reads happen
                    # only in this worker and cleanup shuts its socket down.
                    stream_raw = getattr(getattr(response, "fp", None), "raw", None)
                    stream_socket = getattr(stream_raw, "_sock", None)
                    with self.lock:
                        self.response, self.stream_socket, self.stream_raw = response, stream_socket, stream_raw
                        if stream_socket is not None:
                            stream_socket.settimeout(None)
                    if self.stopped.is_set() or self.terminal_state is not None:
                        break
                    for jpeg in iter_jpeg_frames(response, boundary):
                        if self.stopped.is_set() or self.terminal_state is not None:
                            break
                        frame = self.decoder(jpeg)
                        if frame is None:
                            raise ValueError("The local feed sent an invalid JPEG image.")
                        if self.produced == 0:
                            # Headers may precede YouTube resolution. The status
                            # reflects the actual FPS once its first frame exists.
                            self._fetch_status()
                        self._publish(frame)
                        failures = 0
                    if self.stopped.is_set() or self.terminal_state is not None:
                        break
                    status = self._fetch_status()
                    if self._accept_terminal(status):
                        break
                    if not self.status_supported:
                        self.terminal_state = "finished"
                        break
                    raise EOFError("The local video feed disconnected before its adapter finished.")
                except ValueError as exc:
                    self.error = safe_message(exc)
                    self.terminal_state = "error"
                    break
                except (OSError, EOFError) as exc:
                    if self.stopped.is_set() or self.terminal_state is not None:
                        break
                    status = self._fetch_status()
                    if (status and (status["state"] != "finished" or self.produced > 0)
                            and self._accept_terminal(status)):
                        break
                    if failures >= self.max_retries:
                        self.error = (f"Could not read the local video adapter after {self.max_retries} retries. "
                                      f"Start feed/video_adapter.py on the selected port. {safe_message(exc)}")
                        self.terminal_state = "error"
                        break
                    failures += 1
                    self.reconnections += 1
                finally:
                    with self.lock:
                        self.response = self.stream_socket = self.stream_raw = None
                    if response is not None:
                        response.close()
                if self.stopped.wait(self.retry_delay):
                    break
        except Exception as exc:
            self.error = safe_message(exc)
            self.terminal_state = "error"
        finally:
            self.done.set()

    def read(self, timeout: float = 0.05):
        try:
            return True, self.frames.get(timeout=timeout)
        except Empty:
            return (False if self.done.is_set() else None), None

    def stats(self) -> dict:
        return {"capacity_frames": 1, "received_frames": self.produced,
                "dropped_frames": self.dropped, "reconnections": self.reconnections,
                "source_fps": self.source_fps}

    def close(self) -> None:
        self.stopped.set()
        self._shutdown_stream()
        self.thread.join(timeout=self.connect_timeout + 0.5)
        self.status_thread.join(timeout=min(self.connect_timeout, 0.5) + 0.1)
