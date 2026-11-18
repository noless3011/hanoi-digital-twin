"""Read the iHanoi camera transport observed in a captured live session.

The observed binary layout is a 12-byte wrapper followed by one HEVC NAL.
Wrapper meanings are unknown; this is an empirical adapter, not a VTS spec.
Optional media and transport dependencies are imported only when selected.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import ssl
from threading import Event
import time
from typing import Callable
from urllib.parse import urlsplit, urlunsplit


START_CODE = b"\x00\x00\x00\x01"
WRAPPER_BYTES = 12
PARAMETER_TYPES = {32, 33, 34}  # VPS, SPS, PPS
MAX_IMAGE_WIDTH = 3840
MAX_IMAGE_HEIGHT = 2160
FRAME_TIMEOUT = 20
RECONNECT_DELAY = 1
REFRESH_GUIDANCE = "Reopen this camera in iHanoi and update its captured URL in the feed configuration."


def websocket_url(value: str) -> str:
    """Normalize a captured upgrade URL without changing its session path."""
    try:
        if not isinstance(value, str) or not value or any(char.isspace() for char in value):
            raise ValueError
        parts = urlsplit(value)
        schemes = {"http": "ws", "https": "wss", "ws": "ws", "wss": "wss"}
        if (parts.scheme not in schemes or not parts.hostname or parts.username is not None
                or parts.password is not None or parts.fragment):
            raise ValueError
        parts.port  # Validate an explicit port without including it in an error.
        return urlunsplit((schemes[parts.scheme], parts.netloc, parts.path, parts.query, ""))
    except (ValueError, TypeError):
        raise ValueError("Supply an HTTP(S) or WS(S) camera URL without credentials or a fragment.") from None


def unwrap_hevc(message: bytes) -> tuple[int, bytes]:
    """Validate the NAL header at the observed offset; retain trailing zeros."""
    if not isinstance(message, bytes) or len(message) < WRAPPER_BYTES + 2:
        raise ValueError("Not a complete wrapped HEVC NAL")
    nal = message[WRAPPER_BYTES:]
    nal_type = (nal[0] >> 1) & 63
    if nal[0] & 128 or not nal[1] & 7 or nal_type > 40:
        raise ValueError("Invalid or unsupported HEVC NAL header at offset 12")
    return nal_type, nal


class _SourceError(RuntimeError):
    """Only fixed adapter messages are safe to propagate to public status."""


def _check_frame(frame) -> None:
    if not 0 < frame.width <= MAX_IMAGE_WIDTH or not 0 < frame.height <= MAX_IMAGE_HEIGHT:
        raise _SourceError("The iHanoi video dimensions are unsupported; maximum size is 3840 x 2160.")


class HevcDecoder:
    """Wait for VPS/SPS/PPS and an IRAP before parsing successive HEVC NALs."""

    def __init__(self):
        try:
            import av
        except ImportError:
            raise RuntimeError("The iHanoi source requires PyAV; install the feed requirements.") from None
        self._av = av
        self.codec = av.CodecContext.create("hevc", "r")
        self.codec.thread_count = 1
        self.parameters = {}
        self.nal_counts = Counter()
        self.errors = 0
        self.skipped_messages = 0
        self.ready = False
        self.flushed = False

    def _parse_decode(self, data):
        frames = []
        try:
            packets = self.codec.parse(data)
        except self._av.error.FFmpegError:
            self.errors += 1
            return frames
        for packet in packets:
            try:
                decoded = self.codec.decode(packet)
            except self._av.error.FFmpegError:
                self.errors += 1
                continue
            for frame in decoded:
                _check_frame(frame)
            frames.extend(decoded)
        return frames

    def feed(self, message: bytes):
        if self.flushed:
            raise RuntimeError("Cannot feed a flushed decoder")
        try:
            nal_type, nal = unwrap_hevc(message)
        except ValueError:
            self.skipped_messages += 1
            return []
        self.nal_counts[nal_type] += 1
        if nal_type in PARAMETER_TYPES:
            self.parameters[nal_type] = nal
        if not self.ready:
            if 16 <= nal_type <= 23 and PARAMETER_TYPES <= self.parameters.keys():
                self.ready = True
                parameters = b"".join(START_CODE + self.parameters[t] for t in sorted(PARAMETER_TYPES))
                return self._parse_decode(parameters + START_CODE + nal)
            self.skipped_messages += 1
            return []
        return self._parse_decode(START_CODE + nal)

    def flush(self):
        if self.flushed or not self.ready:
            return []
        self.flushed = True
        frames = self._parse_decode(b"")
        try:
            decoded = self.codec.decode(None)
        except self._av.error.FFmpegError:
            self.errors += 1
            return frames
        for frame in decoded:
            _check_frame(frame)
        return frames + decoded


def stream_frames(url: str, stopped: Event, on_frame: Callable, *, ca_file: Path | None = None,
                  resend_on_stall: bool = True) -> None:
    """Decode a captured session until cancellation, restarting stopped streams.

    Resend the init command once on inactivity and reconnect a closed stream
    that delivered frames. Expired URLs still fail with refresh guidance.
    With resend_on_stall=False, inactivity fails instead, so the caller can close
    the connection and open a new one.
    """
    if stopped.is_set():
        return
    url = websocket_url(url)
    try:
        from websockets.exceptions import ConnectionClosed, InvalidHandshake
        from websockets.sync.client import connect
    except ImportError:
        raise RuntimeError("The iHanoi source requires websockets; install the feed requirements.") from None
    try:
        tls = ssl.create_default_context()
        if ca_file:
            tls.load_verify_locations(cafile=str(ca_file))
    except (OSError, ssl.SSLError):
        raise RuntimeError("Could not load the verified iHanoi TLS trust configuration.") from None
    frames_received = 0
    last_frame = time.monotonic()
    restarting = False

    def publish(frames):
        nonlocal frames_received, last_frame, restarting
        for frame in frames:
            if stopped.is_set():
                return
            _check_frame(frame)  # Enforce the bound before callers allocate BGR arrays.
            try:
                on_frame(frame)
            except Exception:
                raise _SourceError("Could not process an iHanoi video frame.") from None
            frames_received += 1
            last_frame = time.monotonic()
            restarting = False

    try:
        while not stopped.is_set():
            decoder = HevcDecoder()
            frames_received = 0
            restarting = False
            try:
                with connect(url, ssl=tls if url.startswith("wss:") else None, proxy=None,
                             open_timeout=10, close_timeout=2, max_size=8 * 1024 * 1024,
                             max_queue=4, ping_interval=None, compression=None,
                             user_agent_header="okhttp/4.12.0",
                             additional_headers={"Accept-Encoding": "gzip"}) as ws:
                    if stopped.is_set():
                        return
                    ws.send("mobile:")
                    last_frame = time.monotonic()
                    while not stopped.is_set():
                        if time.monotonic() - last_frame >= FRAME_TIMEOUT:
                            if not resend_on_stall:
                                raise _SourceError(f"No iHanoi video frame arrived for {FRAME_TIMEOUT} seconds. " + REFRESH_GUIDANCE)
                            if restarting:
                                raise _SourceError("No iHanoi video frame arrived after restarting the stream. " + REFRESH_GUIDANCE)
                            ws.send("mobile:")
                            decoder = HevcDecoder()
                            restarting = True
                            last_frame = time.monotonic()
                        try:
                            message = ws.recv(timeout=1)
                        except TimeoutError:
                            continue
                        if stopped.is_set():
                            return
                        if isinstance(message, bytes):
                            publish(decoder.feed(message))
            except ConnectionClosed:
                if stopped.is_set():
                    return
                if not frames_received:
                    raise _SourceError("The iHanoi camera connection closed before delivering frames. " + REFRESH_GUIDANCE) from None
                publish(decoder.flush())
                if stopped.wait(RECONNECT_DELAY):
                    return
    except _SourceError as exc:
        raise RuntimeError(str(exc)) from None
    except ssl.SSLCertVerificationError:
        raise RuntimeError("iHanoi TLS certificate verification failed. Check the additional trusted CA PEM bundle.") from None
    except InvalidHandshake:
        raise RuntimeError("The iHanoi server refused the WebSocket handshake. " + REFRESH_GUIDANCE) from None
    except Exception:
        if stopped.is_set():
            return
        raise RuntimeError("The iHanoi camera transport failed. " + REFRESH_GUIDANCE) from None
