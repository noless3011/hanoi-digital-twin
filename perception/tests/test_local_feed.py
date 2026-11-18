"""Exercise the local feed protocol without importing model/GPU dependencies."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
from threading import Event, Lock, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_feed


def multipart(*frames: bytes, closed: bool = True) -> bytes:
    body = b"".join(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                    + str(len(frame)).encode() + b"\r\n\r\n" + frame + b"\r\n" for frame in frames)
    return body + (b"--frame--\r\n" if closed else b"")


class Response(io.BytesIO):
    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = headers or {}


class FakeAdapter:
    def __init__(self, *, state="finished", frames=(b"first", b"second", b"last"), failures=0,
                 status_supported=True, stream_body=None):
        self.state, self.frames, self.failures = state, frames, failures
        self.status_supported, self.stream_body = status_supported, stream_body
        self.stream_attempts = 0
        self.lock = Lock()

    def open(self, request, timeout):
        if request.full_url.endswith("/status"):
            if not self.status_supported:
                raise HTTPError(request.full_url, 404, "Not found", {}, None)
            return Response(json.dumps({"mode": "video", "state": self.state, "fps": 24,
                                        "frames": len(self.frames), "error": "Capture failed"}).encode())
        with self.lock:
            self.stream_attempts += 1
            if self.stream_attempts <= self.failures:
                raise URLError("Connection refused")
        return Response(self.stream_body if self.stream_body is not None else multipart(*self.frames),
                        {"Content-Type": 'multipart/x-mixed-replace; boundary="frame"', "X-Stream-Fps": "24"})


class MultipartTests(unittest.TestCase):
    def test_complete_frames_and_quoted_boundary(self):
        # A boundary-looking byte sequence inside a JPEG is payload, not framing.
        frames = (b"jpeg\r\n--frame\r\nstill jpeg", b"tail")
        boundary = local_feed.stream_boundary('multipart/x-mixed-replace; boundary="frame"')
        self.assertEqual(list(local_feed.iter_jpeg_frames(io.BytesIO(multipart(*frames)), boundary)), list(frames))

    def test_partial_reads_are_reassembled(self):
        class Fragmented(io.BytesIO):
            def read(self, count=-1):
                return super().read(min(count, 2))
        self.assertEqual(list(local_feed.iter_jpeg_frames(Fragmented(multipart(b"jpeg bytes")), b"--frame")), [b"jpeg bytes"])

    def test_reject_invalid_headers_and_oversized_frames(self):
        for headers in (b"Content-Type: image/jpeg\r\n", b"Content-Type: image/png\r\nContent-Length: 4\r\n",
                        b"Content-Type: image/jpeg\r\nContent-Length: 8388609\r\n",
                        b"Content-Type: image/jpeg\r\nContent-Length: -2\r\n",
                        b"Content-Type: image/jpeg\r\nContent-Length: 4\r\nContent-Length: 4\r\n"):
            with self.subTest(headers=headers), self.assertRaises(ValueError):
                list(local_feed.iter_jpeg_frames(io.BytesIO(b"--frame\r\n" + headers + b"\r\njpeg\r\n"), b"--frame"))

    def test_partial_jpeg_is_never_published(self):
        body = b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: 20\r\n\r\nshort"
        with self.assertRaises(EOFError):
            list(local_feed.iter_jpeg_frames(io.BytesIO(body), b"--frame"))

    def test_invalid_stream_content_type(self):
        for content_type in ("image/jpeg", "multipart/x-mixed-replace", "multipart/x-mixed-replace; boundary=é"):
            with self.subTest(content_type=content_type), self.assertRaises(ValueError):
                local_feed.stream_boundary(content_type)


class JpegDecodeTests(unittest.TestCase):
    @staticmethod
    def jpeg_header(width, height):
        # A tiny SOF fixture makes the allocation guard observable without
        # constructing a large image or importing OpenCV/model dependencies.
        frame = (b"\x08" + height.to_bytes(2, "big") + width.to_bytes(2, "big")
                 + b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00")
        return b"\xff\xd8\xff\xc0\x00\x11" + frame + b"\xff\xd9"

    def decoder_modules(self, frame):
        cv2 = SimpleNamespace(imdecode=Mock(return_value=frame), IMREAD_COLOR=1)
        np = SimpleNamespace(frombuffer=Mock(return_value="encoded bytes"), uint8="uint8")
        return cv2, np

    def test_oversized_dimensions_never_reach_decoder(self):
        for width, height in ((3841, 2160), (3840, 2161), (65535, 65535), (0, 10)):
            with self.subTest(width=width, height=height):
                cv2, np = self.decoder_modules(None)
                payload = self.jpeg_header(width, height)
                self.assertLess(len(payload), 30)
                with self.assertRaisesRegex(ValueError, "3840x2160"):
                    local_feed.decode_jpeg(payload, cv2, np)
                cv2.imdecode.assert_not_called()
                np.frombuffer.assert_not_called()

    def test_dimensions_at_boundaries_reach_decoder(self):
        for width, height in ((3840, 2160), (1, 1)):
            with self.subTest(width=width, height=height):
                frame = SimpleNamespace(shape=(height, width, 3))
                cv2, np = self.decoder_modules(frame)
                payload = self.jpeg_header(width, height)
                self.assertIs(local_feed.decode_jpeg(payload, cv2, np), frame)
                cv2.imdecode.assert_called_once_with("encoded bytes", 1)
                np.frombuffer.assert_called_once_with(payload, "uint8")

    def test_malformed_jpeg_is_actionable_before_decode(self):
        for payload in (b"not jpeg", b"\xff\xd8\xff\xd9",
                        b"\xff\xd8\xff\xc0\x00\x01\xff\xd9"):
            with self.subTest(payload=payload):
                cv2, np = self.decoder_modules(None)
                with self.assertRaisesRegex(ValueError, "JPEG"):
                    local_feed.decode_jpeg(payload, cv2, np)
                cv2.imdecode.assert_not_called()

    def test_decoder_rejects_invalid_or_mismatched_pixels(self):
        payload = self.jpeg_header(640, 480)
        for frame in (None, SimpleNamespace(shape=(500, 640, 3))):
            with self.subTest(frame=frame):
                cv2, np = self.decoder_modules(frame)
                with self.assertRaisesRegex(ValueError, "could not be decoded"):
                    local_feed.decode_jpeg(payload, cv2, np)


class ReaderTests(unittest.TestCase):
    def reader(self, adapter, **kwargs):
        reader = local_feed.LatestFrameReader("http://127.0.0.1:8766/stream.mjpg", lambda data: data,
                                           opener=adapter.open, poll_interval=0.01, retry_delay=0.001, **kwargs)
        self.addCleanup(reader.close)
        return reader

    def test_latest_frame_and_eof_preserve_last_complete_frame(self):
        adapter = FakeAdapter()
        reader = self.reader(adapter)
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(reader.read(timeout=0), (True, b"last"))
        self.assertEqual(reader.read(timeout=0), (False, None))
        self.assertEqual(reader.terminal_state, "finished")
        self.assertEqual(reader.dropped, 2)
        self.assertEqual(reader.stats()["received_frames"], 3)
        self.assertEqual(reader.source_fps, 24)

    def test_initial_connection_retry_recovers(self):
        adapter = FakeAdapter(failures=1)
        reader = self.reader(adapter)
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(reader.reconnections, 1)
        self.assertEqual(adapter.stream_attempts, 2)
        self.assertIsNone(reader.error)
        self.assertEqual(reader.read(timeout=0), (True, b"last"))

    def test_connection_failure_is_bounded_and_actionable(self):
        adapter = FakeAdapter(failures=10, state="waiting", frames=())
        reader = self.reader(adapter)
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(adapter.stream_attempts, 4)
        self.assertEqual(reader.terminal_state, "error")
        self.assertIn("Start feed/video_adapter.py", reader.error)

    def test_stop_and_adapter_error_are_distinct(self):
        for state in ("stopped", "error"):
            with self.subTest(state=state):
                reader = self.reader(FakeAdapter(state=state, frames=()))
                self.assertTrue(reader.done.wait(1))
                self.assertEqual(reader.terminal_state, state)
                self.assertEqual(reader.error, "Capture failed" if state == "error" else None)

    def test_optional_generic_mjpeg_eof(self):
        reader = self.reader(FakeAdapter(status_supported=False, frames=(b"jpeg",)), require_status=False)
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(reader.terminal_state, "finished")
        self.assertEqual(reader.read(timeout=0), (True, b"jpeg"))

    def test_invalid_feed_fails_even_if_adapter_reports_finished(self):
        reader = self.reader(FakeAdapter(stream_body=b"--frame\r\nContent-Type: image/png\r\nContent-Length: 4\r\n\r\npng!"))
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(reader.terminal_state, "error")
        self.assertIn("image/jpeg", reader.error)

    def test_decoder_failure_is_reported(self):
        reader = local_feed.LatestFrameReader("http://localhost:8766/stream.mjpg", lambda data: None,
                                           opener=FakeAdapter().open, retry_delay=0)
        self.addCleanup(reader.close)
        self.assertTrue(reader.done.wait(1))
        self.assertEqual(reader.terminal_state, "error")
        self.assertIn("invalid JPEG", reader.error)


class WaitingFeedTests(unittest.TestCase):
    def test_waiting_twin_fps_update_and_bounded_cleanup(self):
        connected, release = Event(), Event()
        state = {"mode": "twin", "state": "waiting", "fps": 17, "frames": 0}

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                if self.path == "/status":
                    body = json.dumps(state).encode()
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("X-Stream-Fps", "30")
                    self.end_headers()
                    self.wfile.flush()
                    connected.set()
                    release.wait(2)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        server.daemon_threads = True
        server_thread = Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        reader = local_feed.LatestFrameReader(f"http://127.0.0.1:{server.server_port}/stream.mjpg", lambda data: data,
                                           poll_interval=0.01)
        try:
            self.assertTrue(connected.wait(1))
            deadline = time.monotonic() + 1
            while reader.source_fps != 17 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(reader.source_fps, 17)
            self.assertEqual(reader.read(timeout=0.02), (None, None))
            self.assertEqual(reader.reconnections, 0)
            started = time.monotonic()
            reader.close()
            self.assertLess(time.monotonic() - started, 0.8)
            self.assertFalse(reader.thread.is_alive())
            self.assertFalse(reader.status_thread.is_alive())
        finally:
            release.set()
            reader.close()
            server.shutdown()
            server.server_close()
            server_thread.join(1)


class EndpointTests(unittest.TestCase):
    def test_defaults_and_custom_loopback_source(self):
        self.assertEqual(local_feed.local_endpoint(None), "http://127.0.0.1:8766/stream.mjpg")
        self.assertEqual(local_feed.local_endpoint(None, 9876), "http://127.0.0.1:9876/stream.mjpg")
        self.assertEqual(local_feed.local_endpoint("https://[::1]:8443/camera"), "https://[::1]:8443/camera")

    def test_reject_remote_sources_and_invalid_ports(self):
        for source in ("https://www.youtube.com/watch?v=abc", "video.mp4", "0", "http://192.168.0.10/video",
                       "http://localhost:wrong/stream", "http://user:password@localhost/stream"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                local_feed.local_endpoint(source)
        for port in (0, 65536):
            with self.subTest(port=port), self.assertRaises(ValueError):
                local_feed.local_endpoint(None, port)

    def test_redirect_cannot_escape_loopback(self):
        with self.assertRaises(ValueError):
            local_feed.LocalRedirectHandler().redirect_request(None, None, 302, "", {}, "https://example.com/stream")


if __name__ == "__main__":
    unittest.main()
