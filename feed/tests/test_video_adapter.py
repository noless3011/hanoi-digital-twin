"""Exercise the real loopback MJPEG protocol and source adapter without YOLO."""

from contextlib import contextmanager
import errno
from http.client import HTTPConnection
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
from threading import Event, Thread
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_adapter as adapter


def make_jpeg(value: int = 32) -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), value, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


@contextmanager
def listening(mode="twin"):
    store = adapter.FrameStore(mode)
    server = adapter.create_server("127.0.0.1", 0, store)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield store, server.server_port
    finally:
        store.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


def request(port, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=3)
    connection.request(method, path, body=body, headers=headers or {})
    response = connection.getresponse()
    data = response.read()
    result = response.status, dict(response.getheaders()), json.loads(data) if data else None
    connection.close()
    return result


def read_part(response):
    boundary = response.readline()
    if boundary == b"--frame--\r\n":
        return None
    if boundary != b"--frame\r\n":
        raise AssertionError(f"Unexpected multipart boundary: {boundary!r}")
    content_type = response.readline()
    if content_type != b"Content-Type: image/jpeg\r\n":
        raise AssertionError(f"Unexpected frame type: {content_type!r}")
    length = int(response.readline().split(b":", 1)[1])
    if response.readline() != b"\r\n":
        raise AssertionError("Missing multipart header separator")
    jpeg = response.read(length)
    if response.read(2) != b"\r\n":
        raise AssertionError("Missing multipart body separator")
    return jpeg


class HTTPAdapterTests(unittest.TestCase):
    def upload(self, port, store, body, **extra_headers):
        return request(port, "POST", "/frame", body,
                       {"Content-Type": "image/jpeg", "X-Stream-Session": store.session, **extra_headers})

    def test_waiting_stream_delivers_latest_frame_and_stop_closes_it(self):
        with listening() as (store, port):
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader("X-Stream-Fps"), "15")
            self.assertEqual(response.getheader("Content-Type"), "multipart/x-mixed-replace; boundary=frame")
            jpeg = make_jpeg()
            status, _, value = self.upload(port, store, jpeg)
            self.assertEqual(status, 200)
            self.assertEqual(value["state"], "streaming")
            self.assertEqual(read_part(response), jpeg)
            old_session = store.session
            status, _, value = request(port, "POST", "/stop", headers={"X-Stream-Session": old_session})
            self.assertEqual(status, 200)
            self.assertEqual(value["state"], "stopped")
            self.assertNotEqual(value["session"], old_session)
            self.assertIsNone(read_part(response))
            self.assertEqual(response.read(), b"")
            connection.close()

    def test_latest_store_skips_history_for_new_client(self):
        with listening() as (store, port):
            frames = [make_jpeg(value) for value in (10, 40, 90)]
            for frame in frames:
                self.assertEqual(self.upload(port, store, frame)[0], 200)
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            self.assertEqual(read_part(response), frames[-1])
            self.assertEqual(store.frames, 3)
            store.stop(store.session)
            self.assertIsNone(read_part(response))
            connection.close()

    def test_stop_restart_rejects_stale_inflight_upload(self):
        with listening() as (store, port):
            old_session = store.session
            started, release = Event(), Event()
            real_validator = adapter.validate_jpeg

            def delayed_validator(payload):
                real_validator(payload)
                started.set()
                self.assertTrue(release.wait(3))

            results = []
            with patch.object(adapter, "validate_jpeg", delayed_validator):
                upload = Thread(target=lambda: results.append(request(port, "POST", "/frame", make_jpeg(), {
                    "Content-Type": "image/jpeg", "X-Stream-Session": old_session,
                })))
                upload.start()
                self.assertTrue(started.wait(3))
                self.assertEqual(request(port, "POST", "/stop", headers={"X-Stream-Session": old_session})[0], 200)
                release.set()
                upload.join(timeout=3)
            self.assertEqual(results[0][0], 409)
            self.assertEqual(store.state, "stopped")
            self.assertEqual(store.frames, 0)
            self.assertEqual(self.upload(port, store, make_jpeg())[0], 200)
            self.assertEqual(store.state, "streaming")
            self.assertEqual(store.frames, 1)
            self.assertEqual(request(port, "POST", "/stop", headers={"X-Stream-Session": old_session})[0], 409)
            self.assertEqual(store.state, "streaming")

    def test_upload_validation_and_video_mode_rejections(self):
        with listening() as (store, port):
            self.assertEqual(request(port, "POST", "/frame", make_jpeg(), {"Content-Type": "image/jpeg"})[0], 409)
            self.assertEqual(self.upload(port, store, b"not a JPEG")[0], 400)
            self.assertEqual(self.upload(port, store, make_jpeg(), **{"Content-Type": "image/png"})[0], 415)
            status, _, body = request(port, "POST", "/frame", b"", {
                "Content-Type": "image/jpeg", "Content-Length": str(adapter.MAX_JPEG_BYTES + 1),
                "X-Stream-Session": store.session,
            })
            self.assertEqual(status, 413)
            huge_header = bytearray(make_jpeg())
            index = huge_header.index(b"\xff\xc0")
            huge_header[index + 7:index + 9] = (65535).to_bytes(2, "big")
            with patch.object(cv2, "imdecode") as decoder:
                self.assertEqual(self.upload(port, store, bytes(huge_header))[0], 400)
                decoder.assert_not_called()
            self.assertEqual(store.frames, 0)
            self.assertEqual(request(port, "POST", "/stop", b"body", {"X-Stream-Session": store.session})[0], 400)
        with listening("video") as (store, port):
            self.assertEqual(self.upload(port, store, make_jpeg())[0], 409)
            self.assertEqual(request(port, "POST", "/stop")[0], 409)

    def test_local_cors_and_private_network_preflight(self):
        with listening() as (store, port):
            for origin in ("http://localhost:5173", "https://127.0.0.1:9000", "http://[::1]:5173"):
                status, headers, _ = request(port, "OPTIONS", "/frame", headers={
                    "Origin": origin, "Access-Control-Request-Private-Network": "true",
                    "Access-Control-Request-Method": "POST",
                })
                self.assertEqual(status, 204)
                self.assertEqual(headers["Access-Control-Allow-Origin"], origin)
                self.assertEqual(headers["Access-Control-Allow-Private-Network"], "true")
                self.assertIn("X-Stream-Session", headers["Access-Control-Allow-Headers"])
                self.assertNotIn("Access-Control-Allow-Credentials", headers)
            for origin in ("https://example.org", "null", "http://localhost.example.org:5173", "http://localhost/path"):
                status, headers, _ = request(port, "GET", "/status", headers={"Origin": origin})
                self.assertEqual(status, 403)
                self.assertNotIn("Access-Control-Allow-Origin", headers)
            self.assertEqual(store.frames, 0)

    def test_finished_stream_drains_latest_and_status_survives(self):
        with listening("video") as (store, port):
            jpeg = make_jpeg()
            store.publish(jpeg)
            store.set_state("finished")
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            self.assertEqual(read_part(response), jpeg)
            self.assertIsNone(read_part(response))
            self.assertEqual(request(port, "GET", "/status")[2]["state"], "finished")
            connection.close()

    def test_shutdown_unblocks_stream_with_no_first_frame(self):
        with listening() as (store, port):
            connection = HTTPConnection("127.0.0.1", port, timeout=3)
            connection.request("GET", "/stream.mjpg")
            response = connection.getresponse()
            store.close()
            self.assertIsNone(read_part(response))
            connection.close()


class ServerBindingTests(unittest.TestCase):
    def test_second_live_adapter_is_rejected_and_shutdown_releases_port(self):
        # No accepted connections remain active during the restart. This checks
        # the real OS bind semantics rather than mocking socket options.
        with listening("video") as (_, port):
            with self.assertRaisesRegex(OSError, "already in use.*Ctrl\\+C.*same --port") as failure:
                second = adapter.create_server("127.0.0.1", port, adapter.FrameStore("video"))
                self.addCleanup(second.server_close)
            self.assertEqual(failure.exception.errno, errno.EADDRINUSE)
        restarted = adapter.create_server("127.0.0.1", port, adapter.FrameStore("video"))
        self.addCleanup(restarted.server_close)
        self.assertEqual(restarted.server_port, port)

    @unittest.skipUnless(os.name == "nt", "Windows-exclusive socket semantics")
    def test_windows_listener_is_exclusive_without_address_reuse(self):
        server = adapter.create_server("127.0.0.1", 0, adapter.FrameStore("video"))
        self.addCleanup(server.server_close)
        self.assertEqual(server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE), 1)
        self.assertEqual(server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR), 0)

    def test_other_bind_errors_keep_their_original_exception(self):
        failure = PermissionError(errno.EACCES, "Permission denied")
        with patch.object(adapter, "AdapterServer", side_effect=failure), self.assertRaises(PermissionError) as caught:
            adapter.create_server("127.0.0.1", 8766, adapter.FrameStore("video"))
        self.assertIs(caught.exception, failure)


class SourceTests(unittest.TestCase):
    def args(self, source):
        return adapter.parser().parse_args(["--source", source])

    def test_youtube_urls_and_direct_sources(self):
        expected = "https://www.youtube.com/watch?v=AbC_12-xy"
        for url in ("https://youtu.be/AbC_12-xy?t=4", "https://www.youtube.com/live/AbC_12-xy",
                    "https://youtube.com/shorts/AbC_12-xy", "https://youtube.com/watch?v=AbC_12-xy&list=123"):
            self.assertEqual(adapter.youtube_url(url), expected)
        self.assertIsNone(adapter.youtube_url("https://youtube.com.example.com/watch?v=test"))
        with self.assertRaises(ValueError):
            adapter.youtube_url("https://youtube.com/playlist?list=test")
        self.assertEqual(adapter.resolve_source(self.args("0")), ([{"url": 0}], {"source": "0", "is_live": True}))
        self.assertFalse(adapter.resolve_source(self.args(r"C:\videos\test.mp4"))[1]["is_live"])
        self.assertFalse(adapter.resolve_source(self.args("https://example.org/test.mp4?token=secret"))[1]["is_live"])
        self.assertTrue(adapter.resolve_source(self.args("https://example.org/live.m3u8?token=secret"))[1]["is_live"])

    def test_youtube_resolution_selects_video_and_keeps_secrets_in_memory(self):
        captured = {}

        class Downloader:
            def __init__(self, options):
                captured["options"] = options

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def extract_info(self, url, download):
                captured.update(url=url, download=download)
                return {"id": "abc", "title": "Live", "is_live": True, "formats": [
                    {"url": "https://cdn/video?secret=1", "vcodec": "avc1", "protocol": "https", "height": 720},
                    {"url": "https://cdn/audio?secret=2", "vcodec": "none", "protocol": "https", "height": None},
                    {"url": "https://cdn/drm?secret=3", "vcodec": "avc1", "protocol": "https", "has_drm": True},
                    {"url": "https://cdn/4k?secret=4", "vcodec": "avc1", "protocol": "https", "height": 2160},
                ]}

        fake_module = SimpleNamespace(YoutubeDL=Downloader, utils=SimpleNamespace(DownloadError=RuntimeError))
        with patch.dict(sys.modules, {"yt_dlp": fake_module}), patch.object(adapter, "node_path", return_value="node"):
            formats, metadata = adapter.resolve_source(self.args("https://youtu.be/abc"))
        self.assertFalse(captured["download"])
        self.assertTrue(captured["options"]["noplaylist"])
        self.assertFalse(captured["options"]["cachedir"])
        self.assertEqual(captured["options"]["js_runtimes"], {"node": {"path": "node"}})
        self.assertEqual(formats[0]["height"], 720)
        self.assertEqual(len(formats), 2)
        self.assertNotIn("secret", json.dumps(metadata))
        self.assertEqual(adapter.safe_message("bad https://cdn/video?secret=123"), "bad [URL]")

    def test_import_is_ml_free_and_opencv_lazy(self):
        code = "import sys, video_adapter; assert all(x not in sys.modules for x in ('cv2','numpy','torch','ultralytics','yt_dlp'))"
        subprocess.run([sys.executable, "-c", code], cwd=Path(adapter.__file__).resolve().parent,
                       check=True, capture_output=True)

    def test_rejects_nonloopback_binding(self):
        for host in ("0.0.0.0", "192.168.1.2", "example.org"):
            with self.assertRaises(ValueError):
                adapter.create_server(host, 0, adapter.FrameStore("twin"))
        self.assertEqual(adapter.loopback_host("localhost"), "127.0.0.1")
        self.assertEqual(adapter.loopback_host("::1"), "::1")

    def test_finite_video_worker_is_paced_and_stays_queryable_after_eof(self):
        frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (10, 30, 50)]

        class Capture:
            def __init__(self):
                self.remaining = iter(frames[1:])
                self.released = False

            def get(self, prop):
                return 20

            def read(self):
                frame = next(self.remaining, None)
                return frame is not None, frame

            def release(self):
                self.released = True

        capture = Capture()
        store = adapter.FrameStore("video")
        started = time.monotonic()
        with patch.object(adapter, "resolve_source", return_value=([{"url": "mock.mp4"}], {"is_live": False})), \
             patch.object(adapter, "open_capture", return_value=(capture, frames[0], {})):
            adapter.video_worker(self.args("mock.mp4"), store, Event())
        self.assertEqual(store.frames, 3)
        self.assertEqual(store.state, "finished")
        self.assertGreaterEqual(time.monotonic() - started, 0.13)
        self.assertTrue(capture.released)
        self.assertEqual(adapter.jpeg_dimensions(store.jpeg), (32, 24))

    def test_live_fifo_drains_eof_and_latest_consumer_is_bounded(self):
        frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in range(5)]

        class Capture:
            def __init__(self):
                self.remaining = iter(frames[1:])
                self.released = False

            def read(self):
                frame = next(self.remaining, None)
                return frame is not None, frame

            def release(self):
                self.released = True

        capture = Capture()
        reader = adapter.LatestFrameReader(capture, 100, 0.02, frames[0])
        self.assertTrue(reader.playback_done.wait(2))
        self.assertLessEqual(reader.frames.qsize(), 1)
        ok, last = reader.read()
        self.assertTrue(ok)
        self.assertTrue(np.array_equal(last, frames[-1]))
        self.assertEqual(reader.read()[0], False)
        self.assertEqual(reader.dropped, 4)
        reader.close()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
