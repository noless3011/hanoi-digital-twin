"""Verify iHanoi configuration, transport and real recorded video without YOLO."""
from contextlib import redirect_stderr
from http.client import HTTPConnection
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import video_adapter as adapter
import ihanoi_source as ihanoi

FEED_DIR = Path(adapter.__file__).resolve().parent
CAPTURE = FEED_DIR.parent / "experiments" / "ihanoi_ws" / "captures" / "mobile_text_01"
TOKEN = "private-camera-token-should-not-leak"


class ConfigurationTests(unittest.TestCase):
    def test_existing_source_default_and_exclusive_named_feed(self):
        defaults = adapter.parser().parse_args([])
        self.assertEqual(defaults.source, adapter.DEFAULT_SOURCE)
        self.assertIsNone(defaults.ihanoi)
        self.assertEqual(defaults.ihanoi_feeds, FEED_DIR / "ihanoi.json")
        self.assertEqual(defaults.ihanoi_ca_file, FEED_DIR / "certs" / "vtscloud-ca.pem")
        self.assertEqual(adapter.parser().parse_args(["--ihanoi", "street"]).ihanoi, "street")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            adapter.parser().parse_args(["--source", "twin", "--ihanoi", "street"])

    def test_utf8_bom_map_and_url_normalization(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feeds.json"
            path.write_text(json.dumps({"Đường phố": "https://camera.invalid/evup/session/token?channel=1"}),
                            encoding="utf-8-sig")
            self.assertEqual(adapter.resolve_ihanoi_feed("Đường phố", path),
                             "wss://camera.invalid/evup/session/token?channel=1")
        for value, expected in (("http://127.0.0.1:1234/live", "ws://127.0.0.1:1234/live"),
                                ("ws://localhost/live", "ws://localhost/live"),
                                ("wss://camera.invalid/live", "wss://camera.invalid/live")):
            self.assertEqual(ihanoi.websocket_url(value), expected)

    def test_invalid_maps_names_and_urls_fail_without_token_leaks(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "feeds.json"
            with self.assertRaises(FileNotFoundError):
                adapter.resolve_ihanoi_feed("street", path)
            for raw in ("{broken JSON", "[]", "null", '{"street": 42}', '{"street": ""}', '{"": "https://camera.invalid"}'):
                with self.subTest(raw=raw):
                    path.write_text(raw, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        adapter.resolve_ihanoi_feed("street", path)
            path.write_text(json.dumps({"street": "https://camera.invalid/" + TOKEN}), encoding="utf-8")
            for name in ("missing", "", "  "):
                with self.assertRaises(ValueError) as error:
                    adapter.resolve_ihanoi_feed(name, path)
                self.assertNotIn(TOKEN, str(error.exception))
            for url in ("file:///" + TOKEN, "https://user:" + TOKEN + "@camera.invalid/live",
                        "wss://camera.invalid/live#" + TOKEN, "wss://", "https://camera.invalid:99999/" + TOKEN):
                with self.subTest(url=url):
                    path.write_text(json.dumps({"street": url}), encoding="utf-8")
                    with self.assertRaises(ValueError) as error:
                        adapter.resolve_ihanoi_feed("street", path)
                    self.assertNotIn(TOKEN, str(error.exception))

    def test_invalid_feed_fails_before_http_listener_starts(self):
        with tempfile.TemporaryDirectory() as temporary:
            args = adapter.parser().parse_args(["--ihanoi", "missing", "--ihanoi-feeds", str(Path(temporary) / "absent.json")])
            with patch.object(adapter, "create_server") as listener, self.assertRaises(FileNotFoundError):
                adapter.run(args)
            listener.assert_not_called()

    def test_import_keeps_media_and_model_dependencies_lazy(self):
        code = ("import sys, video_adapter, ihanoi_source; "
                "assert all(x not in sys.modules for x in "
                "('cv2','numpy','av','websockets','torch','ultralytics','yt_dlp'))")
        subprocess.run([sys.executable, "-c", code], cwd=FEED_DIR, check=True, capture_output=True)


class TransportTests(unittest.TestCase):
    def closed(self, code=1000):
        from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError
        from websockets.frames import Close
        cls = ConnectionClosedOK if code in (1000, 1001) else ConnectionClosedError
        return cls(Close(code, ""), Close(code, ""), True)

    def test_exact_text_startup_reconnects_closed_stream_with_fresh_decoder(self):
        frame = SimpleNamespace(width=384, height=216)
        tail = SimpleNamespace(width=384, height=216)
        received, stopped = [], Event()
        socket = MagicMock()
        socket.recv.side_effect = ['{"status":"ready"}', b"wrapped NAL", self.closed()]
        decoder = MagicMock()
        decoder.feed.return_value = [frame]
        decoder.flush.return_value = [tail]
        next_socket, next_decoder = MagicMock(), MagicMock()
        next_socket.recv.return_value = b"new session NAL"
        next_decoder.feed.return_value = [frame]

        def receive(frame):
            received.append(frame)
            if len(received) == 3:
                stopped.set()

        with patch("websockets.sync.client.connect") as connect, \
             patch.object(ihanoi, "HevcDecoder", side_effect=[decoder, next_decoder]), \
             patch.object(ihanoi, "RECONNECT_DELAY", 0):
            connect.return_value.__enter__.side_effect = [socket, next_socket]
            ihanoi.stream_frames("ws://127.0.0.1:1234/feed", stopped, receive)
        socket.send.assert_called_once_with("mobile:")
        self.assertIsInstance(socket.send.call_args.args[0], str)
        decoder.feed.assert_called_once_with(b"wrapped NAL")
        next_socket.send.assert_called_once_with("mobile:")
        next_decoder.feed.assert_called_once_with(b"new session NAL")
        self.assertEqual(received, [frame, tail, frame])
        decoder.flush.assert_called_once()
        self.assertEqual(connect.call_count, 2)
        self.assertEqual(connect.call_args_list[0], connect.call_args_list[1])
        self.assertEqual(connect.return_value.__exit__.call_count, 2)

    def test_stop_before_connect_and_receive_timeout_cancel_promptly(self):
        stopped = Event()
        stopped.set()
        with patch("websockets.sync.client.connect") as connect:
            ihanoi.stream_frames("ws://127.0.0.1:1234/feed", stopped, lambda frame: self.fail("Unexpected frame"))
            connect.assert_not_called()
        stopped.clear()
        socket, decoder = MagicMock(), MagicMock()

        def timeout(**kwargs):
            stopped.set()
            raise TimeoutError()

        socket.recv.side_effect = timeout
        with patch("websockets.sync.client.connect") as connect, patch.object(ihanoi, "HevcDecoder", return_value=decoder):
            connect.return_value.__enter__.return_value = socket
            ihanoi.stream_frames("ws://127.0.0.1:1234/feed", stopped, lambda frame: self.fail("Unexpected frame"))
        decoder.flush.assert_not_called()
        self.assertEqual(socket.recv.call_count, 1)
        self.assertLessEqual(socket.recv.call_args.kwargs["timeout"], 1)
        connect.return_value.__exit__.assert_called_once()

    def test_empty_close_and_connection_failure_are_actionable_and_private(self):
        socket, decoder = MagicMock(), MagicMock()
        socket.recv.side_effect = self.closed()
        decoder.flush.return_value = []
        with patch("websockets.sync.client.connect") as connect, patch.object(ihanoi, "HevcDecoder", return_value=decoder):
            connect.return_value.__enter__.return_value = socket
            with self.assertRaises(RuntimeError) as error:
                ihanoi.stream_frames("ws://127.0.0.1:1234/" + TOKEN, Event(), lambda frame: None)
        self.assertNotIn(TOKEN, str(error.exception))
        self.assertTrue(str(error.exception))
        with patch("websockets.sync.client.connect", side_effect=OSError("bad wss://camera.invalid/" + TOKEN)):
            with self.assertRaises(RuntimeError) as error:
                ihanoi.stream_frames("ws://127.0.0.1:1234/" + TOKEN, Event(), lambda frame: None)
        self.assertNotIn(TOKEN, str(error.exception))

    def test_inactivity_timeout_has_no_unbounded_retry(self):
        socket, decoder = MagicMock(), MagicMock()
        socket.recv.side_effect = TimeoutError()
        with patch("websockets.sync.client.connect") as connect, patch.object(ihanoi, "HevcDecoder", return_value=decoder), \
             patch.object(ihanoi.time, "monotonic", side_effect=[0, 0, 21, 21, 42]):
            connect.return_value.__enter__.return_value = socket
            with self.assertRaises(RuntimeError):
                ihanoi.stream_frames("ws://127.0.0.1:1234/feed", Event(), lambda frame: None)
        self.assertLessEqual(socket.recv.call_count, 2)
        self.assertEqual(socket.send.call_args_list, [unittest.mock.call("mobile:"), unittest.mock.call("mobile:")])

    def test_inactivity_without_resend_fails_so_caller_reconnects(self):
        socket, decoder = MagicMock(), MagicMock()
        socket.recv.side_effect = TimeoutError()
        with patch("websockets.sync.client.connect") as connect, patch.object(ihanoi, "HevcDecoder", return_value=decoder), \
             patch.object(ihanoi.time, "monotonic", side_effect=[0, 0, 21]):
            connect.return_value.__enter__.return_value = socket
            with self.assertRaisesRegex(RuntimeError, "No iHanoi video frame arrived for 20 seconds"):
                ihanoi.stream_frames("ws://127.0.0.1:1234/feed", Event(), lambda frame: None, resend_on_stall=False)
        socket.send.assert_called_once_with("mobile:")  # only the startup command, never a resend
        connect.return_value.__exit__.assert_called_once()  # the connection was closed

    def test_inactivity_resends_init_and_resumes_frames(self):
        stopped, received = Event(), []
        socket, decoder, restarted_decoder = MagicMock(), MagicMock(), MagicMock()
        frame = SimpleNamespace(width=384, height=216)
        restarted_decoder.feed.return_value = [frame]
        socket.recv.return_value = b"restarted NAL"

        def receive(frame):
            received.append(frame)
            stopped.set()

        with patch("websockets.sync.client.connect") as connect, \
             patch.object(ihanoi, "HevcDecoder", side_effect=[decoder, restarted_decoder]), \
             patch.object(ihanoi.time, "monotonic", side_effect=[0, 0, 21, 21, 22]):
            connect.return_value.__enter__.return_value = socket
            ihanoi.stream_frames("ws://127.0.0.1:1234/feed", stopped, receive)
        self.assertEqual(received, [frame])
        self.assertEqual(socket.send.call_count, 2)
        for call in socket.send.call_args_list:
            self.assertEqual(call.args, ("mobile:",))
        decoder.feed.assert_not_called()
        restarted_decoder.feed.assert_called_once_with(b"restarted NAL")

    def test_oversized_decoded_frame_is_rejected_before_callback(self):
        socket, decoder, received = MagicMock(), MagicMock(), []
        socket.recv.return_value = b"wrapped NAL"
        decoder.feed.return_value = [SimpleNamespace(width=3841, height=216)]
        with patch("websockets.sync.client.connect") as connect, patch.object(ihanoi, "HevcDecoder", return_value=decoder):
            connect.return_value.__enter__.return_value = socket
            with self.assertRaises(RuntimeError):
                ihanoi.stream_frames("ws://127.0.0.1:1234/feed", Event(), received.append)
        self.assertEqual(received, [])


class WorkerTests(unittest.TestCase):
    def test_shared_jpeg_publish_honors_resize_and_fps(self):
        import numpy as np

        args = adapter.parser().parse_args(["--ihanoi", "street", "--max-height", "16", "--fps", "12"])
        frame = MagicMock()
        frame.to_ndarray.return_value = np.full((32, 64, 3), 80, dtype=np.uint8)

        def stream(url, stopped, on_frame, **kwargs):
            on_frame(frame)

        store = adapter.FrameStore("ihanoi")
        with patch.object(ihanoi, "stream_frames", side_effect=stream):
            adapter.ihanoi_worker(args, store, Event(), url="ws://127.0.0.1/feed")
        frame.to_ndarray.assert_called_once_with(format="bgr24")
        self.assertEqual(adapter.jpeg_dimensions(store.jpeg), (32, 16))
        self.assertEqual((store.frames, store.state, store.fps), (1, "finished", 12))

    def test_worker_error_and_cancellation_states_keep_tokens_private(self):
        args = adapter.parser().parse_args(["--ihanoi", "street"])
        store, stopped = adapter.FrameStore("ihanoi"), Event()
        output = io.StringIO()
        with patch.object(ihanoi, "stream_frames", side_effect=RuntimeError("bad wss://camera.invalid/" + TOKEN)), redirect_stderr(output):
            adapter.ihanoi_worker(args, store, stopped, url="ws://127.0.0.1/feed")
        self.assertEqual(store.state, "error")
        self.assertNotIn(TOKEN, json.dumps(store.status()) + output.getvalue())
        cancelled = adapter.FrameStore("ihanoi")

        def stream(url, stopped, on_frame, **kwargs):
            stopped.set()
            raise RuntimeError("Socket interrupted during stop")

        with patch.object(ihanoi, "stream_frames", side_effect=stream):
            adapter.ihanoi_worker(args, cancelled, stopped, url="ws://127.0.0.1/feed")
        self.assertEqual(cancelled.state, "stopped")
        self.assertIsNone(cancelled.error)


@unittest.skipUnless((CAPTURE / "messages.jsonl").is_file(), "Retained real camera sample unavailable")
class RecordedMediaTests(unittest.TestCase):
    def recorded_messages(self):
        entries = [json.loads(line) for line in (CAPTURE / "messages.jsonl").read_text().splitlines()]
        return [(CAPTURE / entry["file"]).read_bytes() for entry in entries if entry["type"] == "binary"]

    def test_real_hevc_waits_for_random_access_and_decodes_cleanly(self):
        decoder, frames = ihanoi.HevcDecoder(), []
        for index, message in enumerate(self.recorded_messages()):
            frames.extend(decoder.feed(message))
            if index < 5:
                self.assertFalse(decoder.ready)
                self.assertEqual(frames, [])
        frames.extend(decoder.flush())
        self.assertEqual(len(frames), 310)
        self.assertEqual(decoder.errors, 0)
        self.assertEqual({(frame.width, frame.height) for frame in frames}, {(384, 216)})
        self.assertGreater(float(frames[0].to_ndarray(format="bgr24").std()), 30)
        self.assertEqual(decoder.flush(), [])

    def test_invalid_wrapped_messages_do_not_start_or_error_decoder(self):
        decoder = ihanoi.HevcDecoder()
        for payload in (b"", b"short", bytes(12) + b"\x40\x00", bytes(12) + b"\xc0\x01"):
            self.assertEqual(decoder.feed(payload), [])
        self.assertFalse(decoder.ready)
        self.assertEqual(decoder.errors, 0)
        self.assertEqual(decoder.skipped_messages, 4)

    def test_loopback_websocket_worker_publishes_real_jpeg_over_mjpeg(self):
        import cv2
        import numpy as np
        from websockets.sync.server import serve

        observed, failures = [], []
        source_closed = Event()
        messages = self.recorded_messages()[:100]

        def handler(socket):
            try:
                observed.append(socket.recv(timeout=3))
                for message in messages:
                    socket.send(message)
                socket.close(code=1000)
            except Exception as error:
                failures.append(error)
            finally:
                source_closed.set()

        with serve(handler, "127.0.0.1", 0) as source_server, tempfile.TemporaryDirectory() as temporary:
            source_thread = Thread(target=source_server.serve_forever, daemon=True)
            source_thread.start()
            config = Path(temporary) / "ihanoi.json"
            config.write_text(json.dumps({"street": f"ws://127.0.0.1:{source_server.socket.getsockname()[1]}/feed"}))
            args = adapter.parser().parse_args(["--ihanoi", "street", "--ihanoi-feeds", str(config)])
            stopped, store = Event(), adapter.FrameStore("ihanoi", 25)
            http_server = adapter.create_server("127.0.0.1", 0, store)
            http_thread = Thread(target=http_server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            worker = Thread(target=adapter.ihanoi_worker, args=(args, store, stopped), daemon=True)
            http_thread.start()
            worker.start()
            connection = None
            try:
                with store.condition:
                    self.assertTrue(store.condition.wait_for(lambda: store.frames > 50, timeout=10))
                self.assertTrue(source_closed.wait(timeout=3))
                stopped.set()
                worker.join(timeout=3)
                self.assertFalse(worker.is_alive(), "Worker did not stop after cancellation")
                self.assertEqual(observed, ["mobile:"])
                self.assertEqual(failures, [])
                self.assertEqual(store.state, "stopped")
                self.assertGreater(store.frames, 50)
                connection = HTTPConnection("127.0.0.1", http_server.server_port, timeout=3)
                connection.request("GET", "/stream.mjpg")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                self.assertEqual(response.getheader("X-Stream-Fps"), "25")
                self.assertEqual(response.readline(), b"--frame\r\n")
                self.assertEqual(response.readline(), b"Content-Type: image/jpeg\r\n")
                size = int(response.readline().split(b":", 1)[1])
                self.assertEqual(response.readline(), b"\r\n")
                jpeg = response.read(size)
                decoded = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape, (216, 384, 3))
                self.assertGreater(float(decoded.std()), 30)
                self.assertEqual(response.read(2), b"\r\n")
                self.assertEqual(response.readline(), b"--frame--\r\n")
                connection.close()
                connection = HTTPConnection("127.0.0.1", http_server.server_port, timeout=3)
                connection.request("GET", "/status")
                status = json.loads(connection.getresponse().read())
                self.assertEqual((status["mode"], status["state"]), ("ihanoi", "stopped"))
                self.assertNotIn("session", status)
            finally:
                if connection:
                    connection.close()
                stopped.set()
                store.close()
                http_server.shutdown()
                http_server.server_close()
                http_thread.join(timeout=1)
                source_server.shutdown()
                source_thread.join(timeout=1)
                worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
