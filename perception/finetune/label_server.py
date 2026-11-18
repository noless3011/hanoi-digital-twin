"""Local bounding-box labelling tool. Standard library only; serves label_ui.html on localhost.

    .venv\\Scripts\\python.exe perception\\finetune\\label_server.py --project data\\labeling\\hanoi-v1
    then open http://localhost:8000

API (JSON):
    GET  /api/project            classes and the list of frames with their status and box count
    GET  /api/labels/<id>        one frame's boxes and status
    PUT  /api/labels/<id>        save {"status": "todo"|"done"|"skip", "boxes": [{x1, y1, x2, y2, cls}]}
    GET  /images/<id>.jpg        the frame
Every save is written atomically to labels/<id>.json, so the server can be stopped at any time.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

UI = Path(__file__).with_name("label_ui.html")
ID = re.compile(r"^[0-9a-f]{6,40}$")
STATUSES = {"todo", "done", "skip"}


class Project:
    def __init__(self, root: Path):
        self.root = root
        meta = json.loads((root / "project.json").read_text(encoding="utf-8"))
        self.classes = meta["classes"]
        self.items = {i["id"]: i for i in json.loads((root / "items.json").read_text(encoding="utf-8"))}

    def label_path(self, item_id):
        return self.root / "labels" / f"{item_id}.json"

    def labels(self, item_id):
        path = self.label_path(item_id)
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return {"status": "todo", "boxes": []}

    def save(self, item_id, data):
        item = self.items[item_id]
        if data.get("status") not in STATUSES or not isinstance(data.get("boxes"), list):
            raise ValueError("expected status in todo/done/skip and a list of boxes")
        boxes = []
        for b in data["boxes"]:
            x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
            x1, x2 = sorted((max(0.0, min(x1, item["width"])), max(0.0, min(x2, item["width"]))))
            y1, y2 = sorted((max(0.0, min(y1, item["height"])), max(0.0, min(y2, item["height"]))))
            if b.get("cls") not in self.classes:
                raise ValueError(f"unknown class {b.get('cls')!r}")
            if x2 - x1 >= 1 and y2 - y1 >= 1:
                boxes.append({"x1": round(x1, 1), "y1": round(y1, 1), "x2": round(x2, 1), "y2": round(y2, 1),
                              "cls": b["cls"]})
        old = self.labels(item_id)
        out = {"status": data["status"], "prelabelled": old.get("prelabelled", False), "boxes": boxes}
        fd, tmp = tempfile.mkstemp(dir=self.root / "labels", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(out, f)
        os.replace(tmp, self.label_path(item_id))
        return out

    def summary(self):
        rows = []
        for item in self.items.values():
            lab = self.labels(item["id"])
            rows.append({**item, "status": lab["status"], "boxes": len(lab["boxes"])})
        return {"classes": self.classes, "items": rows}


def make_handler(project: Project):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):   # quiet: one line per save only
            pass

        def send(self, code, body=b"", ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, obj, code=HTTPStatus.OK):
            self.send(code, json.dumps(obj).encode())

        def item_id(self, prefix, suffix=""):
            raw = self.path.split("?")[0][len(prefix):]
            if suffix:
                if not raw.endswith(suffix):
                    return None
                raw = raw[: -len(suffix)]
            return raw if ID.match(raw) and raw in project.items else None

        def do_GET(self):
            path = self.path.split("?")[0]
            if path in ("/", "/index.html"):
                return self.send(HTTPStatus.OK, UI.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/project":
                return self.send_json(project.summary())
            if path.startswith("/api/labels/"):
                item_id = self.item_id("/api/labels/")
                return self.send_json(project.labels(item_id)) if item_id else self.send_json({"error": "unknown frame"}, 404)
            if path.startswith("/images/"):
                item_id = self.item_id("/images/", ".jpg")
                if item_id:
                    return self.send(HTTPStatus.OK, (project.root / "images" / f"{item_id}.jpg").read_bytes(), "image/jpeg")
            self.send_json({"error": "not found"}, 404)

        def do_PUT(self):
            if not self.path.startswith("/api/labels/"):
                return self.send_json({"error": "not found"}, 404)
            item_id = self.item_id("/api/labels/")
            if not item_id:
                return self.send_json({"error": "unknown frame"}, 404)
            try:
                length = int(self.headers.get("Content-Length", 0))
                if length > 2_000_000:
                    raise ValueError("too large")
                saved = project.save(item_id, json.loads(self.rfile.read(length)))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                return self.send_json({"error": str(e)}, 400)
            print(f"saved {item_id}: {saved['status']}, {len(saved['boxes'])} boxes", flush=True)
            self.send_json(saved)

    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    project = Project(args.project)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(project))
    print(f"{len(project.items)} frames in {args.project}. Open http://localhost:{args.port}  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
