"""A small local web app: pick a footprint screenshot, get the data back.

    python -m orderflow.webapp [--host 127.0.0.1] [--port 8000]

The page (`web/index.html`) posts the raw image bytes to `POST /api/parse` and renders the JSON that
`parse_screenshot` + `analyze` produce. Standard library only (no web framework); meant for one person on their own
machine, not for exposure to the internet: there is no authentication, and reading a chart takes 10-30 s of CPU.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from .analytics import analyze
from .assemble import parse_screenshot
from .models import load_split_classifiers

PAGE = Path(__file__).parent / "web" / "index.html"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MIN_SIZE = (800, 400)  # (width, height) of anything that could hold a footprint chart
MAX_CONCURRENT_READS = 2  # reading is CPU bound; more at once only makes everyone slower

_slots = threading.BoundedSemaphore(MAX_CONCURRENT_READS)
_models = None
_models_lock = threading.Lock()


def _split_models():
    """The split-layout classifiers, loaded once (they take a moment to read from disk)."""
    global _models
    with _models_lock:
        if _models is None:
            _models = load_split_classifiers()
        return _models


class ReadError(Exception):
    """A problem with the uploaded picture, reported to the user as is."""

    def __init__(self, message: str, status: HTTPStatus = HTTPStatus.UNPROCESSABLE_ENTITY):
        super().__init__(message)
        self.status = status


def _jsonable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not serialisable: {type(obj)}")


def read_chart(data: bytes, filename: str = "screenshot.png", with_analysis: bool = True) -> dict:
    """Parse the bytes of a screenshot; returns {"parsed": ..., "analysis": ... | None}."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ReadError("That file is not a readable picture (use a PNG or JPEG screenshot).", HTTPStatus.BAD_REQUEST)
    h, w = img.shape[:2]
    if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
        raise ReadError(f"The picture is {w}x{h} px; a footprint chart screenshot is at least {MIN_SIZE[0]}x{MIN_SIZE[1]}.")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "upload.png"
        cv2.imwrite(str(path), img)  # re-encoded: the parser reads from a path, and this normalises the format
        try:
            doc = parse_screenshot(path, _split_models())
        except (ValueError, IndexError, ZeroDivisionError) as exc:
            raise ReadError(
                "This does not look like a GoCharting footprint screenshot this reader supports "
                f"({exc}). See docs/CHART_SETTINGS.md for the chart settings it was built for."
            ) from exc
    doc["source"] = re.sub(r"[^\w .\-]", "_", filename)[:80]
    analysis = None
    if with_analysis:
        try:
            analysis = analyze(doc)
        except Exception as exc:  # the data is still worth returning
            analysis = {"error": f"analysis failed: {exc}"}
    return {"parsed": doc, "analysis": analysis}


class Handler(BaseHTTPRequestHandler):
    server_version = "orderflow"

    def _send(self, status: HTTPStatus, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: HTTPStatus, payload: dict) -> None:
        self._send(status, json.dumps(payload, default=_jsonable).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        route = urlparse(self.path).path
        if route in ("/", "/index.html"):
            self._send(HTTPStatus.OK, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif route == "/favicon.ico":
            self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        elif route == "/health":
            self._json(HTTPStatus.OK, {"ok": True})
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path != "/api/parse":
            return self._json(HTTPStatus.NOT_FOUND, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            return self._json(HTTPStatus.LENGTH_REQUIRED, {"error": "send the picture as the request body"})
        if length <= 0 or length > MAX_UPLOAD_BYTES:
            return self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                              {"error": f"the picture must be between 1 byte and {MAX_UPLOAD_BYTES // 2**20} MB"})
        data = self.rfile.read(length)
        analyse = parse_qs(url.query).get("analyze", ["1"])[0] != "0"
        name = self.headers.get("X-Filename", "screenshot.png")
        if not _slots.acquire(timeout=120):
            return self._json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "the reader is busy, try again in a minute"})
        try:
            result = read_chart(data, name, analyse)
        except ReadError as exc:
            return self._json(exc.status, {"error": str(exc)})
        except Exception as exc:  # never leave the browser hanging on a bug
            return self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"unexpected error: {exc}"})
        finally:
            _slots.release()
        self._json(HTTPStatus.OK, result)

    def log_message(self, fmt: str, *args) -> None:  # quieter than the default
        print(f"{self.address_string()} {fmt % args}")


def make_server(host: str = "127.0.0.1", port: int = 8000) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    ap = argparse.ArgumentParser(description="Local web app for the footprint screenshot reader")
    ap.add_argument("--host", default="127.0.0.1", help="interface to listen on (default: this machine only)")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    _split_models()  # load before the first request, not during it
    server = make_server(args.host, args.port)
    print(f"Footprint reader on http://{args.host}:{server.server_port}/  (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
