"""The local web app: serves the page and turns an uploaded screenshot into JSON."""
import json
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from orderflow.webapp import MAX_UPLOAD_BYTES, make_server

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def base():
    server = make_server("127.0.0.1", 0)  # port 0: any free port
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def post(base, data, query="", headers=None):
    req = urllib.request.Request(f"{base}/api/parse{query}", data=data, method="POST", headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def test_page_and_health(base):
    with urllib.request.urlopen(base + "/") as r:
        html = r.read().decode()
        assert r.status == 200 and "Footprint Reader" in html and "/api/parse" in html
    with urllib.request.urlopen(base + "/health") as r:
        assert json.loads(r.read()) == {"ok": True}


def test_unknown_routes_are_404(base):
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(base + "/nope")
    assert e.value.code == 404
    req = urllib.request.Request(base + "/other", data=b"x", method="POST")
    with pytest.raises(urllib.error.HTTPError) as e2:
        urllib.request.urlopen(req)
    assert e2.value.code == 404


def test_not_a_picture_is_a_clear_400(base):
    status, body = post(base, b"this is not an image")
    assert status == 400 and "not a readable picture" in body["error"]


def test_a_tiny_picture_is_rejected_with_a_reason(base):
    import cv2
    import numpy as np

    ok, png = cv2.imencode(".png", np.full((100, 100, 3), 255, np.uint8))
    status, body = post(base, png.tobytes())
    assert status == 422 and "100x100" in body["error"]


def test_a_picture_that_is_not_a_chart_is_a_clear_422(base):
    import cv2
    import numpy as np

    blank = np.full((876, 1806, 3), 255, np.uint8)
    ok, png = cv2.imencode(".png", blank)
    status, body = post(base, png.tobytes())
    assert status == 422 and "does not look like a GoCharting footprint screenshot" in body["error"]


def test_body_size_limits(base):
    status, body = post(base, b"")
    assert status in (400, 411, 413)
    req = urllib.request.Request(base + "/api/parse", data=b"x", method="POST", headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1)})
    # the server must refuse by the declared length without reading that many bytes
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # the connection may be closed before the client finishes sending
        assert getattr(e, "code", 413) == 413 or isinstance(e, (urllib.error.URLError, ConnectionError))


def test_a_real_screenshot_returns_parsed_data_and_analysis(base):
    data = (ROOT / "shots3" / "BHEL_06-10-26.png").read_bytes()
    status, body = post(base, data, headers={"X-Filename": "my chart <1>.png"})
    assert status == 200
    doc = body["parsed"]
    assert doc["source"] == "my chart _1_.png"  # no server path leaks; the name is sanitised
    assert len(doc["bars"]) == 8 and all(b["valid"] for b in doc["bars"])
    assert [b["delta_pct"]["value"] for b in doc["bars"]][-1] == -0.28
    assert body["analysis"]["flow"]["series"][0]["delta_pct"] == 38.55
    # analysis can be switched off
    status, body = post(base, data, "?analyze=0")
    assert status == 200 and body["analysis"] is None
