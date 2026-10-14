"""The in-browser black-area detector (browser/black-areas.js), run in headless Chromium.

Skipped when Playwright or a Chromium build is not available (it is not a dependency of the Python package).
"""
import base64
from pathlib import Path

import pytest

playwright = pytest.importorskip("playwright.sync_api")

from orderflow.split import load_split_shot  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "browser" / "black-areas.js"


@pytest.fixture(scope="module")
def page():
    with playwright.sync_playwright() as p:
        exe = Path("/opt/pw-browsers/chromium")
        try:
            browser = p.chromium.launch(executable_path=str(exe)) if exe.exists() else p.chromium.launch()
        except Exception as e:  # no browser installed here
            pytest.skip(f"no Chromium available: {e}")
        pg = browser.new_page()
        pg.goto("about:blank")
        pg.add_script_tag(path=str(SCRIPT))
        yield pg
        browser.close()


def scan(page, image: Path, roi, **opts):
    data = base64.b64encode(image.read_bytes()).decode()
    return page.evaluate(
        """async ([data, roi, opts]) => {
             const im = new Image(); im.src = 'data:image/png;base64,' + data; await im.decode();
             return BlackAreasCore.scanSource(im, Object.assign({roi}, opts));
           }""", [data, roi, opts])


def test_solid_black_poc_boxes_of_the_trading_page_screenshot(page):
    r = scan(page, ROOT / "tests" / "data" / "black_solid_poc.png", {"x": 0, "y": 46, "w": 1790, "h": 600}, colors=["black"])
    got = [(b["x"], b["y"], b["w"], b["h"], b["kind"]) for b in r["boxes"]]
    assert got == [(792, 400, 85, 12, "solid"), (386, 452, 85, 10, "solid")]


@pytest.mark.parametrize("name,min_recall", [("split1", 1.0), ("split10", 0.9), ("split13", 0.7)])
def test_black_frames_match_the_python_poc_detector(page, name, min_recall):
    """The POC of the split layout is a black frame; the browser detector should find most of what the
    Python reader finds (a frame whose edges are hidden under a price line can be missed)."""
    image = ROOT / "tests" / "data" / f"{name}.png"
    geo = load_split_shot(image)
    poc = geo.find_poc_rows()
    r = scan(page, image, {"x": 0, "y": 46, "w": 1790, "h": geo.table.y_top - 46})
    hit = 0
    for col, row in poc.items():
        x0, x1 = geo.table.col_x(col)
        cx, cy = (x0 + x1) / 2, geo.grid.row_top(row) + geo.grid.pitch / 2
        hit += any(b["x"] - 6 <= cx <= b["x"] + b["w"] + 6 and b["y"] - 6 <= cy <= b["y"] + b["h"] + 6 for b in r["boxes"])
    assert hit / len(poc) >= min_recall, f"{hit} of {len(poc)} frames"
    # and nothing else: every box sits on a POC row
    for b in r["boxes"]:
        assert any(abs(b["y"] + b["h"] / 2 - (geo.grid.row_top(row) + geo.grid.pitch / 2)) < 20 for row in poc.values()), b


def test_transparent_pixels_are_not_black(page):
    n = page.evaluate("""() => {
        const w = 800, h = 400, data = new Uint8ClampedArray(w * h * 4);   // all (0, 0, 0, 0): a transparent layer
        return BlackAreasCore.detectBlackAreas(data, w, h, {}).length; }""")
    assert n == 0


def test_a_small_black_box_and_a_thin_line(page):
    boxes = page.evaluate("""() => {
        const w = 400, h = 200, data = new Uint8ClampedArray(w * h * 4).fill(255);
        const set = (x, y) => { const i = (y * w + x) * 4; data[i] = data[i + 1] = data[i + 2] = 0; };
        for (let y = 50; y < 62; y++) for (let x = 100; x < 190; x++) set(x, y);   // 90x12 solid
        for (let x = 0; x < 400; x++) set(x, 150);                                 // a 1 px rule: not an area
        for (let y = 100; y < 106; y++) for (let x = 20; x < 30; x++) set(x, y);   // 10x6: too small
        return BlackAreasCore.detectBlackAreas(data, w, h, {}); }""")
    assert [(b["x"], b["y"], b["w"], b["h"], b["kind"]) for b in boxes] == [(100, 50, 90, 12, "solid")]


def test_detection_is_fast(page):
    ms = page.evaluate("""() => {
        const w = 1800, h = 860, data = new Uint8ClampedArray(w * h * 4).fill(255);
        const t0 = performance.now();
        for (let k = 0; k < 10; k++) BlackAreasCore.detectBlackAreas(data, w, h, {});
        return (performance.now() - t0) / 10; }""")
    assert ms < 30, f"{ms:.1f} ms per 1800x860 scan"


def test_a_previous_canvas_does_not_bleed_into_the_next_scan(page):
    """Pages draw a chart in several stacked, partly transparent canvases and the detector scans them one after
    another through one scratch canvas: a transparent canvas must come back empty, not show the previous one."""
    res = page.evaluate("""() => {
        const make = (draw) => { const c = document.createElement('canvas'); c.width = 600; c.height = 300; draw(c.getContext('2d')); return c; };
        const withBox = make((g) => { g.fillStyle = '#fff'; g.fillRect(0, 0, 600, 300); g.fillStyle = '#000'; g.fillRect(100, 100, 90, 14); });
        const empty = make(() => {});                                    // fully transparent
        const textOnly = make((g) => { g.fillStyle = '#000'; g.font = '12px sans-serif'; g.fillText('1,980 2,000', 20, 40); });
        const r1 = BlackAreasCore.scanSource(withBox, {}).boxes.length;
        const r2 = BlackAreasCore.scanSource(empty, {}).boxes.length;      // would still see the first canvas's box
        const r3 = BlackAreasCore.scanSource(textOnly, {}).boxes.length;   // text is not an area
        return [r1, r2, r3]; }""")
    assert res == [1, 0, 0]


def test_dark_green_areas(page):
    """The dark end of the buy colour scale (0, 83, 0) is found; the bright green box next to it is not."""
    r = scan(page, ROOT / "tests" / "data" / "dark_green_poc.png", {"x": 0, "y": 46, "w": 1280, "h": 600})
    got = [(b["x"], b["y"], b["w"], b["h"], b["kind"], b["color"]) for b in r["boxes"]]
    assert got == [(160, 356, 65, 23, "solid", "darkgreen")]


def test_colors_can_be_chosen(page):
    image = ROOT / "tests" / "data" / "black_solid_poc.png"
    roi = {"x": 0, "y": 46, "w": 1790, "h": 600}
    both = scan(page, image, roi)["boxes"]
    black = scan(page, image, roi, colors=["black"])["boxes"]
    green = scan(page, image, roi, colors=["darkgreen"])["boxes"]
    assert {b["color"] for b in both} == {"black", "darkgreen"}
    assert [(b["x"], b["y"], b["w"], b["h"]) for b in black] == [(792, 400, 85, 12), (386, 452, 85, 10)]
    assert [(b["x"], b["y"]) for b in green] == [(81, 412), (182, 438)]


def test_both_colours_scan_is_still_fast(page):
    ms = page.evaluate("""() => {
        const w = 1800, h = 860, data = new Uint8ClampedArray(w * h * 4).fill(255);
        const t0 = performance.now();
        for (let k = 0; k < 10; k++) BlackAreasCore.detectAreas(data, w, h, {});
        return (performance.now() - t0) / 10; }""")
    assert ms < 40, f"{ms:.1f} ms per 1800x860 scan for both colours"


def test_the_mid_dark_green_of_the_second_candle(page):
    """(0, 136, 0) is the next step up the buy scale from (0, 83, 0): included; the light greens and the teal candles are not."""
    r = scan(page, ROOT / "tests" / "data" / "green_second_candle.png", {"x": 0, "y": 46, "w": 840, "h": 600})
    got = [(b["x"], b["y"], b["w"], b["h"], b["color"]) for b in r["boxes"]]
    assert got == [(119, 356, 59, 36, "darkgreen")]
