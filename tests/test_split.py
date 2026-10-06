"""Split-box layout (bid | ask boxes, black POC rectangle, delta|volume profile).

Everything here is regression-tested on nine screenshots; the cross-checks (cells vs table vs
profile vs candles vs imbalance rule) are the real safety net, so most tests assert that they hold.
"""
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from orderflow.assemble import parse_screenshot
from orderflow.calibrate import calibrate_table
from orderflow.cells import CellText
from orderflow.layout import detect_layout
from orderflow.models import load_split_classifiers
from orderflow.pipeline import ShotGeometry
from orderflow.profile2 import read_profile2
from orderflow.repair import infer_unreadable
from orderflow.split import SplitShotGeometry, load_split_shot
from orderflow.table import read_table
from orderflow.validate import validate_columns

DATA = Path(__file__).parent / "data"
STEMS = [f"split{k}" for k in range(1, 14)]
STEPS = dict(zip(STEMS, [0.9, 0.55, 0.7, 1.0, 15.0, 15.0, 2.0, 5.0, 2.0, 5.0, 0.85, 5.0, 0.65]))
BARS = dict(zip(STEMS, [13, 13, 13, 13, 5, 13, 13, 13, 13, 15, 13, 13, 14]))
CLIPPED = {"split10": [0], "split13": [0]}  # bars cut off by the left image edge: skipped, never failed
MODELS = load_split_classifiers()


@pytest.fixture(scope="module")
def docs():
    return {s: parse_screenshot(DATA / f"{s}.png", MODELS) for s in STEMS}


@pytest.fixture(scope="module")
def geos():
    return {s: load_split_shot(DATA / f"{s}.png") for s in STEMS}


def test_layout_is_detected():
    for stem in STEMS:
        img = cv2.imread(str(DATA / f"{stem}.png"))
        assert detect_layout(img, calibrate_table(img)) == "split", stem
    for n in range(1, 6):
        img = cv2.imread(str(DATA / f"shot{n}.png"))
        assert detect_layout(img, calibrate_table(img)) == "single", n


@pytest.mark.parametrize("stem", STEMS)
def test_row_grid_and_step(geos, docs, stem):
    assert geos[stem].grid.price_step == STEPS[stem]
    assert docs[stem]["layout"] == "split" and docs[stem]["price"]["step_per_row"] == STEPS[stem]
    # row prices are multiples of the step
    g = geos[stem]
    for r in range(g.k_first, g.k_first + 5):
        price = g.grid.row_price(r, g.axis)
        assert price / g.grid.price_step == pytest.approx(round(price / g.grid.price_step), abs=1e-3)


@pytest.mark.parametrize("stem", STEMS)
def test_every_bar_validates(docs, stem):
    d = docs[stem]
    assert d["summary"]["bars"] == BARS[stem]
    assert d["summary"]["flagged_bars"] == [] and d["summary"]["skipped_bars"] == CLIPPED.get(stem, [])
    assert d["summary"]["valid_bars"] == BARS[stem] - len(CLIPPED.get(stem, []))
    assert d["summary"]["flagged_ohlc"] == []


@pytest.mark.parametrize("stem", STEMS)
def test_poc_box_is_the_largest_cell_of_every_bar(docs, stem):
    for b in docs[stem]["bars"]:
        if b["index"] in CLIPPED.get(stem, []):
            continue
        cells = [c for c in b["cells"] if "total" in c]
        assert b["poc_price"] is not None
        poc = next(c for c in cells if c["price"] == b["poc_price"])
        assert poc["is_poc"]
        assert poc["total"] >= max(c["total"] for c in cells) * 0.97 - 1  # displayed K values are rounded


@pytest.mark.parametrize("stem", STEMS)
def test_imbalance_flags_follow_the_300_percent_rule(docs, stem):
    for b in docs[stem]["bars"]:
        if b["index"] in CLIPPED.get(stem, []):
            continue
        if not docs[stem]["imbalance"]["shown"]:
            continue  # flags are computed from the rule here, so there is nothing drawn to check
        assert any(c["name"] == "imbalance" and c["status"] == "ok" for c in b["checks"]), (stem, b["index"])


@pytest.mark.parametrize("stem", STEMS)
def test_candles_present_and_valid(docs, stem):
    for b in docs[stem]["bars"]:
        o = b["ohlc"]
        if (stem == "split2" and b["index"] == 12) or b["index"] in CLIPPED.get(stem, []):
            continue  # no candle in the visible window / bar cut off by the image edge
        assert o is not None and o["valid"], (stem, b["index"])
        assert o["low"] <= min(o["open"], o["close"]) + 1e-9 and max(o["open"], o["close"]) <= o["high"] + 1e-9


def test_the_pink_line_is_the_profile_poc_not_the_last_close(docs):
    seen = 0
    for stem in STEMS:
        d = docs[stem]
        line, ps = d["price"]["poc_line"], d.get("profile_summary")
        if line is None or ps is None:
            continue
        seen += 1
        assert abs(line - ps["poc"]) <= 0.1 * d["price"]["step_per_row"], stem
    assert seen == 12
    # ... while it is often NOT the last close (up to ~9 rows away): it marks the profile POC
    gaps = [abs(docs[s]["price"]["poc_line"] - docs[s]["price"]["last_close"]) / docs[s]["price"]["step_per_row"]
            for s in STEMS if docs[s]["price"]["poc_line"]]
    assert max(gaps) > 8 and sum(g > 2 for g in gaps) >= 3


def test_profile_without_a_profile_is_empty(docs):
    assert docs["split9"]["profile"] == [] and "profile_summary" not in docs["split9"]


@pytest.mark.parametrize("stem", [s for s in STEMS if s != "split9"])
def test_profile_structure(docs, stem):
    d = docs[stem]
    rows = d["profile"]
    ps = d["profile_summary"]
    assert [r["price"] for r in rows] == sorted((r["price"] for r in rows), reverse=True)
    peaks = [r for r in rows if r["zone"] == "peak"]
    assert len(peaks) == 1 and peaks[0]["price"] == ps["poc"]
    assert peaks[0]["volume"] == max(r["volume"] for r in rows)
    va = [r["price"] for r in rows if r["in_value_area"]]
    assert ps["value_area_low"] == min(va) and ps["value_area_high"] == max(va) and ps["value_area_low"] <= ps["poc"] <= ps["value_area_high"]
    # (the colour-coded value area is not always contiguous: the platform seems to colour at a finer resolution)


def test_every_profile_row_agrees_with_the_cells(docs):
    assert all(docs[s]["summary"]["flagged_profile_prices"] == [] for s in STEMS)


def test_a_misread_is_corrected_only_when_the_checks_single_out_one_fix(docs):
    # split3, 167.3: a bold blue '48K' used to be read as '43K' (a 3/8 confusion); with the retrained
    # classifier it reads right, and the final value must be right either way
    d = docs["split3"]
    cell = next(c for b in d["bars"] for c in b["cells"] if c.get("price") == 167.3 and c.get("ask_text") == "48K")
    assert cell["ask"] == 48000.0
    # split12, 1645: a '48' read as '43', found from the column total alone (the profile row is too coarse)
    d12 = docs["split12"]
    assert d12["summary"]["corrected_cells"] == 1
    cell = next(c for b in d12["bars"] for c in b["cells"] if c.get("corrected_from"))
    assert (cell["price"], cell["bid_text"], cell["bid"]) == (1645.0, "48", 48.0)
    assert all(docs[s]["summary"]["corrected_cells"] == 0 for s in STEMS if s != "split12")


# --- hand-read ground truth ------------------------------------------------------------------

HAND = {  # (stem, col) -> [(price, bid, ask)] read by eye from the screenshots
    ("split1", 0): [(298.8, 0, 3100), (297.9, 0, 6000), (297.0, 6400, 28000), (296.1, 19000, 3600),
                    (295.2, 6500, 2800), (294.3, 26000, 11000), (293.4, 2300, 2600), (292.5, 987, 0)],
    ("split1", 1): [(296.1, 0, 2700), (295.2, 8600, 8300), (294.3, 7500, 963), (293.4, 1100, 271)],
    ("split2", 2): [(364.1, 1700, 1300), (363.55, 3700, 4300), (363.0, 17000, 20000), (362.45, 52000, 54000),
                    (361.9, 34000, 17000), (361.35, 0, 0)],
}


@pytest.mark.parametrize("key", sorted(HAND))
def test_cells_match_hand_reading(geos, key):
    stem, col = key
    g = geos[stem]
    got = [(round(g.grid.row_price(r, g.axis), 2), c.bid, c.ask) for r, c in g.read_cells(col, MODELS.cells)]
    assert got == [(p, float(b), float(a)) for p, b, a in HAND[key]]


def test_blue_17k_on_orange_is_not_read_as_12k(geos):
    """The row/column checks caught this 7/2 confusion once; keep it fixed."""
    g = geos["split2"]
    cells = dict(g.read_cells(2, MODELS.cells))
    row = next(r for r in cells if abs(g.grid.row_price(r, g.axis) - 361.9) < 0.01)
    assert cells[row].ask == 17000 and cells[row].buy_imbalance


def test_table_reads_with_m_suffix(geos):
    g = geos["split3"]
    t = read_table(g.img, g.table, MODELS.table)
    assert (t[0].volume.value, t[7].volume.value, t[11].cum.value, t[12].cum.value) == (1_120_000, 3_480_000, 1_460_000, 1_730_000)


def test_profile_reads_m_values(geos):
    g = geos["split3"]
    rows = read_profile2(g, MODELS.profile)
    assert [r.volume for r in rows][2:7] == [1_300_000, 2_400_000, 1_500_000, 1_200_000, 1_800_000]


# --- inference of covered cells --------------------------------------------------------------

def test_covered_cell_is_solved_from_the_profile_row(docs):
    d = docs["split5"]
    assert d["summary"]["inferred_cells"] == 1
    cell = next(c for c in d["bars"][0]["cells"] if c.get("inferred"))
    assert cell["price"] == 4635.0
    # the text underneath reads "9.4K | 17K"; the answer must be consistent with that, within its stated uncertainty
    assert abs(cell["bid"] - 9400) <= cell["uncertainty"] and abs(cell["ask"] - 17000) <= cell["uncertainty"]
    assert cell["uncertainty"] < 4000


def test_inference_refuses_when_two_cells_on_a_row_are_unknown():
    class FakeCell:
        def __init__(self, bid, ask): self.bid, self.ask, self.bid_tol, self.ask_tol = bid, ask, 0.0, 0.0

    class Rep:
        def __init__(self, col, cells): self.col, self.cells = col, cells

    reports = [Rep(0, [(5, FakeCell(None, None))]), Rep(1, [(5, FakeCell(None, None))])]

    class P:
        row, delta, volume, delta_raw, volume_raw = 5, 100.0, 1000.0, "100", "1000"

    class Value:
        value = None

    class Col:
        clipped = False
        volume = delta = Value()
    assert infer_unreadable(None, reports, [Col(), Col()], [P()]) == []


# --- the checks really catch errors ----------------------------------------------------------

def test_imbalance_check_catches_a_misread_cell(geos):
    g = geos["split1"]
    original = SplitShotGeometry.read_cells

    def corrupted(self, col, clf):
        cells = original(self, col, clf)
        if col == 0:  # the '26K | 11K' row: make the bid tiny so its drawn sell flag is impossible
            for i, (row, c) in enumerate(cells):
                if c.sell_imbalance:
                    cells[i] = (row, CellText(c.raw, 100.0, c.ask, c.sell_imbalance, c.buy_imbalance, False, 1.0, c.bid_tol, c.ask_tol))
        return cells

    SplitShotGeometry.read_cells = corrupted
    try:
        table = read_table(g.img, g.table, MODELS.table)
        rep = validate_columns(g, table, MODELS.cells)[0]
    finally:
        SplitShotGeometry.read_cells = original
    assert any(c.name == "imbalance" and c.status == "fail" for c in rep.checks)
    assert not rep.ok


def test_json_roundtrip(docs):
    for d in docs.values():
        assert json.loads(json.dumps(d)) == d


# --- the variant with pale fills, centre-gap candles, a clipped first bar and a dashed price line ---

def test_dashed_red_line_is_the_current_price(docs):
    # the right-axis tag of split10 reads 1,403.9; the line is measured to within a pixel (0.2)
    assert docs["split10"]["price"]["current"] == pytest.approx(1403.9, abs=0.5)
    assert docs["split10"]["price"]["poc_line"] == pytest.approx(1395.0, abs=0.5)  # the solid pink line, labelled 1395
    for stem in STEMS:
        if stem not in ("split10", "split11", "split12", "split13"):
            assert docs[stem]["price"]["current"] is None  # the other charts draw no dashed line


def test_candle_zone_is_measured_per_image(geos):
    assert all(geos[s].candle_window[0] < -30 for s in ("split10", "split11", "split12", "split13"))  # candles in the centre gap
    assert all(geos[s].candle_window[0] > 0 for s in STEMS if s not in ("split10", "split11", "split12", "split13"))  # elsewhere: at the column's left edge
    assert geos["split10"].box.left[0] <= 3  # boxes start at the column edge (no fixed margin)


def test_clipped_first_bar_cells_are_solved_from_the_profile(docs):
    bar0 = docs["split10"]["bars"][0]
    assert bar0["clipped"] and not bar0["valid"]
    solved = [c for c in bar0["cells"] if c.get("inferred")]
    assert {c["price"] for c in solved} >= {1355.0, 1350.0, 1345.0}
    assert all(c["uncertainty"] < 4000 for c in solved)


def test_profile_text_misread_is_corrected(docs):
    # the top delta label is '8K'; a classifier trained on fewer images read it as '5K' (breaking the row
    # total) and the correction pass fixed it. The final value must be right either way.
    row = next(p for p in docs["split10"]["profile"] if p["price"] == 1405.0)
    assert (row["delta_text"], row["delta"], row["valid"]) == ("8K", 8000.0, True)


def test_the_dashed_line_does_not_corrupt_the_text_under_it(docs):
    top = {c["price"]: c for c in docs["split10"]["bars"][13]["cells"] if "bid" in c}
    assert (top[1405.0]["bid"], top[1405.0]["ask"]) == (19000.0, 51000.0)  # '19K | 51K' sits right under the line
    assert docs["split10"]["bars"][14]["valid"]


def test_divergence_bands_match_candle_vs_delta():
    """The platform's pale bands mark bars whose candle direction opposes the delta sign."""
    from orderflow.assemble import parse_screenshot

    expect = {"split11": {0, 4, 12}, "split12": {1, 3, 12}, "split13": {4, 7, 9}, "split1": set()}
    for stem, idx in expect.items():
        doc = parse_screenshot(f"tests/data/{stem}.png")
        marked = {b["index"] for b in doc["bars"] if b["platform_divergence"]}
        assert marked == idx, stem
        for b in doc["bars"]:
            if b["platform_divergence"]:
                up = b["ohlc"]["direction"] == "up"
                assert (b["platform_divergence"] == "red") == (up and b["delta"]["value"] < 0)


def test_current_price_line_variants(docs):
    """Dashed current-price line: red (split11, split12) or teal (split13), tagged on the axis."""
    for stem, tag in (("split11", 286.4), ("split12", 1660.0), ("split13", 308.5)):
        cur = docs[stem]["price"]["current"]
        assert cur is not None and abs(cur - tag) < 0.3, stem


def test_imbalance_off_charts_use_computed_flags(docs):
    for stem in ("split11", "split12", "split13"):
        assert docs[stem]["imbalance"] == {"shown": False, "source": "computed", "ratio": 3.0}
    assert docs["split1"]["imbalance"]["source"] == "drawn"


def test_cum_delta_pane_candles_chain_through_the_table(docs):
    """Pane candles: open = previous bar's cumulative delta, close = this bar's, within the pixel fit."""
    for stem in STEMS:
        d = docs[stem]
        assert d["cum_delta_pane"]["invalid_bars"] == [], stem
        bars = d["bars"]
        for i, b in enumerate(bars):
            c = b["cum_delta_candle"]
            cum = b["cum_delta"]["value"] if b.get("cum_delta") else None
            if c is None or cum is None or b.get("clipped"):
                continue  # a clipped bar's table text is not trusted
            tol = 3 * d["cum_delta_pane"]["units_per_px"]
            assert abs(c["close"] - cum) <= tol, (stem, i)
            assert c["low"] <= min(c["open"], c["close"]) + 1 and c["high"] >= max(c["open"], c["close"]) - 1
            assert (c["direction"] == "up") == (c["close"] >= c["open"]), (stem, i)


def test_small_text_row_pitch_and_step():
    """EBGNG (shots/): 7 px text, rows 15.4 px apart. Positions are whole pixels (15, 15, 16, ...), which a
    median read as 14.65 px and a step of 0.95 instead of 1.0."""
    g = load_split_shot(Path(__file__).resolve().parents[1] / "shots" / "EBGNG_06-10-26.png")
    assert g.grid.pitch == pytest.approx(15.43, abs=0.05)
    assert g.grid.price_step == 1.0
