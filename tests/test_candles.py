from pathlib import Path

import pytest

from orderflow.assemble import parse_screenshot
from orderflow.candles import bucket
from orderflow.models import load_classifiers
from orderflow.pipeline import load_shot
from orderflow.candles import find_candles

DATA = Path(__file__).parent / "data"
MODELS = load_classifiers()


@pytest.fixture(scope="module")
def docs():
    return {n: parse_screenshot(DATA / f"shot{n}.png", MODELS) for n in range(1, 6)}


def test_bucket_snaps_to_the_row_containing_the_price():
    assert bucket(512.5, 1.0) == 513.0  # (512, 513]
    assert bucket(512.0, 1.0) == 512.0  # exactly on the boundary belongs to the lower row
    assert bucket(2880.5, 5.0, jitter=0.25) == 2885.0  # (2880, 2885]
    assert bucket(512.03, 1.0, jitter=0.05) == 512.0  # within pixel jitter above a boundary
    assert bucket(272.0, 0.55) == pytest.approx(272.25)  # rows are multiples of 0.55


@pytest.mark.parametrize("shot,present", [(1, 13), (2, 4), (3, 7), (4, 4), (5, 5)])
def test_candle_per_visible_bar(docs, shot, present):
    ohlc = [b["ohlc"] for b in docs[shot]["bars"]]
    assert sum(o is not None for o in ohlc) == present
    if shot == 5:  # the first two bars are clipped / off the left edge
        assert ohlc[0] is None and ohlc[1] is None


@pytest.mark.parametrize("shot", [1, 2, 3, 4, 5])
def test_ohlc_is_internally_consistent(docs, shot):
    for b in docs[shot]["bars"]:
        o = b["ohlc"]
        if o is None:
            continue
        assert o["low"] <= min(o["open"], o["close"]) + 1e-9
        assert max(o["open"], o["close"]) <= o["high"] + 1e-9
        if o["body_px"] > 0:
            assert (o["close"] > o["open"]) == (o["direction"] == "up")
        assert o["high_row"] >= o["low_row"]


@pytest.mark.parametrize("shot", [2, 3, 4, 5])
def test_candles_chain_and_cover_the_traded_rows(docs, shot):
    d = docs[shot]
    assert d["summary"]["flagged_ohlc"] == []
    for b in d["bars"]:
        o = b["ohlc"]
        traded = [c["price"] for c in b["cells"] if c.get("total", 0) > 0]
        if o and traded:
            assert o["high_clipped"] or o["high_row"] >= max(traded) - 1e-6
            assert o["low_clipped"] or o["low_row"] <= min(traded) + 1e-6


def test_each_open_chains_from_the_previous_close_within_a_session(docs):
    for shot in (1, 2, 3, 4, 5):
        d = docs[shot]
        step = d["price"]["step_per_row"]
        bars = d["bars"]
        for prev, cur in zip(bars, bars[1:]):
            if not (prev["ohlc"] and cur["ohlc"]) or cur["new_session"] or not cur["ohlc"]["valid"]:
                continue
            assert abs(cur["ohlc"]["open"] - prev["ohlc"]["close"]) <= 0.8 * step + 0.3, (shot, cur["index"])


def test_shot1_live_bar_candle_is_flagged_not_trusted(docs):
    assert docs[1]["summary"]["flagged_ohlc"] == [12]
    live = docs[1]["bars"][12]["ohlc"]
    assert not live["valid"] and live["checks"][0]["name"] == "continuity"


def test_session_gap_is_allowed(docs):
    # shot 3: bar 1 closes 271.01 yesterday, bar 2 opens 270.02 (overnight gap), not flagged
    bars = docs[3]["bars"]
    assert bars[2]["new_session"] and abs(bars[2]["ohlc"]["open"] - bars[1]["ohlc"]["close"]) > 0.9
    assert bars[2]["ohlc"]["valid"]


def test_known_candles_shot2(docs):
    b0, b1, b2, b3 = (b["ohlc"] for b in docs[2]["bars"])
    assert (b0["direction"], b1["direction"], b2["direction"], b3["direction"]) == ("up", "up", "down", "down")
    assert b0["low_clipped"] and not b0["high_clipped"]  # the wick runs off the bottom of the plot
    assert b1["open"] == pytest.approx(505.55, abs=0.15) and b1["close"] == pytest.approx(506.14, abs=0.15)
    assert b3["high"] == pytest.approx(504.25, abs=0.15) and b3["low"] == pytest.approx(501.29, abs=0.15)


def test_candles_found_directly():
    geo = load_shot(DATA / "shot4.png")
    candles = find_candles(geo)
    assert [c.direction for c in candles] == ["up", "up", "down", "up"]
