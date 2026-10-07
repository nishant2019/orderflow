"""Regression guard on a fixed sample of the uploaded GoCharting screenshots (shots/, shots2/, shots3/).

These images have no hand-read ground truth (apart from the table text in table_truth.json), so the test pins
what the parser is trusted to deliver: bar counts, the row price step, which optional parts the chart has, and a
floor on how many bars pass every cross-check. `tests/data/shots_regression.json` holds the recorded values;
raise the floors when the parser improves, never lower them to make a test pass.

The sample covers the three folders and the font sizes seen: 3-row and 4-row summary tables (Delta %), small text
(EBGNG, GRAPHITE, NUVAMA), early-session charts with 3 or 8 wide columns, `M` volumes, thousands separators.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from orderflow.assemble import parse_screenshot

ROOT = Path(__file__).resolve().parents[1]
RECORDED = json.loads((ROOT / "tests" / "data" / "shots_regression.json").read_text())


def _path(key: str) -> Path:
    return ROOT / f"{key}_06-10-26.png"


def _parse(key):
    return key, parse_screenshot(_path(key))


@pytest.fixture(scope="module")
def docs():
    with ProcessPoolExecutor(4) as pool:  # sixteen screenshots at several seconds each: do them side by side
        return dict(pool.map(_parse, sorted(RECORDED)))


@pytest.mark.parametrize("key", sorted(RECORDED))
def test_structure(docs, key):
    rec, d = RECORDED[key], docs[key]
    assert d["layout"] == "split"
    assert len(d["bars"]) == rec["bars"]
    assert sum(bool(b.get("clipped")) for b in d["bars"]) == rec["clipped"]
    assert d["price"]["step_per_row"] == rec["step"]
    assert d["imbalance"]["shown"] == rec["imbalance_shown"]
    assert (d["bars"][0]["delta_pct"] is not None) == rec["delta_pct"]
    assert ("cum_delta_pane" in d) == rec["has_cum_pane"]


@pytest.mark.parametrize("key", sorted(RECORDED))
def test_valid_bars_floor(docs, key):
    got = sum(b["valid"] for b in docs[key]["bars"])
    assert got >= RECORDED[key]["valid"], f"{key}: {got} valid bars, recorded {RECORDED[key]['valid']}"


@pytest.mark.parametrize("key", sorted(RECORDED))
def test_cum_delta_pane_and_divergence_markers_are_present_and_consistent(docs, key):
    d = docs[key]
    assert d["cum_delta_pane"]["invalid_bars"] == []
    for b in d["bars"]:
        if b["platform_divergence"] and b["ohlc"] and b["delta"]["value"] is not None:
            up = b["ohlc"]["direction"] == "up"
            assert (b["platform_divergence"] == "red") == (up and b["delta"]["value"] < 0), (key, b["index"])


def test_four_row_table_has_a_checked_delta_pct(docs):
    for key in (k for k in RECORDED if k.startswith("shots3/")):
        for b in docs[key]["bars"]:
            assert b["delta_pct"] is not None
            chk = [c for c in b["checks"] if c["name"] == "pct"]
            assert chk and all(c["status"] != "fail" for c in chk), (key, b["index"], chk)
