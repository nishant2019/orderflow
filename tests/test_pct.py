"""The optional 4th table row, Delta % (= delta / volume * 100), of the shots3 screenshots."""
from pathlib import Path

import pytest

from orderflow.assemble import parse_screenshot
from orderflow.calibrate import calibrate_table
import cv2

ROOT = Path(__file__).resolve().parents[1]
BHEL = ROOT / "shots3" / "BHEL_06-10-26.png"
PCT = [38.55, 8.9, 5.62, 2.75, 17.52, 9.11, 33.93, -0.28]  # hand-read; the sign follows the delta (red cell)


@pytest.fixture(scope="module")
def doc():
    return parse_screenshot(BHEL)


def test_table_row_count_is_detected():
    assert calibrate_table(cv2.imread(str(BHEL))).n_rows == 4
    assert calibrate_table(cv2.imread(str(ROOT / "tests" / "data" / "split1.png"))).n_rows == 3
    assert calibrate_table(cv2.imread(str(ROOT / "shots2" / "BHEL_06-10-26.png"))).n_rows == 3


def test_delta_pct_is_read_and_signed(doc):
    assert [b["delta_pct"]["value"] for b in doc["bars"]] == PCT


def test_delta_pct_matches_delta_over_volume(doc):
    for b in doc["bars"]:
        chk = next(c for c in b["checks"] if c["name"] == "pct")
        assert chk["status"] == "ok", chk
        assert abs(100 * b["delta"]["value"] / b["volume"]["value"] - b["delta_pct"]["value"]) < 0.3


def test_charts_without_the_row_have_no_delta_pct():
    d = parse_screenshot(ROOT / "tests" / "data" / "split1.png")
    assert all(b["delta_pct"] is None for b in d["bars"])
    assert all(c["name"] != "pct" for b in d["bars"] for c in b["checks"])
