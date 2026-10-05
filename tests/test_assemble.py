import json
import subprocess
import sys
from pathlib import Path

import pytest

from orderflow.assemble import infer_times, parse_screenshot
from orderflow.models import load_classifiers
from orderflow.timeaxis import BarLabel

DATA = Path(__file__).parent / "data"
MODELS = load_classifiers()


@pytest.fixture(scope="module")
def parsed():
    return {n: parse_screenshot(DATA / f"shot{n}.png", MODELS) for n in range(1, 6)}


def test_json_serialisable(parsed):
    for doc in parsed.values():
        assert json.loads(json.dumps(doc)) == doc


@pytest.mark.parametrize(
    "shot,step,current",
    [(1, 1.0, 405.0), (2, 1.0, None), (3, 0.55, 266.48), (4, 5.0, 2885.0), (5, 1.0, None)],
)
def test_price_block(parsed, shot, step, current):
    price = parsed[shot]["price"]
    assert price["step_per_row"] == step
    if current is None:
        assert price["current"] is None
    else:
        assert price["current"] == pytest.approx(current, abs=0.1 * step + 0.02)


@pytest.mark.parametrize(
    "shot,bars,valid,flagged,skipped,flagged_prices",
    [
        (1, 13, 12, [11], [], [405.0]),  # live bar's cells sit under the translucent profile
        (2, 4, 4, [], [], []),
        (3, 7, 7, [], [], []),
        (4, 4, 4, [], [], []),
        (5, 7, 5, [], [0, 1], []),  # clipped first bar; bar 1 has no cells in the price range
    ],
)
def test_summaries(parsed, shot, bars, valid, flagged, skipped, flagged_prices):
    s = parsed[shot]["summary"]
    assert (s["bars"], s["valid_bars"]) == (bars, valid)
    assert s["flagged_bars"] == flagged
    assert s["skipped_bars"] == skipped
    assert s["flagged_profile_prices"] == flagged_prices


def test_times_and_sessions(parsed):
    def times(n):
        return [(b["time"], b["time_inferred"], b["new_session"]) for b in parsed[n]["bars"]]

    assert times(2) == [("09:15", True, True), ("09:45", False, False), ("10:15", False, False), ("10:45", False, False)]
    # previous-day bars come first, then the session-start bar whose time is inferred
    assert [t[0] for t in times(3)] == ["14:45", "15:15", "09:15", "09:45", "10:15", "10:45", "11:15"]
    assert times(3)[2] == ("09:15", True, True)
    assert times(1)[-1][0] == "15:15" and len(times(1)) == 13


def test_bar_contents_shot2(parsed):
    bar = parsed[2]["bars"][0]
    assert bar["volume"]["value"] == 196510 and bar["delta"]["value"] == 19810
    assert bar["poc_price"] == 507.0
    top = bar["cells"][1]  # 512: '184 X 12K', red bid and blue ask
    assert (top["price"], top["bid"], top["ask"]) == (512.0, 184.0, 12000.0)
    assert top["sell_imbalance"] and top["buy_imbalance"]
    prices = [c["price"] for c in bar["cells"]]
    assert prices == sorted(prices, reverse=True)
    assert any(c["is_poc"] and c["price"] == 507.0 for c in bar["cells"])


def test_ellipsis_cells_are_marked_hidden(parsed):
    hidden = [c for b in parsed[1]["bars"] for c in b["cells"] if c.get("hidden")]
    assert len(hidden) == 5  # the "..." rows in shot 1: cols 1 (x2), 2, 9 and 12


def test_profile_rows(parsed):
    prof = parsed[2]["profile"]
    assert [p["delta"] for p in prof][:3] == [0.0, 12000.0, -945.0]
    assert all(p["valid"] for p in prof)


def test_infer_times_extrapolates_and_handles_gaps():
    mk = lambda m: BarLabel("x", m, None if m is not None else "5 Oct 26", 1.0)
    labels = [mk(None), mk(585), mk(615), mk(645)]
    assert [t for t, _ in infer_times(labels)] == [555, 585, 615, 645]
    labels = [mk(585), mk(615), mk(None)]
    assert [t for t, _ in infer_times(labels)] == [585, 615, 645]


def test_cli_prints_json():
    out = subprocess.run(
        [sys.executable, "-m", "orderflow", str(DATA / "shot4.png"), "--compact"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert json.loads(out)["summary"]["bars"] == 4
