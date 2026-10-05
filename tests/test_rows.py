from pathlib import Path

import cv2
import pytest

from orderflow.axis import calibrate_price_axis
from orderflow.calibrate import calibrate_table
from orderflow.overlays import find_current_price_line
from orderflow.rows import fit_row_grid

DATA = Path(__file__).parent / "data"
EXPECTED_STEP = {1: 1.0, 2: 1.0, 3: 0.55, 4: 5.0, 5: 1.0}
# price shown in the pink line's label on the screenshot (None: no line drawn)
LINE_PRICE = {1: 405.0, 2: None, 3: 266.48, 4: 2885.0, 5: None}


def load(shot):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    axis = calibrate_price_axis(img)
    geo = calibrate_table(img)
    return img, axis, geo


@pytest.mark.parametrize("shot", EXPECTED_STEP)
def test_price_step(shot):
    img, axis, geo = load(shot)
    grid = fit_row_grid(img, axis, geo)
    assert grid.price_step == pytest.approx(EXPECTED_STEP[shot])
    assert 14 < grid.pitch < 30


@pytest.mark.parametrize("shot", EXPECTED_STEP)
def test_row_prices_are_step_multiples(shot):
    img, axis, geo = load(shot)
    grid = fit_row_grid(img, axis, geo)
    step = grid.price_step
    k0 = grid.row_index(300)
    for k in range(k0, k0 + 6):
        price = grid.row_price(k, axis)
        assert price / step == pytest.approx(round(price / step), abs=1e-3)
        # consecutive rows are exactly one step apart
        assert grid.row_price(k, axis) - grid.row_price(k + 1, axis) == pytest.approx(step)


@pytest.mark.parametrize("shot", LINE_PRICE)
def test_current_price_line(shot):
    img, axis, geo = load(shot)
    line = find_current_price_line(img, axis, y_limit=geo.y_top)
    expected = LINE_PRICE[shot]
    if expected is None:
        assert line is None
    else:
        assert line.price == pytest.approx(expected, abs=0.1 * EXPECTED_STEP[shot] + 0.02)
