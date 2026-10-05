from pathlib import Path

import cv2
import pytest

from orderflow.calibrate import calibrate_table

DATA = Path(__file__).parent / "data"
# shot -> expected number of bar columns (counted by eye from the screenshots)
EXPECTED_COLS = {1: 13, 2: 4, 3: 7, 4: 4, 5: 7}


@pytest.mark.parametrize("shot,cols", EXPECTED_COLS.items())
def test_table_columns(shot, cols):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    geo = calibrate_table(img)
    assert geo.n_cols == cols
    assert 20 <= geo.row_height <= 30


def test_clipped_first_column():
    geo = calibrate_table(cv2.imread(str(DATA / "shot5.png")))
    assert geo.first_clipped
    first, second = geo.col_x(0), geo.col_x(1)
    assert first[1] - first[0] < geo.pitch  # partial column
    assert abs((second[1] - second[0]) - geo.pitch) < 2


@pytest.mark.parametrize("shot", [1, 2, 3, 4])
def test_unclipped(shot):
    assert not calibrate_table(cv2.imread(str(DATA / f"shot{shot}.png"))).first_clipped
