from pathlib import Path

import cv2
import pytest

from orderflow.calibrate import calibrate_table

DATA = Path(__file__).parent / "data"
# shot -> expected number of bar columns (counted by eye from the screenshots)
EXPECTED_COLS = {1: 13, 2: 4, 3: 7, 4: 4}


@pytest.mark.parametrize("shot,cols", EXPECTED_COLS.items())
def test_table_columns(shot, cols):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    geo = calibrate_table(img)
    assert geo.n_cols == cols
    assert 20 <= geo.row_height <= 30
