import json
from pathlib import Path

import cv2
import pytest

from orderflow.axis import calibrate_price_axis, parse_price

DATA = Path(__file__).parent / "data"
TRUTH = json.loads((DATA / "axis_labels.json").read_text())


@pytest.mark.parametrize("shot", sorted(TRUTH))
def test_axis_fit(shot):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    axis = calibrate_price_axis(img)
    expected = {parse_price(t) for t in TRUTH[shot]}
    read = {p for _, p in axis.labels}
    # every label read must be a true label; only the frame-clipped one may be missing
    assert read <= expected
    assert len(expected - read) <= 1
    assert axis.slope < 0
    assert axis.max_residual_px < 1.0
    # round trip
    for y, price in axis.labels:
        assert abs(axis.y_to_price(y) - price) < abs(axis.slope) * 1.0
        assert abs(axis.price_to_y(price) - y) < 1.0
