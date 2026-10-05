"""Footprint row grid: pitch, phase and price-per-row, measured from the cell bars.

The price step per row is *measured* (row pitch in px x price per px), not assumed:
some charts aggregate several ticks per row, giving steps such as 0.55.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .axis import PriceAxis
from .calibrate import TableGeometry

SAT_MIN, VAL_MIN = 90, 120  # HSV thresholds for saturated bar fills (green/red/orange)
MIN_BAR_W = 8
MIN_BAR_H, MAX_BAR_H = 11, 40  # excludes text glyphs (smaller) and candle wicks (taller)
MIN_FILL = 0.45  # bbox fill ratio; border boxes are hollow rings and fall below this


@dataclass(frozen=True)
class RowGrid:
    pitch: float  # row height in px
    origin: float  # y of the top of row 0 (any row; rows extend both ways)
    bar_height: float  # drawn bar height (pitch minus the gap)
    price_step: float  # price per row, snapped to 2 decimals (can be e.g. 0.55)

    def row_top(self, k: int) -> float:
        return self.origin + k * self.pitch

    def row_index(self, y: float) -> int:
        return int(np.floor((y - self.origin) / self.pitch))

    def row_price(self, k: int, axis: PriceAxis) -> float:
        """Price level of row k: its top edge, snapped to a multiple of the price step."""
        raw = axis.y_to_price(self.row_top(k))
        return round(round(raw / self.price_step) * self.price_step, 6)

    def price_row(self, price: float, axis: PriceAxis) -> int:
        """Index of the row whose price level contains `price` (row k spans price-step..price)."""
        y = axis.price_to_y(price)
        return self.row_index(y)


def _bar_boxes(img: np.ndarray, geo: TableGeometry, y_lo: int, y_hi: int) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > SAT_MIN) & (hsv[..., 2] > VAL_MIN))[y_lo:y_hi, geo.x_left : geo.x_right + 1]
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)
    keep = []
    for x, y, w, h, area in stats[1:]:
        if w >= MIN_BAR_W and MIN_BAR_H <= h <= MAX_BAR_H and area / (w * h) >= MIN_FILL:
            keep.append((y + y_lo, h))
    return np.array(keep, dtype=float)


def fit_row_grid(img: np.ndarray, axis: PriceAxis, geo: TableGeometry) -> RowGrid:
    ys = [y for y, _ in axis.labels]
    boxes = _bar_boxes(img, geo, int(min(ys)) - 20, int(max(ys)) + 20)
    if len(boxes) < 4:
        raise ValueError("too few cell bars to fit the row grid")
    # drop merged blobs (e.g. a bar plus its border box): keep the dominant height class
    heights = boxes[:, 1]
    mode_h = np.bincount(heights.astype(int)).argmax()
    boxes = boxes[np.abs(heights - mode_h) <= 2]
    tops = np.unique(boxes[:, 0])
    diffs = np.diff(tops)
    diffs = diffs[diffs >= MIN_BAR_H - 2]
    if len(diffs) == 0:
        raise ValueError("cannot estimate row pitch")
    smallest = diffs.min()
    pitch = float(np.median(diffs[diffs <= smallest * 1.25]))
    for _ in range(3):  # refine by regression of top against row index
        k = np.round((tops - tops[0]) / pitch)
        pitch, origin = np.polyfit(k, tops, 1)
    k = np.round((tops - tops[0]) / pitch)
    resid = np.abs(tops - (origin + k * pitch)).max()
    if resid > 2.0:
        raise ValueError(f"cell bars do not sit on a regular grid (max error {resid:.1f}px)")
    return RowGrid(
        pitch=float(pitch),
        origin=float(origin),
        bar_height=float(np.median(boxes[:, 1])),
        price_step=round(float(pitch * abs(axis.slope)), 2),
    )
