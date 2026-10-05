"""Point-of-control (POC) detection: the orange-red box drawn around one cell per bar."""
from __future__ import annotations

import cv2
import numpy as np

from .pipeline import ShotGeometry

RING_HUE = (10, 14)  # the box's orange-red; the imbalance fill is hue ~19
MIN_LINE_FRACTION = 0.5  # of the column width, for a ring edge row


def _ring_mask(img: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    return (h >= RING_HUE[0]) & (h <= RING_HUE[1]) & (s > 200) & (v > 200)


def find_poc_rows(geo: ShotGeometry) -> dict[int, int]:
    """Map bar column -> row index of its POC cell (columns without a box are omitted)."""
    ring = _ring_mask(geo.img)
    y_lo = int(geo.grid.row_top(geo.k_first)) - 6
    y_hi = int(geo.grid.row_top(geo.k_last + 1)) + 6
    out: dict[int, int] = {}
    for col in range(geo.table.n_cols):
        x0, x1 = geo.table.col_x(col)
        counts = ring[y_lo:y_hi, x0:x1].sum(axis=1)
        ys = np.where(counts >= MIN_LINE_FRACTION * (x1 - x0))[0] + y_lo
        if len(ys) == 0:
            continue
        # group consecutive rows into edge lines (the box edge is ~3 px thick)
        lines, cur = [], [ys[0]]
        for y in ys[1:]:
            if y - cur[-1] <= 1:
                cur.append(y)
            else:
                lines.append(float(np.mean(cur)))
                cur = [y]
        lines.append(float(np.mean(cur)))
        if len(lines) >= 2:
            out[col] = geo.grid.row_index((lines[0] + lines[1]) / 2)
    for col in range(geo.table.n_cols):
        if col not in out:  # an edge hidden by the price line: use the box's left side instead
            row = _row_from_left_side(ring, geo, col, y_lo, y_hi)
            if row is not None:
                out[col] = row
    return out


def _row_from_left_side(ring: np.ndarray, geo: ShotGeometry, col: int, y_lo: int, y_hi: int) -> int | None:
    x0, _ = geo.table.col_x(col)
    counts = ring[y_lo:y_hi, x0 : x0 + 8].sum(axis=1)
    ys = np.where(counts >= 2)[0] + y_lo
    if len(ys) == 0:
        return None
    runs, cur = [], [ys[0]]
    for y in ys[1:]:
        if y - cur[-1] <= 1:
            cur.append(y)
        else:
            runs.append(cur)
            cur = [y]
    runs.append(cur)
    best = max(runs, key=len)
    if len(best) < 0.8 * geo.grid.pitch:
        return None
    return geo.grid.row_index((best[0] + best[-1]) / 2)
