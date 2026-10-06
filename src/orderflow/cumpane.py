"""The cumulative-delta candle pane drawn under the price rows.

Each bar has a candle whose open is the previous bar's cumulative delta and whose close is this bar's
(the `Cum` row of the table); the wicks are the extremes reached inside the bar. The pane's amber dashed
line is the zero level. The pixel -> value scale is not read from the pane's axis labels: it is fitted
through the zero line and the table's cumulative values, and each candle is then checked against it
(the body ends must land on the table's open/close), so the fit is both the calibration and the test.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .candles import DOWN_BGR, UP_BGR

AMBER_BGR = np.array([0, 75, 150])
COLOR_TOL = 30
MIN_ZERO_FRACTION = 0.3  # the dashed zero line covers about half the pane width
MIN_BODY_W = 3
MAX_RESIDUAL_PX = 2.5  # a body end this far from where the table's value puts it fails the check
PANE_TOP_MARGIN = 12  # px below the last price-axis label


@dataclass(frozen=True)
class CumCandle:
    direction: str
    open: float
    high: float
    low: float
    close: float
    residual_px: float  # worst distance between a body end and the table's value, in pixels
    valid: bool


@dataclass(frozen=True)
class CumPane:
    zero_y: float
    units_per_px: float
    candles: list  # CumCandle | None per bar
    fit_points: int


def find_zero_line(raw: np.ndarray, y_top: int, y_from: int) -> float | None:
    win = raw[y_from:y_top, 20:-20].astype(np.int32)
    hit = np.abs(win - AMBER_BGR).sum(axis=2) < 40
    rows = np.where(hit.sum(axis=1) > MIN_ZERO_FRACTION * win.shape[1])[0]
    return float(rows.mean()) + y_from + 0.5 if len(rows) else None


def _measure(raw: np.ndarray, x0: int, x1: int, y0: int, y1: int):
    """(direction, top, body_top, body_bottom, bottom) pixel rows (bottom edges exclusive) of the candle."""
    win = raw[y0:y1, x0:x1].astype(np.int32)
    up = np.abs(win - UP_BGR).sum(axis=2) < COLOR_TOL
    down = np.abs(win - DOWN_BGR).sum(axis=2) < COLOR_TOL
    if max(up.sum(), down.sum()) < 4:
        return None
    direction = "up" if up.sum() >= down.sum() else "down"
    mask = up if direction == "up" else down
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    lo = stats[best, cv2.CC_STAT_LEFT]
    hi = lo + stats[best, cv2.CC_STAT_WIDTH]
    keep = lab == best
    for i in range(1, n):  # wick and body can be split by an anti-aliased pixel row
        if i != best and stats[i, cv2.CC_STAT_LEFT] <= hi and stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] >= lo:
            keep |= lab == i
    rows = np.where(keep.any(axis=1))[0]
    widths = keep.sum(axis=1)
    body = np.where(widths >= MIN_BODY_W)[0]
    top, bottom = int(rows.min()), int(rows.max()) + 1
    if len(body):
        bt, bb = int(body.min()), int(body.max()) + 1
    else:  # doji
        bt = bb = (top + bottom) // 2
    return direction, y0 + top, y0 + bt, y0 + bb, y0 + bottom


def read_cum_pane(raw: np.ndarray, table_geo, cum_values: list, y_from: int) -> CumPane | None:
    """`cum_values[i]` is bar i's table cumulative delta (None when unreadable)."""
    y_top = int(table_geo.y_top)
    zero = find_zero_line(raw, y_top, y_from)
    if zero is None:
        return None
    marks = []
    for i in range(table_geo.n_cols):
        x0, x1 = table_geo.col_x(i)
        marks.append(_measure(raw, max(x0, 0), min(x1, raw.shape[1]), y_from, y_top - 2))
    # scale: y = zero - v / k, fitted on body ends against table values (close = cum[i], open = cum[i-1])
    pts = []
    for i, m in enumerate(marks):
        if m is None or cum_values[i] is None:
            continue
        d, _, bt, bb, _ = m
        close_y, open_y = (bt, bb) if d == "up" else (bb, bt)
        pts.append((close_y, cum_values[i]))
        if i > 0 and cum_values[i - 1] is not None:
            pts.append((open_y, cum_values[i - 1]))
    pts = [(y, v) for y, v in pts if abs(v) > 0]
    if len(pts) < 4:
        return None
    ys = np.array([zero - y for y, _ in pts], float)  # pixels above the zero line
    vs = np.array([v for _, v in pts], float)
    keep = np.ones(len(pts), bool)
    k = float(np.dot(ys, vs) / np.dot(ys, ys))  # value per px (least squares through the zero line)
    for _ in range(4):
        k = float(np.dot(ys[keep], vs[keep]) / np.dot(ys[keep], ys[keep]))
        keep = np.abs(ys - vs / k) <= MAX_RESIDUAL_PX
        if keep.sum() < 4:
            return None
    to_val = lambda y: (zero - y) * k  # noqa: E731
    candles = []
    for i, m in enumerate(marks):
        if m is None:
            candles.append(None)
            continue
        d, top, bt, bb, bottom = m
        upper, lower = to_val(bt), to_val(bb)
        open_, close = (lower, upper) if d == "up" else (upper, lower)
        res = 0.0
        if cum_values[i] is not None:
            res = abs(zero - (bt if d == "up" else bb) - cum_values[i] / k)
            if i > 0 and cum_values[i - 1] is not None:
                res = max(res, abs(zero - (bb if d == "up" else bt) - cum_values[i - 1] / k))
        candles.append(CumCandle(d, open_, to_val(top), to_val(bottom), close, round(res, 2), res <= MAX_RESIDUAL_PX))
    return CumPane(zero, k, candles, int(keep.sum()))
