"""Per-bar OHLC from the thin candle drawn at the left edge of each footprint column.

Up candles are teal and down candles red (GoCharting defaults). The wick is 1 px wide and the
body ~6 px; y positions map to prices through the price axis.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .pipeline import ShotGeometry

UP_BGR = np.array([125, 157, 18])
DOWN_BGR = np.array([50, 60, 233])
COLOR_TOL = 30  # L1 distance
X_BEFORE, X_AFTER = 8, 16  # search window around the column's left edge
MIN_BODY_W = 3  # rows at least this wide are body; narrower rows are wick
MIN_EXTENT = 4  # px: smaller marks are not candles
CLIP_MARGIN = 2  # px from the search window's top/bottom that counts as clipped
MAX_GAP = 2  # px between a wick's visible end and the black POC edge that may be covering it


def bucket(price: float, step: float, jitter: float = 0.0) -> float:
    """Row price (bucket top edge) whose bucket (p - step, p] contains `price`.

    `jitter` (price units, normally one pixel's worth) absorbs measurement noise when a
    price sits right on a row boundary: a price within `jitter` above a boundary stays in
    the bucket below it.
    """
    import math

    return round(math.ceil((price - jitter) / step - 1e-9) * step, 6)


@dataclass(frozen=True)
class Candle:
    direction: str  # "up" | "down" | "flat"
    open: float
    high: float
    low: float
    close: float
    high_clipped: bool  # wick runs off the visible plot: the true high is higher
    low_clipped: bool
    body_px: int  # body height in pixels (0: a doji drawn as a line)
    high_hidden: bool = False  # the wick end was under the POC rectangle: high is estimated to its far edge
    low_hidden: bool = False


def _plot_window(geo: ShotGeometry) -> tuple[int, int]:
    """Pixel rows [y0, y1) in which candles can appear (the price pane)."""
    ys = [y for y, _ in geo.axis.labels]
    spacing = float(np.median(np.diff(sorted(ys)))) if len(ys) > 1 else geo.grid.pitch
    top = max(46, int(min(ys) - spacing))
    return top, int(max(ys) + spacing * 0.6)


def _hidden_extension(black: np.ndarray, end: int, col: int, step: int) -> int:
    """How far the wick continues under a black POC edge beyond its visible `end`.

    Starting at most MAX_GAP px past the visible end, a run of at least 2 black pixels in the
    wick's own column means the wick is covered there; the covered length (gap + run) is returned.
    """
    n = black.shape[0]
    gap = 0
    while gap <= MAX_GAP:
        y = end + step * (gap + 1)
        if not (0 <= y < n):
            return 0
        if black[y, col]:
            break
        gap += 1
    else:
        return 0
    run = 0
    y = end + step * (gap + 1)
    while 0 <= y < n and black[y, col]:
        run += 1
        y += step
    return gap + run if run >= 2 else 0


def find_candle(geo: ShotGeometry, col: int) -> Candle | None:
    x0, _ = geo.table.col_x(col)
    before, after = getattr(geo, "candle_window", (X_BEFORE, X_AFTER))
    xa, xb = max(x0 - before, 0), x0 + after
    y0, y1 = _plot_window(geo)
    win = geo.img[y0:y1, xa:xb].astype(np.int32)
    up = np.abs(win - UP_BGR).sum(axis=2) < COLOR_TOL
    down = np.abs(win - DOWN_BGR).sum(axis=2) < COLOR_TOL
    if max(up.sum(), down.sum()) < MIN_EXTENT:
        return None
    direction = "up" if up.sum() >= down.sum() else "down"
    mask = up if direction == "up" else down

    # keep the tallest connected piece: the candle, not specks of the same colour
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_HEIGHT]))
    # a ring or cell fill can split the candle into pieces (wick above, body below): merge every
    # piece whose x-range overlaps the candle so far, until nothing more joins
    keep = lab == best
    lo = stats[best, cv2.CC_STAT_LEFT]
    hi = lo + stats[best, cv2.CC_STAT_WIDTH]
    joined = {best}
    grew = True
    while grew:
        grew = False
        for i in range(1, n):
            if i in joined:
                continue
            left = stats[i, cv2.CC_STAT_LEFT]
            right = left + stats[i, cv2.CC_STAT_WIDTH]
            if left <= hi and right >= lo:
                joined.add(i)
                keep |= lab == i
                lo, hi = min(lo, left), max(hi, right)
                grew = True
    rows = np.where(keep.any(axis=1))[0]
    if rows.max() - rows.min() + 1 < MIN_EXTENT:
        return None
    widths = keep.sum(axis=1)
    top_px, bottom_px = int(rows.min()), int(rows.max())
    # the black POC rectangle can cover the end of a wick: if a black run continues the wick's
    # own column beyond its visible end, the wick ends somewhere inside it; take the run's far end
    wick_col = int(np.argmax(keep.sum(axis=0) * (keep.sum(axis=0) < 0.9 * len(rows)) + keep.sum(axis=0) * 0.001))
    black = win.max(axis=2) < 40
    high_hidden = low_hidden = False
    ext = _hidden_extension(black, top_px, wick_col, -1)
    if ext:
        top_px -= ext
        high_hidden = True
    ext = _hidden_extension(black, bottom_px, wick_col, +1)
    if ext:
        bottom_px += ext
        low_hidden = True
    body_rows = np.where(widths >= MIN_BODY_W)[0]
    if len(body_rows):
        body_top, body_bottom = int(body_rows.min()), int(body_rows.max())
        body_px = body_bottom - body_top + 1
    else:  # a doji: no row is wide enough; the body collapses to a point
        body_top = body_bottom = (top_px + bottom_px) // 2
        body_px = 0

    def price(y_edge: float) -> float:
        return float(geo.axis.y_to_price(y0 + y_edge))

    high, low = price(top_px), price(bottom_px + 1)
    upper, lower = price(body_top), price(body_bottom + 1)
    open_, close = (lower, upper) if direction == "up" else (upper, lower)
    if body_px == 0:
        open_ = close = (upper + lower) / 2
    return Candle(
        direction=direction, open=open_, high=high, low=low, close=close,
        high_clipped=top_px <= CLIP_MARGIN, low_clipped=bottom_px >= (y1 - y0) - 1 - CLIP_MARGIN,
        body_px=body_px, high_hidden=high_hidden, low_hidden=low_hidden,
    )


def find_candles(geo: ShotGeometry) -> list[Candle | None]:
    return [find_candle(geo, c) for c in range(geo.table.n_cols)]
