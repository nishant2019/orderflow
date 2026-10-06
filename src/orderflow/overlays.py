"""Chart overlays: the pink current-price line."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .axis import PriceAxis

RIGHT_STRIP = 110  # px at the right, holding the axis and profile labels
LEFT_FRAME = 12
MIN_LINE_FRACTION = 0.5  # share of the plot width a pink row must cover


@dataclass(frozen=True)
class PriceLine:
    y: float  # pixel centre of the line
    price: float


def _is_pink(img: np.ndarray) -> np.ndarray:
    b, g, r = (img[..., i].astype(int) for i in range(3))
    return (r > 220) & (g < 150) & (90 < b) & (b < 190) & (r - g > 80)


def find_current_price_line(img: np.ndarray, axis: PriceAxis, y_limit: float | None = None) -> PriceLine | None:
    """The horizontal pink line marking the current price, or None if the chart has none.

    `y_limit` restricts the search to above that pixel row (the table below also has pink).
    """
    h, w = img.shape[:2]
    plot_w = w - RIGHT_STRIP - LEFT_FRAME
    covered = _is_pink(img)[:, LEFT_FRAME : w - RIGHT_STRIP].sum(axis=1) > MIN_LINE_FRACTION * plot_w
    rows = np.where(covered)[0]
    if y_limit is not None:
        rows = rows[rows < y_limit]
    if len(rows) == 0:
        return None
    # the line is 2 px thick; take the first contiguous group
    group = [rows[0]]
    for y in rows[1:]:
        if y == group[-1] + 1:
            group.append(y)
        else:
            break
    y = float(np.mean(group)) + 0.5  # pixel row index -> centre of the pixel
    return PriceLine(y=y, price=axis.y_to_price(y))


def remove_price_line(img: np.ndarray, axis: PriceAxis, table) -> np.ndarray:
    """A copy of the screenshot with the pink current-price line patched out.

    The 2 px line runs through text and would read as ink; each of its rows is replaced by the
    nearest row above/below that is not part of the line (strokes it crosses lose 1-2 px).
    """
    h, w = img.shape[:2]
    pink = _is_pink(img)[:, LEFT_FRAME : w - RIGHT_STRIP].sum(axis=1) > MIN_LINE_FRACTION * (w - RIGHT_STRIP - LEFT_FRAME)
    rows = np.where(pink)[0]
    rows = rows[rows < table.y_top]
    out = img.copy()
    if len(rows) == 0:
        return out
    lo, hi = int(rows.min()), int(rows.max())
    above, below = max(lo - 1, 0), min(hi + 1, h - 1)
    for y in range(lo, hi + 1):
        src = above if (y - lo) <= (hi - y) else below
        out[y, LEFT_FRAME : w - RIGHT_STRIP] = img[src, LEFT_FRAME : w - RIGHT_STRIP]
    return out


DASH_BGR = np.array([82, 91, 237])  # the dashed line's dark first pixel row (its fainter second row is shared with the solid pink line)
DASH_TOL = 30
MIN_DASH_FRACTION = 0.35  # dashes cover roughly 60-70 % of the plot width


def _dashed_rows(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    plot = img[:, LEFT_FRAME : w - RIGHT_STRIP].astype(np.int32)
    hit = np.abs(plot - DASH_BGR).sum(axis=2) < DASH_TOL
    rows = np.where(hit.sum(axis=1) > MIN_DASH_FRACTION * plot.shape[1])[0]
    return np.concatenate([rows, rows + 1]) if len(rows) else rows  # + the faint second row below it


def find_dashed_price_line(img: np.ndarray, axis: PriceAxis, y_limit: float | None = None) -> PriceLine | None:
    """The red *dashed* horizontal line (with a price tag on the right axis): the current price."""
    rows = _dashed_rows(img)
    if y_limit is not None:
        rows = rows[rows < y_limit]
    if len(rows) == 0:
        return None
    group = [rows[0]]
    for y in rows[1:]:
        if y == group[-1] + 1:
            group.append(y)
        else:
            break
    y = float(np.mean(group)) + 0.5
    return PriceLine(y=y, price=axis.y_to_price(y))


def remove_dashed_price_line(img: np.ndarray, y_limit: float | None = None) -> np.ndarray:
    """Copy of the image with the dashed current-price line patched out (rows copied from the
    nearest rows above/below that are not part of the line)."""
    h, w = img.shape[:2]
    rows = _dashed_rows(img)
    if y_limit is not None:
        rows = rows[rows < y_limit]
    out = img.copy()
    if len(rows) == 0:
        return out
    lo, hi = int(rows.min()), int(rows.max())
    above, below = max(lo - 1, 0), min(hi + 1, h - 1)
    for y in range(lo, hi + 1):
        src = above if (y - lo) <= (hi - y) else below
        out[y, LEFT_FRAME : w - RIGHT_STRIP] = img[src, LEFT_FRAME : w - RIGHT_STRIP]
    return out
