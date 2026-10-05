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
