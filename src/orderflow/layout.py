"""Tell the two footprint layouts apart.

* "single": one column per bar, "bid X ask" text, orange-red POC ring.
* "split":  two boxes per bar and row (bid | ask), black POC rectangle.

The POC rectangle is the cleanest cue (its colour differs); if none is visible the box
structure decides.
"""
from __future__ import annotations

import cv2
import numpy as np

from .calibrate import TableGeometry

MIN_RUN = 70  # px: a POC edge is a long horizontal line (about the width of a bar)
BLACK_MAX = 40


def _long_rows(mask: np.ndarray, y1: int) -> int:
    """Number of pixel rows above y1 holding a horizontal run of at least MIN_RUN set pixels."""
    n = 0
    for y in range(48, y1):
        row = mask[y]
        if row.sum() < MIN_RUN:
            continue
        run = best = 0
        for v in row:
            run = run + 1 if v else 0
            best = max(best, run)
        n += best >= MIN_RUN
    return n


def detect_layout(img: np.ndarray, table: TableGeometry) -> str:
    h, w = img.shape[:2]
    plot = img[:, 12 : w - 120]
    hsv = cv2.cvtColor(plot, cv2.COLOR_BGR2HSV)
    black = plot.max(axis=2) < BLACK_MAX
    orange_ring = (hsv[..., 0] >= 10) & (hsv[..., 0] <= 14) & (hsv[..., 1] > 200) & (hsv[..., 2] > 200)
    n_black = _long_rows(black, table.y_top)
    n_orange = _long_rows(orange_ring, table.y_top)
    if n_black or n_orange:
        return "split" if n_black > n_orange else "single"
    return "single"
