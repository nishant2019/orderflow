"""The platform's delta-divergence markers: pale vertical bands drawn through a bar's candle.

Pale red (BGR 207,210,255) and pale green (213,239,178) bands, 3 px wide with a 1 px gap, span the plot
height.  Each band is assigned to the bar column it falls in.  Charts without the markers return nothing.
"""
from __future__ import annotations

import numpy as np

RED_BAND_BGR = (207, 210, 255)
GREEN_BAND_BGR = (213, 239, 178)
MIN_HEIGHT_FRACTION = 0.15  # a band covers at least this share of the plot height


def find_divergence_bands(raw: np.ndarray, col_edges, y_top: int) -> dict[int, str]:
    """Map bar index -> "red" | "green" for every bar whose column holds a marker band."""
    plot = raw[: max(int(y_top), 1)]
    need = plot.shape[0] * MIN_HEIGHT_FRACTION
    found: dict[int, str] = {}
    for colour, bgr in (("red", RED_BAND_BGR), ("green", GREEN_BAND_BGR)):
        xs = np.where((plot == bgr).all(axis=2).sum(axis=0) > need)[0]
        have = set(int(x) for x in xs)
        for x in sorted(have):
            if not ({x + 1, x + 3} <= have and x + 2 not in have):  # the marker is drawn 2 px + gap + 1 px
                continue
            for i in range(len(col_edges) - 1):
                if col_edges[i] <= x < col_edges[i + 1]:
                    found[i] = colour
                    break
    return found
