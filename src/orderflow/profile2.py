"""Centre-axis volume profile (split layout): per-price delta on the left of a vertical axis,
total volume on the right.

Left bars are green (delta > 0) or orange-red (delta < 0), length proportional to |delta|; right
bars are coloured by value-area membership: teal = inside the value area, blue = outside, pink =
the highlighted (maximum-volume) row. Numbers are blue: delta right-aligned against the axis,
volume left-aligned from it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import Glyph, GlyphClassifier, glyph_features, segment_glyphs
from .pipeline import ShotGeometry
from .table import parse_signed

GREEN_BGR = np.array([92, 229, 136])
RED_BGR = np.array([57, 90, 246])
VA_BGR = np.array([224, 241, 167])  # teal: inside the value area
OUT_BGR = np.array([240, 199, 164])  # blue: outside the value area
PEAK_BGR = np.array([120, 101, 250])  # pink: the highlighted row
COLOR_TOL = 36
MIN_COLUMN_PIXELS = 6
TEXT_WIDTH = 70  # px either side of the axis searched for the numbers
BLUE_B_MIN, BLUE_R_MAX, BLUE_G_MAX = 170, 120, 150


def _near(img: np.ndarray, color: np.ndarray) -> np.ndarray:
    return np.abs(img.astype(np.int32) - color).sum(axis=2) < COLOR_TOL


@dataclass(frozen=True)
class ProfileRow2:
    row: int
    price: float
    delta_raw: str
    delta: float | None
    volume_raw: str
    volume: float | None
    delta_px: int  # left bar length
    volume_px: int  # right bar length
    zone: str  # "value_area" | "outside" | "peak" | "none"
    confidence: float
    delta_glyphs: tuple = ()  # the segmented glyphs, kept for training
    volume_glyphs: tuple = ()
    corrected_from: str = ""  # "delta|volume" text before a single-glyph correction by the checksums


def find_axis(geo: ShotGeometry) -> int | None:
    """x of the vertical axis (the right edge of the delta bars), or None when the chart has
    no profile."""
    img = geo.img
    y0 = int(geo.grid.row_top(geo.k_first))
    y1 = int(geo.grid.row_top(geo.k_last + 1))
    delta = (_near(img, GREEN_BGR) | _near(img, RED_BGR))[y0:y1]
    cols = delta.sum(axis=0)
    xs = np.where(cols >= MIN_COLUMN_PIXELS)[0]
    xs = xs[xs > img.shape[1] // 2]
    if len(xs) == 0:
        return None
    return int(xs.max())


def _blue(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    crop = img[y0:y1, x0:x1].astype(np.int32)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    return (b > BLUE_B_MIN) & (r < BLUE_R_MAX) & (g < BLUE_G_MAX)


def _glyphs(geo: ShotGeometry, row: int, x0: int, x1: int) -> list[Glyph]:
    y0 = int(round(geo.grid.row_top(row))) + 2
    y1 = int(round(geo.grid.row_top(row + 1))) - 2
    blue = _blue(geo.img, x0, x1, y0, y1)
    zeros = np.zeros_like(blue)
    return segment_glyphs({"black": zeros, "red": zeros, "blue": blue}, geo.grid.pitch, geo.glyph_h, blue.astype(np.float32))


def _read(geo: ShotGeometry, glyphs: list[Glyph], clf: GlyphClassifier) -> tuple[str, float]:
    chars, dists = [], []
    for g in glyphs:
        ch, d = clf.classify(glyph_features(g, geo.glyph_h))
        chars.append(ch)
        dists.append(d)
    conf = float(np.mean([1 / (1 + max(0.0, d - 2.0)) for d in dists])) if dists else 0.0
    return "".join(chars), conf


def read_profile2(geo: ShotGeometry, clf: GlyphClassifier) -> list[ProfileRow2]:
    """One entry per row that has any profile content, top to bottom."""
    img = geo.img
    axis_x = find_axis(geo)
    if axis_x is None:
        return []
    green, red = _near(img, GREEN_BGR), _near(img, RED_BGR)
    va, out, peak = _near(img, VA_BGR), _near(img, OUT_BGR), _near(img, PEAK_BGR)
    x_floor = img.shape[1] // 2
    rows = []
    for row in range(geo.k_first, geo.k_last + 1):
        y0 = int(round(geo.grid.row_top(row))) + 3
        y1 = int(round(geo.grid.row_top(row + 1))) - 2
        need = max(2, (y1 - y0) // 2)

        def extent(mask: np.ndarray, lo: int, hi: int) -> np.ndarray:
            return np.where(mask[y0:y1, lo:hi].sum(axis=0) >= need)[0] + lo

        left = np.concatenate([extent(green, x_floor, axis_x + 1), extent(red, x_floor, axis_x + 1)])
        px_left = int(axis_x - left.min() + 1) if len(left) else 0
        zones = {"value_area": va, "outside": out, "peak": peak}
        best_zone, px_right = "none", 0
        for name, mask in zones.items():
            cols = extent(mask, axis_x + 1, img.shape[1] - 12)
            if len(cols) and cols.max() - axis_x > px_right:
                best_zone, px_right = name, int(cols.max() - axis_x)
        d_gl = _glyphs(geo, row, axis_x - TEXT_WIDTH, axis_x + 1)
        v_gl = _glyphs(geo, row, axis_x + 3, axis_x + TEXT_WIDTH + 3)
        d_raw, d_conf = _read(geo, d_gl, clf)
        v_raw, v_conf = _read(geo, v_gl, clf)
        if not d_raw and not v_raw and px_left == 0 and px_right == 0:
            continue
        conf = min(c for c, t in ((d_conf, d_raw), (v_conf, v_raw)) if t) if (d_raw or v_raw) else 0.0
        rows.append(ProfileRow2(
            row, geo.grid.row_price(row, geo.axis), d_raw, parse_signed(d_raw) if d_raw else None,
            v_raw, parse_signed(v_raw) if v_raw else None, px_left, px_right, best_zone, conf,
            tuple(d_gl), tuple(v_gl),
        ))
    return rows
