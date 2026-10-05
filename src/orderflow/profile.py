"""Right-hand per-price delta profile: horizontal bars anchored at the plot's right edge.

Green bars are positive delta (ask > bid), orange-red negative. Bar length is proportional
to |delta| and the value is printed in blue, right-aligned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import Glyph, GlyphClassifier, glyph_features, segment_glyphs
from .pipeline import ShotGeometry
from .table import parse_signed

GREEN_BGR = np.array([92, 229, 136])
RED_BGR = np.array([57, 90, 246])
COLOR_TOL = 40  # L1 distance; footprint fills are far from these exact fills
MIN_COLUMN_PIXELS = 6
TEXT_WINDOW = 84  # px left of the bar anchor searched for the value text
BLUE_B_MIN, BLUE_R_MAX, BLUE_G_MAX = 170, 120, 150


@dataclass(frozen=True)
class ProfileBar:
    row: int
    sign: int  # +1 green, -1 red, 0 no bar
    length_px: int  # bar length from the anchor (0 for no bar)


def _color_masks(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = img.astype(np.int32)
    green = np.abs(flat - GREEN_BGR).sum(axis=2) < COLOR_TOL
    red = np.abs(flat - RED_BGR).sum(axis=2) < COLOR_TOL
    return green, red


def find_anchor(geo: ShotGeometry) -> int:
    """x of the bars' common right edge."""
    green, red = _color_masks(geo.img)
    y0 = int(geo.grid.row_top(geo.k_first))
    y1 = int(geo.grid.row_top(geo.k_last + 1))
    cols = (green | red)[y0:y1].sum(axis=0)
    xs = np.where(cols >= MIN_COLUMN_PIXELS)[0]
    xs = xs[xs > geo.img.shape[1] // 2]
    if len(xs) == 0:
        raise ValueError("no profile bars found")
    return int(xs.max())


def find_bars(geo: ShotGeometry) -> list[ProfileBar]:
    """One entry per footprint row in view, top to bottom."""
    green, red = _color_masks(geo.img)
    anchor = find_anchor(geo)
    x_floor = geo.img.shape[1] // 3
    bars = []
    for row in range(geo.k_first, geo.k_last + 1):
        y0 = int(round(geo.grid.row_top(row))) + 3
        y1 = int(round(geo.grid.row_top(row + 1))) - 2
        strip_g = green[y0:y1, x_floor : anchor + 1]
        strip_r = red[y0:y1, x_floor : anchor + 1]
        need = max(2, (y1 - y0) // 2)
        cols_g = np.where(strip_g.sum(axis=0) >= need)[0]
        cols_r = np.where(strip_r.sum(axis=0) >= need)[0]
        n_g, n_r = len(cols_g), len(cols_r)
        if n_g == 0 and n_r == 0:
            bars.append(ProfileBar(row, 0, 0))
            continue
        cols, sign = (cols_g, 1) if n_g >= n_r else (cols_r, -1)
        bars.append(ProfileBar(row, sign, anchor - (int(cols.min()) + x_floor) + 1))
    return bars


def blue_mask(img: np.ndarray, x0: int, x1: int, y0: int, y1: int) -> np.ndarray:
    crop = img[y0:y1, x0:x1].astype(np.int32)
    b, g, r = crop[..., 0], crop[..., 1], crop[..., 2]
    return (b > BLUE_B_MIN) & (r < BLUE_R_MAX) & (g < BLUE_G_MAX)


def row_glyphs(geo: ShotGeometry, row: int, anchor: int, glyph_h: float) -> list[Glyph]:
    y0 = int(round(geo.grid.row_top(row)))
    y1 = int(round(geo.grid.row_top(row + 1)))
    x0, x1 = anchor - TEXT_WINDOW, anchor + 3
    blue = blue_mask(geo.img, x0, x1, y0, y1)
    zeros = np.zeros_like(blue)
    return segment_glyphs({"black": zeros, "red": zeros, "blue": blue}, geo.grid.pitch, glyph_h, blue.astype(np.float32))


def profile_glyph_height(geo: ShotGeometry, anchor: int) -> int:
    import cv2

    heights: list[int] = []
    for row in range(geo.k_first, geo.k_last + 1):
        y0 = int(round(geo.grid.row_top(row)))
        y1 = int(round(geo.grid.row_top(row + 1)))
        blue = blue_mask(geo.img, anchor - TEXT_WINDOW, anchor + 3, y0, y1)
        _, _, stats, _ = cv2.connectedComponentsWithStats(blue.astype(np.uint8), connectivity=8)
        heights += [int(s[3]) for s in stats[1:] if s[3] >= 6]
    return int(np.bincount(heights).argmax())


@dataclass(frozen=True)
class ProfileRow:
    row: int
    price: float
    raw: str
    value: float | None  # printed value (rounded: 4.3K -> 4300)
    sign: int  # bar colour: +1 green, -1 red, 0 none
    length_px: int
    confidence: float


def read_profile(geo: ShotGeometry, clf: GlyphClassifier) -> list[ProfileRow]:
    anchor = find_anchor(geo)
    gh = profile_glyph_height(geo, anchor)
    rows = []
    for bar in find_bars(geo):
        glyphs = row_glyphs(geo, bar.row, anchor, gh)
        chars, dists = [], []
        for g in glyphs:
            ch, d = clf.classify(glyph_features(g, gh))
            chars.append(ch)
            dists.append(d)
        raw = "".join(chars)
        value = parse_signed(raw) if raw else None
        conf = 0.0 if value is None else float(np.mean([1 / (1 + max(0.0, d - 2.0)) for d in dists]))
        rows.append(ProfileRow(bar.row, geo.grid.row_price(bar.row, geo.axis), raw, value, bar.sign, bar.length_px, conf))
    return rows
