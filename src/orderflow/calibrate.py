"""Per-image calibration: locate the Volume/Delta/Cum table and the bar columns.

Screenshots share one layout but differ in pixel dimensions, so geometry is
measured from each image instead of being hard-coded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WHITE_THRESHOLD = 235  # a pixel with min(B, G, R) below this is "ink"
RIGHT_STRIP = 110  # right-hand label strip excluded when scanning rows
MIN_TABLE_HEIGHT = 40
MIN_PITCH, MAX_PITCH = 60, 320


@dataclass(frozen=True)
class TableGeometry:
    y_top: int
    y_bottom: int  # inclusive
    x_left: int
    x_right: int  # inclusive
    pitch: float  # column width in px
    n_cols: int

    @property
    def row_height(self) -> float:
        return (self.y_bottom - self.y_top + 1) / 3

    def row_y(self, row: int) -> tuple[int, int]:
        """Pixel y-range [start, end) of table row 0 (volume), 1 (delta), 2 (cum)."""
        h = self.row_height
        return round(self.y_top + row * h), round(self.y_top + (row + 1) * h)

    def col_x(self, col: int) -> tuple[int, int]:
        """Pixel x-range [start, end) of bar column `col`."""
        return round(self.x_left + col * self.pitch), round(self.x_left + (col + 1) * self.pitch)


def _contiguous(ys: list[int]) -> list[tuple[int, int]]:
    groups, start, prev = [], ys[0], ys[0]
    for y in ys[1:]:
        if y != prev + 1:
            groups.append((start, prev))
            start = y
        prev = y
    groups.append((start, prev))
    return groups


def find_table_rows(img: np.ndarray) -> tuple[int, int]:
    """Return the (top, bottom) pixel rows of the three-row summary table."""
    h, w = img.shape[:2]
    ink = img.min(axis=2) < WHITE_THRESHOLD
    score = ink[:, 15 : w - RIGHT_STRIP].sum(axis=1)
    rows = [y for y in range(h // 2, h - 20) if score[y] > 0.18 * w]
    if not rows:
        raise ValueError("summary table not found")
    blocks = [b for b in _contiguous(rows) if b[1] - b[0] + 1 >= MIN_TABLE_HEIGHT]
    if not blocks:
        raise ValueError("summary table not found")
    return max(blocks, key=lambda b: b[1] - b[0])


def _x_extent(img: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    w = img.shape[1]
    ink = (img[y0 : y1 + 1, : w - RIGHT_STRIP].min(axis=2) < WHITE_THRESHOLD).mean(axis=0)
    xs = np.where(ink > 0.5)[0]
    xs = xs[xs > 15]
    return int(xs.min()), int(xs.max())


def _boundary_candidates(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    """x positions where the flat cell colour changes in the text-free top margin of a row."""
    h = (y1 - y0 + 1) / 3
    found: list[int] = []
    for r in range(3):
        y = round(y0 + r * h) + 2  # margin above the glyphs
        line = img[y, x0 : x1 + 2].astype(int)
        jump = np.abs(np.diff(line, axis=0)).sum(axis=1)
        found.extend((np.where(jump > 12)[0] + 1 + x0).tolist())
    return np.array(sorted(set(found)), dtype=float)


def _fit_pitch(bounds: np.ndarray, x0: int, span: int) -> float:
    """Largest pitch p in range such that every boundary sits near x0 + k*p."""
    best_p, best_err = None, None
    for p in np.arange(MIN_PITCH, min(MAX_PITCH, span) + 0.01, 0.25):
        k = np.round((bounds - x0) / p)
        err = np.abs(bounds - (x0 + k * p))
        if len(bounds) and np.median(err) > 1.5:
            continue
        frac = (err < 2.0).mean() if len(bounds) else 1.0
        if frac < 0.9:
            continue
        n = span / p
        if abs(n - round(n)) > 0.06:
            continue
        # Prefer the coarsest consistent grid (a finer one would also fit).
        if best_p is None or p > best_p + 0.5:
            best_p, best_err = float(p), err
    if best_p is None:
        raise ValueError("could not fit column pitch")
    return best_p


def calibrate_table(img: np.ndarray) -> TableGeometry:
    y0, y1 = find_table_rows(img)
    x0, x1 = _x_extent(img, y0, y1)
    span = x1 - x0 + 1
    bounds = _boundary_candidates(img, y0, y1, x0, x1)
    bounds = bounds[(bounds > x0 + 20) & (bounds < x1 - 20)]
    pitch = _fit_pitch(bounds, x0, span)
    return TableGeometry(y0, y1, x0, x1, pitch, round(span / pitch))
