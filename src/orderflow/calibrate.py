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
TABLE_ROW_HEIGHTS = (20.3, 25.0)  # px per table row in the two font sizes seen (3 rows: 61 or 75 px; 4 rows: 81 px)


def count_table_rows(height: int) -> int:
    """3 (Volume, Delta, Cum) or 4 (+ Delta %): whichever gives a known row height."""
    return min((3, 4), key=lambda n: min(abs(height / n - h) for h in TABLE_ROW_HEIGHTS))
MIN_PITCH, MAX_PITCH = 60, 320
CLIP_MARGIN = 20  # table starting this close to the image's left edge is clipped


@dataclass(frozen=True)
class TableGeometry:
    y_top: int
    y_bottom: int  # inclusive
    col_edges: tuple[int, ...]  # x of each column boundary, left to right
    pitch: float  # nominal column width in px
    first_clipped: bool  # leftmost column is cut by the image edge
    n_rows: int = 3  # Volume, Delta, Cum (+ a Delta % row when the chart shows it)

    @property
    def n_cols(self) -> int:
        return len(self.col_edges) - 1

    @property
    def x_left(self) -> int:
        return self.col_edges[0]

    @property
    def x_right(self) -> int:
        return self.col_edges[-1] - 1

    @property
    def row_height(self) -> float:
        return (self.y_bottom - self.y_top + 1) / self.n_rows

    def row_y(self, row: int) -> tuple[int, int]:
        """Pixel y-range [start, end) of table row 0 (volume), 1 (delta), 2 (cum), 3 (delta %)."""
        h = self.row_height
        return round(self.y_top + row * h), round(self.y_top + (row + 1) * h)

    def col_x(self, col: int) -> tuple[int, int]:
        """Pixel x-range [start, end) of bar column `col` (the first may be clipped)."""
        return self.col_edges[col], self.col_edges[col + 1]


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
    return blocks[-1]  # the table is the lowest dense block (dense footprint cells can form taller ones above it)


def _x_extent(img: np.ndarray, y0: int, y1: int) -> tuple[int, int]:
    w = img.shape[1]
    ink = (img[y0 : y1 + 1, : w - RIGHT_STRIP].min(axis=2) < WHITE_THRESHOLD).mean(axis=0)
    xs = np.where(ink > 0.5)[0]
    xs = xs[xs > 15]
    return int(xs.min()), int(xs.max())


def _boundary_candidates(img: np.ndarray, y0: int, y1: int, x0: int, x1: int, n_rows: int = 3) -> np.ndarray:
    """x positions where the flat cell colour changes in the text-free top margin of a row."""
    h = (y1 - y0 + 1) / n_rows
    found: list[int] = []
    for r in range(n_rows):
        y = round(y0 + r * h) + 2  # margin above the glyphs
        line = img[y, x0 : x1 + 2].astype(int)
        jump = np.abs(np.diff(line, axis=0)).sum(axis=1)
        found.extend((np.where(jump > 12)[0] + 1 + x0).tolist())
    # a colour transition spans 1-3 px; merge each cluster to its mean position
    clusters: list[list[int]] = []
    for x in sorted(set(found)):
        if clusters and x - clusters[-1][-1] <= 3:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return np.array([np.mean(c) for c in clusters], dtype=float)


def _fit_pitch(bounds: np.ndarray) -> float:
    """Largest pitch p such that every boundary sits near bounds[0] + k*p."""
    for p in np.arange(MAX_PITCH, MIN_PITCH - 0.01, -0.25):
        err = np.abs((bounds - bounds[0]) / p - np.round((bounds - bounds[0]) / p)) * p
        if err.max() < 2.0:
            return float(p)
    raise ValueError("could not fit column pitch")


def calibrate_table(img: np.ndarray) -> TableGeometry:
    y0, y1 = find_table_rows(img)
    x0, x1 = _x_extent(img, y0, y1)
    clipped = x0 <= CLIP_MARGIN  # table runs off the left edge of the image
    n_rows = count_table_rows(y1 - y0 + 1)
    inner = _boundary_candidates(img, y0, y1, x0, x1, n_rows)
    inner = inner[(inner > x0 + 20) & (inner < x1 - 20)]
    anchors = np.concatenate([[] if clipped else [x0], inner, [x1 + 1]])
    if len(anchors) < 2:
        raise ValueError("could not fit column pitch")
    pitch = _fit_pitch(anchors)
    origin = anchors[0]
    ks = np.arange(-int(np.ceil((origin - x0) / pitch)), int((x1 + 1 - origin) / pitch + 0.5) + 1)
    edges = [int(round(origin + k * pitch)) for k in ks if x0 + 20 < origin + k * pitch < x1 - 20]
    return TableGeometry(y0, y1, tuple([x0, *edges, x1 + 1]), pitch, clipped, n_rows)
