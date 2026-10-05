"""Shared per-shot setup: calibrate once, then enumerate the footprint cells."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .axis import PriceAxis, calibrate_price_axis
from .calibrate import TableGeometry, calibrate_table
from .cells import CellRect, cell_rect, estimate_glyph_height, glyph_features, ink_masks, segment_glyphs, soft_ink_map, Glyph
from .rows import RowGrid, fit_row_grid


@dataclass
class ShotGeometry:
    img: np.ndarray
    axis: PriceAxis
    table: TableGeometry
    grid: RowGrid
    k_first: int  # first and last row index inside the plotted price range
    k_last: int
    glyph_h: int

    def cell_glyphs(self, col: int, row: int) -> list[Glyph]:
        rect = cell_rect(self.table, self.grid, col, row)
        masks = ink_masks(self.img, rect)
        return segment_glyphs(masks, self.grid.pitch, self.glyph_h, soft_ink_map(self.img, rect, masks))

    def text_cells(self, col: int) -> list[tuple[int, list[Glyph]]]:
        """(row, glyphs) of every cell in the column that holds text, top to bottom."""
        out = []
        for row in range(self.k_first, self.k_last + 1):
            glyphs = self.cell_glyphs(col, row)
            if glyphs:
                out.append((row, glyphs))
        return out


def load_shot(path: str) -> ShotGeometry:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    axis = calibrate_price_axis(img)
    table = calibrate_table(img)
    grid = fit_row_grid(img, axis, table)
    ys = [y for y, _ in axis.labels]
    k_first, k_last = grid.row_index(min(ys)), grid.row_index(max(ys))
    rects = [cell_rect(table, grid, c, k) for c in range(table.n_cols) for k in range(k_first, k_last + 1)]
    return ShotGeometry(img, axis, table, grid, k_first, k_last, estimate_glyph_height(img, rects, grid.pitch))
