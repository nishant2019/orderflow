"""Summary table (Volume / Delta / Cum rows) reader.

Cells are flat colours; the text is white on saturated cells and dark on light ones, so
ink is "far from the cell's own background colour".
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibrate import TableGeometry
from .cells import Glyph, GlyphClassifier, glyph_features, segment_glyphs

ROW_NAMES = ("volume", "delta", "cum")
INK_DIST = 90  # L1 colour distance from the background that counts as ink
X_INSET = 3  # px trimmed from each side of a cell (column separators)
MIN_GLYPH_H = 8
_NUM = re.compile(r"^(-?\d+(?:\.\d+)?)([KM]?)$")
_SCALE = {"": 1.0, "K": 1e3, "M": 1e6}


def parse_signed(text: str) -> float | None:
    m = _NUM.match(text)
    return None if not m else float(m.group(1)) * _SCALE[m.group(2)]


def _cell_box(table: TableGeometry, row: int, col: int) -> tuple[int, int, int, int]:
    y0, y1 = table.row_y(row)
    x0, x1 = table.col_x(col)
    return y0 + 1, y1 - 1, x0 + X_INSET, x1 - X_INSET


def cell_ink(img: np.ndarray, table: TableGeometry, row: int, col: int) -> tuple[np.ndarray, np.ndarray]:
    """(bool ink mask, soft coverage 0..1) of one table cell."""
    y0, y1, x0, x1 = _cell_box(table, row, col)
    crop = img[y0:y1, x0:x1].astype(np.int32)
    flat = (crop >> 3).reshape(-1, 3)
    vals, counts = np.unique(flat, axis=0, return_counts=True)
    bg = vals[counts.argmax()] * 8 + 4
    dist = np.abs(crop - bg).sum(axis=2).astype(np.float32)
    ink = dist > INK_DIST
    soft = np.clip(dist / max(float(dist.max()), 1.0), 0.0, 1.0)
    return ink, np.where(ink, soft, 0.0).astype(np.float32)


def table_glyph_height(img: np.ndarray, table: TableGeometry) -> int:
    heights: list[int] = []
    for row in range(3):
        for col in range(table.n_cols):
            ink, _ = cell_ink(img, table, row, col)
            _, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
            heights += [int(s[3]) for s in stats[1:] if s[3] >= MIN_GLYPH_H]
    return int(np.bincount(heights).argmax())


def cell_glyphs(img: np.ndarray, table: TableGeometry, row: int, col: int, glyph_h: float) -> list[Glyph]:
    ink, soft = cell_ink(img, table, row, col)
    # faintly antialiased strokes can break into pieces one pixel apart (a K into stem and arms): rejoin them
    ink = cv2.morphologyEx(ink.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)).astype(bool)
    zeros = np.zeros_like(ink)
    return segment_glyphs({"black": ink, "red": zeros, "blue": zeros}, table.pitch, glyph_h, soft)


@dataclass(frozen=True)
class TableCell:
    raw: str
    value: float | None
    confidence: float


@dataclass(frozen=True)
class TableColumn:
    volume: TableCell
    delta: TableCell
    cum: TableCell
    clipped: bool  # the column is cut by the image edge; its text is unreliable


def read_table(img: np.ndarray, table: TableGeometry, clf: GlyphClassifier) -> list[TableColumn]:
    gh = table_glyph_height(img, table)
    columns = []
    for col in range(table.n_cols):
        cells = []
        for row in range(3):
            glyphs = cell_glyphs(img, table, row, col, gh)
            chars, dists = [], []
            for g in glyphs:
                ch, d = clf.classify(glyph_features(g, gh))
                chars.append(ch)
                dists.append(d)
            raw = "".join(chars)
            value = parse_signed(raw)
            conf = 0.0 if value is None else float(np.mean([1 / (1 + max(0.0, d - 2.0)) for d in dists]))
            cells.append(TableCell(raw, value, conf))
        columns.append(TableColumn(*cells, clipped=table.first_clipped and col == 0))
    return columns


def load_table_classifier() -> GlyphClassifier:
    data = np.load(Path(__file__).parent / "data" / "table_glyphs.npz", allow_pickle=False)
    return GlyphClassifier(data["features"], data["labels"])
