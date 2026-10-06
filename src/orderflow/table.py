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
_PCT = re.compile(r"^(\d+(?:\.\d+)?)%$")
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


def estimate_advance(img: np.ndarray, table: TableGeometry, glyph_h: float) -> float:
    """The monospace character advance, from the spacing of neighbouring digits in the Volume row
    (dark text on a pale fill, which does not break apart): the left edges of tall glyphs that sit
    next to each other are one advance apart (two when a period lies between them)."""
    diffs: list[float] = []
    for col in range(table.n_cols):
        if table.first_clipped and col == 0:
            continue
        ink, _ = cell_ink(img, table, 0, col)
        _, _, stats, _ = cv2.connectedComponentsWithStats(ink.astype(np.uint8), connectivity=8)
        tall = sorted(int(x) for x, y, w, h, area in stats[1:] if h >= 0.8 * glyph_h and w >= 3)
        diffs += [b - a for a, b in zip(tall, tall[1:])]
    d = np.array([x for x in diffs if 0.5 * glyph_h <= x <= 2.6 * glyph_h], float)
    if len(d) < 5:
        return float(0.9 * glyph_h)
    base = np.percentile(d, 20)
    near = d[(d >= 0.85 * base) & (d <= 1.15 * base)]
    return float(np.median(near))


def _slot_glyphs(ink: np.ndarray, soft: np.ndarray, advance: float) -> list[Glyph]:
    """Cut a cell's text into equal character slots (monospace font): immune to strokes that
    break apart or touch, which defeats connected-component segmentation on thin white text."""
    cols = np.where(ink.any(axis=0))[0]
    if len(cols) == 0:
        return []
    a, b = int(cols.min()), int(cols.max()) + 1
    n = max(int(round((b - a + 1.0) / advance)), 1)
    start = (a + b) / 2 - n * advance / 2
    rows = np.where(ink.any(axis=1))[0]
    band_top = int(rows.min())
    out = []
    for i in range(n):
        x0, x1 = int(round(start + i * advance)), int(round(start + (i + 1) * advance))
        x0, x1 = max(x0, 0), min(x1, ink.shape[1])
        sub = ink[:, x0:x1]
        r = np.where(sub.any(axis=1))[0]
        c = np.where(sub.any(axis=0))[0]
        if len(r) == 0:
            continue
        y0, y1, xa, xb = int(r.min()), int(r.max()) + 1, int(c.min()), int(c.max()) + 1
        mask = sub[y0:y1, xa:xb]
        out.append(Glyph(x0 + xa, y0, xb - xa, y1 - y0, "black", mask, soft[y0:y1, x0 + xa : x0 + xb] * mask, float(y0 - band_top)))
    return out


def cell_glyphs(img: np.ndarray, table: TableGeometry, row: int, col: int, glyph_h: float,
                advance: float | None = None) -> list[Glyph]:
    ink, soft = cell_ink(img, table, row, col)
    if advance:
        return _slot_glyphs(ink, soft, advance)
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
    delta_pct: TableCell | None = None  # the optional 4th row, delta / volume in percent (signed like the delta)


def read_table(img: np.ndarray, table: TableGeometry, clf: GlyphClassifier) -> list[TableColumn]:
    gh = table_glyph_height(img, table)
    adv = estimate_advance(img, table, gh)
    columns = []
    for col in range(table.n_cols):
        cells = []
        pct = None
        for row in range(table.n_rows):
            glyphs = cell_glyphs(img, table, row, col, gh, adv)
            chars, dists = [], []
            for g in glyphs:
                ch, d = clf.classify(glyph_features(g, gh))
                chars.append(ch)
                dists.append(d)
            raw = "".join(chars)
            if row == 3:  # "38.55%": the colour shows the sign, the text does not
                m = _PCT.match(raw)
                value = None if not m else float(m.group(1))
            else:
                value = parse_signed(raw)
            conf = 0.0 if value is None else float(np.mean([1 / (1 + max(0.0, d - 2.0)) for d in dists]))
            cell = TableCell(raw, value, conf)
            if row == 3:
                pct = cell
            else:
                cells.append(cell)
        if pct is not None and pct.value is not None and cells[1].value is not None and cells[1].value < 0:
            pct = TableCell(pct.raw, -pct.value, pct.confidence)
        columns.append(TableColumn(*cells, clipped=table.first_clipped and col == 0, delta_pct=pct))
    return columns


def load_table_classifier() -> GlyphClassifier:
    data = np.load(Path(__file__).parent / "data" / "table_glyphs.npz", allow_pickle=False)
    return GlyphClassifier(data["features"], data["labels"])
