"""Bar timestamps from the time-axis labels under the summary table.

Each bar has its label centred under it. Most labels are "H:MM"/"HH:MM"; the first bar
of a session shows its date ("5 Oct 26") instead of its time.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibrate import TableGeometry
from .cells import Glyph, GlyphClassifier, glyph_features, segment_glyphs

BAND_TOP = 6  # px below the table where the search starts
MAX_LABEL_OFFSET = 90  # px below the table the search covers
MIN_LABEL_ROWS = 7  # a label is at least this many ink rows tall
BLACKHAT_MIN = 40
_TIME = re.compile(r"^(\d{1,2}):(\d{2})$")
_DATE = re.compile(r"^(\d{1,2})([A-Za-z0]{3})(\d{2})$")


def _blackhat(img: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    value = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2]
    closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return (closed.astype(int) - value) > BLACKHAT_MIN


def find_label_band(img: np.ndarray, table: TableGeometry) -> tuple[int, int]:
    """Pixel rows [y0, y1) of the time labels: the first text-height ink band under the table."""
    y_start = table.y_bottom + BAND_TOP
    y_end = min(img.shape[0] - 4, table.y_bottom + MAX_LABEL_OFFSET)
    ink = _blackhat(img, y_start, y_end, table.x_left, table.x_right + 1)
    rows = np.where(ink.sum(axis=1) >= 3)[0]
    if len(rows) == 0:
        raise ValueError("time-axis labels not found")
    start = prev = rows[0]
    for y in list(rows[1:]) + [None]:
        if y is None or y > prev + 2:
            if prev - start + 1 >= MIN_LABEL_ROWS:
                return y_start + start - 1, y_start + prev + 2
            if y is None:
                break
            start = y
        if y is not None:
            prev = y
    raise ValueError("time-axis labels not found")


def label_mask(img: np.ndarray, table: TableGeometry, col: int) -> np.ndarray:
    x0, x1 = table.col_x(col)
    y0, y1 = find_label_band(img, table)
    return _blackhat(img, y0, y1, x0, x1)


def label_glyph_height(img: np.ndarray, table: TableGeometry) -> int:
    heights: list[int] = []
    for col in range(table.n_cols):
        _, _, stats, _ = cv2.connectedComponentsWithStats(label_mask(img, table, col).astype(np.uint8), connectivity=8)
        heights += [int(s[3]) for s in stats[1:] if s[3] >= 6]
    return int(np.bincount(heights).argmax())


def label_glyphs(img: np.ndarray, table: TableGeometry, col: int, glyph_h: float) -> list[Glyph]:
    ink = label_mask(img, table, col)
    zeros = np.zeros_like(ink)
    return segment_glyphs({"black": ink, "red": zeros, "blue": zeros}, table.pitch, glyph_h, ink.astype(np.float32))


@dataclass(frozen=True)
class BarLabel:
    raw: str  # classified characters without spaces, e.g. "10:15" or "5Oct26"
    minutes: int | None  # minutes after midnight, for a time label
    date: str | None  # e.g. "5 Oct 26", for a session-start label
    confidence: float


def read_labels(img: np.ndarray, table: TableGeometry, clf: GlyphClassifier) -> list[BarLabel]:
    gh = label_glyph_height(img, table)
    out = []
    for col in range(table.n_cols):
        glyphs = label_glyphs(img, table, col, gh)
        chars, dists = [], []
        for g in glyphs:
            ch, d = clf.classify(glyph_features(g, gh))
            chars.append(ch)
            dists.append(d)
        raw = "".join(chars)
        if ":" in raw:  # a clock time has no letters: the capital O is a zero
            raw = raw.replace("O", "0")
        conf = float(np.mean([1 / (1 + max(0.0, d - 2.0)) for d in dists])) if dists else 0.0
        t, d = _TIME.match(raw), _DATE.match(raw)
        minutes = int(t.group(1)) * 60 + int(t.group(2)) if t and int(t.group(2)) < 60 and int(t.group(1)) < 24 else None
        date = f"{d.group(1)} {d.group(2).replace('0', 'O')} {d.group(3)}" if d else None
        out.append(BarLabel(raw, minutes, date, conf))
    return out


def load_label_classifier() -> GlyphClassifier:
    data = np.load(Path(__file__).parent / "data" / "label_glyphs.npz", allow_pickle=False)
    return GlyphClassifier(data["features"], data["labels"])
