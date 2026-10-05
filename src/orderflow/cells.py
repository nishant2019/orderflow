"""Footprint cell text: locate cells, build a text-ink mask, segment glyphs.

Text is black, or red/blue on orange imbalance cells (red = sell-side bid, blue =
buy-side ask). The ink test therefore depends on the cell's background colour.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .calibrate import TableGeometry
from .rows import RowGrid

DARK_RATIO = 0.8  # black text: value below this fraction of the surrounding (closed) value
BLACKHAT_MIN = 45  # and at least this much darker than it
ORANGE_H = (16, 23)  # OpenCV hue of the orange imbalance fill (the POC border ring is hue ~12)


@dataclass(frozen=True)
class CellRect:
    col: int
    row: int  # row index in the RowGrid
    x0: int
    x1: int  # exclusive
    y0: int
    y1: int  # exclusive


def cell_rect(geo: TableGeometry, grid: RowGrid, col: int, row: int) -> CellRect:
    x0, x1 = geo.col_x(col)
    top = grid.row_top(row)
    return CellRect(col, row, x0, x1, int(round(top)), int(round(top + grid.pitch)))


def _is_orange(hsv_px: np.ndarray) -> np.ndarray:
    h, s, v = hsv_px[..., 0], hsv_px[..., 1], hsv_px[..., 2]
    return (h >= ORANGE_H[0]) & (h <= ORANGE_H[1]) & (s > 150) & (v > 180)


def ink_masks(img: np.ndarray, rect: CellRect, min_orange_share: float | None = None) -> dict[str, np.ndarray]:
    """Boolean masks (cell-shaped) of black, red and blue text pixels.

    Red and blue text only occurs on orange imbalance fills. By default a few orange pixels
    anywhere in the rectangle count (single-column layout, where the orange bar can be tiny);
    `min_orange_share` instead requires that share of the rectangle's middle band (split
    layout, where the neighbouring row's orange box can leak into the margins).
    """
    crop = img[rect.y0 : rect.y1, rect.x0 : rect.x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    b, g, r = (crop[..., i].astype(int) for i in range(3))
    value = hsv[..., 2]
    # Black text = thin dark strokes. A black-hat (closing - image) responds to strokes up to
    # ~4 px but ignores large flat bar fills, even ones darker than the white beside them.
    closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    black = ((closed.astype(int) - value) > BLACKHAT_MIN) & (value < DARK_RATIO * closed) & (closed > 100)
    blue = (b > 170) & (r < 110) & (g < 110)
    red = (r > 190) & (g < 90) & (b < 110)
    # red/blue text only occurs on orange imbalance fills; elsewhere red is a bar fill
    if min_orange_share is None:
        on_orange = _is_orange(hsv).sum() >= 4  # even a tiny orange bar marks an imbalance cell
    else:
        h = hsv.shape[0]
        band = _is_orange(hsv[h // 4 : h - h // 4])
        on_orange = band.mean() >= min_orange_share
    if not on_orange:
        red = np.zeros_like(red)
        blue = np.zeros_like(blue)
    else:
        # a red *fill* (sell-delta bar) is also red; text strokes are thin, fills are solid
        kernel = np.ones((5, 5), np.uint8)
        solid = cv2.dilate(cv2.erode(red.astype(np.uint8), kernel), kernel).astype(bool)
        red = red & ~solid
    return {"black": black, "red": red, "blue": blue}


def soft_ink_map(img: np.ndarray, rect: CellRect, masks: dict[str, np.ndarray]) -> np.ndarray:
    """Anti-aliased ink coverage (0..1): black-hat strength for black text, 1 on red/blue text."""
    crop = img[rect.y0 : rect.y1, rect.x0 : rect.x1]
    value = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)[..., 2]
    closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(np.float32)
    soft = np.clip((closed - value) / np.maximum(closed, 1.0), 0.0, 1.0)
    soft = np.where(masks["black"], soft, 0.0)
    coloured = masks["red"] | masks["blue"]
    return np.where(coloured, 1.0, soft).astype(np.float32)


@dataclass(frozen=True)
class Glyph:
    x0: int  # cell-local bbox
    y0: int
    w: int
    h: int
    color: str  # dominant ink colour: black | red | blue
    mask: np.ndarray  # bool, bbox-shaped
    soft: np.ndarray | None = None  # float 0..1 ink coverage, bbox-shaped
    rel_y: float = 0.0  # glyph top below the text line's top, in px (separates '.' from '-')


TYPICAL_W = 0.8  # digit width as a fraction of digit height (typical across the shots)


def _longest_run(col: np.ndarray) -> int:
    best = cur = 0
    for v in col:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def _remove_non_text(union: np.ndarray, glyph_h: float) -> np.ndarray:
    """Drop candle bodies/wicks and POC border lines without touching glyphs they abut.

    Columns whose ink is a vertical run taller than a glyph are candle ink; rows whose
    ink is a horizontal run much wider than a glyph are border lines.
    """
    out = union.copy()
    tall = np.array([_longest_run(union[:, x]) > glyph_h * 1.25 for x in range(union.shape[1])])
    out[:, tall] = False
    wide = np.array([_longest_run(out[y]) > glyph_h * 2.6 for y in range(out.shape[0])])
    out[wide, :] = False
    return out


def _keep_text_band(union: np.ndarray, glyph_h: float) -> tuple[np.ndarray, int]:
    """Keep only the rows of the one text line (a window one glyph tall); bar edges and
    gridline fragments above/below it are dropped."""
    rows = union.sum(axis=1)
    h = int(round(glyph_h))
    if len(rows) <= h:
        return union, 0
    sums = np.convolve(rows, np.ones(h + 1), mode="valid")
    y0 = int(sums.argmax())
    out = np.zeros_like(union)
    out[y0 : y0 + h + 1] = union[y0 : y0 + h + 1]
    return out, y0


def _column_runs(cols: np.ndarray) -> list[tuple[int, int]]:
    runs, start = [], None
    for x, on in enumerate(cols):
        if on and start is None:
            start = x
        elif not on and start is not None:
            runs.append((start, x))
            start = None
    if start is not None:
        runs.append((start, len(cols)))
    return runs


def _merge_close_runs(runs: list[tuple[int, int]], max_w: float) -> list[tuple[int, int]]:
    """Join runs split by a 1 px gap when together they are no wider than one glyph
    (thin K/X strokes sometimes break apart under the ink threshold)."""
    out: list[tuple[int, int]] = []
    for r in runs:
        if out and r[0] - out[-1][1] == 1 and r[1] - out[-1][0] <= max_w:
            out[-1] = (out[-1][0], r[1])
        else:
            out.append(r)
    return out


def _peel_dot(union: np.ndarray, x0: int, x1: int, typ_w: float) -> list[tuple[int, int]]:
    """A '.' kerned against the preceding digit forms one run; peel it off the right end
    when its last 1-3 columns hold ink only in the bottom rows."""
    rows = np.where(union[:, x0:x1].any(axis=1))[0]
    if len(rows) == 0:
        return [(x0, x1)]
    top, bottom = rows.min(), rows.max()
    low = bottom - 0.3 * (bottom - top)
    dot_cols = 0
    for x in range(x1 - 1, x0, -1):
        ys = np.where(union[:, x])[0]
        if 2 <= len(ys) <= 3 and ys.min() >= low:  # a period is a filled block >= 2 rows tall
            dot_cols += 1
        else:
            break
    if 1 <= dot_cols <= 3 and (x1 - x0 - dot_cols) >= 0.6 * typ_w and (x1 - x0) > 1.2 * typ_w:
        return [(x0, x1 - dot_cols), (x1 - dot_cols, x1)]
    return [(x0, x1)]


def _split_run(proj: np.ndarray, x0: int, x1: int, typ_w: float) -> list[tuple[int, int]]:
    """Split a run that is wider than one glyph at the weakest columns (single-column layout)."""
    n = int(round((x1 - x0) / typ_w))
    if n <= 1:
        return [(x0, x1)]
    cuts = []
    for k in range(1, n):
        mid = x0 + (x1 - x0) * k / n
        lo, hi = max(x0 + 1, int(mid - 1.5)), min(x1 - 1, int(mid + 1.5) + 1)
        cuts.append(lo + int(np.argmin(proj[lo : hi + 1])) if hi >= lo else int(mid))
    edges = [x0, *cuts, x1]
    return [(edges[i], edges[i + 1]) for i in range(n)]


def _split_run_advance(proj: np.ndarray, x0: int, x1: int, advance: float) -> list[tuple[int, int]]:
    """Split a run of touching glyphs using the font's measured advance (split layout).

    n glyphs occupy about n * advance - 1 px; a count that would leave a sliver narrower than
    0.6 of a slot is rejected in favour of fewer glyphs.
    """
    n = max(1, int(round((x1 - x0 + 1) / advance)))
    while n > 1:
        cuts = []
        for k in range(1, n):
            mid = x0 + (x1 - x0) * k / n
            lo, hi = max(x0 + 1, int(mid - 1.5)), min(x1 - 1, int(mid + 1.5) + 1)
            cuts.append(lo + int(np.argmin(proj[lo : hi + 1])) if hi >= lo else int(mid))
        edges = [x0, *cuts, x1]
        if min(b - a for a, b in zip(edges, edges[1:])) >= 0.6 * advance:
            return [(edges[i], edges[i + 1]) for i in range(n)]
        n -= 1
    return [(x0, x1)]


def segment_glyphs(
    masks: dict[str, np.ndarray], pitch: float, glyph_h: float, soft: np.ndarray | None = None,
    advance: float | None = None,
) -> list[Glyph]:
    """Glyphs of a cell, left to right: runs of ink columns, splitting touching glyphs."""
    union = np.zeros_like(masks["black"])
    for m in masks.values():
        union |= m
    union = _remove_non_text(union, glyph_h)
    union, band_top = _keep_text_band(union, glyph_h)
    proj = union.sum(axis=0)
    typ_w = advance if advance else TYPICAL_W * glyph_h  # glyph slot width, used to split touching glyphs
    out = []
    for r0, r1 in _merge_close_runs(_column_runs(proj > 0), 1.1 * typ_w):
        if (r1 - r0) > 1.4 * typ_w:
            pieces = _split_run_advance(proj, r0, r1, advance) if advance else _split_run(proj, r0, r1, typ_w)
        else:
            pieces = _peel_dot(union, r0, r1, typ_w)
        for x0, x1 in pieces:
            sub = union[:, x0:x1]
            rows = np.where(sub.any(axis=1))[0]
            if len(rows) == 0:
                continue
            y0, y1 = int(rows.min()), int(rows.max()) + 1
            mask = sub[y0:y1]
            counts = {c: int((m[y0:y1, x0:x1] & mask).sum()) for c, m in masks.items()}
            patch = None if soft is None else soft[y0:y1, x0:x1] * mask
            out.append(Glyph(x0, y0, x1 - x0, y1 - y0, max(counts, key=counts.get), mask, patch, float(y0 - band_top)))
    return _drop_strays(out, glyph_h)


DOT_GAP = 0.9  # max gap between a period and its neighbours, as a fraction of glyph height


def _drop_strays(glyphs: list[Glyph], glyph_h: float) -> list[Glyph]:
    """Remove candle-fringe slivers and isolated specks. A period always sits between two
    glyphs, so a dot-sized glyph without neighbours on both sides is noise. A cell made
    only of dots (the chart's "..." ellipsis) is left untouched."""
    # a period is small and roughly square; a minus sign is about 3:1
    is_dot = [g.h <= 0.4 * glyph_h and g.w <= 0.5 * glyph_h and g.w <= 2 * g.h for g in glyphs]
    if glyphs and all(is_dot):
        return glyphs if len(glyphs) >= 3 else []  # "..." is an ellipsis; a lone speck is noise
    out = []
    for i, g in enumerate(glyphs):
        if g.w == 1 and g.h >= 3:
            continue  # vertical sliver
        if is_dot[i]:
            left = i > 0 and g.x0 - (glyphs[i - 1].x0 + glyphs[i - 1].w) <= DOT_GAP * glyph_h
            right = i + 1 < len(glyphs) and glyphs[i + 1].x0 - (g.x0 + g.w) <= DOT_GAP * glyph_h
            if not (left and right):
                continue
        out.append(g)
    return _drop_far_ends(out, glyph_h)


END_GAP = 2.0  # glyphs this many glyph-heights away from the main text are candle remnants


def _drop_far_ends(glyphs: list[Glyph], glyph_h: float) -> list[Glyph]:
    glyphs = list(glyphs)
    limit = END_GAP * glyph_h
    while len(glyphs) > 1 and glyphs[1].x0 - (glyphs[0].x0 + glyphs[0].w) > limit:
        glyphs.pop(0)
    while len(glyphs) > 1 and glyphs[-1].x0 - (glyphs[-2].x0 + glyphs[-2].w) > limit:
        glyphs.pop()
    return glyphs


FEAT_H, FEAT_W = 14, 10
EXTRA_WEIGHT = 6.0  # weight of size/position features against the 140 shape values


def estimate_glyph_height(img: np.ndarray, rects: list[CellRect], pitch: float) -> int:
    """Digit height of the shot: the most common component height in the cell text."""
    heights: list[int] = []
    for rect in rects:
        m = ink_masks(img, rect)
        union = m["black"] | m["red"] | m["blue"]
        _, _, stats, _ = cv2.connectedComponentsWithStats(union.astype(np.uint8), connectivity=8)
        heights += [int(s[3]) for s in stats[1:] if 6 <= s[3] <= pitch * 0.7]
    return int(np.bincount(heights).argmax())


def glyph_features(glyph: Glyph, glyph_h: float) -> np.ndarray:
    """Size-normalised shape vector: resized bbox plus width and height relative to a digit."""
    source = glyph.mask.astype(np.float32) if glyph.soft is None else glyph.soft
    shape = cv2.resize(source, (FEAT_W, FEAT_H), interpolation=cv2.INTER_AREA)
    extra = np.array(
        [glyph.w / glyph_h * EXTRA_WEIGHT, glyph.h / glyph_h * EXTRA_WEIGHT, glyph.rel_y / glyph_h * EXTRA_WEIGHT], dtype=np.float32
    )
    return np.concatenate([shape.ravel(), extra])


class GlyphClassifier:
    """k-nearest-neighbour classifier over size-normalised glyph shapes."""

    def __init__(self, features: np.ndarray, labels: np.ndarray, k: int = 3):
        self.features, self.labels, self.k = features, labels, k

    @classmethod
    def load(cls) -> "GlyphClassifier":
        data = np.load(Path(__file__).parent / "data" / "cell_glyphs.npz", allow_pickle=False)
        return cls(data["features"], data["labels"])

    def classify(self, feat: np.ndarray) -> tuple[str, float]:
        d = ((self.features - feat[None]) ** 2).sum(axis=1)
        idx = np.argsort(d)[: self.k]
        votes: dict[str, float] = {}
        for i in idx:
            votes[str(self.labels[i])] = votes.get(str(self.labels[i]), 0.0) + 1.0 / (1.0 + d[i])
        best = max(votes, key=votes.get)
        return best, float(d[idx[0]])


@dataclass(frozen=True)
class CellText:
    raw: str  # classified characters, e.g. "3.5KX20K"
    bid: float | None  # volume (rounded as displayed: 3.5K -> 3500.0)
    ask: float | None
    sell_imbalance: bool  # bid drawn in red
    buy_imbalance: bool  # ask drawn in blue
    ellipsis: bool  # chart printed "..." (cell too small for text)
    confidence: float  # 0..1; low when glyphs matched poorly or the text did not parse
    bid_tol: float = 0.0  # half the last displayed digit: 3.5K is +-50, 326 is +-0.5
    ask_tol: float = 0.0
    inferred: bool = False  # solved from the chart's totals because the text was unreadable


_NUM = re.compile(r"^(\d+(?:\.\d+)?)([KM]?)$")
_SCALE = {"": 1.0, "K": 1e3, "M": 1e6}
GOOD_DIST = 2.0  # nearest-neighbour distance at which a glyph is considered a clean match


def display_tolerance(text: str) -> float:
    """Half a unit of the last displayed digit, in volume units."""
    m = _NUM.match(text)
    if not m:
        return 0.0
    decimals = len(m.group(1).split(".")[1]) if "." in m.group(1) else 0
    return 0.5 * 10.0 ** (-decimals) * _SCALE[m.group(2)]


def parse_number(text: str) -> float | None:
    m = _NUM.match(text)
    if not m:
        return None
    return float(m.group(1)) * _SCALE[m.group(2)]


def _read(glyphs: list[Glyph], glyph_h: float, clf: GlyphClassifier) -> tuple[list[str], list[float]]:
    chars, dists = [], []
    for g in glyphs:
        ch, d = clf.classify(glyph_features(g, glyph_h))
        chars.append(ch)
        dists.append(d)
    return chars, dists


def _try_parse(chars: list[str]) -> tuple[float, float, int] | None:
    sep = [i for i, c in enumerate(chars) if c == "X"]
    if len(sep) != 1:
        return None
    bid, ask = parse_number("".join(chars[: sep[0]])), parse_number("".join(chars[sep[0] + 1 :]))
    return None if bid is None or ask is None else (bid, ask, sep[0])


REPAIR_PENALTY = 0.7  # confidence factor when a stray end glyph had to be dropped


def parse_cell(glyphs: list[Glyph], glyph_h: float, clf: GlyphClassifier) -> CellText:
    """Read one footprint cell: "<bid> X <ask>" with colour flags.

    A glyph that is not part of the text (a candle remnant at either end) makes the text
    unparseable; in that case up to two end glyphs are dropped and the shortest repair is used.
    """
    if glyphs and all(g.h <= 0.4 * glyph_h for g in glyphs):
        return CellText("...", None, None, False, False, True, 1.0)
    chars, dists = _read(glyphs, glyph_h, clf)
    raw = "".join(chars)
    penalty = 1.0
    parsed = _try_parse(chars)
    lo, hi = 0, len(glyphs)
    if parsed is None:
        for drops in range(1, 3):
            for left in range(drops + 1):
                cand = chars[left : len(chars) - (drops - left)]
                parsed = _try_parse(cand)
                if parsed is not None:
                    lo, hi, penalty = left, len(chars) - (drops - left), REPAIR_PENALTY
                    break
            if parsed is not None:
                break
    if parsed is None:
        return CellText(raw, None, None, False, False, False, 0.0)
    bid, ask, sep = parsed
    kept = glyphs[lo:hi]
    left_g, right_g = kept[:sep], kept[sep + 1 :]
    left_red = sum(g.color == "red" for g in left_g) > len(left_g) / 2
    right_blue = sum(g.color == "blue" for g in right_g) > len(right_g) / 2
    quality = float(np.mean([1.0 / (1.0 + max(0.0, d - GOOD_DIST)) for d in dists[lo:hi]]))
    text = "".join(chars[lo:hi])
    bid_text, ask_text = text.split("X")
    return CellText(
        text, bid, ask, left_red, right_blue, False, quality * penalty,
        display_tolerance(bid_text), display_tolerance(ask_text),
    )
