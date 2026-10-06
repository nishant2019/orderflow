"""Split-box footprint layout: each bar is two columns of boxes per price row, bid on the left
and ask on the right, one number per box.

A box is orange when that side has an imbalance (red text on the bid side, blue on the ask
side); the point of control is a black rectangle around the pair of boxes of one row.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .axis import PriceAxis, calibrate_price_axis
from .calibrate import TableGeometry, calibrate_table
from .cells import CellRect, Glyph, _is_orange, ink_masks, segment_glyphs, soft_ink_map
from .overlays import remove_dashed_price_line, remove_price_line
from .rows import RowGrid, snap_step

WHITE_MAX = 235  # a pixel with min(B,G,R) below this is part of a box (or ink)
BLACK_MAX = 40  # pure-black ring pixels are not box fill
MIN_BOX_H, MAX_BOX_H = 10, 48
MIN_FILL = 0.45
LEFT_MARGIN = 0  # px from the column's left edge before boxes can start (candles are whitened beforehand)
OUTER_INSET = 3  # px trimmed from the pair's outer edges when reading text


@dataclass(frozen=True)
class BoxGeometry:
    left: tuple[int, int]  # x range [start, end) of the bid box, relative to the column's left edge
    right: tuple[int, int]  # x range of the ask box

    def half(self, table: TableGeometry, col: int, side: str) -> tuple[int, int]:
        x0, _ = table.col_x(col)
        lo, hi = self.left if side == "bid" else self.right
        # the POC rectangle's side bars sit just outside the pair: stay clear of them
        return (x0 + lo + OUTER_INSET, x0 + hi) if side == "bid" else (x0 + lo, x0 + hi - OUTER_INSET)


def _box_mask(img: np.ndarray) -> np.ndarray:
    m = img.min(axis=2) < WHITE_MAX
    return m & (img.max(axis=2) > BLACK_MAX)  # drop black ring / text-black pixels


def find_box_geometry(img: np.ndarray, table: TableGeometry, y0: int, y1: int) -> BoxGeometry:
    """Where the two boxes sit inside a column, measured from the fills of all columns."""
    mask = _box_mask(img)[y0:y1]
    lefts, mids_l, mids_r, rights = [], [], [], []
    for col in range(table.n_cols):
        x0, x1 = table.col_x(col)
        if table.first_clipped and col == 0:
            continue
        proj = mask[:, x0:x1].sum(axis=0).astype(float)
        if proj.max() < 20:
            continue
        on = proj >= 0.45 * proj.max()
        on[:LEFT_MARGIN] = False
        xs = np.where(on)[0]
        if len(xs) < 20:
            continue
        lefts.append(xs.min())
        rights.append(xs.max() + 1)
        centre = (xs.min() + xs.max()) // 2
        gap = [x for x in range(centre - 6, centre + 7) if not on[x]]
        if gap:
            mids_l.append(min(gap))
            mids_r.append(max(gap) + 1)
    if not lefts:
        raise ValueError("no boxes found")
    m = lambda v: int(round(float(np.median(v))))
    return BoxGeometry((m(lefts), m(mids_l)), (m(mids_r), m(rights)))


def _blackhat(img: np.ndarray, y0: int, y1: int, x0: int, x1: int, thr: int = 40) -> np.ndarray:
    value = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)[..., 2]
    closed = cv2.morphologyEx(value, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    return (closed.astype(int) - value) > thr


def fit_text_row_grid(img: np.ndarray, axis: PriceAxis, table: TableGeometry, box: BoxGeometry) -> tuple[RowGrid, int, int, int]:
    """Row pitch and phase from the vertical positions of the numbers.

    Box fills are a poor lattice (pale boxes read as white, stacked boxes merge), but every box
    holds one centred number, so digit tops sit on an exact row lattice. Returns
    (grid, k_first, k_last, glyph_h).
    """
    ys = [y for y, _ in axis.labels]
    y_lo, y_hi = int(min(ys)) - 25, int(max(ys)) + 25
    tops: list[int] = []
    heights: list[int] = []
    for col in range(table.n_cols):
        for side in ("bid", "ask"):
            a, b = box.half(table, col, side)
            mask = _blackhat(img, y_lo, y_hi, a, b).astype(np.uint8)
            _, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for x, y, w, h, area in stats[1:]:
                if 7 <= h <= 12 and w >= 3:
                    tops.append(y + y_lo)
                    heights.append(h)
    if len(tops) < 8:
        raise ValueError("too little text to fit the row grid")
    glyph_h = int(np.bincount(heights).argmax())
    uniq = np.unique([t for t, h in zip(tops, heights) if abs(h - glyph_h) <= 1]).astype(float)
    diffs = np.diff(uniq)
    diffs = diffs[diffs >= MIN_BOX_H]
    pitch = float(np.median(diffs[diffs <= diffs.min() * 1.25]))
    for _ in range(4):  # regression, dropping outliers each round
        k = np.round((uniq - uniq[0]) / pitch)
        slope, origin = np.polyfit(k, uniq, 1)
        resid = np.abs(uniq - (origin + k * slope))
        keep = resid <= 1.5
        if keep.all() or keep.sum() < 4:
            break
        uniq = uniq[keep]
        pitch = slope
    k = np.round((uniq - uniq[0]) / pitch)
    pitch, text_origin = np.polyfit(k, uniq, 1)
    # the number is vertically centred in its row: row top = text centre - pitch / 2
    row_origin = text_origin + glyph_h / 2 - pitch / 2
    grid = RowGrid(float(pitch), float(row_origin), float(pitch) - 2, snap_step(float(pitch * abs(axis.slope))))
    return grid, grid.row_index(min(ys)), grid.row_index(max(ys)), glyph_h


# --- the split-box shot -------------------------------------------------------------------------

from .cells import CellText, GlyphClassifier, display_tolerance, glyph_features, parse_number  # noqa: E402
from .pipeline import ShotGeometry  # noqa: E402

ORANGE_FILL_MIN = 0.25  # share of a box that must be orange to count as an imbalance box
ROW_INSET = 2  # px trimmed from the top and bottom of a box when reading text
MIN_ORANGE_PIXELS = 300  # fewer orange fill pixels in the footprint area: the chart has imbalances switched off
POC_LINE_FRACTION = 0.7  # of the pair's width, for a black row to count as an edge of the POC rectangle
MIN_FILLED = 0.5  # share of a box that must be non-white for the row to exist at all


@dataclass
class SplitShotGeometry(ShotGeometry):
    box: BoxGeometry = None  # type: ignore[assignment]
    raw: np.ndarray = None  # type: ignore[assignment]  # the screenshot as read; `img` has the price line patched out
    text_img: np.ndarray = None  # type: ignore[assignment]  # `img` with candle strokes whitened, for text reading
    advance: float = 0.0  # px between consecutive glyphs of the (monospace) cell font
    imbalance_ratio: float = 3.0  # the chart's imbalance setting (Ratio, 300 %); see docs/CHART_SETTINGS.md
    imbalance_shown: bool = True  # False when the chart was taken with "Show Imbalances" off (no orange anywhere)
    overrides: dict = field(default_factory=dict)  # (col, row) -> CellText, e.g. inferred cells
    candle_window: tuple = (1, 8)  # px left/right of a column's left edge searched for its candle (measured per image)

    layout = "split"

    def half_rect(self, col: int, row: int, side: str) -> CellRect:
        a, b = self.box.half(self.table, col, side)
        top = self.grid.row_top(row)
        # stay inside the box vertically: neighbouring rows' fills must not leak into the rectangle
        return CellRect(col, row, a, b, int(round(top)) + ROW_INSET, int(round(top + self.grid.pitch)) - ROW_INSET)

    def half_glyphs(self, col: int, row: int, side: str) -> list[Glyph]:
        rect = self.half_rect(col, row, side)
        masks = ink_masks(self.text_img, rect, min_orange_share=ORANGE_FILL_MIN)
        return segment_glyphs(masks, self.grid.pitch, self.glyph_h, soft_ink_map(self.text_img, rect, masks),
                              advance=self.advance or None)

    def half_stats(self, col: int, row: int, side: str) -> tuple[float, float]:
        """(non-white share, orange share) of a box."""
        rect = self.half_rect(col, row, side)
        crop = self.text_img[rect.y0 : rect.y1, rect.x0 : rect.x1]
        filled = float((crop.min(axis=2) < WHITE_MAX).mean())
        orange = float(_is_orange(cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)).mean())
        return filled, orange

    def find_poc_rows(self) -> dict[int, int]:
        """The POC is a black rectangle (3 px) around the pair of boxes of one row."""
        black = (self.raw.max(axis=2) < BLACK_MAX)
        y_lo = int(self.grid.row_top(self.k_first)) - 6
        y_hi = int(self.grid.row_top(self.k_last + 1)) + 6
        out: dict[int, int] = {}
        extents: list[tuple[int, int]] = []
        for col in range(self.table.n_cols):
            a = self.box.half(self.table, col, "bid")[0] - OUTER_INSET - 4
            b = self.box.half(self.table, col, "ask")[1] + OUTER_INSET + 4
            counts = black[y_lo:y_hi, a:b].sum(axis=1)
            ys = np.where(counts >= POC_LINE_FRACTION * (b - a))[0] + y_lo
            if len(ys) == 0:
                continue
            lines, cur = [], [ys[0]]
            for y in ys[1:]:
                if y - cur[-1] <= 1:
                    cur.append(y)
                else:
                    lines.append(float(np.mean(cur)))
                    cur = [y]
            lines.append(float(np.mean(cur)))
            if len(lines) >= 2:
                out[col] = self.grid.row_index((lines[0] + lines[1]) / 2)
                xs = np.where(black[int(round(lines[0])), a - 8 : b + 10])[0] + a - 8
                if len(xs):
                    extents.append((xs.min() - self.table.col_x(col)[0], xs.max() - self.table.col_x(col)[0]))
        sides = None
        if extents:  # where the rectangle's outer edges sit relative to the column's left edge
            sides = (int(np.median([e[0] for e in extents])), int(np.median([e[1] for e in extents])))
        for col in range(self.table.n_cols):
            if col not in out and sides:  # an edge hidden by the price line: use the side bars
                row = self._poc_from_sides(black, col, y_lo, y_hi, sides)
                if row is not None:
                    out[col] = row
        return out

    def _poc_from_sides(self, black: np.ndarray, col: int, y_lo: int, y_hi: int, sides: tuple[int, int]) -> int | None:
        """POC row from the rectangle's two vertical side bars. Both must be present and agree:
        a single bar can belong to the neighbouring column's rectangle."""
        x0, _ = self.table.col_x(col)
        centres = []
        for xa, xb in ((x0 + sides[0], x0 + sides[0] + 4), (x0 + sides[1] - 3, x0 + sides[1] + 1)):
            ys = np.where(black[y_lo:y_hi, xa:xb].sum(axis=1) >= 2)[0] + y_lo
            if len(ys) == 0:
                return None
            runs, cur = [], [ys[0]]
            for y in ys[1:]:
                if y - cur[-1] <= 1 + 0.2 * self.grid.pitch:  # a label drawn over the bar leaves a short gap
                    cur.append(y)
                else:
                    runs.append(cur)
                    cur = [y]
            runs.append(cur)
            run = max(runs, key=lambda r: r[-1] - r[0])
            if run[-1] - run[0] + 1 < 0.8 * self.grid.pitch:
                return None
            centres.append((run[0] + run[-1]) / 2)
        if abs(centres[0] - centres[1]) > 3:
            return None
        return self.grid.row_index(sum(centres) / 2)

    def text_cells(self, col: int):  # not meaningful for two boxes; kept for interface parity
        raise NotImplementedError("use read_cells for the split layout")

    def read_cells(self, col: int, clf: GlyphClassifier) -> list[tuple[int, CellText]]:
        out = []
        for row in range(self.k_first, self.k_last + 1):
            cell = self.overrides.get((col, row)) or self.read_cell(col, row, clf)
            if cell is not None:
                if self.table.first_clipped and col == 0 and not cell.inferred:
                    # a column cut by the image edge shows only part of its boxes: its text cannot be
                    # trusted, so the cell counts as unreadable (and may be solved from the totals)
                    cell = CellText("clipped", None, None, False, False, False, 0.0)
                out.append((row, cell))
        if not self.imbalance_shown:
            out = derive_imbalance_flags(out, self.imbalance_ratio)
        return out

    def read_cell(self, col: int, row: int, clf: GlyphClassifier) -> CellText | None:
        halves = {}
        stats = {side: self.half_stats(col, row, side) for side in ("bid", "ask")}
        if all(stats[side][0] < MIN_FILLED for side in stats):
            return None  # no boxes on this row for this bar
        for side in ("bid", "ask"):
            filled, orange = stats[side]
            # a box can be (nearly) white, e.g. a zero volume: it still holds text when its sibling is drawn
            glyphs = self.half_glyphs(col, row, side)
            chars, dists = [], []
            for g in glyphs:
                ch, d = clf.classify(glyph_features(g, self.glyph_h))
                chars.append(ch)
                dists.append(d)
            red = sum(g.color == "red" for g in glyphs) > len(glyphs) / 2 if glyphs else False
            blue = sum(g.color == "blue" for g in glyphs) > len(glyphs) / 2 if glyphs else False
            halves[side] = ("".join(chars), dists, orange >= ORANGE_FILL_MIN, red, blue)
        if halves["bid"] is None and halves["ask"] is None:
            return None  # no boxes on this row for this bar
        bid_t, bid_d, bid_or, bid_red, _ = halves["bid"] or ("", [], False, False, False)
        ask_t, ask_d, ask_or, _, ask_blue = halves["ask"] or ("", [], False, False, False)
        bid, ask = parse_number(bid_t), parse_number(ask_t)
        dists = bid_d + ask_d
        conf = float(np.mean([1.0 / (1.0 + max(0.0, d - 2.0)) for d in dists])) if dists else 0.0
        ok = bid is not None and ask is not None
        if not ok:  # keep the invariant used downstream: a cell either parses fully or not at all
            bid = ask = None
        return CellText(
            f"{bid_t}X{ask_t}", bid, ask, bid_or, ask_or, False, conf if ok else 0.0,
            display_tolerance(bid_t), display_tolerance(ask_t),
        )


_CANDLE_COLORS = (np.array([125, 157, 18]), np.array([50, 60, 233]))  # up (teal) / down (red)
CANDLE_PAD = 2  # px added around the measured candle zone


def find_candle_zone(img: np.ndarray, table: TableGeometry, y0: int, y1: int) -> tuple[int, int]:
    """Where, relative to a column's left edge, the candles are drawn: at the left edge in some
    chart settings, in the centre gap between the bid and ask boxes in others. Measured from the
    exact candle colours over all columns."""
    hist = np.zeros(table.pitch.__int__() + 40, dtype=float)
    shift = 10
    for col in range(table.n_cols):
        x0, x1 = table.col_x(col)
        if table.first_clipped and col == 0:
            continue
        a, b = max(x0 - shift, 0), min(x1 + 10, img.shape[1])
        win = img[y0:y1, a:b].astype(np.int32)
        for color in _CANDLE_COLORS:
            m = np.abs(win - color).sum(axis=2) < 30
            for off, n in zip(range(a - x0 + shift, b - x0 + shift), m.sum(axis=0)):
                if 0 <= off < len(hist):
                    hist[off] += n
    if hist.max() < 20:
        return (0, 7)  # no candles found: the left-edge default
    peak = int(hist.argmax())
    on = hist >= 0.2 * hist.max()
    lo = hi = peak
    while lo > 0 and on[lo - 1]:
        lo -= 1
    while hi < len(hist) - 1 and on[hi + 1]:
        hi += 1
    return lo - shift, hi - shift + 1


def _whiten_candles(img: np.ndarray, table: TableGeometry, zone: tuple[int, int]) -> np.ndarray:
    """Copy of the image with candle strokes painted white, so a wick that touches a box
    cannot be mistaken for a digit (and does not distort the box measurements)."""
    out = img.copy()
    for col in range(table.n_cols):
        x0, _ = table.col_x(col)
        a, b = max(x0 + zone[0] - CANDLE_PAD, 0), x0 + zone[1] + CANDLE_PAD
        region = out[:, a:b]
        for color in _CANDLE_COLORS:
            region[np.abs(region.astype(np.int32) - color).sum(axis=2) < 30] = 255
    return out


def load_split_shot(path: str) -> SplitShotGeometry:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(path)
    axis = calibrate_price_axis(img)
    table = calibrate_table(img)
    clean = remove_dashed_price_line(remove_price_line(img, axis, table), table.y_top)
    ys = [y for y, _ in axis.labels]
    y_lo, y_hi = int(min(ys)) - 25, int(max(ys)) + 25
    zone = find_candle_zone(clean, table, y_lo, y_hi)
    text_img = _whiten_candles(clean, table, zone)
    box = find_box_geometry(text_img, table, y_lo, y_hi)
    grid, k_first, k_last, glyph_h = fit_text_row_grid(text_img, axis, table, box)
    geo = SplitShotGeometry(clean, axis, table, grid, k_first, k_last, glyph_h, box, raw=img, text_img=text_img,
                            candle_window=(CANDLE_PAD - zone[0], zone[1] + CANDLE_PAD))
    geo.advance = _estimate_advance(geo)
    geo.imbalance_shown = _count_orange(text_img, table, y_lo, y_hi) >= MIN_ORANGE_PIXELS
    return geo


def derive_imbalance_flags(cells: list[tuple[int, CellText]], ratio: float) -> list[tuple[int, CellText]]:
    """Imbalance flags from the chart's rule, for charts that do not draw them.

    The rule (verified on every drawn flag of the charts that do draw them): sell imbalance when
    bid[r] >= ratio * ask[row above]; buy imbalance when ask[r] >= ratio * bid[row below]; only
    between rows that are both drawn. Computed from the displayed (rounded) numbers.
    """
    from dataclasses import replace

    by_row = {r: c for r, c in cells if c.bid is not None and c.ask is not None}
    out = []
    for r, c in cells:
        if r not in by_row or c.inferred:
            out.append((r, c))
            continue
        up, down = by_row.get(r - 1), by_row.get(r + 1)
        sell = bool(up and c.bid > 0 and c.bid >= ratio * up.ask)
        buy = bool(down and c.ask > 0 and c.ask >= ratio * down.bid)
        out.append((r, replace(c, sell_imbalance=sell, buy_imbalance=buy)))
    return out


def _count_orange(img: np.ndarray, table: TableGeometry, y0: int, y1: int) -> int:
    hsv = cv2.cvtColor(img[y0:y1, table.x_left : table.x_right + 1], cv2.COLOR_BGR2HSV)
    return int(_is_orange(hsv).sum())


def _estimate_advance(geo: SplitShotGeometry) -> float:
    """The cell font is monospace, but glyph *starts* jitter by a pixel with the digit and the
    sub-pixel position, so the advance is fractional (about 8.5 px). It is the mean spacing over
    two-glyph spans (which damps the jitter), taken around the median, over the whole chart."""
    spans: list[float] = []
    for col in range(geo.table.n_cols):
        for row in range(geo.k_first, geo.k_last + 1):
            for side in ("bid", "ask"):
                g = geo.half_glyphs(col, row, side)
                spans += [(c.x0 - a.x0) / 2 for a, c in zip(g, g[2:])]
    spans = [d for d in spans if 0.6 * geo.glyph_h <= d <= 1.3 * geo.glyph_h]
    if len(spans) < 10:
        return float(geo.glyph_h)
    med = float(np.median(spans))
    return float(np.mean([d for d in spans if abs(d - med) <= 1.0]))
