"""Price-axis calibration: read the right-hand price labels and fit pixel y -> price.

The axis font is a fixed-size monospace with a 9 px advance, right-aligned, so
each label is cut into 9 px character cells and matched against templates.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

STRIP_WIDTH = 85  # px left of the right edge scanned for labels
STRIP_MARGIN = 12  # px at the very right (frame) excluded
INK_THRESHOLD = 170  # min(B,G,R) below this counts as text ink
CELL_W, CELL_H = 9, 12
GLYPH_ROWS = 9  # height of a full digit
TEMPLATE_FILE = Path(__file__).parent / "data" / "axis_templates.npz"
MAX_FIT_RESIDUAL_PX = 1.5
PRICE_RE = re.compile(r"^-?[\d,]+\.\d+$")


@dataclass(frozen=True)
class LabelBand:
    y_top: int
    y_bottom: int  # inclusive
    right_edge: int  # x of the rightmost ink column
    left_edge: int  # x of the leftmost ink column inside the label strip

    @property
    def y_center(self) -> float:
        return self.y_top + (GLYPH_ROWS - 1) / 2


def _runs(idx: np.ndarray, gap: int) -> list[tuple[int, int]]:
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for v in idx[1:]:
        if v > prev + gap:
            out.append((int(start), int(prev)))
            start = v
        prev = v
    out.append((int(start), int(prev)))
    return out


def find_label_bands(img: np.ndarray) -> list[LabelBand]:
    """Text bands in the right-hand label strip, top to bottom (frame lines removed)."""
    h, w = img.shape[:2]
    x0, x1 = w - STRIP_WIDTH, w - STRIP_MARGIN
    ink = img[:, x0:x1].min(axis=2) < INK_THRESHOLD
    ink[ink.mean(axis=1) > 0.9] = False  # horizontal frame lines
    bands = []
    for top, bottom in _runs(np.where(ink.any(axis=1))[0], gap=2):
        cols = np.where(ink[top : bottom + 1].any(axis=0))[0]
        bands.append(LabelBand(top, bottom, int(cols.max()) + x0, int(cols.min()) + x0))
    # Labels are right-aligned, so their true edge is shared; a faint antialiased
    # column can make a single band's measured edge off by one pixel.
    if bands:
        common = Counter(b.right_edge for b in bands).most_common(1)[0][0]
        bands = [
            replace(b, right_edge=common) if abs(b.right_edge - common) <= 1 else b for b in bands
        ]
    return bands


def cell_patch(img: np.ndarray, band: LabelBand, k: int) -> np.ndarray:
    """Float patch (CELL_H x CELL_W, 0..1 ink) of the k-th character from the right."""
    x_end = band.right_edge - CELL_W * k
    x_start = x_end - CELL_W + 1
    if x_start < 0 or x_end < band.left_edge:  # beyond the label: other chart ink
        return np.zeros((CELL_H, CELL_W), dtype=np.float32)
    crop = img[band.y_top : band.y_top + CELL_H, x_start : x_end + 1].min(axis=2)
    ink = 1.0 - crop.astype(np.float32) / 255.0
    ink[ink < 0.3] = 0.0  # drop faint gridline/tick residue
    return ink


def band_patches(img: np.ndarray, band: LabelBand, max_chars: int = 10) -> list[np.ndarray]:
    """Character patches left-to-right."""
    patches = [cell_patch(img, band, k) for k in range(max_chars)]
    while patches and patches[-1].sum() < 1.0:  # trim empty leading cells
        patches.pop()
    return patches[::-1]


def normalize(patch: np.ndarray) -> np.ndarray:
    """Scale to unit peak so anti-aliasing brightness differences between shots cancel."""
    peak = float(patch.max())
    return patch / peak if peak > 0.3 else np.zeros_like(patch)


class GlyphTemplates:
    def __init__(self, chars: list[str], patches: np.ndarray):
        self.chars = chars
        self.patches = patches

    @classmethod
    def load(cls, path: Path = TEMPLATE_FILE) -> "GlyphTemplates":
        data = np.load(path, allow_pickle=False)
        return cls(list(data["chars"]), data["patches"])

    def save(self, path: Path = TEMPLATE_FILE) -> None:
        np.savez(path, chars=np.array(self.chars), patches=self.patches)

    def classify(self, patch: np.ndarray) -> tuple[str, float]:
        """Nearest template, tolerant to sub-pixel horizontal offsets of the glyph."""
        best_i, best_d = 0, float("inf")
        for variant in _shifted_variants(normalize(patch)):
            d = ((self.patches - variant[None]) ** 2).sum(axis=(1, 2))
            i = int(d.argmin())
            if d[i] < best_d:
                best_i, best_d = i, float(d[i])
        return self.chars[best_i], best_d


def _shift_x(patch: np.ndarray, dx: int) -> np.ndarray:
    out = np.zeros_like(patch)
    if dx > 0:
        out[:, dx:] = patch[:, :-dx]
    elif dx < 0:
        out[:, :dx] = patch[:, -dx:]
    else:
        out[:] = patch
    return out


def _shifted_variants(patch: np.ndarray) -> list[np.ndarray]:
    """The patch shifted horizontally in quarter-pixel steps over +-1 px (linear interpolation)."""
    out = []
    for shift in np.arange(-1.0, 1.01, 0.25):
        whole = int(np.floor(shift))
        frac = float(shift - whole)
        out.append((1 - frac) * _shift_x(patch, whole) + frac * _shift_x(patch, whole + 1))
    return out


def read_band(img: np.ndarray, band: LabelBand, tpl: GlyphTemplates, max_dist: float = 1.5) -> str | None:
    if band.y_bottom - band.y_top + 1 < GLYPH_ROWS:
        return None  # clipped by the frame
    out = ""
    for p in band_patches(img, band):
        ch, dist = tpl.classify(p)
        if dist > max_dist:
            return None
        out += "" if ch == " " else ch
    return out


@dataclass(frozen=True)
class PriceAxis:
    slope: float  # price per pixel (negative: price falls as y grows)
    intercept: float
    max_residual_px: float
    labels: tuple[tuple[float, float], ...]  # (y_center, price)

    def y_to_price(self, y: float) -> float:
        return self.slope * y + self.intercept

    def price_to_y(self, price: float) -> float:
        return (price - self.intercept) / self.slope


def parse_price(text: str) -> float:
    return float(text.replace(",", ""))


def calibrate_price_axis(img: np.ndarray, tpl: GlyphTemplates | None = None) -> PriceAxis:
    tpl = tpl or GlyphTemplates.load()
    pts = []
    for band in find_label_bands(img):
        text = read_band(img, band, tpl)
        if text and PRICE_RE.match(text):
            pts.append((band.y_center, parse_price(text)))
    if len(pts) < 3:
        raise ValueError(f"only {len(pts)} price labels read; cannot calibrate")
    # Other label columns (e.g. the lower pane's "0.000") also look like prices; the true
    # axis labels are evenly spaced, so drop the worst outlier until the fit is tight.
    while True:
        ys = np.array([p[0] for p in pts])
        ps = np.array([p[1] for p in pts])
        slope, intercept = np.polyfit(ys, ps, 1)
        resid = np.abs(ys - (ps - intercept) / slope)
        if resid.max() <= MAX_FIT_RESIDUAL_PX or len(pts) <= 3:
            break
        pts.pop(int(resid.argmax()))
    if resid.max() > MAX_FIT_RESIDUAL_PX:
        raise ValueError("price labels are not evenly spaced")
    return PriceAxis(float(slope), float(intercept), float(resid.max()), tuple(pts))
