"""Train the split-layout glyph classifiers from the screenshots themselves.

Hand-transcribing every number in every screenshot does not scale, but the chart prints its own
checksums: each bar's cells must add up to the table, and each price row's cells must add up to the
profile. A glyph is accepted as a training label only when everything it belongs to agrees with
those checks, so a misread that passes both a column sum and a row sum is very unlikely.
The classifiers are then retrained on the accepted glyphs plus the hand-labelled legacy sets, and
the process repeats so that cells that failed before get another chance.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .cells import GlyphClassifier, glyph_features
from .profile2 import read_profile2
from .split import SplitShotGeometry, load_split_shot
from .table import cell_glyphs as table_cell_glyphs
from .table import read_table, table_glyph_height
from .validate import ColumnReport, validate_columns, validate_profile2

DATA = Path(__file__).parent / "data"


@dataclass
class Samples:
    features: list = field(default_factory=list)
    labels: list = field(default_factory=list)

    def add(self, glyphs, chars: str, glyph_h: float) -> None:
        for g, ch in zip(glyphs, chars):
            self.features.append(glyph_features(g, glyph_h))
            self.labels.append(ch)

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self.features, dtype=np.float32), np.array(self.labels)


@dataclass
class Classifiers3:
    cells: GlyphClassifier
    table: GlyphClassifier
    profile: GlyphClassifier


@dataclass
class Verified:
    cells: Samples = field(default_factory=Samples)
    table: Samples = field(default_factory=Samples)
    profile: Samples = field(default_factory=Samples)
    n_cells: int = 0
    n_cells_total: int = 0
    n_table: int = 0
    n_table_total: int = 0
    n_profile: int = 0
    n_profile_total: int = 0


def _side_chars(geo: SplitShotGeometry, col: int, row: int, side: str, clf: GlyphClassifier):
    glyphs = geo.half_glyphs(col, row, side)
    chars = "".join(clf.classify(glyph_features(g, geo.glyph_h))[0] for g in glyphs)
    return glyphs, chars


def verify_image(path: str | Path, models: Classifiers3) -> Verified:
    """Read one screenshot and return the glyphs that every cross-check agrees with."""
    geo = load_split_shot(path)
    table = read_table(geo.img, geo.table, models.table)
    reports: list[ColumnReport] = validate_columns(geo, table, models.cells)
    profile = read_profile2(geo, models.profile)
    pchecks = validate_profile2(profile, reports)
    row_ok = {p.row: p.ok for p in pchecks}
    has_profile = bool(profile)
    out = Verified()

    for col, rep in enumerate(reports):
        col_ok = rep.ok and not table[col].clipped and any(c.name == "volume" and c.status == "ok" for c in rep.checks)
        for row, cell in rep.cells:
            out.n_cells_total += 1
            if cell.bid is None or not col_ok:
                continue
            if has_profile and row in row_ok and not row_ok[row]:
                continue
            for side in ("bid", "ask"):
                glyphs, chars = _side_chars(geo, col, row, side, models.cells)
                out.cells.add(glyphs, chars, geo.glyph_h)
            out.n_cells += 1

    gh = table_glyph_height(geo.img, geo.table)
    for col, rep in enumerate(reports):
        col_ok = rep.ok and not table[col].clipped and any(c.name in ("cum",) and c.status == "ok" for c in rep.checks)
        for r, name in enumerate(("volume", "delta", "cum")):
            out.n_table_total += 1
            cell = getattr(table[col], name)
            if not col_ok or cell.value is None:
                continue
            glyphs = table_cell_glyphs(geo.img, geo.table, r, col, gh)
            if len(glyphs) == len(cell.raw):
                out.table.add(glyphs, cell.raw, gh)
                out.n_table += 1

    for row in profile:
        out.n_profile_total += 1
        if not row_ok.get(row.row, False) or row.delta is None or row.volume is None:
            continue
        if not row.delta_glyphs or not row.volume_glyphs:
            continue
        out.profile.add(row.delta_glyphs, row.delta_raw, geo.glyph_h)
        out.profile.add(row.volume_glyphs, row.volume_raw, geo.glyph_h)
        out.n_profile += 1
    return out


def manual_profile_samples(stem: str, truth: list[list[str]], models: Classifiers3) -> Samples:
    """Glyphs of profile rows whose (delta, volume) text was transcribed by hand."""
    geo = load_split_shot(DATA.parent.parent.parent / "tests" / "data" / f"{stem}.png")
    rows = [r for r in read_profile2(geo, models.profile) if r.delta_glyphs or r.volume_glyphs]
    out = Samples()
    if len(rows) != len(truth):
        return out
    for row, (d_text, v_text) in zip(rows, truth):
        if len(row.delta_glyphs) == len(d_text) and len(row.volume_glyphs) == len(v_text):
            out.add(row.delta_glyphs, d_text, geo.glyph_h)
            out.add(row.volume_glyphs, v_text, geo.glyph_h)
    return out


def _clf(features: np.ndarray, labels: np.ndarray) -> GlyphClassifier:
    return GlyphClassifier(features, labels)


def _merge(base: tuple[np.ndarray, np.ndarray], extra: Samples) -> tuple[np.ndarray, np.ndarray]:
    if not extra.labels:
        return base
    f, l = extra.arrays()
    return np.concatenate([base[0], f]), np.concatenate([base[1], l])


def train_rounds(stems: list[str], base: dict[str, tuple[np.ndarray, np.ndarray]], tests_dir: Path,
                 manual_profile: dict[str, list[list[str]]], rounds: int = 3, verbose: bool = True):
    """Iteratively grow the classifiers from verified glyphs. Returns (classifiers, extras)
    where extras maps 'cells'/'table'/'profile' to the verified (features, labels) added on top
    of `base`."""
    models = Classifiers3(_clf(*base["cells"]), _clf(*base["table"]), _clf(*base["profile"]))
    extras: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for rnd in range(1, rounds + 1):
        acc = {"cells": Samples(), "table": Samples(), "profile": Samples()}
        totals = [0] * 6
        for stem in stems:
            try:
                v = verify_image(tests_dir / f"{stem}.png", models)
            except ValueError as e:  # a screenshot the geometry cannot be fitted to yet: skip it this round
                if verbose:
                    print(f"  skipped {stem}: {e}")
                continue
            for key in acc:
                acc[key].features += getattr(v, key).features
                acc[key].labels += getattr(v, key).labels
            for i, n in enumerate((v.n_cells, v.n_cells_total, v.n_table, v.n_table_total, v.n_profile, v.n_profile_total)):
                totals[i] += n
        for stem, truth in manual_profile.items():
            m = manual_profile_samples(stem, truth, models)
            acc["profile"].features += m.features
            acc["profile"].labels += m.labels
        if verbose:
            print(f"round {rnd}: verified cells {totals[0]}/{totals[1]}, table cells {totals[2]}/{totals[3]}, "
                  f"profile rows {totals[4]}/{totals[5]}")
        extras = {k: acc[k].arrays() for k in acc if acc[k].labels}
        models = Classifiers3(
            _clf(*_merge(base["cells"], acc["cells"])),
            _clf(*_merge(base["table"], acc["table"])),
            _clf(*_merge(base["profile"], acc["profile"])),
        )
    return models, extras
