"""Loads the trained glyph classifiers shipped in orderflow/data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cells import GlyphClassifier

DATA = Path(__file__).parent / "data"


def _load(name: str) -> GlyphClassifier:
    data = np.load(DATA / f"{name}.npz", allow_pickle=False)
    return GlyphClassifier(data["features"], data["labels"])


@dataclass(frozen=True)
class Classifiers:
    cells: GlyphClassifier
    table: GlyphClassifier
    profile: GlyphClassifier
    labels: GlyphClassifier


def load_classifiers() -> Classifiers:
    return Classifiers(_load("cell_glyphs"), _load("table_glyphs"), _load("profile_glyphs"), _load("label_glyphs"))


@dataclass(frozen=True)
class SplitClassifiers:
    cells: GlyphClassifier
    table: GlyphClassifier
    profile: GlyphClassifier
    labels: GlyphClassifier


def _load_merged(*names: str) -> GlyphClassifier:
    parts = [np.load(DATA / f"{n}.npz", allow_pickle=False) for n in names]
    return GlyphClassifier(np.concatenate([p["features"] for p in parts]), np.concatenate([p["labels"] for p in parts]))


def load_split_classifiers() -> SplitClassifiers:
    """Classifiers for the split-box layout: the hand-labelled legacy sets plus glyphs that were
    verified against the charts' own checksums (see `orderflow.selftrain`)."""
    return SplitClassifiers(
        cells=_load_merged("cell_glyphs", "split_cells_glyphs"),
        table=_load_merged("table_glyphs", "split_table_glyphs"),
        profile=_load_merged("profile_glyphs", "split_profile_glyphs"),
        labels=_load("label_glyphs"),
    )
