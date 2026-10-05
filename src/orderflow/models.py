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
