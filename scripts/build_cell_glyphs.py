"""Build the cell-glyph training set from hand-transcribed cells in tests/data/cell_truth.json.

A cell is used only when its segmented glyph count equals its transcription length, so
labels are always exact. Cells that do not align are reported and skipped.
"""
import json
from pathlib import Path

import numpy as np

from orderflow.cells import glyph_features
from orderflow.pipeline import load_shot

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
truth = json.loads((DATA / "cell_truth.json").read_text())
features, labels, shots = [], [], []
for shot, cols in sorted(truth.items()):
    geo = load_shot(DATA / f"shot{shot}.png")
    used = skipped = 0
    for col, texts in cols.items():
        cells = geo.text_cells(int(col))
        if len(cells) != len(texts):
            print(f"shot {shot} col {col}: {len(cells)} text cells found, {len(texts)} transcribed - skipped")
            continue
        for (_, glyphs), text in zip(cells, texts):
            chars = text.replace(" ", "")
            if len(glyphs) != len(chars):
                skipped += 1
                print(f"shot {shot} col {col}: '{text}' has {len(glyphs)} glyphs - skipped")
                continue
            used += 1
            for g, ch in zip(glyphs, chars):
                features.append(glyph_features(g, geo.glyph_h))
                labels.append(ch)
                shots.append(int(shot))
    print(f"shot {shot}: {used} cells used, {skipped} skipped")
np.savez(
    ROOT / "src" / "orderflow" / "data" / "cell_glyphs.npz",
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels),
    shots=np.array(shots),
)
print(len(labels), "glyphs;", "".join(sorted(set(labels))))
