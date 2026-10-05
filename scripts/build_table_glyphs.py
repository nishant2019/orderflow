"""Build the table-glyph training set from tests/data/table_truth.json (exact labels only)."""
import json
from pathlib import Path

import cv2
import numpy as np

from orderflow.axis import calibrate_price_axis  # noqa: F401  (import check)
from orderflow.calibrate import calibrate_table
from orderflow.cells import glyph_features
from orderflow.table import ROW_NAMES, cell_glyphs, table_glyph_height

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
truth = json.loads((DATA / "table_truth.json").read_text())
features, labels, shots = [], [], []
for shot, rows in sorted(truth.items()):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    table = calibrate_table(img)
    gh = table_glyph_height(img, table)
    used = skipped = 0
    for r, name in enumerate(ROW_NAMES):
        for col, text in enumerate(rows[name]):
            if text is None:
                continue
            glyphs = cell_glyphs(img, table, r, col, gh)
            if len(glyphs) != len(text):
                skipped += 1
                print(f"shot {shot} {name}[{col}] '{text}': {len(glyphs)} glyphs - skipped")
                continue
            used += 1
            for g, ch in zip(glyphs, text):
                features.append(glyph_features(g, gh))
                labels.append(ch)
                shots.append(int(shot))
    print(f"shot {shot}: glyph height {gh}, {used} cells used, {skipped} skipped")
np.savez(
    ROOT / "src" / "orderflow" / "data" / "table_glyphs.npz",
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels),
    shots=np.array(shots),
)
print(len(labels), "glyphs;", "".join(sorted(set(labels))))
