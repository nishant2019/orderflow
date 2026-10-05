"""Build the time-axis glyph training set from tests/data/time_truth.json."""
import json
from pathlib import Path

import cv2
import numpy as np

from orderflow.calibrate import calibrate_table
from orderflow.cells import glyph_features
from orderflow.timeaxis import label_glyph_height, label_glyphs

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
truth = json.loads((DATA / "time_truth.json").read_text())
features, labels, shots = [], [], []
for shot, texts in sorted(truth.items()):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    table = calibrate_table(img)
    gh = label_glyph_height(img, table)
    used = skipped = 0
    for col, text in enumerate(texts):
        if text is None:
            continue
        chars = text.replace(" ", "")
        glyphs = label_glyphs(img, table, col, gh)
        if len(glyphs) != len(chars):
            skipped += 1
            print(f"shot {shot} col {col}: '{text}' has {len(glyphs)} glyphs - skipped")
            continue
        used += 1
        for g, ch in zip(glyphs, chars):
            features.append(glyph_features(g, gh))
            labels.append(ch)
            shots.append(int(shot))
    print(f"shot {shot}: glyph height {gh}, {used} labels used, {skipped} skipped")

# The price axis uses the same font and has every digit plus '.' and ',': add its labels.
from orderflow.axis import find_label_bands  # noqa: E402
from orderflow.timeaxis import _blackhat  # noqa: E402
from orderflow.cells import segment_glyphs  # noqa: E402

axis_truth = json.loads((DATA / "axis_labels.json").read_text())
for shot, texts in sorted(axis_truth.items()):
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    used = 0
    for band, text in zip(find_label_bands(img), texts):
        if band.y_bottom - band.y_top + 1 < 9:
            continue  # clipped by the plot frame
        x0, x1 = band.left_edge - 2, band.right_edge + 3
        y0, y1 = band.y_top - 2, band.y_bottom + 4
        ink = _blackhat(img, y0, y1, x0, x1)
        zeros = np.zeros_like(ink)
        glyphs = segment_glyphs({"black": ink, "red": zeros, "blue": zeros}, 120.0, 9, ink.astype(np.float32))
        if len(glyphs) != len(text):
            continue
        used += 1
        for g, ch in zip(glyphs, text):
            features.append(glyph_features(g, 9))
            labels.append(ch)
            shots.append(100 + int(shot))  # 100+ marks price-axis samples
    print(f"price axis shot {shot}: {used} labels used")

np.savez(
    ROOT / "src" / "orderflow" / "data" / "label_glyphs.npz",
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels),
    shots=np.array(shots),
)
print(len(labels), "glyphs;", "".join(sorted(set(labels))))
