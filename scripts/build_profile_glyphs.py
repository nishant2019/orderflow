"""Build the profile-text glyph training set from tests/data/profile_truth.json."""
import json
from pathlib import Path

import numpy as np

from orderflow.cells import glyph_features
from orderflow.pipeline import load_shot
from orderflow.profile import find_anchor, profile_glyph_height, row_glyphs, find_bars

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
truth = json.loads((DATA / "profile_truth.json").read_text())
features, labels, shots = [], [], []
for shot, texts in sorted(truth.items()):
    geo = load_shot(DATA / f"shot{shot}.png")
    anchor = find_anchor(geo)
    gh = profile_glyph_height(geo, anchor)
    lines = []
    for bar in find_bars(geo):
        glyphs = row_glyphs(geo, bar.row, anchor, gh)
        if glyphs:
            lines.append(glyphs)
    if len(lines) != len(texts):
        print(f"shot {shot}: {len(lines)} text rows found, {len(texts)} transcribed - skipped")
        continue
    used = skipped = 0
    for glyphs, text in zip(lines, texts):
        if len(glyphs) != len(text):
            skipped += 1
            print(f"shot {shot}: '{text}' has {len(glyphs)} glyphs - skipped")
            continue
        used += 1
        for g, ch in zip(glyphs, text):
            features.append(glyph_features(g, gh))
            labels.append(ch)
            shots.append(int(shot))
    print(f"shot {shot}: glyph height {gh}, {used} rows used, {skipped} skipped")
np.savez(
    ROOT / "src" / "orderflow" / "data" / "profile_glyphs.npz",
    features=np.array(features, dtype=np.float32),
    labels=np.array(labels),
    shots=np.array(shots),
)
print(len(labels), "glyphs;", "".join(sorted(set(labels))))
