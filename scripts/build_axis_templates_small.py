"""Axis glyph templates for the smaller axis font (7 px advance), from hand-read labels
(tests/data/axis_labels_small.json: image -> {band top y: label text})."""
import json
from pathlib import Path

import cv2
import numpy as np

from orderflow.axis import FONT_SMALL, GlyphTemplates, band_patches, find_label_bands, normalize, read_band

ROOT = Path(__file__).resolve().parents[1]
truth = json.loads((ROOT / "tests" / "data" / "axis_labels_small.json").read_text())
acc: dict[str, list[np.ndarray]] = {}
for rel, labels in truth.items():
    img = cv2.imread(str(ROOT / rel))
    for band in find_label_bands(img):
        text = labels.get(str(band.y_top))
        if text is None:
            continue
        patches = band_patches(img, band, font=FONT_SMALL)
        assert len(patches) >= len(text), (rel, text, len(patches))
        for ch, p in zip(text[::-1], patches[::-1]):  # aligned from the right (a leading blank cell may exist)
            acc.setdefault(ch, []).append(normalize(p))
chars = sorted(acc)
patches = [np.mean(acc[c], axis=0) for c in chars]
chars.append(" ")
patches.append(np.zeros((FONT_SMALL.cell_h, FONT_SMALL.cell_w)))
tpl = GlyphTemplates(chars, np.stack(patches).astype(np.float32))
tpl.save(ROOT / "src" / "orderflow" / "data" / FONT_SMALL.templates)
print("templates:", "".join(chars))
for rel, labels in truth.items():
    img = cv2.imread(str(ROOT / rel))
    ok = sum(read_band(img, b, tpl, max_dist=99, font=FONT_SMALL) == labels[str(b.y_top)]
             for b in find_label_bands(img) if str(b.y_top) in labels)
    print(rel, f"{ok}/{len(labels)} labels read back")
