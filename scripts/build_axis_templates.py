"""Build axis glyph templates from the hand-verified labels in tests/data/axis_labels.json."""
import json
from pathlib import Path

import cv2
import numpy as np

from orderflow.axis import GlyphTemplates, normalize, read_band, band_patches, find_label_bands, CELL_H, CELL_W, GLYPH_ROWS

DATA = Path(__file__).resolve().parents[1] / "tests" / "data"
truth = json.loads((DATA / "axis_labels.json").read_text())
# One template per (character, shot): rendering differs slightly between shots
# (anti-aliasing phase), so averaging across shots blurs the classes together.
acc: dict[tuple[str, str], list[np.ndarray]] = {}
for shot, labels in truth.items():
    img = cv2.imread(str(DATA / f"shot{shot}.png"))
    bands = find_label_bands(img)
    assert len(bands) >= len(labels), (shot, len(bands))
    for band, text in zip(bands, labels):
        if band.y_bottom - band.y_top + 1 < GLYPH_ROWS:
            continue  # label clipped by the plot frame
        patches = band_patches(img, band)
        assert len(patches) == len(text), (shot, text, len(patches))
        for ch, p in zip(text, patches):
            acc.setdefault((ch, shot), []).append(normalize(p))


def build(exclude_shot: str | None = None) -> GlyphTemplates:
    # '.' and ',' are classified from their own shot only; others from all but the held-out one
    keys = sorted(k for k in acc if k[1] != exclude_shot)
    chars = [k[0] for k in keys]
    patches = [np.mean(acc[k], axis=0) for k in keys]
    chars.append(" ")
    patches.append(np.zeros((CELL_H, CELL_W)))
    return GlyphTemplates(chars, np.stack(patches).astype(np.float32))


# Leave-one-shot-out check of how well templates generalise to an unseen shot.
for held in truth:
    tpl = build(exclude_shot=held)
    img = cv2.imread(str(DATA / f"shot{held}.png"))
    ok = total = 0
    for band, text in zip(find_label_bands(img), truth[held]):
        if band.y_bottom - band.y_top + 1 < GLYPH_ROWS:
            continue
        got = read_band(img, band, tpl, max_dist=99)
        total += 1
        ok += got == text
    print(f"held-out shot {held}: {ok}/{total} labels read correctly")

final = build()
final.save()
print("saved templates for:", "".join(sorted(set(final.chars))))
