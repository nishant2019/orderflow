"""Self-train the split-layout classifiers from the screenshots in tests/data (split1..splitN) and shots*/.

Glyphs are kept as training labels only when their cell, column and profile row agree with the
chart's own checksums (see orderflow.selftrain). Hand-transcribed data (table_truth.json,
profile2_truth.json) supplies glyphs the images cannot label themselves, such as the 'M' suffix.
Writes src/orderflow/data/split_{cells,table,profile}_glyphs.npz.
"""
import json
from pathlib import Path

import numpy as np

from orderflow.selftrain import train_rounds

ROOT = Path(__file__).resolve().parents[1]
R, T = ROOT / "src" / "orderflow" / "data", ROOT / "tests" / "data"


def load(name):
    d = np.load(R / name)
    return d["features"], d["labels"]


stems = sorted(p.stem for p in T.glob("split[0-9]*.png"))
for d in ("shots", "shots2", "shots3"):  # the uploaded GoCharting shots
    stems += sorted(f"../../{d}/{p.stem}" for p in (ROOT / d).glob("*.png"))
base = {"cells": load("cell_glyphs.npz"), "table": load("table_glyphs.npz"), "profile": load("profile_glyphs.npz")}
manual = json.loads((T / "profile2_truth.json").read_text())
_, extras = train_rounds(stems, base, T, manual, rounds=4)
for key, (features, labels) in extras.items():
    np.savez(R / f"split_{key}_glyphs.npz", features=features, labels=labels)
    print(key, features.shape, "".join(sorted(set(labels))))
