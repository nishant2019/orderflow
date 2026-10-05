"""Leave-one-image-out evaluation of the split-layout readers.

For each split image: self-train the classifiers on the other images (dropping any hand-transcribed
data that came from the held-out image), then read the held-out image and report how much of it
passes the chart's own cross-checks. No label for the held-out image is used anywhere.
"""
import json
import sys
from pathlib import Path

import numpy as np

from orderflow.cells import GlyphClassifier
from orderflow.profile2 import read_profile2
from orderflow.selftrain import train_rounds
from orderflow.split import load_split_shot
from orderflow.table import read_table
from orderflow.validate import validate_columns, validate_profile2

ROOT = Path(__file__).resolve().parents[1]
R, T = ROOT / "src" / "orderflow" / "data", ROOT / "tests" / "data"
stems = sorted(p.stem for p in T.glob("split[0-9].png"))
manual = json.loads((T / "profile2_truth.json").read_text())


def load(name):
    d = np.load(R / name)
    return d["features"], d["labels"], (d["shots"] if "shots" in d.files else None)


cells_f, cells_l, _ = load("cell_glyphs.npz")
table_f, table_l, table_s = load("table_glyphs.npz")
prof_f, prof_l, _ = load("profile_glyphs.npz")

totals = dict(cols=0, bad_cols=0, rows=0, bad_rows=0, cells=0, unparsed=0)
for held in stems:
    num = int(held.replace("split", ""))
    keep = np.ones(len(table_l), bool) if table_s is None else table_s != 100 + num
    base = {"cells": (cells_f, cells_l), "table": (table_f[keep], table_l[keep]), "profile": (prof_f, prof_l)}
    models, _ = train_rounds([s for s in stems if s != held], base, T,
                             {k: v for k, v in manual.items() if k != held}, rounds=2, verbose=False)
    geo = load_split_shot(T / f"{held}.png")
    table = read_table(geo.img, geo.table, models.table)
    reports = validate_columns(geo, table, models.cells)
    profile = read_profile2(geo, models.profile)
    pchecks = validate_profile2(profile, reports)
    bad_cols = [r.col for r in reports if not r.ok]
    bad_rows = [p.raw for p in pchecks if not p.ok]
    n_cells = sum(len(r.cells) for r in reports)
    unparsed = sum(1 for r in reports for _, c in r.cells if c.bid is None)
    print(f"{held}: columns failing {len(bad_cols)}/{len(reports)} {bad_cols}  profile rows failing "
          f"{len(bad_rows)}/{len(pchecks)}  unparsed cells {unparsed}/{n_cells}", flush=True)
    for k, v in zip(totals, (len(reports), len(bad_cols), len(pchecks), len(bad_rows), n_cells, unparsed)):
        totals[k] += v
print("TOTAL", totals)
