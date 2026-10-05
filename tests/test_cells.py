"""Cell text parsing. Shots 2-5 are in the training set (cell_truth.json); shot 1 is a true
hold-out (cell_holdout.json) that never enters training."""
import json
from pathlib import Path

import numpy as np
import pytest

from orderflow.cells import GlyphClassifier, parse_cell, parse_number
from orderflow.pipeline import load_shot

DATA = Path(__file__).parent / "data"
TRAIN_TRUTH = json.loads((DATA / "cell_truth.json").read_text())
HOLDOUT = json.loads((DATA / "cell_holdout.json").read_text())
NPZ = np.load(Path(__file__).parents[1] / "src" / "orderflow" / "data" / "cell_glyphs.npz")


def classifier(exclude_shot=None):
    keep = np.ones(len(NPZ["labels"]), bool) if exclude_shot is None else NPZ["shots"] != exclude_shot
    return GlyphClassifier(NPZ["features"][keep], NPZ["labels"][keep])


def score(shot, truth, clf):
    geo = load_shot(DATA / f"shot{shot}.png")
    ok = total = 0
    wrong = []
    for col, texts in truth[str(shot)].items():
        cells = geo.text_cells(int(col))
        assert len(cells) == len(texts), (shot, col, len(cells), len(texts))
        for (_, glyphs), text in zip(cells, texts):
            cell = parse_cell(glyphs, geo.glyph_h, clf)
            total += 1
            if text == "...":
                good = cell.ellipsis
            else:
                bid, ask = (parse_number(p.strip()) for p in text.split("X"))
                good = (cell.bid, cell.ask) == (bid, ask)
            ok += good
            if not good:
                wrong.append((text, cell.raw))
    return ok, total, wrong


@pytest.mark.parametrize("shot", [2, 3, 4, 5])
def test_training_shots_parse_exactly(shot):
    ok, total, wrong = score(shot, TRAIN_TRUTH, classifier())
    assert ok == total, wrong


@pytest.mark.parametrize("shot", [4, 5])
def test_unseen_shot_with_known_font(shot):
    ok, total, wrong = score(shot, TRAIN_TRUTH, classifier(exclude_shot=shot))
    assert ok == total, wrong


def test_true_holdout_shot1():
    ok, total, wrong = score(1, HOLDOUT, classifier())
    assert ok / total >= 0.97, wrong


def test_parse_number():
    assert parse_number("3.5K") == 3500
    assert parse_number("326") == 326
    assert parse_number("1.2M") == 1_200_000
    assert parse_number("3..5K") is None


def test_imbalance_colours():
    clf = classifier()
    geo = load_shot(DATA / "shot5.png")
    cells = [parse_cell(g, geo.glyph_h, clf) for _, g in geo.text_cells(2)]
    by_raw = {c.raw: c for c in cells}
    assert by_raw["3.5KX20K"].sell_imbalance and not by_raw["3.5KX20K"].buy_imbalance
    assert by_raw["9.3KX13K"].buy_imbalance and not by_raw["9.3KX13K"].sell_imbalance
    assert not by_raw["8.8KX9.9K"].sell_imbalance and not by_raw["8.8KX9.9K"].buy_imbalance
