import json
from pathlib import Path

import numpy as np
import pytest

from orderflow.cells import GlyphClassifier
from orderflow.pipeline import load_shot
from orderflow.profile import find_bars, read_profile
from orderflow.table import load_table_classifier, parse_signed, read_table
from orderflow.validate import validate_columns, validate_profile

DATA = Path(__file__).parent / "data"
TRUTH = json.loads((DATA / "profile_truth.json").read_text())
_ROOT = Path(__file__).parents[1] / "src" / "orderflow" / "data"
_CELLS = np.load(_ROOT / "cell_glyphs.npz")
_PROFILE = np.load(_ROOT / "profile_glyphs.npz")


def profile_clf(exclude_shot=None):
    keep = np.ones(len(_PROFILE["labels"]), bool) if exclude_shot is None else _PROFILE["shots"] != exclude_shot
    return GlyphClassifier(_PROFILE["features"][keep], _PROFILE["labels"][keep])


def read(shot, clf):
    geo = load_shot(DATA / f"shot{shot}.png")
    return geo, [r for r in read_profile(geo, clf) if r.raw]


@pytest.mark.parametrize("shot", sorted(TRUTH))
def test_profile_rows_read_exactly(shot):
    _, rows = read(shot, profile_clf())
    assert [r.value for r in rows] == [parse_signed(t) for t in TRUTH[shot]]


@pytest.mark.parametrize("shot", sorted(TRUTH))
def test_bar_colour_matches_value_sign(shot):
    _, rows = read(shot, profile_clf())
    for r in rows:
        if r.value and r.sign:  # a tiny bar (e.g. -173) can be too thin to see
            assert (r.value > 0) == (r.sign > 0), (r.raw, r.sign)


def checks(shot, clf):
    geo = load_shot(DATA / f"shot{shot}.png")
    reports = validate_columns(
        geo, read_table(geo.img, geo.table, load_table_classifier()),
        GlyphClassifier(_CELLS["features"], _CELLS["labels"]),
    )
    return validate_profile(read_profile(geo, clf), reports)


@pytest.mark.parametrize("shot", [2, 3, 4, 5])
def test_profile_validates_against_cells(shot):
    bad = [(p.raw, [(c.name, c.detail) for c in p.checks if c.status == "fail"]) for p in checks(shot, profile_clf()) if not p.ok]
    assert not bad, bad


def test_shot1_flags_rows_hidden_under_the_overlay():
    # the live bar's cells sit under the translucent profile, so row -137K cannot add up
    bad = [p.raw for p in checks(1, profile_clf()) if not p.ok]
    assert bad == ["-137K"]


def test_misreads_by_an_unseen_shot_classifier_are_caught():
    """With shot 5 held out, '-4.8K' and '-4.5K' are misread; validation must flag them."""
    held = checks(5, profile_clf(exclude_shot=5))
    _, rows = read(5, profile_clf(exclude_shot=5))
    wrong = {r.raw for r, t in zip(rows, TRUTH["5"]) if r.value != parse_signed(t)}
    assert wrong, "expected the held-out classifier to misread something"
    flagged = {p.raw for p in held if not p.ok}
    assert wrong <= flagged
