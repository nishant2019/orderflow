import json
from pathlib import Path

import pytest

from orderflow.analytics import Thresholds, analyze, build_levels, stacked_imbalances, _prepare

DATA = Path(__file__).parent / "data"


def cell(price, bid, ask, buy=False, sell=False):
    return {
        "price": float(price), "bid": float(bid), "ask": float(ask), "total": float(bid + ask),
        "delta": float(ask - bid), "buy_imbalance": buy, "sell_imbalance": sell,
        "is_poc": False, "confidence": 1.0,
    }


def ohlc(o, h, l, c, valid=True, clip_hi=False, clip_lo=False, step=1.0):
    import math

    return {
        "open": o, "high": h, "low": l, "close": c, "direction": "up" if c >= o else "down",
        "high_row": float(math.ceil(h / step) * step), "low_row": float(math.ceil(l / step) * step),
        "high_clipped": clip_hi, "low_clipped": clip_lo, "valid": valid,
        "checks": [] if valid else [{"name": "continuity", "status": "fail", "detail": "x"}],
    }


def bar(index, cells, poc=None, valid=True, clipped=False, failed=(), candle=None):
    vol = sum(c["total"] for c in cells)
    delta = sum(c["delta"] for c in cells)
    return {
        "index": index, "time": f"{9 + index // 2:02d}:{(index % 2) * 30:02d}", "clipped": clipped,
        "volume": {"value": vol}, "delta": {"value": delta}, "poc_price": poc,
        "cells": sorted(cells, key=lambda c: -c["price"]), "valid": valid, "failed_checks": list(failed),
        "ohlc": candle,
    }


def doc(*bars, current=None):
    return {
        "source": "synthetic",
        "price": {"step_per_row": 1.0, "last_close": current, "visible": {"high": 200.0, "low": 50.0}},
        "bars": list(bars),
    }


def prepared(b):
    return _prepare(doc(b), b)


def signals(result, index, kind=None):
    s = next(b for b in result["bars"] if b["index"] == index)["signals"]
    return [x for x in s if kind is None or x["type"] == kind]


# --- stacked imbalances ----------------------------------------------------------------------

def test_stacked_imbalance_needs_consecutive_rows():
    cells = [cell(p, 100, 400, buy=True) for p in (101, 102, 103, 104)] + [cell(106, 100, 400, buy=True), cell(107, 100, 400, buy=True)]
    out = stacked_imbalances(prepared(bar(0, cells)), 1.0, Thresholds())
    assert len(out) == 1  # 101-104 qualifies; the pair at 106-107 is too short and not adjacent
    s = out[0]
    assert (s["price"], s["price_to"], s["evidence"]["rows"], s["bias"]) == (101.0, 104.0, 4, "bullish")


def test_stacked_sell_imbalance_is_bearish_and_gap_breaks_the_run():
    cells = [cell(p, 400, 100, sell=True) for p in (100, 101, 103, 104, 105)]  # gap at 102
    out = stacked_imbalances(prepared(bar(0, cells)), 1.0, Thresholds())
    assert [(s["price"], s["price_to"]) for s in out] == [(103.0, 105.0)]
    assert out[0]["bias"] == "bearish"


def test_stack_threshold_is_configurable():
    cells = [cell(p, 100, 400, buy=True) for p in (101, 102)]
    assert not stacked_imbalances(prepared(bar(0, cells)), 1.0, Thresholds())
    assert stacked_imbalances(prepared(bar(0, cells)), 1.0, Thresholds(stack_min_rows=2))


# --- absorption and exhaustion ---------------------------------------------------------------

def absorption_bar(index, low=100):
    """Ten rows; heavy aggressive buying (ask) in the top two rows, little elsewhere."""
    cells = [cell(low + 9, 500, 6000), cell(low + 8, 600, 5000)]
    cells += [cell(low + k, 700, 800) for k in range(8)]
    return bar(index, cells, poc=low + 9)


def test_absorption_at_high_with_confirmation():
    held = bar(1, [cell(p, 800, 700) for p in range(100, 108)])  # next bar never exceeds 109
    res = analyze(doc(absorption_bar(0), held))
    (sig,) = signals(res, 0, "absorption")
    assert (sig["bias"], sig["location"], sig["price"]) == ("bearish", "high", 109.0)
    assert sig["confirmation"] == "held"
    assert sig["evidence"]["zone_delta"] > 0


def test_absorption_broken_when_next_bar_makes_a_higher_high():
    higher = bar(1, [cell(p, 800, 700) for p in range(105, 114)])
    (sig,) = signals(analyze(doc(absorption_bar(0), higher)), 0, "absorption")
    assert sig["confirmation"] == "broken"


def test_no_absorption_without_one_sided_aggression():
    cells = [cell(109, 3000, 3100), cell(108, 3000, 3000)] + [cell(100 + k, 700, 800) for k in range(8)]
    assert not signals(analyze(doc(bar(0, cells, poc=109))), 0, "absorption")


def test_absorption_needs_more_than_an_even_spread():
    """With 4 rows the top two hold half the volume by chance alone; that is not absorption."""
    cells = [cell(103, 100, 1300), cell(102, 100, 1300), cell(101, 100, 1300), cell(100, 100, 1300)]
    assert not signals(analyze(doc(bar(0, cells, poc=103))), 0, "absorption")


def test_bullish_absorption_at_low():
    cells = [cell(100, 6000, 500), cell(101, 5000, 600)] + [cell(102 + k, 800, 700) for k in range(8)]
    (sig,) = signals(analyze(doc(bar(0, cells, poc=100))), 0, "absorption")
    assert (sig["bias"], sig["location"]) == ("bullish", "low")


def test_exhaustion_volume_thinning_into_the_high():
    cells = [cell(109, 0, 40), cell(108, 200, 400), cell(107, 800, 1500)] + [cell(100 + k, 3000, 4000) for k in range(7)]
    (sig,) = signals(analyze(doc(bar(0, cells, poc=105))), 0, "exhaustion")
    assert (sig["bias"], sig["location"], sig["price"]) == ("bearish", "high", 109.0)
    assert sig["evidence"]["taper"] == [40.0, 600.0, 2300.0]


def test_unfinished_vs_finished_extreme():
    cells = [cell(109, 300, 500), cell(108, 800, 900)] + [cell(100 + k, 800, 700) for k in range(7)] + [cell(99, 0, 600)]
    res = analyze(doc(bar(0, cells, poc=105)))
    assert signals(res, 0, "unfinished_extreme")[0]["location"] == "high"  # both sides traded at 109
    assert signals(res, 0, "finished_extreme")[0]["location"] == "low"  # nobody sold at 99


def test_bars_with_too_few_rows_get_no_extreme_signals():
    res = analyze(doc(bar(0, [cell(100, 10, 20), cell(101, 20, 30)], poc=101)))
    assert [s["type"] for s in signals(res, 0)] == ["poc"]


# --- exclusion, partial data -----------------------------------------------------------------

def test_invalid_bars_are_excluded_with_a_reason():
    good = absorption_bar(0)
    bad = bar(1, [cell(100, 1, 1)], valid=False, failed=["volume"])
    clipped = bar(2, [], valid=False, clipped=True)
    empty = bar(3, [], valid=False)
    res = analyze(doc(good, bad, clipped, empty))
    reasons = {b["index"]: b.get("reason") for b in res["bars"] if b["status"] == "excluded"}
    assert reasons == {
        1: "failed validation: volume", 2: "clipped by the image edge",
        3: "no footprint cells in the visible price range",
    }
    assert res["summary"]["analyzed_bars"] == 1 and res["summary"]["excluded_bars"] == [1, 2, 3]


def test_hidden_cells_reduce_confidence_and_are_noted():
    b = absorption_bar(0)
    b["cells"].append({"price": 95.0, "hidden": True})
    res = analyze(doc(b))
    (sig,) = signals(res, 0, "absorption")
    assert sig["confidence"] == 0.7
    assert any("hidden" in n for n in res["bars"][0]["notes"])


def test_view_edge_is_noted():
    b = bar(0, [cell(p, 800, 700) for p in range(195, 201)], poc=197)
    assert any("edge" in n for n in analyze(doc(b))["bars"][0]["notes"])


# --- levels ----------------------------------------------------------------------------------

def test_levels_merge_nearby_prices_and_assign_roles():
    analysed = [
        {"index": 0, "stats": {"range": [90, 120]}, "signals": [{"type": "poc", "price": 100.0}]},
        {"index": 1, "stats": {"range": [95, 125]}, "signals": [{"type": "poc", "price": 101.0}]},
        {"index": 2, "stats": {"range": [90, 99]}, "signals": [{"type": "poc", "price": 110.0}]},
    ]
    levels = build_levels(analysed, 1.0, current=105.0, cfg=Thresholds())
    assert len(levels) == 2
    low, high = sorted(levels, key=lambda lv: lv["price"])
    assert low["price"] in (100.0, 101.0) and low["role"] == "support" and low["bars"] == [0, 1]
    # merged price 100.5 snaps to 100; only bar 1 (95-125) retests it, bar 2 (90-99) stops short
    assert low["strength"] == 2.0 and low["retests"] == 1
    assert high["role"] == "resistance" and high["distance_steps"] == 5.0


def test_level_retests_count_later_bars_touching_the_price():
    analysed = [
        {"index": 0, "stats": {"range": [95, 105]}, "signals": [{"type": "poc", "price": 100.0}]},
        {"index": 1, "stats": {"range": [98, 103]}, "signals": []},
        {"index": 2, "stats": {"range": [110, 115]}, "signals": []},
    ]
    (level,) = build_levels(analysed, 1.0, current=None, cfg=Thresholds())
    assert level["retests"] == 1 and "role" not in level


# --- integration on the real screenshots -----------------------------------------------------

@pytest.fixture(scope="module")
def real():
    from orderflow.assemble import parse_screenshot
    from orderflow.models import load_classifiers

    m = load_classifiers()
    return {n: parse_screenshot(DATA / f"shot{n}.png", m) for n in range(1, 6)}


def test_real_screenshots_analyse_and_serialise(real):
    for n, d in real.items():
        a = analyze(d)
        assert json.loads(json.dumps(a)) == a
        assert a["summary"]["analyzed_bars"] == d["summary"]["valid_bars"], n


def test_real_exclusions_follow_validation(real):
    assert analyze(real[1])["summary"]["excluded_bars"] == [11]
    assert analyze(real[5])["summary"]["excluded_bars"] == [0, 1]


def test_shot2_known_signals(real):
    a = analyze(real[2])
    bar0 = signals(a, 0)
    stack = [s for s in bar0 if s["type"] == "stacked_imbalance"]
    assert [(s["price"], s["price_to"]) for s in stack] == [(496.0, 498.0)]  # 496/497/498 all buy-imbalanced
    assert [s["location"] for s in bar0 if s["type"] == "unfinished_extreme"] == ["high"]
    (ex,) = signals(a, 3, "exhaustion")  # '0 X 3' at the high of the last bar
    assert (ex["bias"], ex["price"]) == ("bearish", 505.0)


def test_thresholds_change_results(real):
    strict = analyze(real[3], Thresholds(absorption_min_share=0.9))
    default = analyze(real[3])
    assert strict["summary"]["signal_counts"].get("absorption:bearish", 0) < default["summary"]["signal_counts"]["absorption:bearish"]


# --- candle-based signals --------------------------------------------------------------------

def plain_cells(low=100, rows=8, bid=800, ask=700):
    return [cell(low + k, bid, ask) for k in range(rows)]


def test_rejection_from_a_long_upper_wick():
    b = bar(0, plain_cells(), poc=104, candle=ohlc(103, 112, 101, 104))  # wick 8 of range 11
    (sig,) = signals(analyze(doc(b)), 0, "rejection")
    assert (sig["bias"], sig["location"], sig["price"]) == ("bearish", "high", 112.0)
    assert sig["strength"] == pytest.approx(8 / 11, abs=0.01)


def test_rejection_from_a_long_lower_wick():
    b = bar(0, plain_cells(), poc=104, candle=ohlc(109, 110, 100, 108))
    (sig,) = signals(analyze(doc(b)), 0, "rejection")
    assert (sig["bias"], sig["location"]) == ("bullish", "low")


def test_no_rejection_when_the_wick_runs_off_the_plot():
    b = bar(0, plain_cells(), poc=104, candle=ohlc(103, 112, 101, 104, clip_hi=True))
    assert not signals(analyze(doc(b)), 0, "rejection")


def test_delta_divergence_up_candle_negative_delta():
    cells = [cell(100 + k, 900, 600) for k in range(8)]  # net delta clearly negative
    b = bar(0, cells, poc=104, candle=ohlc(101, 108, 100, 107))
    (sig,) = signals(analyze(doc(b)), 0, "delta_divergence")
    assert sig["bias"] == "bearish" and sig["evidence"]["delta"] < 0


def test_delta_divergence_down_candle_positive_delta():
    cells = [cell(100 + k, 600, 900) for k in range(8)]
    b = bar(0, cells, poc=104, candle=ohlc(107, 108, 100, 101))
    (sig,) = signals(analyze(doc(b)), 0, "delta_divergence")
    assert sig["bias"] == "bullish"


def test_no_divergence_for_agreeing_delta_or_a_doji():
    agree = bar(0, [cell(100 + k, 600, 900) for k in range(8)], poc=104, candle=ohlc(101, 108, 100, 107))
    doji = bar(1, [cell(100 + k, 900, 600) for k in range(8)], poc=104, candle=ohlc(104, 108, 100, 104.2))
    res = analyze(doc(agree, doji))
    assert not signals(res, 0, "delta_divergence") and not signals(res, 1, "delta_divergence")


def test_invalid_candle_is_ignored_and_noted():
    b = bar(0, plain_cells(), poc=104, candle=ohlc(103, 112, 101, 104, valid=False))
    res = analyze(doc(b))
    assert not signals(res, 0, "rejection")
    assert any("failed validation" in n for n in res["bars"][0]["notes"])
    assert "open" not in res["bars"][0]["stats"]


def test_absorption_confirmation_uses_the_next_bars_candle():
    # next bar's traded rows stay below the high, but its wick pokes above it: absorption broken
    nxt = bar(1, [cell(p, 800, 700) for p in range(100, 108)], candle=ohlc(104, 111, 100, 105))
    (sig,) = signals(analyze(doc(absorption_bar(0), nxt)), 0, "absorption")
    assert sig["confirmation"] == "broken"
    quiet = bar(1, [cell(p, 800, 700) for p in range(100, 108)], candle=ohlc(104, 107, 100, 105))
    (sig,) = signals(analyze(doc(absorption_bar(0), quiet)), 0, "absorption")
    assert sig["confirmation"] == "held"


def test_stats_include_candle_geometry():
    b = bar(0, plain_cells(), poc=104, candle=ohlc(102, 110, 100, 108))
    stats = analyze(doc(b))["bars"][0]["stats"]
    assert (stats["open"], stats["close"], stats["direction"]) == (102, 108, "up")
    assert stats["close_location"] == pytest.approx(0.8) and stats["body_share"] == pytest.approx(0.6)


# --- volume profile --------------------------------------------------------------------------

def profile_doc(vols, deltas, va=None, last_close=None, step=1.0, base=100.0):
    rows = []
    for i, (v, d) in enumerate(zip(vols, deltas)):
        in_va = (va is None) or (va[0] <= i <= va[1])
        rows.append({"price": base + i * step, "volume": float(v), "delta": float(d), "in_value_area": in_va,
                     "zone": "value_area" if in_va else "outside", "valid": True})
    rows.sort(key=lambda r: -r["price"])
    return {"source": "synthetic", "layout": "split", "profile_kind": "delta_volume", "profile": rows, "bars": [],
            "price": {"step_per_row": step, "last_close": last_close, "visible": {"high": 999.0, "low": 0.0}}}


VOLS = [5, 12, 30, 60, 100, 70, 40, 15, 45, 25, 6]
DELTAS = [4, 0, 0, 30, 5, -35, 0, 0, 0, 0, 0]


def test_profile_poc_value_area_nodes_and_tails():
    from orderflow.analytics import profile_analysis

    p = profile_analysis(profile_doc([v * 1000 for v in VOLS], [d * 1000 for d in DELTAS], va=(2, 8), last_close=104.0))
    assert p["poc"]["price"] == 104.0 and p["poc"]["volume"] == 100000
    assert p["value_area"]["low"] == 102.0 and p["value_area"]["high"] == 108.0
    assert [n["price"] for n in p["high_volume_nodes"]] == [108.0]  # 45K: local peak above 1.25x the median row
    assert [n["price"] for n in p["low_volume_nodes"]] == [107.0]  # 15K: a trough between two bigger rows
    assert p["thin_tails"] == {"high_rows": 1, "low_rows": 1}  # 6K and 5K are <= 10% of the POC row
    assert p["price_vs_value_area"] == "inside" and p["price_distance_from_poc_steps"] == 0.0
    assert p["reference_price"] == {"price": 104.0, "source": "last_close"}


def test_profile_aggressive_rows_need_volume_and_one_sidedness():
    from orderflow.analytics import profile_analysis

    p = profile_analysis(profile_doc([v * 1000 for v in VOLS], [d * 1000 for d in DELTAS]))
    got = {(a["price"], a["side"]) for a in p["aggressive_levels"]}
    # 103: 30K of 60K delta is buying; 105: -35K of 70K is selling. The 100 row (4K of 5K) is too thin to count.
    assert got == {(103.0, "buying"), (105.0, "selling")}


def test_profile_shape_and_last_close_position():
    from orderflow.analytics import profile_analysis

    top_heavy = profile_doc([1, 2, 3, 5, 8, 20, 60, 100, 70, 30, 10], [0] * 11)
    bottom_heavy = profile_doc([10, 30, 70, 100, 60, 20, 8, 5, 3, 2, 1], [0] * 11)
    balanced = profile_doc([5, 10, 20, 40, 80, 100, 80, 40, 20, 10, 5], [0] * 11)
    assert profile_analysis(top_heavy)["shape"]["type"] == "P"
    assert profile_analysis(bottom_heavy)["shape"]["type"] == "b"
    assert profile_analysis(balanced)["shape"]["type"] == "D"
    above = profile_analysis(profile_doc(VOLS, DELTAS, va=(2, 8), last_close=112.0))
    below = profile_analysis(profile_doc(VOLS, DELTAS, va=(2, 8), last_close=99.0))
    assert above["price_vs_value_area"] == "above" and below["price_vs_value_area"] == "below"


def test_profile_ignores_unvalidated_rows_and_single_column_charts():
    from orderflow.analytics import profile_analysis

    d = profile_doc([v * 1000 for v in VOLS], [d * 1000 for d in DELTAS])
    for r in d["profile"]:
        r["valid"] = False
    assert profile_analysis(d) is None
    d["profile_kind"] = "delta"
    assert profile_analysis(d) is None


def test_profile_points_join_the_level_clustering_and_confluence_merges():
    base = profile_doc([v * 1000 for v in VOLS], [d * 1000 for d in DELTAS], va=(2, 8), last_close=106.0)
    base["bars"] = [bar(0, [cell(100 + k, 600 + 10 * k, 700) for k in range(8)], poc=104)]
    res = analyze(base)
    kinds = {tuple(lv["kinds"]) for lv in res["levels"]}
    assert any("profile_poc" in k and "poc" in k for k in kinds)  # bar POC and profile POC at 104 form one level
    assert not any("low_volume_node" in k for k in kinds)
    lv = next(lv for lv in res["levels"] if "profile_poc" in lv["kinds"])
    assert lv["price"] == 104.0 and lv["role"] == "support" and lv["distance_steps"] == -2.0


def test_real_profiles(real_split):
    for stem, (doc_, a) in real_split.items():
        pr = a["profile"]
        assert json.loads(json.dumps(pr)) == pr
        assert pr["poc"]["price"] == doc_["profile_summary"]["poc"]
        assert 0.4 <= pr["value_area"]["share_of_volume"] <= 0.95
        assert pr["poc"]["price"] not in [n["price"] for n in pr["high_volume_nodes"]]
        assert pr["shape"]["type"] in "PbD"
        assert any("profile_poc" in lv["kinds"] for lv in a["levels"])
        assert abs(pr["total_delta"] - doc_["profile_summary"]["total_delta"]) < 1e-6


@pytest.fixture(scope="module")
def real_split():
    from orderflow.assemble import parse_screenshot

    out = {}
    for k in range(1, 9):
        d = parse_screenshot(DATA / f"split{k}.png")
        out[f"split{k}"] = (d, analyze(d))
    return out


def test_current_price_line_is_preferred_over_the_last_close():
    d = profile_doc(VOLS, DELTAS, va=(2, 8), last_close=104.0)
    d["price"]["current"] = 111.0
    from orderflow.analytics import profile_analysis

    p = profile_analysis(d)
    assert p["reference_price"] == {"price": 111.0, "source": "current"} and p["price_vs_value_area"] == "above"
    d["bars"] = [bar(0, [cell(100 + k, 600, 700) for k in range(8)], poc=104)]
    lv = next(lv for lv in analyze(d)["levels"] if lv["price"] == 104.0)
    assert lv["role"] == "support" and lv["distance_steps"] == -7.0  # measured from 111, not 104
