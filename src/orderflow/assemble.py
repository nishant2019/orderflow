"""Assemble every parser into one JSON-ready description of a footprint screenshot."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from .candles import Candle, bucket, find_candles
from .cells import CellText
from .models import Classifiers, load_classifiers
from .overlays import find_current_price_line
from .pipeline import ShotGeometry, load_shot
from .poc import find_poc_rows
from .profile import read_profile
from .table import TableColumn, read_table
from .timeaxis import BarLabel, read_labels
from .validate import Check, ColumnReport, ProfileCheck, validate_candles, validate_columns, validate_profile

SCHEMA_VERSION = 1


def _fmt_time(minutes: int | None) -> str | None:
    return None if minutes is None else f"{minutes // 60:02d}:{minutes % 60:02d}"


def infer_times(labels: list[BarLabel]) -> list[tuple[int | None, bool]]:
    """(minutes, inferred) per bar. A session-start bar shows its date instead of its time;
    its time follows from the neighbouring bars and the bar interval."""
    known = [(i, lab.minutes) for i, lab in enumerate(labels) if lab.minutes is not None]
    steps = [
        (m2 - m1) // (i2 - i1)
        for (i1, m1), (i2, m2) in zip(known, known[1:])
        if m2 > m1 and (m2 - m1) % (i2 - i1) == 0
    ]
    interval = statistics.median(steps) if steps else None
    out: list[tuple[int | None, bool]] = []
    for i, lab in enumerate(labels):
        if lab.minutes is not None:
            out.append((lab.minutes, False))
            continue
        guess = None
        if interval:
            after = next(((j, m) for j, m in known if j > i), None)
            before = next(((j, m) for j, m in reversed(known) if j < i), None)
            if after:
                guess = int(after[1] - interval * (after[0] - i))
            elif before:
                guess = int(before[1] + interval * (i - before[0]))
        out.append((guess, guess is not None))
    return out


def _split(raw: str) -> tuple[str, str]:
    bid, ask = raw.split("X")
    return bid, ask


def _cell_json(geo: ShotGeometry, row: int, cell: CellText, is_poc: bool) -> dict:
    price = geo.grid.row_price(row, geo.axis)
    if cell.ellipsis:
        return {"price": price, "hidden": True}
    if cell.bid is None:
        return {"price": price, "unreadable": True, "raw": cell.raw}
    bid_text, ask_text = _split(cell.raw)
    return {
        "price": price,
        "bid": cell.bid,
        "ask": cell.ask,
        "bid_text": bid_text,
        "ask_text": ask_text,
        "delta": cell.ask - cell.bid,
        "total": cell.bid + cell.ask,
        "sell_imbalance": cell.sell_imbalance,
        "buy_imbalance": cell.buy_imbalance,
        "is_poc": is_poc,
        "confidence": round(cell.confidence, 3),
    }


def _checks_json(checks: list[Check]) -> list[dict]:
    return [{"name": c.name, "status": c.status, "detail": c.detail} for c in checks]


def _table_value(cell) -> dict:
    return {"value": cell.value, "text": cell.raw}


def _ohlc_json(candle: Candle | None, checks: list[Check], step: float, jitter: float) -> dict | None:
    if candle is None:
        return None
    failed = [c.name for c in checks if c.status == "fail"]
    return {
        "open": round(candle.open, 3),
        "high": round(candle.high, 3),
        "low": round(candle.low, 3),
        "close": round(candle.close, 3),
        "high_row": bucket(candle.high, step, jitter),  # row price whose bucket holds the high
        "low_row": bucket(candle.low, step, jitter),
        "direction": candle.direction,
        "body_px": candle.body_px,
        "high_clipped": candle.high_clipped,  # the wick runs off the visible plot
        "low_clipped": candle.low_clipped,
        "checks": _checks_json(checks),
        "valid": not failed,
    }


def _bar_json(
    geo: ShotGeometry, index: int, tcol: TableColumn, rep: ColumnReport, label: BarLabel,
    time: tuple[int | None, bool], poc_row: int | None, ohlc: dict | None = None,
) -> dict:
    cells = [_cell_json(geo, row, cell, row == poc_row) for row, cell in rep.cells]
    traded = [c["price"] for c in cells if c.get("total", 0) > 0]
    failed = sorted({c.name for c in rep.checks if c.status == "fail"})
    return {
        "index": index,
        "time": _fmt_time(time[0]),
        "time_inferred": time[1],
        "new_session": label.date is not None,
        "date_label": label.date,
        "clipped": tcol.clipped,
        "volume": _table_value(tcol.volume),
        "delta": _table_value(tcol.delta),
        "cum_delta": _table_value(tcol.cum),
        "ohlc": ohlc,
        "poc_price": None if poc_row is None else geo.grid.row_price(poc_row, geo.axis),
        "traded_range": {"low": min(traded), "high": max(traded)} if traded else None,
        "cells": sorted(cells, key=lambda c: -c["price"]),
        "checks": _checks_json(rep.checks),
        "valid": rep.ok and not tcol.clipped and bool(cells),
        "failed_checks": failed,
    }


def _profile_json(rows: list, checks: list[ProfileCheck]) -> list[dict]:
    out = []
    for r, c in zip(rows, checks):
        if not r.raw:
            continue
        out.append({
            "price": r.price,
            "delta": r.value,
            "text": r.raw,
            "bar_px": r.length_px,
            "checks": _checks_json(c.checks),
            "valid": c.ok,
        })
    return sorted(out, key=lambda p: -p["price"])


def parse_screenshot(path: str | Path, models: Classifiers | None = None) -> dict:
    """Parse one GoCharting footprint screenshot into a JSON-serialisable dict."""
    models = models or load_classifiers()
    geo = load_shot(path)
    table = read_table(geo.img, geo.table, models.table)
    reports = validate_columns(geo, table, models.cells)
    poc = find_poc_rows(geo)
    labels = read_labels(geo.img, geo.table, models.labels)
    times = infer_times(labels)
    profile_rows = read_profile(geo, models.profile)
    profile_checks = validate_profile(profile_rows, reports)
    line = find_current_price_line(geo.img, geo.axis, y_limit=geo.table.y_top)

    candles = find_candles(geo)
    traded = [
        [geo.grid.row_price(row, geo.axis) for row, c in rep.cells if c.bid is not None and c.bid + c.ask > 0]
        for rep in reports
    ]
    candle_checks = validate_candles(
        candles, traded, geo.grid.price_step, abs(geo.axis.slope), [lab.date is not None for lab in labels]
    )
    bars = [
        _bar_json(
            geo, i, table[i], reports[i], labels[i], times[i], poc.get(i),
            _ohlc_json(candles[i], candle_checks[i], geo.grid.price_step, abs(geo.axis.slope)),
        )
        for i in range(geo.table.n_cols)
    ]
    profile = _profile_json(profile_rows, profile_checks)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": str(path),
        "image": {"height": int(geo.img.shape[0]), "width": int(geo.img.shape[1])},
        "price": {
            "step_per_row": geo.grid.price_step,
            "row_height_px": round(geo.grid.pitch, 3),
            "current": None if line is None else round(line.price, 3),
            # bars extending beyond this range are cut off by the view
            "visible": {
                "high": geo.grid.row_price(geo.k_first, geo.axis),
                "low": geo.grid.row_price(geo.k_last, geo.axis),
            },
        },
        "bars": bars,
        "profile": profile,
        "summary": {
            "bars": len(bars),
            "valid_bars": sum(b["valid"] for b in bars),
            "flagged_bars": [b["index"] for b in bars if b["failed_checks"]],
            "skipped_bars": [b["index"] for b in bars if not b["valid"] and not b["failed_checks"]],
            "profile_rows": len(profile),
            "flagged_ohlc": [b["index"] for b in bars if b["ohlc"] and not b["ohlc"]["valid"]],
            "flagged_profile_prices": [p["price"] for p in profile if not p["valid"]],
            "approximate": "cell volumes are as displayed (3.5K = 3500 +-50); table and profile values likewise",
        },
    }


def to_json(path: str | Path, pretty: bool = True) -> str:
    return json.dumps(parse_screenshot(path), indent=2 if pretty else None)
