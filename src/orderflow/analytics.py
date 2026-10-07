"""Rule-based order-flow analytics on the JSON produced by `parse_screenshot`.

Works on data, not pixels. Only bars that passed validation are analysed; the rest are listed
with the reason they were excluded. Values are as displayed on the chart (rounded), so every
threshold is deliberately coarse. All rules are heuristics with documented, tunable thresholds
(`Thresholds`); each signal carries its numeric evidence so a trader can judge it.

Each bar's candle (open/high/low/close, when it passed validation) adds direction, close
location, wick rejection and delta divergence, and sharpens absorption confirmation. Without a
valid candle the rules fall back to the highest and lowest rows that traded.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Thresholds:
    stack_min_rows: int = 3  # consecutive imbalance rows that make a "stacked imbalance"
    zone_rows: int = 2  # rows examined at each bar extreme
    min_traded_rows: int = 4  # fewer traded rows: too little structure to judge extremes
    absorption_min_share: float = 0.25  # extreme-zone volume as a share of the bar's cell volume
    absorption_min_delta_ratio: float = 0.15  # |zone delta| / zone volume
    absorption_vs_uniform: float = 1.5  # zone share must also exceed this x its share if volume were spread evenly
    exhaustion_max_extreme_ratio: float = 0.20  # extreme row total vs the POC row total
    taper_rows: int = 3  # rows that must thin out towards the extreme
    rejection_min_wick_share: float = 0.5  # wick as a share of the candle range
    divergence_min_delta_ratio: float = 0.05  # |bar delta| / bar volume
    divergence_min_body_share: float = 0.25  # body as a share of the range (not a doji)
    dominance_min_ratio: float = 0.30  # |delta| / volume (the chart's Delta %) for a one-sided bar
    dominance_full_ratio: float = 0.60  # ... at which the signal's strength reaches 1
    flow_min_ratio: float = 0.10  # a bar counts towards a run of same-sided delta from this |delta %|
    flow_min_bars: int = 3  # consecutive same-sided bars that make a "persistent_delta" run
    hvn_min_ratio: float = 1.25  # profile row volume vs the median row for a high-volume node
    lvn_max_ratio: float = 0.5  # ... and for a low-volume node (a thin spot inside the range)
    tail_ratio: float = 0.10  # row volume vs the POC row below which the profile edge is a thin tail
    aggressive_delta_ratio: float = 0.4  # |delta| / volume of a profile row for one-sided aggression
    aggressive_min_volume_ratio: float = 0.25  # ... only on rows carrying at least this share of the POC row's volume
    shape_skew: float = 0.65  # volume centroid above / below (1 - this) of the range => P / b profile
    level_merge_steps: float = 1.5  # price levels closer than this (in row steps) are merged
    partial_confidence: float = 0.7  # confidence factor for bars with hidden "..." cells


@dataclass
class _Bar:
    index: int
    time: str | None
    rows: list[dict]  # traded rows (total > 0), price descending
    all_rows: list[dict]  # every readable cell row, price descending
    poc: float | None
    volume: float | None
    delta: float | None
    partial: bool
    touches_view_edge: bool
    ohlc: dict | None = None  # validated candle, or None
    delta_ratio: float | None = None  # delta / volume: the chart's Delta % when it shows it, else computed
    delta_ratio_source: str | None = None  # "shown" | "computed"
    platform_divergence: str | None = None  # the platform's own marker: "red" | "green"


def _step(doc: dict) -> float:
    return doc["price"]["step_per_row"]


def _prepare(doc: dict, bar: dict) -> _Bar:
    readable = [c for c in bar["cells"] if "bid" in c]
    ohlc = bar.get("ohlc")
    visible = doc["price"].get("visible")
    prices = [c["price"] for c in readable]
    edge = bool(visible and prices and (max(prices) >= visible["high"] or min(prices) <= visible["low"]))
    ratio, source = _delta_ratio(bar)
    return _Bar(
        index=bar["index"],
        time=bar["time"],
        rows=[c for c in readable if c["total"] > 0],
        all_rows=readable,
        poc=bar["poc_price"],
        volume=bar["volume"]["value"],
        delta=bar["delta"]["value"],
        partial=any(c.get("hidden") for c in bar["cells"]),
        touches_view_edge=edge,
        ohlc=ohlc if ohlc and ohlc["valid"] else None,
        delta_ratio=ratio,
        delta_ratio_source=source,
        platform_divergence=bar.get("platform_divergence"),
    )


def _delta_ratio(bar: dict) -> tuple[float | None, str | None]:
    """Net delta as a share of the bar's volume (-1..1). The chart's own Delta % row is used when it has one:
    it carries two more significant digits than the rounded volume and delta it is made from."""
    pct = bar.get("delta_pct")
    if pct and pct.get("value") is not None:
        return pct["value"] / 100.0, "shown"
    vol, dlt = bar["volume"]["value"], bar["delta"]["value"]
    if vol and dlt is not None:
        return dlt / vol, "computed"
    return None, None


def _signal(kind: str, bias: str, location: str, price: float, strength: float, evidence: dict, text: str,
            price_to: float | None = None, confidence: float = 1.0, confirmation: str | None = None) -> dict:
    out = {
        "type": kind,
        "bias": bias,
        "location": location,
        "price": price,
        "strength": round(max(0.0, min(1.0, strength)), 3),
        "confidence": round(confidence, 3),
        "evidence": evidence,
        "text": text,
    }
    if price_to is not None:
        out["price_to"] = price_to
    if confirmation is not None:
        out["confirmation"] = confirmation
    return out


# --- per-bar detectors -------------------------------------------------------------------------

def stacked_imbalances(bar: _Bar, step: float, cfg: Thresholds) -> list[dict]:
    """Runs of consecutive price rows that all carry the same imbalance flag."""
    signals = []
    for flag, bias, word in (("buy_imbalance", "bullish", "buy"), ("sell_imbalance", "bearish", "sell")):
        flagged = sorted((c for c in bar.all_rows if c[flag]), key=lambda c: c["price"])
        run: list[dict] = []
        for cell in flagged + [None]:  # sentinel flushes the last run
            if cell is not None and run and abs(cell["price"] - run[-1]["price"] - step) < step * 0.01:
                run.append(cell)
                continue
            if len(run) >= cfg.stack_min_rows:
                lo, hi = run[0]["price"], run[-1]["price"]
                vol = sum(c["total"] for c in run)
                signals.append(_signal(
                    "stacked_imbalance", bias, "mid", lo, 0.4 + 0.15 * (len(run) - cfg.stack_min_rows),
                    {"side": word, "rows": len(run), "volume": vol}, f"{len(run)} stacked {word} imbalances {lo:g}-{hi:g}",
                    price_to=hi,
                ))
            run = [cell] if cell is not None else []
    return signals


def _zone(rows: list[dict], n: int, top: bool) -> list[dict]:
    return rows[:n] if top else rows[-n:]


def extreme_signals(bar: _Bar, nxt: _Bar | None, cfg: Thresholds, step: float) -> list[dict]:
    """Absorption, exhaustion and finished/unfinished state at the bar's high and low."""
    if len(bar.rows) < cfg.min_traded_rows:
        return []
    total = sum(c["total"] for c in bar.rows)
    poc_total = max(c["total"] for c in bar.rows)
    signals: list[dict] = []
    conf = cfg.partial_confidence if bar.partial else 1.0
    for top in (True, False):
        loc = "high" if top else "low"
        rows = bar.rows if top else list(reversed(bar.rows))  # always ordered from the extreme inwards
        extreme = rows[0]

        # absorption: heavy one-sided aggression at the extreme that did not carry price further
        zone = rows[: cfg.zone_rows]
        zvol = sum(c["total"] for c in zone)
        zdelta = sum(c["delta"] for c in zone)
        share, ratio = zvol / total, zdelta / zvol
        aggressive_ok = ratio >= cfg.absorption_min_delta_ratio if top else ratio <= -cfg.absorption_min_delta_ratio
        needed = max(cfg.absorption_min_share, cfg.absorption_vs_uniform * len(zone) / len(bar.rows))
        if share >= needed and aggressive_ok:
            confirmation = None
            if nxt is not None and nxt.ohlc:  # the next bar's real high/low
                held = nxt.ohlc["high_row"] <= extreme["price"] if top else nxt.ohlc["low_row"] >= extreme["price"]
                confirmation = "held" if held else "broken"
            elif nxt is not None and nxt.rows:  # fall back to the rows that traded
                held = (max(c["price"] for c in nxt.rows) <= extreme["price"] + step * 0.5) if top else (
                    min(c["price"] for c in nxt.rows) >= extreme["price"] - step * 0.5)
                confirmation = "held" if held else "broken"
            who = "buyers absorbed by passive sellers" if top else "sellers absorbed by passive buyers"
            signals.append(_signal(
                "absorption", "bearish" if top else "bullish", loc, extreme["price"],
                min(1.0, share / 0.5) * min(1.0, abs(ratio) / 0.5),
                {"zone_volume": zvol, "share_of_bar": round(share, 3), "share_needed": round(needed, 3),
                 "zone_delta": zdelta, "delta_ratio": round(ratio, 3)},
                f"absorption at the {loc}: {who}", confidence=conf, confirmation=confirmation,
            ))

        # exhaustion: volume thinning out towards the extreme
        seq = [c["total"] for c in rows[: cfg.taper_rows]]
        thin = extreme["total"] <= cfg.exhaustion_max_extreme_ratio * poc_total
        if len(seq) == cfg.taper_rows and all(a < b for a, b in zip(seq, seq[1:])) and thin:
            signals.append(_signal(
                "exhaustion", "bearish" if top else "bullish", loc, extreme["price"],
                1.0 - extreme["total"] / (cfg.exhaustion_max_extreme_ratio * poc_total + 1e-9),
                {"extreme_volume": extreme["total"], "poc_volume": poc_total, "taper": seq},
                f"volume tapering into the {loc}: {'buying' if top else 'selling'} interest fading", confidence=conf,
            ))

        # finished / unfinished auction at the extreme (both sides traded => likely revisited)
        unfinished = extreme["bid"] > 0 and extreme["ask"] > 0
        signals.append(_signal(
            "unfinished_extreme" if unfinished else "finished_extreme", "neutral", loc, extreme["price"],
            0.5 if unfinished else 0.3, {"bid": extreme["bid"], "ask": extreme["ask"]},
            f"{'unfinished' if unfinished else 'finished'} auction at the {loc}", confidence=conf,
        ))
    return signals


def candle_signals(bar: _Bar, cfg: Thresholds) -> list[dict]:
    """Signals that need the bar's open/close: wick rejection and delta divergence."""
    o = bar.ohlc
    if not o:
        return []
    rng = o["high"] - o["low"]
    if rng <= 0:
        return []
    out: list[dict] = []
    top_body, bottom_body = max(o["open"], o["close"]), min(o["open"], o["close"])
    upper, lower = (o["high"] - top_body) / rng, (bottom_body - o["low"]) / rng
    body = (top_body - bottom_body) / rng
    if upper >= cfg.rejection_min_wick_share and not o["high_clipped"]:
        out.append(_signal("rejection", "bearish", "high", o["high_row"], upper,
                           {"wick_share": round(upper, 3), "close_location": round((o["close"] - o["low"]) / rng, 3)},
                           "long upper wick: higher prices were rejected"))
    if lower >= cfg.rejection_min_wick_share and not o["low_clipped"]:
        out.append(_signal("rejection", "bullish", "low", o["low_row"], lower,
                           {"wick_share": round(lower, 3), "close_location": round((o["close"] - o["low"]) / rng, 3)},
                           "long lower wick: lower prices were rejected"))
    if bar.delta_ratio is not None and body >= cfg.divergence_min_body_share:
        ratio = bar.delta_ratio
        if abs(ratio) >= cfg.divergence_min_delta_ratio and ((o["direction"] == "up" and ratio < 0) or (o["direction"] == "down" and ratio > 0)):
            up = o["direction"] == "up"
            out.append(_signal(
                "delta_divergence", "bearish" if up else "bullish", "bar", o["close"], abs(ratio) / 0.3,
                {"direction": o["direction"], "delta": bar.delta, "delta_ratio": round(ratio, 4), "body_share": round(body, 3),
                 "delta_ratio_source": bar.delta_ratio_source, "platform_marker": bar.platform_divergence},
                f"price closed {'up' if up else 'down'} but net delta was {'negative' if up else 'positive'}",
            ))
    return out


def delta_dominance(bar: _Bar, cfg: Thresholds) -> list[dict]:
    """A bar whose net delta is a large share of its volume: one side did most of the aggressive trading.

    Uses the chart's Delta % (delta / volume) when shown. Strength grows linearly from the minimum ratio's
    share to `dominance_full_ratio`. When the candle closed against the delta the signal says so (that is also a
    `delta_divergence`) and its confidence is lowered, since aggression that did not move price is absorbed.
    """
    ratio = bar.delta_ratio
    if ratio is None or abs(ratio) < cfg.dominance_min_ratio:
        return []
    buy = ratio > 0
    o = bar.ohlc
    against = bool(o and ((o["direction"] == "up" and not buy) or (o["direction"] == "down" and buy)))
    evidence = {"delta": bar.delta, "volume": bar.volume, "delta_pct": round(100 * ratio, 2),
                "delta_pct_source": bar.delta_ratio_source}
    if o:
        evidence["candle_direction"] = o["direction"]
    price = o["close"] if o else (bar.poc if bar.poc is not None else 0.0)
    text = f"{'buyers' if buy else 'sellers'} were {abs(ratio) * 100:.0f}% of the bar's net volume"
    if against:
        text += ", yet the candle closed against them"
    return [_signal("delta_dominance", "bullish" if buy else "bearish", "bar", price, abs(ratio) / cfg.dominance_full_ratio,
                    evidence, text, confidence=0.6 if against else 1.0 if not bar.partial else cfg.partial_confidence)]


def flow_runs(bars: list[_Bar], cfg: Thresholds) -> dict:
    """Net delta as a share of volume bar by bar, and the current run of same-sided bars.

    A run is consecutive bars (no gaps) whose |delta %| is at least `flow_min_ratio` with the same sign; one that
    ends at the last analysed bar and has at least `flow_min_bars` bars is `persistent_delta` evidence that the
    same side has been the aggressor for a while.
    """
    series = [{"index": b.index, "delta_pct": None if b.delta_ratio is None else round(100 * b.delta_ratio, 2),
               "source": b.delta_ratio_source} for b in bars]
    run: list[_Bar] = []
    for b in bars:
        r = b.delta_ratio
        if r is None or abs(r) < cfg.flow_min_ratio:
            run = []
            continue
        if run and (b.index != run[-1].index + 1 or (r > 0) != (run[-1].delta_ratio > 0)):
            run = []
        run.append(b)
    out: dict = {"series": series, "run": None}
    if len(run) >= cfg.flow_min_bars and run[-1].index == bars[-1].index:
        avg = float(np.mean([b.delta_ratio for b in run]))
        out["run"] = {
            "bias": "bullish" if avg > 0 else "bearish",
            "bars": len(run), "from_index": run[0].index, "to_index": run[-1].index,
            "mean_delta_pct": round(100 * avg, 2),
            "text": f"{len(run)} consecutive bars with {'positive' if avg > 0 else 'negative'} delta "
                    f"(mean {100 * avg:+.1f}% of volume)",
        }
    return out


def poc_signal(bar: _Bar) -> list[dict]:
    if bar.poc is None or not bar.rows:
        return []
    hi, lo = max(c["price"] for c in bar.rows), min(c["price"] for c in bar.rows)
    pos = 0.5 if hi == lo else (bar.poc - lo) / (hi - lo)
    where = "upper" if pos >= 2 / 3 else "lower" if pos <= 1 / 3 else "middle"
    return [_signal("poc", "neutral", where, bar.poc, 0.2, {"position_in_range": round(pos, 3)}, f"POC in the {where} third")]


# --- cross-bar levels --------------------------------------------------------------------------

def build_levels(bars: list[dict], step: float, current: float | None, cfg: Thresholds,
                 extra_points: list[tuple[float, float, str]] | None = None) -> list[dict]:
    """Merge POCs, stacked imbalances, absorption and unfinished extremes into price levels."""
    points: list[tuple[float, float, str, int]] = []  # (price, weight, kind, bar index)
    for b in bars:
        for s in b["signals"]:
            kind, price = s["type"], s["price"]
            if kind == "poc":
                points.append((price, 1.0, "poc", b["index"]))
            elif kind == "stacked_imbalance":
                mid = (price + s["price_to"]) / 2
                points.append((mid, 1.5 + 0.25 * (s["evidence"]["rows"] - 3), "stacked_imbalance", b["index"]))
            elif kind == "absorption":
                points.append((price, 2.0 * s["strength"] + 0.5, "absorption", b["index"]))
            elif kind == "unfinished_extreme":
                points.append((price, 1.0, "unfinished_extreme", b["index"]))
            elif kind == "rejection":
                points.append((price, 0.5 + s["strength"], "rejection", b["index"]))
    for price, weight, kind in extra_points or []:  # e.g. the volume profile's POC / value area / nodes
        points.append((price, weight, kind, -1))
    points.sort()
    clusters: list[list[tuple[float, float, str, int]]] = []
    for pt in points:
        if clusters:
            c = clusters[-1]
            mean = sum(p * w for p, w, _, _ in c) / sum(w for _, w, _, _ in c)
            if pt[0] - mean <= cfg.level_merge_steps * step:
                c.append(pt)
                continue
        clusters.append([pt])
    ranges = {b["index"]: b["stats"].get("range") for b in bars}
    levels = []
    for c in clusters:
        weight = float(sum(w for _, w, _, _ in c))
        price = sum(p * w for p, w, _, _ in c) / weight
        price = round(round(price / step) * step, 6)
        idx = sorted({i for _, _, _, i in c if i >= 0})
        first = min(idx) if idx else -1
        later = [i for i, r in ranges.items() if r and i > first and r[0] - step / 2 <= price <= r[1] + step / 2]
        level = {
            "price": price,
            "strength": round(weight, 2),
            "kinds": sorted({k for _, _, k, _ in c}),
            "bars": idx,
            "retests": len(later),
        }
        if current is not None:
            level["role"] = "support" if price < current else "resistance" if price > current else "at_price"
            level["distance_steps"] = round((price - current) / step, 2)
        levels.append(level)
    return sorted(levels, key=lambda lv: -lv["strength"])


# --- entry point -------------------------------------------------------------------------------

def analyze(doc: dict, cfg: Thresholds | None = None) -> dict:
    """Analyse a `parse_screenshot` document."""
    cfg = cfg or Thresholds()
    step = _step(doc)
    prepared: list[tuple[dict, _Bar | None, str | None]] = []
    for bar in doc["bars"]:
        if not bar["valid"]:
            reason = (
                "clipped by the image edge" if bar["clipped"]
                else "failed validation: " + ", ".join(bar["failed_checks"]) if bar["failed_checks"]
                else "no footprint cells in the visible price range"
            )
            prepared.append((bar, None, reason))
        else:
            prepared.append((bar, _prepare(doc, bar), None))
    usable = [p for _, p, _ in prepared if p is not None]

    out_bars = []
    for bar, prep, reason in prepared:
        entry: dict = {"index": bar["index"], "time": bar["time"]}
        if prep is None:
            entry.update(status="excluded", reason=reason, signals=[], stats={})
            out_bars.append(entry)
            continue
        later = [u for u in usable if u.index > prep.index]
        nxt = later[0] if later and later[0].index == prep.index + 1 else None
        signals = (
            stacked_imbalances(prep, step, cfg) + extreme_signals(prep, nxt, cfg, step)
            + candle_signals(prep, cfg) + delta_dominance(prep, cfg) + poc_signal(prep)
        )
        prices = [c["price"] for c in prep.rows]
        stats = {
            "volume": prep.volume,
            "delta": prep.delta,
            "delta_ratio": None if prep.delta_ratio is None else round(prep.delta_ratio, 4),
            "delta_ratio_source": prep.delta_ratio_source,
            "platform_divergence": prep.platform_divergence,
            "range": [min(prices), max(prices)] if prices else None,
            "traded_rows": len(prep.rows),
            "poc": prep.poc,
        }
        if prep.ohlc:
            o = prep.ohlc
            rng = o["high"] - o["low"]
            stats.update({
                "open": o["open"], "high": o["high"], "low": o["low"], "close": o["close"],
                "direction": o["direction"],
                "close_location": round((o["close"] - o["low"]) / rng, 3) if rng > 0 else None,
                "body_share": round(abs(o["close"] - o["open"]) / rng, 3) if rng > 0 else None,
            })
        notes = []
        if prep.ohlc and (prep.ohlc["high_clipped"] or prep.ohlc["low_clipped"]):
            notes.append("candle wick runs off the visible plot: the true high/low is beyond it")
        if bar.get("ohlc") and not bar["ohlc"]["valid"]:
            notes.append("candle failed validation and was ignored: " + ", ".join(
                c["name"] for c in bar["ohlc"]["checks"] if c["status"] == "fail"))
        if prep.partial:
            notes.append("some cells hidden by the chart (...): confidence reduced")
        if prep.touches_view_edge:
            notes.append("bar reaches the edge of the visible price range: its true high/low may be off-screen")
        entry.update(status="analyzed", signals=signals, stats=stats, notes=notes)
        out_bars.append(entry)

    analyzed = [b for b in out_bars if b["status"] == "analyzed"]
    # support/resistance are relative to the current price when the chart draws it (red dashed line),
    # else the last close. The solid pink line is the profile POC, not the price.
    reference = doc["price"].get("current")
    if reference is None:
        reference = doc["price"].get("last_close")
    prof = profile_analysis(doc, cfg)
    levels = build_levels(analyzed, step, reference, cfg,
                          extra_points=profile_points(prof) if prof else None)
    return {
        "schema_version": SCHEMA_VERSION,
        "source": doc.get("source"),
        "thresholds": cfg.__dict__,
        "bars": out_bars,
        "levels": levels,
        "profile": prof,
        "flow": flow_runs(usable, cfg),
        "summary": {
            "analyzed_bars": len(analyzed),
            "excluded_bars": [b["index"] for b in out_bars if b["status"] == "excluded"],
            "signal_counts": _count(analyzed),
        },
    }


def _count(bars: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for b in bars:
        for s in b["signals"]:
            key = f"{s['type']}:{s['bias']}"
            counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


# --- volume profile (delta | volume per price over the visible range) -------------------------

def profile_analysis(doc: dict, cfg: Thresholds | None = None) -> dict | None:
    """What the right-hand profile says: value area, nodes, thin tails, one-sided aggression, shape.

    Uses only profile rows that passed validation. Returns None when the chart has no
    delta|volume profile.
    """
    cfg = cfg or Thresholds()
    if doc.get("profile_kind") != "delta_volume":
        return None
    rows = [r for r in doc["profile"] if r["valid"] and r["volume"] is not None and r["delta"] is not None]
    if len(rows) < 5:
        return None
    rows.sort(key=lambda r: r["price"])  # ascending
    step = doc["price"]["step_per_row"]
    vols = np.array([r["volume"] for r in rows], dtype=float)
    total = float(vols.sum())
    poc_i = int(vols.argmax())
    poc = rows[poc_i]
    median = float(np.median(vols[vols > 0])) if (vols > 0).any() else 0.0
    va = [r for r in rows if r["in_value_area"]]
    last = doc["price"].get("current")  # the chart's own current-price line, when drawn
    reference = "current"
    if last is None:
        last, reference = doc["price"].get("last_close"), "last_close"

    out: dict = {
        "poc": {"price": poc["price"], "volume": poc["volume"], "share_of_volume": round(poc["volume"] / total, 3),
                "delta": poc["delta"], "delta_ratio": round(poc["delta"] / poc["volume"], 3) if poc["volume"] else None},
        "range": {"low": rows[0]["price"], "high": rows[-1]["price"], "rows": len(rows)},
        "total_volume": total,
        "total_delta": float(sum(r["delta"] for r in rows)),
    }
    out["total_delta_ratio"] = round(out["total_delta"] / total, 3) if total else None
    if va:
        vah, val = max(r["price"] for r in va), min(r["price"] for r in va)
        out["value_area"] = {
            "high": vah, "low": val, "rows": len(va),
            "share_of_volume": round(sum(r["volume"] for r in va) / total, 3),
            "note": "colour-coded by the chart; not always contiguous",
        }
        if last is not None:
            out["reference_price"] = {"price": last, "source": reference}
            out["price_vs_value_area"] = (
                "above" if last > vah + step / 2 else "below" if last < val - step / 2 else "inside")
            out["price_distance_from_poc_steps"] = round((last - poc["price"]) / step, 2)

    # high / low volume nodes: local extrema relative to the typical row
    hvn, lvn = [], []
    for i in range(1, len(rows) - 1):
        v = vols[i]
        if median and v >= vols[i - 1] and v >= vols[i + 1] and v >= cfg.hvn_min_ratio * median and i != poc_i:
            hvn.append({"price": rows[i]["price"], "volume": float(v), "ratio_to_median": round(float(v / median), 2)})
        if median and v <= vols[i - 1] and v <= vols[i + 1] and v <= cfg.lvn_max_ratio * median:
            lvn.append({"price": rows[i]["price"], "volume": float(v), "ratio_to_median": round(float(v / median), 2)})
    out["high_volume_nodes"] = hvn
    out["low_volume_nodes"] = lvn

    # thin tails at the edges of the profile (poor highs/lows: little accepted business there)
    def tail(rows_edge):
        n = 0
        for r in rows_edge:
            if r["volume"] <= cfg.tail_ratio * poc["volume"]:
                n += 1
            else:
                break
        return n
    th, tl = tail(reversed(rows)), tail(iter(rows))
    out["thin_tails"] = {"high_rows": th, "low_rows": tl}

    # one-sided aggression: rows where one side dominated the volume
    aggressive = []
    for r in rows:
        if r["volume"] >= cfg.aggressive_min_volume_ratio * poc["volume"] and r["volume"] > 0:
            ratio = r["delta"] / r["volume"]
            if abs(ratio) >= cfg.aggressive_delta_ratio:
                aggressive.append({"price": r["price"], "side": "buying" if ratio > 0 else "selling",
                                   "delta": r["delta"], "volume": r["volume"], "delta_ratio": round(ratio, 3)})
    out["aggressive_levels"] = aggressive

    # shape: where the volume sits within the range (P = top-heavy, b = bottom-heavy, D = balanced)
    span = rows[-1]["price"] - rows[0]["price"]
    centroid = float((np.array([r["price"] for r in rows]) * vols).sum() / total) if total else rows[0]["price"]
    pos = (centroid - rows[0]["price"]) / span if span else 0.5
    out["shape"] = {
        "type": "P" if pos >= cfg.shape_skew else "b" if pos <= 1 - cfg.shape_skew else "D",
        "centroid_price": round(centroid, 4),
        "centroid_position": round(pos, 3),
    }
    # delta above vs below the POC: who has been winning on each side of value
    above = sum(r["delta"] for r in rows if r["price"] > poc["price"])
    below = sum(r["delta"] for r in rows if r["price"] < poc["price"])
    out["delta_around_poc"] = {"above": above, "below": below}
    return out


def profile_points(prof: dict) -> list[tuple[float, float, str]]:
    """(price, weight, kind) points implied by the profile, to be clustered with the bar levels."""
    out = [(prof["poc"]["price"], 3.0, "profile_poc")]
    if "value_area" in prof:
        out.append((prof["value_area"]["high"], 2.0, "value_area_high"))
        out.append((prof["value_area"]["low"], 2.0, "value_area_low"))
    out += [(n["price"], 1.0 + min(1.0, n["ratio_to_median"] / 4), "high_volume_node") for n in prof["high_volume_nodes"]]
    # low-volume nodes are thin spots (prices move quickly through them), not support/resistance:
    # they stay in the profile analysis and out of the level clustering
    out += [(a["price"], 1.0, f"aggressive_{a['side']}") for a in prof["aggressive_levels"]]
    return out
