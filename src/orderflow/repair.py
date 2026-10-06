"""Fill in a cell that cannot be read by solving it from the chart's own totals.

A text label or another overlay can cover a cell (the pink price-line label sits over the first
column's cells). Its bid and ask are then the only unknowns in two equations:

    bid + ask = total volume  - sum of the other cells        (volume)
    ask - bid = total delta   - sum of the other cells        (delta)

The totals come from the profile row (preferred: the table check of the column then stays an
independent test) or, failing that, from the table column. The result is marked `inferred` and
carries the uncertainty accumulated from the displayed rounding of every number involved.
"""
from __future__ import annotations

from dataclasses import dataclass

from .cells import CellText
from .table import TableColumn


@dataclass(frozen=True)
class Inference:
    col: int
    row: int
    source: str  # "profile" | "table"
    cell: CellText


def _solve(volume: float, delta: float, tol: float) -> tuple[float, float] | None:
    ask, bid = (volume + delta) / 2, (volume - delta) / 2
    if ask < -tol or bid < -tol:
        return None  # not a physical solution: the equations are inconsistent
    return max(bid, 0.0), max(ask, 0.0)


def suspect_cells(reports: list, profile: list) -> set[tuple[int, int]]:
    """Cells that parsed but are probably covered or garbled: the one cell where a failing column
    total and a failing profile row meet (so nothing else can explain both)."""
    from .validate import validate_profile2

    bad_rows = {p.row for p in validate_profile2(profile, reports)
                if any(c.name in ("volume", "delta") and c.status == "fail" for c in p.checks)}
    out: set[tuple[int, int]] = set()
    for rep in reports:
        if not any(c.name in ("volume", "delta") and c.status == "fail" for c in rep.checks):
            continue
        hits = [(rep.col, r) for r, c in rep.cells if r in bad_rows and c.bid is not None and not c.inferred]
        if len(hits) == 1:
            out.add(hits[0])
    return out


def infer_unreadable(geo, reports: list, table: list[TableColumn], profile: list, suspects=frozenset()) -> list[Inference]:
    """Inferences for every unreadable cell whose row (or column) has no other unknown."""
    from .validate import table_tolerance

    unreadable = {(rep.col, row) for rep in reports for row, c in rep.cells if c.bid is None or c.ask is None}
    unreadable |= set(suspects)
    if not unreadable:
        return []
    by_row = {p.row: p for p in profile if p.delta is not None and p.volume is not None}
    out: list[Inference] = []
    for col, row in sorted(unreadable):
        others_row = [(rep.col, c) for rep in reports for r, c in rep.cells if r == row and (rep.col, r) != (col, row)]
        solved = None
        if row in by_row and all(c.bid is not None and c.ask is not None for _, c in others_row):
            p = by_row[row]
            vol = p.volume - sum(c.bid + c.ask for _, c in others_row)
            dlt = p.delta - sum(c.ask - c.bid for _, c in others_row)
            tol = table_tolerance(p.volume_raw) + table_tolerance(p.delta_raw.lstrip("-")) + sum(c.bid_tol + c.ask_tol for _, c in others_row)
            sol = _solve(vol, dlt, tol)
            if sol:
                solved = ("profile", sol, tol)
        if solved is None:
            rep = reports[col]
            tcol = table[col]
            others = [c for r, c in rep.cells if r != row]
            if (not tcol.clipped and tcol.volume.value is not None and tcol.delta.value is not None
                    and all(c.bid is not None and c.ask is not None for c in others)
                    and sum(1 for r, c in rep.cells if c.bid is None or c.ask is None) == 1):
                vol = tcol.volume.value - sum(c.bid + c.ask for c in others)
                dlt = tcol.delta.value - sum(c.ask - c.bid for c in others)
                tol = table_tolerance(tcol.volume.raw) + table_tolerance(tcol.delta.raw) + sum(c.bid_tol + c.ask_tol for c in others)
                sol = _solve(vol, dlt, tol)
                if sol:
                    solved = ("table", sol, tol)
        if solved is None:
            continue
        source, (bid, ask), tol = solved
        out.append(Inference(col, row, source, CellText(
            raw="inferred", bid=bid, ask=ask, sell_imbalance=False, buy_imbalance=False, ellipsis=False,
            confidence=0.5, bid_tol=tol, ask_tol=tol, inferred=True,
        )))
    return out


RIVAL_COST_RATIO = 1.5  # a rival reading must be this much farther from the glyph than the winner
MIN_FIX_MARGIN = 0.5  # a correction must beat the runner-up by this much (errors are in units of the allowed tolerance)


# --- correcting misreads with the chart's own checks --------------------------------------------

@dataclass(frozen=True)
class Correction:
    col: int
    row: int
    was: str
    now: str
    cell: CellText


def _with_cell(reports: list, col: int, row: int, cell: CellText) -> list:
    from .validate import ColumnReport

    out = []
    for rep in reports:
        if rep.col != col:
            out.append(rep)
        else:
            out.append(ColumnReport(rep.col, [(r, cell if r == row else c) for r, c in rep.cells]))
    return out


def _alternative_texts(geo, col: int, row: int, side: str, clf, with_cost: bool = False) -> list:
    """Readings of one box that differ from the best guess by a single glyph."""
    from .cells import glyph_features

    glyphs = geo.half_glyphs(col, row, side)
    feats = [glyph_features(g, geo.glyph_h) for g in glyphs]
    best = [clf.classify(f)[0] for f in feats]
    out = []
    for i, f in enumerate(feats):
        if best[i] in ".K" or best[i] == "-":
            continue
        for alt in clf.alternatives(f):
            if alt.isdigit():
                text = "".join(best[:i]) + alt + "".join(best[i + 1 :])
                out.append((text, clf.label_distance(f, alt)) if with_cost else text)
    return out


def correct_misreads(geo, clf, reports: list, table: list[TableColumn], profile: list) -> list[Correction]:
    """Fix single-glyph misreads that break a profile row's totals.

    For every profile row whose volume or delta check fails, each cell on that row is tried with
    each alternative single-glyph reading. A change is accepted only if it is the *only* one that
    makes the row total, the cell's column total and the imbalance rule all pass at once.
    """
    from .cells import display_tolerance, parse_number
    from .validate import ColumnReport, SLACK, _check_imbalance, _check_sums, _row_sums, table_tolerance, validate_profile2

    fixes: list[Correction] = []
    for pc in validate_profile2(profile, reports):
        if pc.ok or not any(c.name in ("volume", "delta") and c.status == "fail" for c in pc.checks):
            continue
        prow = next(p for p in profile if p.row == pc.row)
        found: list[Correction] = []
        for rep in reports:
            for row, cell in rep.cells:
                if row != pc.row or cell.bid is None or cell.ask is None or cell.inferred:
                    continue
                for side in ("bid", "ask"):
                    for alt_text in _alternative_texts(geo, rep.col, row, side, clf):
                        value = parse_number(alt_text)
                        if value is None or alt_text.startswith("0K"):  # "0K" is never displayed
                            continue
                        bid = value if side == "bid" else cell.bid
                        ask = value if side == "ask" else cell.ask
                        bid_t, ask_t = cell.raw.split("X")
                        new_raw = f"{alt_text}X{ask_t}" if side == "bid" else f"{bid_t}X{alt_text}"
                        new = CellText(new_raw, bid, ask, cell.sell_imbalance, cell.buy_imbalance, False, 0.6,
                                       display_tolerance(new_raw.split("X")[0]), display_tolerance(new_raw.split("X")[1]), False, cell.raw)
                        trial = _with_cell(reports, rep.col, row, new)
                        delta, volume, tol, _ = _row_sums(trial)
                        tol_v = tol.get(row, 0.0) + table_tolerance(prow.volume_raw) + SLACK
                        tol_d = tol.get(row, 0.0) + table_tolerance(prow.delta_raw.lstrip("-")) + SLACK
                        if abs(volume[row] - prow.volume) > tol_v or abs(delta[row] - prow.delta) > tol_d:
                            continue
                        tmp = ColumnReport(rep.col, next(t for t in trial if t.col == rep.col).cells)
                        _check_sums(tmp, table[rep.col])
                        ratio = getattr(geo, "imbalance_ratio", None) if getattr(geo, "imbalance_shown", True) else None
                        if ratio:
                            _check_imbalance(tmp, ratio)
                        if any(c.status == "fail" for c in tmp.checks):
                            continue
                        found.append(Correction(rep.col, row, cell.raw, new_raw, new))
        if len(found) == 1:
            fixes.append(found[0])
    return fixes


def correct_column_misreads(geo, clf, reports: list, table: list[TableColumn], profile: list) -> list[Correction]:
    """Fix a single-glyph misread that only the bar's table totals reveal (the profile row is too coarse).

    For a column whose volume/delta check fails, every cell is tried with each single-glyph
    alternative; the change is accepted only if exactly one makes every column check, the imbalance
    rule and the cell's profile row pass.
    """
    from .cells import display_tolerance, parse_number
    from .validate import ColumnReport, _check_imbalance, _check_sums, table_tolerance, validate_profile2

    fixes: list[Correction] = []
    rows_ok = {p.row for p in validate_profile2(profile, reports) if p.ok}
    for rep in reports:
        if not any(c.name in ("volume", "delta") and c.status == "fail" for c in rep.checks):
            continue
        found: list[Correction] = []
        for row, cell in rep.cells:
            if cell.bid is None or cell.ask is None or cell.inferred or cell.ellipsis:
                continue
            for side in ("bid", "ask"):
                for alt_text, cost in _alternative_texts(geo, rep.col, row, side, clf, with_cost=True):
                    value = parse_number(alt_text)
                    if value is None or alt_text.startswith("0K"):
                        continue
                    bid_t, ask_t = cell.raw.split("X")
                    new_raw = f"{alt_text}X{ask_t}" if side == "bid" else f"{bid_t}X{alt_text}"
                    new = CellText(new_raw, value if side == "bid" else cell.bid, value if side == "ask" else cell.ask,
                                   cell.sell_imbalance, cell.buy_imbalance, False, 0.6,
                                   display_tolerance(new_raw.split("X")[0]), display_tolerance(new_raw.split("X")[1]),
                                   False, cell.raw)
                    trial = _with_cell(reports, rep.col, row, new)
                    tmp = ColumnReport(rep.col, next(t for t in trial if t.col == rep.col).cells)
                    _check_sums(tmp, table[rep.col])
                    ratio = getattr(geo, "imbalance_ratio", None) if getattr(geo, "imbalance_shown", True) else None
                    if ratio:
                        _check_imbalance(tmp, ratio)
                    if any(c.status == "fail" for c in tmp.checks):
                        continue
                    if any(r not in {q.row for q in validate_profile2(profile, trial) if q.ok} for r in rows_ok):
                        continue
                    miss = 0.0  # how far the totals still are from the table, in units of the allowed tolerance
                    for got, total in ((table[rep.col].volume, sum(c.bid + c.ask for _, c in tmp.cells if c.bid is not None)),
                                       (table[rep.col].delta, sum(c.ask - c.bid for _, c in tmp.cells if c.bid is not None))):
                        if got.value is not None:
                            miss += abs(total - got.value) / max(table_tolerance(got.raw), 1.0)
                    found.append((cost * (1.0 + miss), Correction(rep.col, row, cell.raw, new_raw, new)))
        found.sort(key=lambda t: t[0])
        # the totals alone are loose (displayed numbers are rounded): the glyph shape arbitrates, and must
        # favour one reading clearly over every other that fits
        if found and (len(found) == 1 or found[1][0] >= RIVAL_COST_RATIO * found[0][0]):
            fixes.append(found[0][1])
    return fixes


def correct_profile_misreads(geo, clf, reports: list, profile: list) -> list:
    """Fix a single-glyph misread in a profile row's own text (e.g. `8K` read as `5K`).

    Only rows whose totals fail are touched, and only when one single-glyph alternative of the
    delta or volume text fits the cells essentially exactly while every other alternative fits
    clearly worse (or is the only one that fits). Returns the profile
    with corrected rows replaced (marked with `corrected_from`).
    """
    from dataclasses import replace

    from .cells import glyph_features
    from .table import parse_signed
    from .validate import SLACK, _row_sums, table_tolerance, validate_profile2

    delta, volume, tol, unknown = _row_sums(reports)
    bad_rows = {p.row for p in validate_profile2(profile, reports) if not p.ok}
    out = []
    for row in profile:
        if row.row not in bad_rows or row.row in unknown or row.row not in volume:
            out.append(row)
            continue
        found = []
        for which, glyphs, raw in (("delta", row.delta_glyphs, row.delta_raw), ("volume", row.volume_glyphs, row.volume_raw)):
            feats = [glyph_features(g, geo.glyph_h) for g in glyphs]
            best = [clf.classify(f)[0] for f in feats]
            for i, f in enumerate(feats):
                if best[i] in ".K-M":
                    continue
                for alt in clf.alternatives(f):
                    if not alt.isdigit():
                        continue
                    text = "".join(best[:i]) + alt + "".join(best[i + 1 :])
                    d_text = text if which == "delta" else row.delta_raw
                    v_text = text if which == "volume" else row.volume_raw
                    d_val, v_val = parse_signed(d_text), parse_signed(v_text)
                    if d_val is None or v_val is None:
                        continue
                    allowed_v = tol.get(row.row, 0.0) + table_tolerance(v_text) + SLACK
                    allowed_d = tol.get(row.row, 0.0) + table_tolerance(d_text.lstrip("-")) + SLACK
                    err_v = abs(volume[row.row] - v_val) / allowed_v
                    err_d = abs(delta[row.row] - d_val) / allowed_d
                    if err_v <= 1.0 and err_d <= 1.0:
                        found.append((err_v + err_d, d_text, v_text, d_val, v_val))
        found = sorted(set(found))
        # accept the best candidate if it is the only one that fits, or beats every rival by a
        # clear margin (errors are summed, in units of the allowed tolerance)
        if found and (len(found) == 1 or found[1][0] - found[0][0] >= MIN_FIX_MARGIN):
            _, d_text, v_text, d_val, v_val = found[0]
            out.append(replace(row, delta_raw=d_text, volume_raw=v_text, delta=d_val, volume=v_val,
                               corrected_from=f"{row.delta_raw}|{row.volume_raw}", confidence=0.6))
        else:
            out.append(row)
    return out
