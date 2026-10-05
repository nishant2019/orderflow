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


def infer_unreadable(geo, reports: list, table: list[TableColumn], profile: list) -> list[Inference]:
    """Inferences for every unreadable cell whose row (or column) has no other unknown."""
    from .validate import table_tolerance

    unreadable = {(rep.col, row) for rep in reports for row, c in rep.cells if c.bid is None or c.ask is None}
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
