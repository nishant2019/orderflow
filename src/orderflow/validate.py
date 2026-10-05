"""Cross-checks between the footprint cells and the summary table.

Every number shown on the chart is rounded, so each check carries a tolerance built from
the displayed precision (3.5K is +-50, 326 is +-0.5). A failed check does not say which
number is wrong - it says this bar's data should not be trusted without a look.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .cells import CellText, display_tolerance, parse_cell, GlyphClassifier
from .pipeline import ShotGeometry
from .poc import find_poc_rows
from .table import TableColumn

SLACK = 1.0  # absolute volume units added to every tolerance


@dataclass(frozen=True)
class Check:
    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str = ""


@dataclass
class ColumnReport:
    col: int
    cells: list[tuple[int, CellText]]  # (row index, parsed cell)
    checks: list[Check] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


def table_tolerance(raw: str) -> float:
    return display_tolerance(raw.lstrip("-"))


def validate_columns(
    geo: ShotGeometry, table: list[TableColumn], clf: GlyphClassifier
) -> list[ColumnReport]:
    poc = find_poc_rows(geo)
    reports = []
    for col, tcol in enumerate(table):
        cells = [(row, parse_cell(glyphs, geo.glyph_h, clf)) for row, glyphs in geo.text_cells(col)]
        rep = ColumnReport(col, cells)
        reports.append(rep)
        if tcol.clipped:
            rep.checks.append(Check("column", "skip", "clipped by the image edge"))
            continue
        if not cells:
            rep.checks.append(Check("column", "skip", "no footprint cells visible (bar outside the price range?)"))
            continue
        _check_sums(rep, tcol)
        _check_poc(rep, poc.get(col))
    _check_cum_chain(reports, table)
    return reports


def _check_sums(rep: ColumnReport, tcol: TableColumn) -> None:
    parsed = [(r, c) for r, c in rep.cells if not c.ellipsis]
    unreadable = [r for r, c in parsed if c.bid is None]
    n_ellipsis = len(rep.cells) - len(parsed)
    if unreadable:
        rep.checks.append(Check("cells", "fail", f"{len(unreadable)} cell(s) did not parse"))
        return
    vol_cells = sum(c.bid + c.ask for _, c in parsed)
    delta_cells = sum(c.ask - c.bid for _, c in parsed)
    tol = sum(c.bid_tol + c.ask_tol for _, c in parsed) + SLACK
    for name, got, cell_sum, extra in (
        ("volume", tcol.volume, vol_cells, 0.0),
        ("delta", tcol.delta, delta_cells, 0.0),
    ):
        if got.value is None:
            rep.checks.append(Check(name, "skip", "table value unreadable"))
            continue
        allowed = tol + table_tolerance(got.raw) + extra
        diff = cell_sum - got.value
        if n_ellipsis:
            # hidden "..." cells hold unknown volume: only a lower bound on volume holds
            if name == "volume":
                status = "ok" if diff <= allowed else "fail"
                rep.checks.append(Check(name, status, f"cells {cell_sum:.0f} vs table {got.value:.0f} (+{n_ellipsis} hidden)"))
            else:
                rep.checks.append(Check(name, "skip", f"{n_ellipsis} hidden cell(s)"))
            continue
        status = "ok" if abs(diff) <= allowed else "fail"
        rep.checks.append(Check(name, status, f"cells {cell_sum:.0f} vs table {got.value:.0f} (tol {allowed:.0f})"))
    if tcol.volume.value is not None and tcol.delta.value is not None and abs(tcol.delta.value) > tcol.volume.value + SLACK:
        rep.checks.append(Check("table", "fail", "|delta| exceeds volume"))


def _check_poc(rep: ColumnReport, poc_row: int | None) -> None:
    if poc_row is None:
        rep.checks.append(Check("poc", "skip", "no POC box found"))
        return
    by_row = {r: c for r, c in rep.cells if c.bid is not None}
    if poc_row not in by_row:
        rep.checks.append(Check("poc", "fail", f"POC box on row {poc_row} has no readable cell"))
        return
    totals = {r: c.bid + c.ask for r, c in by_row.items()}
    tol = by_row[poc_row].bid_tol + by_row[poc_row].ask_tol + SLACK
    top_row = max(totals, key=totals.get)
    ok = totals[poc_row] + tol + 1 >= totals[top_row] - (by_row[top_row].bid_tol + by_row[top_row].ask_tol)
    rep.checks.append(Check("poc", "ok" if ok else "fail", f"POC total {totals[poc_row]:.0f}, max {totals[top_row]:.0f} (row {top_row})"))


def _check_cum_chain(reports: list[ColumnReport], table: list[TableColumn]) -> None:
    """cum[c] == cum[c-1] + delta[c], or == delta[c] when a new session resets the sum."""
    for c, tcol in enumerate(table):
        if tcol.clipped or tcol.cum.value is None or tcol.delta.value is None:
            continue
        tol = table_tolerance(tcol.cum.raw) + table_tolerance(tcol.delta.raw) + SLACK
        if c > 0 and not table[c - 1].clipped and table[c - 1].cum.value is not None:
            tol += table_tolerance(table[c - 1].cum.raw)
            chained = abs(table[c - 1].cum.value + tcol.delta.value - tcol.cum.value) <= tol
        else:
            chained = False
        reset = abs(tcol.delta.value - tcol.cum.value) <= tol
        if chained or reset:
            reports[c].checks.append(Check("cum", "ok", "chain" if chained else "session reset"))
        elif c == 0 or table[c - 1].clipped:
            reports[c].checks.append(Check("cum", "skip", "previous bar not fully visible"))
        else:
            reports[c].checks.append(
                Check("cum", "fail", f"prev cum {table[c-1].cum.value:.0f} + delta {tcol.delta.value:.0f} != {tcol.cum.value:.0f}")
            )
