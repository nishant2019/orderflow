"""Cross-checks between the footprint cells and the summary table.

Every number shown on the chart is rounded, so each check carries a tolerance built from
the displayed precision (3.5K is +-50, 326 is +-0.5). A failed check does not say which
number is wrong - it says this bar's data should not be trusted without a look.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .cells import CellText, display_tolerance, GlyphClassifier
from .pipeline import ShotGeometry
from .table import TableColumn

SLACK = 1.0  # absolute volume units added to every tolerance
LENGTH_REL_TOL = 0.10  # bar length vs fitted scale: catches gross misreads, not rounding
MIN_BAR_PX = 25  # shorter bars are drawn at a minimum visible width, so their length says little


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
    poc = geo.find_poc_rows()
    reports = []
    for col, tcol in enumerate(table):
        cells = geo.read_cells(col, clf)
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
        ratio = getattr(geo, "imbalance_ratio", None)
        if ratio:
            _check_imbalance(rep, ratio)
    _check_cum_chain(reports, table)
    return reports


def _check_sums(rep: ColumnReport, tcol: TableColumn) -> None:
    parsed = [(r, c) for r, c in rep.cells if not c.ellipsis]
    unreadable = [r for r, c in parsed if c.bid is None or c.ask is None]
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


def _check_imbalance(rep: ColumnReport, ratio: float) -> None:
    """The drawn imbalance flags must follow the chart's own rule (settings: Ratio, 300 %).

    Diagonal comparison, verified on every drawn flag of the sample charts: a sell imbalance
    (orange bid) means bid[r] >= ratio * ask[row above]; a buy imbalance (orange ask) means
    ask[r] >= ratio * bid[row below]. A neighbouring row without a cell counts as zero volume inside the
    bar's range (no trades) and as "nothing to compare" beyond its top or bottom row.
    Both directions are checked, within the rounding of the displayed numbers.
    """
    cells = {row: c for row, c in rep.cells if c.bid is not None and c.ask is not None}
    problems = []
    for row, c in cells.items():
        for side, own, own_tol, nb_row, flag in (
            ("sell", c.bid, c.bid_tol, row - 1, c.sell_imbalance),
            ("buy", c.ask, c.ask_tol, row + 1, c.buy_imbalance),
        ):
            nb = cells.get(nb_row)
            if nb is None:
                continue  # neighbour not drawn (a gap, or beyond the bar): the platform's rule there is not visible
            opp, opp_tol = (nb.ask, nb.ask_tol) if side == "sell" else (nb.bid, nb.bid_tol)
            surely = own - own_tol >= ratio * (opp + opp_tol) and own > 0  # imbalanced whatever the rounding
            possibly = own + own_tol >= ratio * max(opp - opp_tol, 0.0) and own > 0
            if flag and not possibly:
                problems.append(f"row {row} {side}: drawn, but {own:g} is not {ratio:g}x {opp:g}")
            elif not flag and surely:
                problems.append(f"row {row} {side}: {own:g} >= {ratio:g}x {opp:g} but not drawn")
    rep.checks.append(Check("imbalance", "fail" if problems else "ok", "; ".join(problems[:3])))


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


@dataclass(frozen=True)
class ProfileCheck:
    row: int
    price: float
    raw: str
    checks: list[Check]

    @property
    def ok(self) -> bool:
        return all(c.status != "fail" for c in self.checks)


def validate_profile(profile: list, reports: list[ColumnReport]) -> list[ProfileCheck]:
    """Check each profile row's printed delta against its bar and against the cells.

    * sign:   the text sign must match the bar colour (no bar <=> zero or tiny value);
    * length: bar length is proportional to |value| (scale fitted from the rows themselves);
    * cells:  the value equals the sum of (ask - bid) over every bar's cell on that row.
    """
    # per-row sums over all bars; a row with an unreadable or "..." cell has unknown total
    sums: dict[int, float] = {}
    tols: dict[int, float] = {}
    unknown: set[int] = set()
    for rep in reports:
        for row, cell in rep.cells:
            if cell.ellipsis or cell.bid is None:
                unknown.add(row)
                continue
            sums[row] = sums.get(row, 0.0) + cell.ask - cell.bid
            tols[row] = tols.get(row, 0.0) + cell.bid_tol + cell.ask_tol

    scale_samples = [
        r.length_px / abs(r.value) for r in profile if r.value and abs(r.value) >= 1000 and r.length_px >= 15
    ]
    scale = float(np.median(scale_samples)) if scale_samples else None

    out = []
    for r in profile:
        checks: list[Check] = []
        out.append(ProfileCheck(r.row, r.price, r.raw, checks))
        if r.value is None:
            if r.raw or r.sign:
                checks.append(Check("text", "fail", f"unreadable profile text '{r.raw}'"))
            continue
        tol_text = table_tolerance(r.raw)
        if r.value != 0 and r.sign != 0 and (r.value > 0) != (r.sign > 0):
            checks.append(Check("sign", "fail", f"text {r.raw} but bar is {'green' if r.sign > 0 else 'red'}"))
        elif r.value == 0 and r.sign != 0:
            checks.append(Check("sign", "fail", "zero value but a bar is drawn"))
        else:
            checks.append(Check("sign", "ok"))
        if scale and abs(r.value) >= 1000:
            expect = scale * abs(r.value)
            slack = 3.0 + LENGTH_REL_TOL * expect + scale * tol_text
            ok = abs(r.length_px - expect) <= slack
            checks.append(Check("length", "ok" if ok else "fail", f"bar {r.length_px}px, expected {expect:.0f}px for {r.raw}"))
        if r.row in unknown:
            checks.append(Check("cells", "skip", "a cell on this row is unreadable or hidden"))
        elif r.row in sums:
            allowed = tols[r.row] + tol_text + SLACK
            diff = sums[r.row] - r.value
            checks.append(
                Check("cells", "ok" if abs(diff) <= allowed else "fail",
                      f"cells {sums[r.row]:.0f} vs profile {r.value:.0f} (tol {allowed:.0f})")
            )
        else:
            checks.append(Check("cells", "ok" if abs(r.value) <= tol_text + SLACK else "fail", "no cells on this row"))
    return out


def validate_candles(candles: list, traded_prices: list[list[float]], step: float, price_per_px: float,
                     new_session: list[bool]) -> list[list[Check]]:
    """Checks on each bar's OHLC.

    * continuity: the open chains from the previous close (not across a new session, where
      overnight gaps are normal);
    * range: the candle's high and low cover the highest and lowest rows that traded
      (a side whose wick runs off the visible plot is not checked).
    `traded_prices[i]` are the row prices with volume in bar i.
    """
    from .candles import bucket

    out: list[list[Check]] = []
    for i, candle in enumerate(candles):
        checks: list[Check] = []
        out.append(checks)
        if candle is None:
            continue
        prev = candles[i - 1] if i > 0 else None
        if prev is not None and not new_session[i]:
            gap = candle.open - prev.close
            tol = 0.8 * step + 2 * price_per_px
            checks.append(Check("continuity", "ok" if abs(gap) <= tol else "fail",
                                f"open {candle.open:.2f} vs previous close {prev.close:.2f} (gap {gap:+.2f}, tol {tol:.2f})"))
        else:
            checks.append(Check("continuity", "skip", "first bar or new session"))
        if traded_prices[i]:
            top, bottom = max(traded_prices[i]), min(traded_prices[i])
            high_ok = candle.high_clipped or bucket(candle.high, step, price_per_px) >= top - 1e-6
            low_ok = candle.low_clipped or bucket(candle.low, step, price_per_px) <= bottom + 1e-6
            checks.append(Check("range", "ok" if high_ok and low_ok else "fail",
                                f"candle {candle.low:.2f}-{candle.high:.2f} vs traded rows {bottom:g}-{top:g}"))
    return out


def _row_sums(reports: list[ColumnReport]) -> tuple[dict[int, float], dict[int, float], dict[int, float], set[int]]:
    """Per row: sum of (ask - bid), sum of (bid + ask), summed display tolerance, and the rows
    that have an unreadable or hidden cell (their totals are unknown)."""
    delta: dict[int, float] = {}
    volume: dict[int, float] = {}
    tol: dict[int, float] = {}
    unknown: set[int] = set()
    for rep in reports:
        for row, cell in rep.cells:
            if cell.ellipsis or cell.bid is None or cell.ask is None:
                unknown.add(row)
                continue
            delta[row] = delta.get(row, 0.0) + cell.ask - cell.bid
            volume[row] = volume.get(row, 0.0) + cell.ask + cell.bid
            tol[row] = tol.get(row, 0.0) + cell.bid_tol + cell.ask_tol
    return delta, volume, tol, unknown


def validate_profile2(profile: list, reports: list[ColumnReport]) -> list[ProfileCheck]:
    """Row checks for the centre-axis profile.

    * delta:   printed delta == sum over bars of (ask - bid) on the row;
    * volume:  printed volume == sum over bars of (bid + ask) on the row (the stronger test:
               volume is never negative and cancels nothing);
    * bars:    bar lengths are proportional to |delta| / volume (scales fitted from the rows);
    * sign:    the delta bar colour agrees with the sign of the printed delta.
    """
    delta, volume, tol, unknown = _row_sums(reports)
    d_scale = [r.delta_px / abs(r.delta) for r in profile if r.delta and abs(r.delta) >= 1000 and r.delta_px >= MIN_BAR_PX]
    v_scale = [r.volume_px / r.volume for r in profile if r.volume and r.volume >= 1000 and r.volume_px >= MIN_BAR_PX]
    ds = float(np.median(d_scale)) if d_scale else None
    vs = float(np.median(v_scale)) if v_scale else None
    out = []
    for r in profile:
        checks: list[Check] = []
        out.append(ProfileCheck(r.row, r.price, f"{r.delta_raw}|{r.volume_raw}", checks))
        if r.delta is None or r.volume is None:
            checks.append(Check("text", "fail", f"unreadable profile text '{r.delta_raw}' | '{r.volume_raw}'"))
            continue
        if r.row in unknown:
            checks.append(Check("cells", "skip", "a cell on this row is unreadable or hidden"))
        elif r.row in delta or r.row in volume:
            allowed_v = tol.get(r.row, 0.0) + table_tolerance(r.volume_raw) + SLACK
            allowed_d = tol.get(r.row, 0.0) + table_tolerance(r.delta_raw.lstrip("-")) + SLACK
            dv, dd = volume.get(r.row, 0.0) - r.volume, delta.get(r.row, 0.0) - r.delta
            checks.append(Check("volume", "ok" if abs(dv) <= allowed_v else "fail",
                                f"cells {volume.get(r.row, 0.0):.0f} vs profile {r.volume:.0f} (tol {allowed_v:.0f})"))
            checks.append(Check("delta", "ok" if abs(dd) <= allowed_d else "fail",
                                f"cells {delta.get(r.row, 0.0):.0f} vs profile {r.delta:.0f} (tol {allowed_d:.0f})"))
        else:
            checks.append(Check("cells", "ok" if abs(r.volume) <= table_tolerance(r.volume_raw) + SLACK else "fail", "no cells on this row"))
        if ds and abs(r.delta) >= 1000 and ds * abs(r.delta) >= MIN_BAR_PX:
            expect = ds * abs(r.delta)
            ok = abs(r.delta_px - expect) <= 4.0 + 0.12 * expect + ds * table_tolerance(r.delta_raw.lstrip("-"))
            checks.append(Check("delta_bar", "ok" if ok else "fail", f"bar {r.delta_px}px, expected {expect:.0f}px"))
        if vs and r.volume >= 1000 and vs * r.volume >= MIN_BAR_PX:
            expect = vs * r.volume
            ok = abs(r.volume_px - expect) <= 4.0 + 0.12 * expect + vs * table_tolerance(r.volume_raw)
            checks.append(Check("volume_bar", "ok" if ok else "fail", f"bar {r.volume_px}px, expected {expect:.0f}px"))
    return out
