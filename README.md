# orderflow

Reads a **GoCharting footprint (order-flow) screenshot** and returns structured data — no LLM, no
external service. Classical computer vision plus small nearest-neighbour glyph classifiers, trained on
the sample screenshots in `tests/data`, `shots/`, `shots2/` and `shots3/`, and a set of cross-checks that tell you
which numbers to trust.

```
python -m orderflow screenshot.png > out.json                 # parsed data (layout auto-detected)
python -m orderflow screenshot.png --analyze > out.json       # {"parsed": ..., "analysis": ...}
python -m orderflow screenshot.png --layout split|single      # force a layout; --compact for one line
```
```python
from orderflow.assemble import parse_screenshot
from orderflow.analytics import analyze, Thresholds
doc = parse_screenshot("screenshot.png")
result = analyze(doc, Thresholds(absorption_min_share=0.3))   # thresholds are tunable
```

## Two chart layouts

| | **split** (the main source) | single (older samples) |
|---|---|---|
| Cells | two boxes per bar and price row: **bid left, ask right**, one number each | one column, text `bid X ask` |
| Imbalance | orange box on that side; red text = sell (bid), blue text = buy (ask) | orange bar, red/blue text |
| POC | **black rectangle** around the pair | orange-red ring |
| Profile | centre axis: **delta on the left, total volume on the right**, value-area colours | delta bars at the right edge |

The layout is detected from the POC rectangle's colour (`orderflow.layout`). Chart settings the split
screenshots were taken with are in `docs/CHART_SETTINGS.md` (imbalance = Ratio, 300 %).

## Chart variants that are handled

Screenshots differ in more than the price: the parser measures what it needs from each image instead of assuming it.

| Variant | Handling |
|---|---|
| Right price-axis font: 9 px or 7 px per character, with or without thousands separators (`1,844.00`) | two template sets (`axis_templates*.npz`); the axis is fitted on the largest set of labels that lie on one line, so misread or foreign labels cannot spoil it |
| Summary table in a large (11 px) or small (9 px) monospace font | text is cut into character slots using the font's measured advance, so thin white strokes that break apart cannot split a glyph |
| Summary table with 3 rows, or 4 with **Delta %** | row count detected from the table height |
| Footprint text from 6 px tall (rows 14 px apart) to 9 px tall (rows 35 px apart) | row grid fitted on glyph baselines, pitch = most common spacing (stray text such as an alert badge is ignored) |
| 3, 8 or 13 bars (early-session charts have few, very wide columns) | column pitch fitted per image |
| Imbalance drawn (orange) or switched off | detected from the orange pixel count; when off the flags are computed from the 300 % rule (`imbalance.source = "computed"`) |
| Current-price line: red or teal, dashed | colour list in `overlays.py` |
| Cumulative-delta pane | scale fitted through the amber zero line and the table's `Cum` row |

**Recommended export:** imbalance display **off**. It removes the orange fills that make the text harder to read and
the only check that depends on them; the imbalance flags are still reported (computed).

## What is extracted

| Part | Output |
|---|---|
| Summary table | per bar: volume, delta, cumulative delta (`1.12M`, `-4.81K`, `831`), and when the chart shows it `delta_pct` (see below) |
| Footprint cells | per bar and price: bid, ask, delta, sell/buy imbalance, POC flag |
| Candles | per bar: open, high, low, close, direction; clipped / hidden-wick flags |
| Time axis | per bar time, session starts (inferred time for the date-labelled bar) |
| Price axis | pixel→price fit, row pitch, **price step per row** (0.9, 0.55, 0.7, 1, 2, 5, 15 …) |
| Profile (split) | per price: delta, volume, value-area membership, peak row; POC, value-area high/low |
| `price.poc_line` | the solid pink horizontal line (the profile POC, see below) |
| `price.current` | the red or teal *dashed* line with a price tag on the right axis (current price), when the chart draws it |
| `imbalance` | `shown` (drawn by the platform or not), `source` (`drawn` / `computed`), `ratio` |
| `bars[].platform_divergence` | the platform's own delta-divergence marker (pale `red` / `green` vertical band) or `null` |
| `bars[].cum_delta_candle`, `cum_delta_pane` | the cumulative-delta candle pane under the price rows (open/high/low/close in delta units) |

All values are **as displayed** (`3.5K` means 3500 ± 50). Each bar and profile row carries its own `checks`
and a `valid` flag; `summary` lists flagged and skipped bars.

### The solid pink line is the profile POC, not the current price
In all 9 split charts that have both, the solid pink line (labelled with its price at the left edge) sits on the
profile's highest-volume row (within 0.03 of a row) and is often far from the last close (up to ~9 rows); the
user confirmed it is the POC line. It is reported as `price.poc_line`. The *current price* is the red **dashed**
line with a tag on the right axis (`price.current`, drawn in some chart settings); the last candle's close is
`price.last_close`.

## Validation: why you can trust a number

Every number on the chart is cross-checked against others the chart prints independently. Tolerances come
from the displayed precision (`3.5K` = ±50).

* cells of a bar sum to the **table** volume and delta; the table's cumulative delta chains bar to bar
  (or resets at a session start);
* cells of a price row sum to the **profile** row's volume and delta; bar lengths are proportional;
* the POC box is the largest cell of its bar;
* drawn imbalance flags follow the chart's own rule: **sell** when `bid[r] ≥ 3 × ask[row above]`, **buy** when
  `ask[r] ≥ 3 × bid[row below]` (every one of 169 drawn flags in the samples satisfies it);
* each candle chains from the previous close (not across a session) and covers the rows that traded.

When a check fails the data is flagged, never silently accepted. Two repairs use the same checks:

* **covered cell** (e.g. the pink price label printed over the first column): its bid and ask are solved from the
  profile row / table totals and marked `inferred` with their uncertainty;
* **single-glyph misread** (`48K` read as `43K`): corrected only if exactly one alternative reading satisfies the row
  total, the column total and the imbalance rule together; marked with `corrected_from`.

## Analytics (`orderflow.analytics`)

Rule-based, on the parsed JSON, only on validated bars. Every signal has numeric evidence, a strength and a
confidence. Per bar: `stacked_imbalance`, `absorption` (with next-bar confirmation), `exhaustion`,
`unfinished/finished_extreme`, `rejection` (wick), `delta_divergence`, `poc`. Cross-bar: support/resistance
`levels` clustered from bar POCs, imbalances, absorption **and the volume profile**.

Volume profile (`analysis.profile`): POC (share of volume, delta at POC), value area (colour-coded by the chart),
last close vs value area, high/low volume nodes, thin tails, rows of one-sided aggression, shape (P / b / D),
delta above vs below the POC. Defaults are in `Thresholds`; they are untested against real price outcomes.

## How well does it work?

**Split screenshots in `tests/data`** (13 images) - leave-one-image-out (retrained without the held-out image and its
hand transcriptions; `scripts/eval_split_holdout.py`; measured **before** the repair steps): **159 / 164 bar columns
and 171 / 180 profile rows pass every check, 16 of 766 cells unparseable** (those are then solved from totals or
flagged). The weak spots were the `M` suffix and the paler or cut-off variants when never seen.

**Uploaded shots** (193 images, bars that fail at least one cross-check; **in-sample** - the classifiers were
self-trained on these same images, so the rate on a new chart will be somewhat worse):

| Folder | Images | Bars | Bars failing a check | Note |
|---|---|---|---|---|
| `shots/` | 51 | 663 (51 clipped first bars, skipped) | 9 | 13 bars per chart, imbalance off |
| `shots2/` | 71 | 213 | 23 | early-session charts, 3 wide bars, imbalance drawn: 19 of these 23 bars fail the imbalance check |
| `shots3/` | 71 | 568 | 6 | 8 bars, imbalance off, 4-row table (Delta %) |

Every one of the 193 images parses (no exceptions) and every row step snaps to a clean value within 0.4 %.
The progression while building this (failing bars over the three folders): 104 -> 90 (row grid fitted on baselines)
-> 57 (touching narrow `1`s split) -> 51 (row pitch robust to stray text) -> 38 (retrained). Most of the remaining
failures are single bars with a volume/delta mismatch, at most 3 per image.

`tests/test_shots.py` pins a fixed sample of 16 of these screenshots (bar counts, row step, optional parts, a floor
on valid bars). The summary-table text of six uploaded images (all rows) and the Delta % row of five more was read
by hand (`table_truth.json`) and is checked exactly in `tests/test_validate.py`; summary-table glyph accuracy on a
held-out image was 100 % for each of the six fully labelled images.

## The optional Delta % row (shots3)

Some charts add a 4th table row, **Delta %** = delta / volume x 100 (e.g. 448.56K / 5.04M = 8.9 %). It is detected
from the table height (`TableGeometry.n_rows`), read like the other rows, and given the delta's sign (the chart shows
the sign only by the cell colour: red = negative). It appears as `bars[].delta_pct` (`null` when the chart has no such
row) and is checked against the table's own volume and delta (`pct` check, within the rounding of the three numbers) -
an independent test of two table numbers.

## Known limits

* GoCharting only, these two layouts. A new font size, theme or settings combination needs new examples;
  a number format never seen in training (e.g. `M` before it was added) will be misread — the checks flag it.
* A clipped first bar (cut by the image edge) is not read; its cells are solved from the profile rows when only that
  column is unknown on a row.
* The colour-coded value area is not always contiguous (the platform seems to colour at a finer resolution).
* Gap rows (no trades) are not drawn; the platform's imbalance rule next to a gap is not visible, so those pairs
  are not checked.
* Candle prices are measured in pixels (±1 px); a wick hidden under the POC rectangle is estimated to its far edge.
* The cumulative-delta pane is read for the split layout (`bars[].cum_delta_candle`, `cum_delta_pane`): its scale is
  fitted through the amber zero line and the table's `Cum` row, and every candle's body ends must land on the
  table's open/close (within 2.5 px) or the bar is listed in `cum_delta_pane.invalid_bars`. Wicks (intrabar
  extremes of cumulative delta) are measured in pixels (about +-1 px, i.e. a few hundred to a few thousand units).
* The amber dashed line in every screenshot is this pane's **zero line**. A separate amber **"VPOC <day>" line** (seen once,
  in an image I could not open as a file) is still not handled: it needs a screenshot that shows it.
* Charts with the imbalance display **off** get *computed* flags (`imbalance.source = "computed"`);
  they follow the 300 % ratio rule, but cannot be cross-checked against drawn orange boxes.
* The uploaded-shots results above are in-sample. A new font size or layout needs examples: add the screenshot,
  hand-read its table row if the glyphs are new, and run `scripts/train_split.py` (about 15 minutes for 3 rounds).
* `platform_divergence` reports the platform's own pale red / green vertical markers per bar; it is `null` on charts
  that do not show them.

## Training data

Glyph classifiers live in `src/orderflow/data/*.npz`.

| Truth (`tests/data`) | Builder (`scripts/`) | Classifier |
|---|---|---|
| `axis_labels.json` | `build_axis_templates.py` | price-axis labels |
| `cell_truth.json` | `build_cell_glyphs.py` | single-layout cell text |
| `table_truth.json`, `profile_truth.json` | `build_table_glyphs.py`, `build_profile_glyphs.py` | table, single-layout profile |
| `time_truth.json` (+ axis labels) | `build_label_glyphs.py` | time labels |
| `axis_labels_small.json` (hand-read axis labels of `shots*/`) | `build_axis_templates_small.py` | price-axis labels, 7 px font |
| `table_truth.json` (+ `delta_pct` rows of five `shots3/` images) | `build_table_glyphs.py` | summary-table glyphs (incl. `M`, `%`) |
| `split1..13.png`, `shots/`, `shots2/`, `shots3/` (+ `profile2_truth.json`) | **`train_split.py`** | split cells / table / profile |

The split classifiers are **self-trained**: a glyph becomes a label only if its cell, column and profile row
agree with the chart's checksums (`orderflow.selftrain`), so thousands of glyphs are labelled without hand
transcription. `cell_holdout.json` (shot 1, single layout) is kept out of training. The hand-read split cells in
`tests/test_split.py` are regression checks of the reading, **not** held out (those images are in the self-training
set): the independent measure for the split layout is the leave-one-image-out run above.

```
pip install -e '.[dev]' && pytest          # about 12 minutes; tests/test_shots.py alone takes 2
```
