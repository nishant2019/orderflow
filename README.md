# orderflow

Reads a **GoCharting footprint (order-flow) screenshot** and returns structured data — no LLM, no
external service. Classical computer vision plus small nearest-neighbour glyph classifiers, trained on
the sample screenshots in `tests/data`, and a set of cross-checks that tell you which numbers to trust.

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

## What is extracted

| Part | Output |
|---|---|
| Summary table | per bar: volume, delta, cumulative delta (`1.12M`, `-4.81K`, `831`) |
| Footprint cells | per bar and price: bid, ask, delta, sell/buy imbalance, POC flag |
| Candles | per bar: open, high, low, close, direction; clipped / hidden-wick flags |
| Time axis | per bar time, session starts (inferred time for the date-labelled bar) |
| Price axis | pixel→price fit, row pitch, **price step per row** (0.9, 0.55, 0.7, 1, 2, 5, 15 …) |
| Profile (split) | per price: delta, volume, value-area membership, peak row; POC, value-area high/low |
| `price.poc_line` | the solid pink horizontal line (the profile POC, see below) |
| `price.current` | the red *dashed* line with a price tag on the right axis (current price), when the chart draws it |

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

Leave-one-image-out on the 9 split screenshots (retrained without the held-out image and its hand
transcriptions; `scripts/eval_split_holdout.py`): **106 / 109 bar columns and 113 / 120 profile rows pass every
check, 1 of 502 cells unparseable** (then solved from totals). Six of the seven failing profile rows are the `M`
suffix in the one image that has it. With all images in training every bar, row and candle validates.

## Known limits

* GoCharting only, these two layouts. A new font size, theme or settings combination needs new examples;
  a number format never seen in training (e.g. `M` before it was added) will be misread — the checks flag it.
* Overlays not handled: the amber **"VPOC <day>" line** (seen once, as an image I could not open as a file).
* A clipped first bar (cut by the image edge) is not read; its cells are solved from the profile rows when only that
  column is unknown on a row.
* The colour-coded value area is not always contiguous (the platform seems to colour at a finer resolution).
* Gap rows (no trades) are not drawn; the platform's imbalance rule next to a gap is not visible, so those pairs
  are not checked.
* Candle prices are measured in pixels (±1 px); a wick hidden under the POC rectangle is estimated to its far edge.
* Not yet extracted: the cumulative-delta candle pane.

## Training data

Glyph classifiers live in `src/orderflow/data/*.npz`.

| Truth (`tests/data`) | Builder (`scripts/`) | Classifier |
|---|---|---|
| `axis_labels.json` | `build_axis_templates.py` | price-axis labels |
| `cell_truth.json` | `build_cell_glyphs.py` | single-layout cell text |
| `table_truth.json`, `profile_truth.json` | `build_table_glyphs.py`, `build_profile_glyphs.py` | table, single-layout profile |
| `time_truth.json` (+ axis labels) | `build_label_glyphs.py` | time labels |
| `split1..9.png` (+ `profile2_truth.json`, split rows of `table_truth.json` for `M`) | **`train_split.py`** | split cells / table / profile |

The split classifiers are **self-trained**: a glyph becomes a label only if its cell, column and profile row
agree with the chart's checksums (`orderflow.selftrain`), so thousands of glyphs are labelled without hand
transcription. `cell_holdout.json` (shot 1, single layout) is kept out of training. The hand-read split cells in
`tests/test_split.py` are regression checks of the reading, **not** held out (those images are in the self-training
set): the independent measure for the split layout is the leave-one-image-out run above.

```
pip install -e '.[dev]' && pytest
```
