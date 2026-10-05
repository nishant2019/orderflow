# orderflow

Reads a **GoCharting footprint (order-flow) screenshot** and returns structured data — no LLM, no
external service. Classical computer vision plus small nearest-neighbour glyph classifiers, all
trained on the sample screenshots in `tests/data`.

```
python -m orderflow screenshot.png > out.json                # parsed data (pretty; --compact for one line)
python -m orderflow screenshot.png --analyze > out.json      # {"parsed": ..., "analysis": ...}
```
```python
from orderflow.assemble import parse_screenshot
doc = parse_screenshot("screenshot.png")

from orderflow.analytics import analyze, Thresholds
result = analyze(doc, Thresholds(absorption_min_share=0.3))   # thresholds are tunable
```

## What is extracted

| Part | Output |
|---|---|
| Summary table | per bar: volume, delta, cumulative delta |
| Footprint cells | per bar and price: bid, ask, delta, sell/buy imbalance (red bid / blue ask on orange), POC flag, `...` hidden cells |
| Time axis | per bar time (`HH:MM`), session starts, inferred times for date-labelled bars |
| Price axis | pixel → price fit, rows, price step per row (e.g. 1, 0.55, 5) |
| Right-hand profile | per-price aggregate delta |
| Overlays | current-price line |

All values are **as displayed** (`3.5K` means 3500 ± 50). Each bar and profile row carries its own
`checks` and a `valid` flag.

## Analytics (`orderflow.analytics`)

Rule-based, runs on the parsed JSON (not pixels), and only on bars that passed validation; the rest
are listed under `excluded` with a reason. Every signal carries its numeric evidence, a strength
(0-1) and a confidence (reduced when the chart hid some cells).

| Signal | Rule (defaults in `Thresholds`) |
|---|---|
| `stacked_imbalance` | >= 3 consecutive price rows with the same buy (blue ask) or sell (red bid) imbalance flag |
| `absorption` | at a bar's high: the top 2 rows hold >= 25% of the bar's volume (and clearly more than an even spread would) with delta >= +15% of that volume (aggressive buying that stalled) -> bearish; mirror at the low -> bullish. `confirmation`: `held` if the next bar did not extend past the extreme, `broken` if it did |
| `exhaustion` | volume thinning over the last 3 rows into the extreme, extreme row <= 20% of the POC row |
| `unfinished_extreme` / `finished_extreme` | both bid and ask traded at the extreme row (likely revisited) / one side is zero |
| `poc` | POC location in the bar's range (upper / middle / lower third) |
| `levels` | POCs, stacked-imbalance zones, absorption and unfinished extremes merged within 1.5 rows; `strength`, `retests` by later bars, `support`/`resistance` vs the current price when one is drawn |

These are heuristics, not trading advice. Open and close are not extracted yet, so "extremes" are the
highest and lowest rows that traded and bars touching the edge of the visible price range are noted
(their true high/low may be off-screen).

## Validation (why you can trust a bar)

Every bar is cross-checked against numbers the chart prints independently:
cell Σ(bid+ask) ≈ table volume; cell Σ(ask−bid) ≈ table delta; the POC cell has the largest total;
cum[c] = cum[c−1] + delta[c] (or a session reset); each profile row = Σ(ask−bid) of that row's cells,
its sign matches the bar colour and its bar length is proportional to |value|. Tolerances come from the
displayed precision. A failed check means *this bar should not be trusted*, not which number is wrong.

`summary` lists `flagged_bars` (a check failed) and `skipped_bars` (could not be checked: clipped by
the image edge, or no cells in the visible price range).

## Known limits

* Only GoCharting, this layout. Dimensions may vary per image (calibrated each time); a new font size or
  theme needs new training examples.
* The live bar can sit under the translucent profile overlay; its cells are then unreadable and flagged.
* Text clipped by an overlay (`78K X 4…`) is read as written and caught by the cell-sum checks.
* Cells outside the visible price range are not in the screenshot.
* Not yet extracted: per-bar OHLC (thin candle), the cumulative-delta candle pane.
* Analytics thresholds are untested against real outcomes; they are starting points to tune.

## Training data

Glyph classifiers live in `src/orderflow/data/*.npz` and are rebuilt from hand-transcribed ground truth:

| Truth file (`tests/data`) | Builder (`scripts/`) | Classifier |
|---|---|---|
| `axis_labels.json` | `build_axis_templates.py` | price-axis labels |
| `cell_truth.json` | `build_cell_glyphs.py` | footprint cell text |
| `table_truth.json` | `build_table_glyphs.py` | summary table |
| `profile_truth.json` | `build_profile_glyphs.py` | profile values |
| `time_truth.json` (+ `axis_labels.json`) | `build_label_glyphs.py` | time labels |

Add a screenshot as `tests/data/shotN.png`, transcribe it into the truth files, re-run the builders.
`cell_holdout.json` (shot 1) is deliberately **not** used for training: it is the out-of-sample check.

```
pip install -e '.[dev]' && pytest
```
