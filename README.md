# orderflow

Reads a **GoCharting footprint (order-flow) screenshot** and returns structured data — no LLM, no
external service. Classical computer vision plus small nearest-neighbour glyph classifiers, all
trained on the sample screenshots in `tests/data`.

```
python -m orderflow screenshot.png > out.json        # pretty JSON; add --compact for one line
```
```python
from orderflow.assemble import parse_screenshot
doc = parse_screenshot("screenshot.png")
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
