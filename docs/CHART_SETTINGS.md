# GoCharting settings the "split" screenshots were taken with

(Read from the platform's settings panel, supplied by the user.)

| Setting | Value | What it means for parsing |
|---|---|---|
| Left cluster | Text format **Sell Volume**, background on, gradient light pink -> red, black text | left box of each pair = bid (sell) volume; fill intensity grows with volume |
| Right cluster | Text format **Buy Volume**, background on, gradient light grey-green -> green, black text | right box = ask (buy) volume |
| Left / right profile | off | no profile drawn inside the cluster |
| Imbalances | on; colours orange / blue / red | orange = imbalance box background, blue text = buy (ask) imbalance, red text = sell (bid) imbalance |
| Imbalance calculation | **Ratio**, **300 %** | a side is flagged when it is >= 3x the opposing side (checked diagonally; see `validate_imbalances`) |
| Volume POC | black, on | black rectangle around the pair of boxes of the highest-volume row |

Box *fill intensity* encodes relative volume (light -> dark), so it can be used to sanity-check
a number the text reader is unsure about. A pale lavender/pink box with `0` marks a zero-volume side.

## Variant B (`split10.png`): paler fills, centred candles, a current-price line

Same layout and the same imbalance / POC settings (Ratio 300 %, black Volume POC), but:

| Difference | Handling |
|---|---|
| Gradient *light* ends are almost white (sell: very pale pink, buy: very pale green), so low-volume boxes are barely tinted | boxes are found by "not white", not by saturation; text is found by a black-hat filter, which does not depend on the fill |
| Imbalance swatches are orange / purple / red | the text colours are unchanged: pure blue (buy) and crimson (sell) on orange |
| Candles are drawn **in the centre gap** between the bid and ask boxes (12 px gap, boxes ~37 px wide) instead of at the column's left edge | the candle zone is measured per image from the exact candle colours; boxes start right at the column edge |
| A **red dashed horizontal line** with a price tag on the right axis | it is the *current price* (`price.current`); it is patched out of the image before the text is read, because it runs through the digits |
| The solid pink line labelled with a price (here `1395`) is still the **profile POC** | `price.poc_line` |
| The first bar is cut off by the left image edge | its cells are not read; they are solved from the profile row totals where possible |
