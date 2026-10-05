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
