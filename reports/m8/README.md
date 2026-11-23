# M8: request logging, quality checks, drift detection, alerts, report

Question: can a deployed SiteSight notice that its inputs (or outputs) no longer look like what it
was built on, without labels, and how often does it cry wolf or stay silent?

## What was built (`sitesight/monitor/`)
- `request_log.py`: JSON-lines log of every API request (also rejected ones). It stores image
  quality statistics, box count, mean confidence, parse rate, review rate, per-resource tally,
  latency and status. It does **not** store the image, file name, captions or box coordinates.
  Enabled with `SITESIGHT_REQUEST_LOG=path` (or `create_app(..., request_log=...)`).
- `quality.py`: per-image width/height, brightness, contrast, sharpness (variance of the Laplacian),
  clipped-shadow and clipped-highlight shares, on the image reduced to 512 px; a `QualityReference`
  that flags an image outside the reference's 1st-99th percentile band.
- `drift.py`: per-feature two-sample Kolmogorov-Smirnov (Bonferroni) plus a multivariate Gaussian
  MMD permutation test. Each family gets half of alpha = 0.01. Batches under 20 are not tested
  (`too_small`), never guessed.
- `report.py`, `cli.py`: self-contained HTML report; `fit` / `check` commands whose exit status is
  0 no drift, 1 alert, 2 too small to decide. `check --log` also tests the model's outputs
  (mean confidence, parse rate, review rate) against a reference log.
- `perturb.py`, `demo.py`: the controlled degradations and the experiment below.

## Experiment (`demo.json`, real images, `python -m sitesight.monitor.demo`)
Protocol and settings were fixed before the run (alpha 0.01; the ten degradations and batch sizes
20/40/80 listed in `perturb.py` / `demo.py`); every setting is reported, including misses.
The reference is a random half A of the 1,002 training images (501); clean pool B is the other
half. 100 random batches per cell. Test split never read.

**Same-site control (false alarms).** Batches drawn from B against A: alert rate 1% / 0% / 0% at
n = 20 / 40 / 80 (budget 1%). 3.4% of clean images are flagged individually by the band rule.

**Detection of degraded batches** (share of batches that alert; whole batch degraded):

| degradation | n=20 | n=40 | n=80 | 25% of batch degraded, n=20 / 40 / 80 |
|---|---|---|---|---|
| blur radius 0.5 | 0.51 | 0.94 | 1.00 | 0.00 / 0.01 / 0.00 |
| blur radius 2 | 1.00 | 1.00 | 1.00 | 0.07 / 0.18 / 1.00 |
| blur radius 5 | 1.00 | 1.00 | 1.00 | 0.07 / 0.23 / 1.00 |
| darker x0.9 | 1.00 | 1.00 | 1.00 | 0.03 / 0.08 / 0.80 |
| darker x0.5 | 1.00 | 1.00 | 1.00 | 0.24 / 0.84 / 1.00 |
| brighter x1.4 | 1.00 | 1.00 | 1.00 | 0.12 / 0.28 / 0.83 |
| rescaled x0.5 | 1.00 | 1.00 | 1.00 | 0.56 / 0.98 / 1.00 |
| rescaled x0.5, restored to size | 1.00 | 1.00 | 1.00 | 0.01 / 0.09 / 0.59 |
| JPEG quality 10 | 0.23 | 0.49 | 0.97 | 0.01 / 0.00 / 0.00 |
| Gaussian noise sigma 10 | 0.10 | 0.21 | 0.72 | 0.02 / 0.01 / 0.01 |

Per-image flag rate (share of degraded images the band rule flags): blur r2/r5 100%, rescaled 100%,
darker x0.5 98%, rescaled-restored 42%, brighter x1.4 23%, blur r0.5 9%, darker x0.9 9%, JPEG q10 8%,
noise 3%.

**New sites (validation vs all training images).** Alert rate 16% / 50% / 95% at n = 20 / 40 / 80.

## What this says
- Whole-batch degradations are caught reliably from n = 40 in most cases, with a false-alarm rate
  at or under the 1% budget for images from already-seen sites.
- **A partial fault is the weak spot.** With 25% of a batch degraded, n = 20 to 40 mostly misses it;
  n = 80 catches the strong ones (blur, rescale, darkening, brightening). Alerts need volume.
  The per-image band flags are the faster signal for blur, rescale and strong darkening.
- **Not detected at all by these features**: mild JPEG and mild noise barely move brightness,
  contrast or the 512-px Laplacian (JPEG q10 and sigma 10 noise reach 97% / 72% only at n = 80 for
  a fully degraded batch, and almost never when partial).
- **The validation split alerts against training** at larger batch sizes. That is not a false alarm
  to tune away: the grouped split puts new sites in validation and the sites differ in brightness
  (mean 0.589 vs 0.543 on 0..1). A new site will look like drift to this monitor, by design;
  the report says what differs and a person decides. The "clean held-out batch does not alert" goal
  therefore holds for the same sites only, not for new ones.

## Caveats
- Features are cheap image statistics, not learned embeddings. An embedding-based drift test
  (e.g. DINOv2 features, which M2 already used) would see semantic shift these cannot; it needs a
  GPU/Colab run for the degraded copies and was **not done**.
- Output drift (confidence, parse rate, review rate) is implemented and tested on synthetic logs
  and a stubbed API, **not** demonstrated on real model outputs: no real prediction log exists
  locally (the M7 predictions stayed in Colab).
- All images in this dataset are 960x540, so the width/height features only react to a rescale.
- The false-alarm control uses random halves of the training split, which share sites with each
  other. It describes traffic from known sites only.
- Batches of 100 draws per cell: a rate of 0.01 means 1 batch in 100; error bars are about
  +-0.1 at the middle of the range, +-0.01 to 0.02 near 0 and 1. alpha is a union bound, not a tuned
  threshold. Thresholds for the per-image band are the reference's own 1st/99th percentiles.
- Alerts are a statistical signal that something changed, not that quality got worse.

## Files
`demo.json` (all numbers above), `example_clean.html`, `example_blur.html`, `example_dark.html` (reports
from `monitor.cli check` on a clean, a blur-radius-2 and a x0.5-darkened batch of 40 same-site
images against a 501-image reference; exit status 0, 1, 1).
