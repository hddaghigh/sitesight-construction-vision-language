# SiteSight: Vision-Language Grounding and Captioning for Construction-Site Drone Imagery

Grounded construction-site understanding from drone images: a described region for every box, a box
for every phrase, and a structured site summary, evaluated honestly, served on CPU and monitored.
Built end to end on the public Bang & Kim UAV dataset (1,431 images, 8,428 region captions).

**Status: all milestones M1 to M10 are built, tested and pushed.** The one human-dependent piece is
still open: the 100-image label audit needs my own labelling ([AUDIT.md](AUDIT.md)), so label
accuracy is **unmeasured**. Read the [technical report](TECHNICAL_REPORT.md) for the whole story,
[DATASHEET.md](DATASHEET.md) for what the data is and its limits, [PLAN.md](PLAN.md) for scope and
decisions, [DATA_LICENSES.md](DATA_LICENSES.md) for licences and citation.

## Headline results
Every number below is read from a committed file (last column); the test
[`tests/test_headline_numbers.py`](tests/test_headline_numbers.py) fails if this table and those files
disagree. All are single runs on one dataset (seeds in the configs); small differences are within noise.

| Claim | Result | Source (committed evidence) |
|---|---|---|
| Caption-field parser coverage | 99.63% of non-empty captions (8,396 of 8,427) | `reports/m2/coverage_report.json` |
| Leakage a random split would have had | 76.6% of test images have a near-duplicate in train (mean of 20 seeds); the grouped split has 0% | `reports/m2/curation_report.json` (`configs/data.toml`, seed 0) |
| Zero-shot grounding, val, mAP@0.5 (`given_phrases`) | OWLv2-base 24.1, Florence-2-large 21.5, Florence-2-base 16.3 | `reports/m5/val_compare.json`, `reports/m4/zero_shot_val.json` |
| Fine-tuned grounding, test (run once), mAP@0.5 | Florence-2-base + LoRA 34.0 vs OWLv2 30.2 vs zero-shot Florence-2-base 18.8 | `reports/m5/test_final.json` (`configs/train.toml`, seed 0) |
| Fine-tuned region captions, test | resource 71.8%, colour 79.3%, count 77.6% (zero-shot: 14.8%, 22.5%, 0.1%) | `reports/m5/test_final.json` |
| Calibration, test | fine-tuning made it worse: ECE 0.459 (OWLv2: 0.048) | `reports/m5/test_final.json` |
| Active learning (3 seeds) | images to reach 90% of full-data score: random 283.0, uncertainty 238.2, uncertainty + diversity 238.5 | `reports/m6/README.md` (`configs/active.toml`; raw per-round rows were not committed) |
| CPU serving, 300 val regions, cold, 2-vCPU Colab | p50 85.1 s PyTorch, 12.1 s ONNX fp32, 9.5 s int8; 1,242 MB vs 574 MB; int8 gives identical captions for 76% | `reports/m7/bench.json` (seed 0) |
| Monitoring, same-site batches of 40 | 0% false alarms; whole-batch blur (radius 2 and 5), darkening and rescaling detected in 100% of batches; 25%-degraded batches need about 80 images | `reports/m8/demo.json` (seed 0) |
| RGB-D odometry, TUM fr1/desk | ATE 8.9 cm (frame-to-frame), 9.1 cm (keyframe) vs 87 cm for a camera that never moved; evo agrees | `reports/m9/metrics.json` |
| Label accuracy of the dataset | **not measured** (audit pending) | `reports/m3/audit_sample.json` fixes the 100 images |

## What is in the repository
| Milestone | What | Where |
|---|---|---|
| M1 | VIA parser, validator, COCO-style schema, resumable verified download | `sitesight/data/` |
| M2 | Caption-field parser, near-duplicate groups, leakage-free split, datasheet | `sitesight/data/`, [`reports/m2/`](reports/m2/) |
| M3 | PyQt6 review tool and audit analysis code | `sitesight/viewer/`, `sitesight/data/audit.py` |
| M4 | Evaluation harness (AP, recall, ECE, BLEU/CIDEr/METEOR, field accuracy), zero-shot baselines | `sitesight/eval/`, `sitesight/baselines/`, [`reports/m4/`](reports/m4/README.md) |
| M5 | LoRA fine-tuning of Florence-2-base, failure classification | `sitesight/train/`, [`reports/m5/`](reports/m5/README.md) |
| M6 | Active-learning simulation | `sitesight/active/`, [`reports/m6/`](reports/m6/README.md) |
| M7 | ONNX fp32/int8 export, FastAPI service, batch CLI, benchmark, Dockerfile | `sitesight/serve/`, [`reports/m7/`](reports/m7/README.md) |
| M8 | Request log, quality checks, KS + MMD drift tests, HTML report | `sitesight/monitor/`, [`reports/m8/`](reports/m8/README.md) |
| M9 | RGB-D visual odometry and ATE/RPE (optional milestone) | `sitesight/geometry/`, [`reports/m9/`](reports/m9/README.md) |
| M10 | CI, this README, the technical report | `.github/workflows/ci.yml`, [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) |

![Review tool showing the original ground-truth boxes and region table for one image](docs/images/viewer_ground_truth.png)

*The review tool on a real image (`3.jpg`) with its original labels. Image and annotations: Bang & Kim, Mendeley Data, CC BY 4.0 ([DATA_LICENSES.md](DATA_LICENSES.md)).*

## Quickstart
```powershell
uv sync --group viewer                       # dev tools + PyQt6 (no torch)
uv run python scripts/download_data.py --data-dir sitesight_data
make data fields                             # parse + caption fields (embeddings/curation: see Makefile)
uv run pytest                                # default suite; the `viewer` and `embed` tests are opt-in
uv run pytest -m viewer                      # GUI tests (offscreen Qt)
make view                                    # open the review tool
```
Serve a model exported by M7 (weights are not in the image or the repository):
```powershell
uv sync --group serve
$env:SITESIGHT_MODEL_DIR = 'export/int8'; $env:SITESIGHT_REQUEST_LOG = 'requests.jsonl'
uv run uvicorn --factory sitesight.serve.app:build_app --port 8000
uv run python -m sitesight.monitor.cli check --reference ref.json --log requests.jsonl --report report.html
```
Heavy steps (training, export, benchmarks, odometry on TUM) run in Colab from token-free notebooks
that embed the code: `uv run python scripts/build_colab_notebook.py --kind m5|m6|m7|m9 --out colab/x.ipynb`
(`m7` also needs `--adapter-dir`). `notebooks/01_data_splits.ipynb` (M2) is written but has not been
run in Colab.
