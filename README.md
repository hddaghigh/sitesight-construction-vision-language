# SiteSight

Grounded construction-site understanding from drone images: region descriptions, a bounding box for each described object, and a structured site summary.

**Status: milestone M3 (data curation, review tool and audit protocol done).** No model has been run yet, so there are no model results to report. See [PLAN.md](PLAN.md) for scope, model choice and milestones, [DATASHEET.md](DATASHEET.md) for what the data is and its limits, [AUDIT.md](AUDIT.md) for the label-quality audit, and [DATA_LICENSES.md](DATA_LICENSES.md) for licences and citation.

## What exists so far
| Milestone | What | Where |
|---|---|---|
| M1 | VIA parser, validator, COCO-style schema, resumable verified download | `sitesight/data/` |
| M2 | Caption-field parser (99.63% coverage), near-duplicate groups, leakage-free split, datasheet | `sitesight/data/`, [`reports/m2/`](reports/m2/) |
| M3 | PyQt6 review tool (GT / prediction boxes, confidence filter, side-by-side comparison, label editing with undo, schema export) and the audit analysis code | `sitesight/viewer/`, `sitesight/data/audit.py` |

The audit itself needs a person: `reports/m3/audit_sample.json` fixes the 100 images, and the agreement numbers will exist only after they are labelled (see [AUDIT.md](AUDIT.md)).

## Review tool
![Review tool showing the original ground-truth boxes and region table for one image](docs/images/viewer_ground_truth.png)

*Screenshot of the tool on a real image (`3.jpg`) with its original labels. Image and annotations: Bang & Kim, Mendeley Data, CC BY 4.0 ([DATA_LICENSES.md](DATA_LICENSES.md)).*

Each panel shows one source: the original ground truth, your edit layer, or a model's predictions (`--predictions a.json b.json`, format in `sitesight/data/predictions.py`). Tick *Compare* to put two sources side by side with synchronised zoom. Boxes are drawn, moved and resized with the mouse; captions autocomplete from the dataset's phrases and are re-parsed into resource, count, colour and state on every edit; everything is exported back in the SiteSight schema and validated before writing.

## Quickstart
```powershell
uv sync --group viewer                       # dev tools + PyQt6 (no torch)
uv run python scripts/download_data.py --data-dir sitesight_data
make data fields                             # parse + caption fields   (embeddings/curation: see Makefile)
uv run pytest                                # 180 tests, about 5 s
uv run pytest -m viewer                      # GUI tests (offscreen Qt)
make view                                    # open the review tool
```
Heavy steps (embeddings on GPU, model baselines, fine-tuning) run in Colab: `notebooks/01_data_splits.ipynb` (written, not yet run).
