# SiteSight

Grounded construction-site understanding from drone images: region descriptions, a bounding box for each described object, and a structured site summary.

**Status: milestone M1 (data parsing, validation, schema).** No models have been run yet, so there are no results to report. See [PLAN.md](PLAN.md) for scope, model choice and milestones, and [DATA_LICENSES.md](DATA_LICENSES.md) for the dataset licence and citation.

## Quickstart (M1)

```powershell
uv sync                                                          # dev tools only; no torch
uv run python scripts/download_data.py --data-dir sitesight_data
uv run python -m sitesight.data.via_parse `
  --via sitesight_data/extracted/via_region_data_final.json `
  --images sitesight_data/extracted/UAVdata `
  --out sitesight_data/derived/annotations.json `
  --report sitesight_data/derived/validation_report.json
uv run pytest
```
