# CONTRIBUTING.md: SiteSight conventions and current status

Read [PLAN.md](PLAN.md) first (scope, models, milestones). This file holds the working rules.

## Rules
- **Never invent numbers.** Every number in docs comes from a committed script + config + seed. If a result is poor, the README says so.
- **No placeholder code or fake data.** Tests use real files (tiny JPEGs built with Pillow, a real local HTTP server), not mocks of our own code.
- **Prove tests can fail**: mutate a constant, confirm a test catches it (done for the parser in M1).
- **Findings are flagged, not dropped.** Clipped / empty / duplicate annotations stay in the file with `flags`; evaluation code decides what to exclude and says so.
- **Explain the maths** (losses, metrics, drift statistics) in docstrings.
- **Disk rule (laptop):** check free space before any download/install; stop if it would leave < 3 GB free. Data, caches and the managed Python live on F: outside the repo.
- **Colab runs the heavy steps** (torch, models, training, benchmarks). Laptop: code, tests, viewer. Notebooks only call CLI entry points.
- **Commits:** small, one per milestone step; commit only when asked.

## Environment (Windows laptop)
```powershell
$env:UV_CACHE_DIR = 'sitesight_data\.uv_cache'
$env:UV_PYTHON_INSTALL_DIR = 'sitesight_data\.uv_python'
$env:UV_PROJECT_ENVIRONMENT = '.venv'
py -3.12 -m uv sync --python 3.11      # `uv` itself was installed with pip --user into Python 3.12
.\.venv\Scripts\python.exe -m pytest   # or: make check
```
Quality gate before any commit: `ruff check`, `ruff format --check`, `mypy`, `pytest` all clean.

## Data (outside the repo: `sitesight_data\`)
`extracted/UAVdata/*.jpg` (1,431 images, all 960x540) and `extracted/via_region_data_final.json` (VIA 1.x export). `derived/annotations.json` and `derived/validation_report.json` are produced by `make parse` and are reproducible (byte-identical for identical input).

## Schema (COCO-style, see `sitesight/data/schema.py`)
`images[{id,file_name,width,height,group_id,split}]`, `annotations[{id,image_id,bbox[x,y,w,h],area,iscrowd,category_id,caption,caption_raw,fields,flags,bbox_raw,source}]`, `categories[]`. `category_id`/`fields` are null until M2; `group_id`/`split` null until M2. `bbox` is absolute pixels, top-left origin, always inside the image (clipped boxes keep the original in `bbox_raw`). Invariants are enforced by `sitesight/data/validate.py`.

## Decisions
- 2026-11-22: Florence-2-base primary (MIT, 463 MB); OWLv2 grounding baseline; Qwen2.5-VL-3B rejected (`qwen-research` licence); see PLAN.md section 3.
- 2026-11-22: runs in Colab, repo stays local; private GitHub repo `hddaghigh/sitesight`; Colab clones it with a read-only token in Colab Secrets.
- M1 parser is stdlib-only (JPEG size read from the SOF marker) so it runs anywhere without dependencies.

## Real-data findings that shape later milestones
8,428 regions (page says 8,601); 795 raw / 746 trimmed / 735 normalised distinct captions; 6 clipped boxes; 12 duplicates; 1 empty caption; 19 images without regions; 9 regions with two captions glued together (e.g. "...on soilA black dust protection..."); 25% of boxes < 1% of image area. The M2 caption-field parser must report these glued captions as unparsed rather than guess.

## Status
- **M1 done** (parser, validator, schema, downloader, tests, CI). Next: **M2** (dedup, grouped split, caption-field parser, datasheet); embeddings/splits run in Colab.
