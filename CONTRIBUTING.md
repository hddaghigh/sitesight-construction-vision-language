# Contributing notes: SiteSight conventions and current status

Read [PLAN.md](PLAN.md) first (scope, models, milestones). This file holds the working rules.

## Rules
- **Never invent numbers.** Every number in docs comes from a committed script + config + seed. If a result is poor, the README says so.
- **No placeholder code or fake data.** Tests use real files (tiny JPEGs built with Pillow, a real local HTTP server), not mocks of our own code.
- **Prove tests can fail**: mutate a constant, confirm a test catches it (done for the parser in M1).
- **Findings are flagged, not dropped.** Clipped / empty / duplicate annotations stay in the file with `flags`; evaluation code decides what to exclude and says so.
- **Explain the maths** (losses, metrics, drift statistics) in docstrings.
- **Disk rule (laptop):** check free space before any download/install; stop if it would leave < 3 GB free. Data, caches and the managed Python live on F: outside the repo.
- **Colab runs the heavy steps** (torch, models, training, benchmarks). Laptop: code, tests, viewer. Notebooks only call CLI entry points. Exception (M2): small jobs may run on the laptop in `.venv-embed` after reporting sizes, when that lets me verify results.
- **Commits:** small, one per milestone step.

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
- 2026-11-22: runs in Colab, repo stays local; private GitHub repo `hddaghigh/sitesight-construction-vision-language`; Colab clones it with a read-only token in Colab Secrets.
- M1 parser is stdlib-only (JPEG size read from the SOF marker) so it runs anywhere without dependencies.

## Real-data findings that shape later milestones
8,428 regions (page says 8,601); 795 raw / 746 trimmed / 735 normalised distinct captions; 6 clipped boxes; 12 duplicates; 1 empty caption; 19 images without regions; 9 regions with two captions glued together (e.g. "...on soilA black dust protection..."); 25% of boxes < 1% of image area. The M2 caption-field parser must report these glued captions as unparsed rather than guess.

## M2 results (details and evidence: DATASHEET.md, reports/m2/)
- Caption-field parser: 8,396 / 8,427 non-empty captions parsed (99.63%); 31 unparsed with reasons; 31 resource categories (`sitesight/data/caption_fields.py`).
- Groups (pHash <= 12 OR DINOv2-small cosine >= 0.90, single linkage): 428 groups, largest 74. Split 1,002 / 215 / 214 images. Naive random split leaks a mean 1,210 train-test near-duplicate pairs (76.6% of test images have a train neighbour); grouped split 0.
- **Residual leak:** under looser similarity (pHash <= 16 or cosine >= 0.85) the grouped split still has 297 train-test pairs (44.9% of test images). Held-out scores mean "unseen scenes", not provably "unseen sites". `safety_fence` and `water_tank` are absent from test.
- Local torch exception: embeddings were computed on the laptop in a separate venv `.venv-embed` (`uv sync --group embed`; 710 MB) with HF cache in `sitesight_data\.hf`, because 1,431 images take only ~4 min. The Colab notebook `notebooks/01_data_splits.ipynb` runs the same CLIs but has **not been executed** yet.
- Opt-in tests: `pytest -m real_data` (needs `SITESIGHT_DATA_DIR`), `pytest -m embed` (run in `.venv-embed`).

## M3 (viewer + audit)
- Qt-free logic: `sitesight/viewer/state.py` (layers original/edit, undo/redo, blind-mode hiding, export), `sitesight/data/audit.py` (sampling, greedy IoU matching, agreement, kappa, bootstrap CI, taxonomy), `sitesight/data/predictions.py` (shared prediction file format for the viewer and M4). Qt layer: `sitesight/viewer/canvas.py`, `app.py`.
- Run with `make view` / `make audit` (blind by default). Tests: default suite is Qt-free; `pytest -m viewer` runs the GUI headless (`QT_QPA_PLATFORM=offscreen`, set `QT_QPA_FONTDIR=C:\Windows\Fonts` to render text in screenshots).
- **The audit has NOT been performed**; `reports/m3/audit_sample.json` is the fixed sample (seed 0). Never fabricate audit numbers; `audit_report.json` must come from a real `audit.json`. Use throwaway audit files for tests/screenshots, never `sitesight_data\audit\audit.json`.
- PyQt6 stub quirks are handled with targeted `# type: ignore[...]` (typed `pyqtSignal`, `QTest` static calls, `QGraphicsItem(parent=)`).

## M4 (zero-shot baselines + evaluation harness)
- Harness (pure numpy, `sitesight/eval/`): `boxes.py` (IoU, NMS), `grounding.py` (AP all-point, COCO-style matching, recall@IoU with `top_k`, size/crowding/kind breakdowns, two protocols `detection` vs `given_phrases`: never compare across them), `calibration.py` (ECE, 10 equal-width bins), `captioning.py` (BLEU, CIDEr-D as in coco-caption, METEOR with **exact matching only**, loose + strict field accuracy), `summary.py` (image-level resource presence F1), `data.py` (GT building + exclusion report), `run.py` (CLI + markdown tables). Every formula has a hand-derived test; mutation-checked (12/13 killed, survivor is an equivalent mutant).
- Baselines (`sitesight/baselines/`): OWLv2-base (all 31 classes, per-class NMS) and **native** `florence-community/Florence-2-base` (no `trust_remote_code`, MIT). Prompts are pre-registered in `prompts.py` and must not be tuned on val. Florence confidence = geometric-mean token probability under greedy decoding.
- **Compute reality (laptop i7-8565U, 4 threads):** Florence-2-base vision tower ~45 s per image (80% of a query; `channels_last` does not help), so image features are cached per image and both Florence tasks run in one pass (~140 s/image total). OWLv2 ~11 s/image for all 31 classes. Full val for Florence is therefore a Colab job (`notebooks/02_baselines.ipynb`, not yet run); locally Florence runs on a seeded 30-image subset (`--limit 30 --seed 0`).
- **Decision 2026-11-22: the M4 baseline runs happen in Colab, not on the laptop CPU.** The local run was stopped (a partial `owlv2_val.json` remains; the runner resumes it). Long model runs stay in Colab (see the Rules section).
- **M4 results exist** (`reports/m4/`): full validation split on a Colab T4 via the token-free notebook built by `scripts/build_colab_notebook.py` (no tokens or passwords are typed anywhere). OWLv2 `given_phrases` mAP@0.5 24.1, Florence-2-large 21.5, base 16.3; zero-shot captions ~0 on template metrics. Prediction files were NOT downloaded (OWLv2's are hundreds of MB); only the 40 KB metrics JSON was.
- Bugs found only by really running on Colab (all fixed, regression-tested): (1) Mendeley returns 403 to urllib's default User-Agent; (2) `python -m sitesight.data.splits` silently did nothing (no `__main__` guard; `tests/test_cli_entrypoints.py` now launches every CLI with `-m`); (3) image-level presence F1 was computed for models queried only where a class is present (99% for Florence): now `None` for those (`sanitize_results`, `--from-metrics`).
- Lesson: unit tests that call `main()` directly or use a local fake server do not prove the real entry point or the real host works; run the actual command and endpoint before declaring a pipeline done.
- Predictions live in `sitesight_data\derived\predictions\` (tens of MB; not committed). Committed: metrics + tables in `reports/m4/`.
- Zero-shot is run on `val` only; `baselines/run.py` refuses `--split test`.
- Opt-in `pytest -m embed` runs the real-model tests (slow: minutes on CPU); it needs `SITESIGHT_DATA_DIR`.

## M5 (LoRA fine-tune of Florence-2-base)
- Code: `sitesight/train/{data,lora}.py`, `eval/gallery.py`, `scripts/build_colab_notebook.py --kind m5`. Run on Colab only.
- Vision features are cached (frozen tower), so no augmentation. Epoch chosen by val loss (epoch 4). Test split evaluated once.
- Results (val/test mAP@0.5 39.2/34.0; ECE 0.43-0.46, worse than zero-shot) and caveats: `reports/m5/README.md`. Adapter (13 MB) is not committed; it lives in `sitesight_data\derived\m5
un\epoch_4`. Gallery images are regenerated, not committed.
- Never pass `--allow-test` again for model selection; the test set is spent for this protocol.

## M6 (active learning)
- Code: `sitesight/active/` (ranking.py strategies + curve summaries, simulate.py runner), `configs/active.toml`, Colab notebook via `scripts/build_colab_notebook.py --kind m6`. Report: `reports/m6/README.md`.
- Result (val, 3 seeds): labels to reach 90% of full-data score: random 283.0, uncertainty 238.2, uncertainty+diversity 238.5. Diversity gave no gain over plain uncertainty. Per-round rows were not committed.
- Unit of cost = image; boxes assumed known; captioning only; the test split is never read.

## M7 (serving)
- Code: `sitesight/serve/` (export.py ONNX+int8, codec.py numpy pre/post, onnx_backend.py greedy loop, api.py FastAPI, cli.py batch, bench.py), `Dockerfile`, CI `docker` job. Colab notebook: `build_colab_notebook.py --kind m7`.
- Result (300 val regions, 2-vCPU Colab, cold): PyTorch p50 85.1 s, ONNX fp32 12.1 s, int8 9.5 s; int8 574 MB vs 1,242 MB; int8 identical caption 76%, field accuracy within 2 regions. Details and caveats in `reports/m7/README.md`.
- Weights must be registered as initializers (model as submodule) or int8 quantisation silently does nothing. Docker image untested locally. API confidence is not calibrated.

## M8 (monitoring)
- Code: `sitesight/monitor/` (request_log, quality, drift, report, cli, perturb, demo). API logs with `SITESIGHT_REQUEST_LOG`; logs hold no images, file names, captions or boxes.
- Drift = per-feature KS (Bonferroni) + MMD permutation test, alpha 0.01 split half/half; batches under 20 are `too_small`. `monitor.cli check` exit 0 ok / 1 alert / 2 too small.
- Result (`reports/m8/`): same-site false alarms 0-1%; whole-batch degradations detected from n=40; 25%-degraded batches need ~80; mild JPEG/noise mostly missed; validation (new sites) alerts vs train. Output drift untested on real predictions; no embedding drift yet.

## M9 (geometry, optional)
- Code: `sitesight/geometry/` (tum.py reader, vo.py ORB+depth+PnP odometry, se3.py ATE/RPE, run.py). Colab notebook: `build_colab_notebook.py --kind m9`; heavy run stays in Colab (dataset not stored locally; F: is near the 3 GB floor).
- Result (`reports/m9/`): ATE 8.9 cm frame-to-frame, 9.1 cm keyframe on fr1/desk (573 frames), evo agrees to 1e-6; keyframes did not help ATE. Odometry only, no loop closure; ground truth only for scoring.

## Status
- **M1 done** (parser, validator, schema, downloader, tests, CI).
- **M2 done** (caption parser, pHash+DINOv2 dedup, grouped split, leakage report, datasheet, Colab notebook not yet run).
- **M3 done except the human part**: viewer, audit analysis, sample and protocol are built and tested; the 100-image audit awaits my own labelling (label accuracy is unmeasured everywhere).
- **M4 to M9 done** (zero-shot baselines and eval harness; LoRA fine-tune; active learning; ONNX/int8 serving; monitoring; RGB-D odometry). Results and caveats: `reports/m4` to `reports/m9`.
- **M10 done**: README headline table with sources, `TECHNICAL_REPORT.md`, `tests/test_headline_numbers.py` (fails if a README number and its committed evidence disagree). When a report changes, update the README row and rerun that test.
- Open items: the 100-image audit (my own labelling), calibration (temperature scaling), KV-cached decoder, embedding drift and real output-drift logs, M6 raw per-round rows were never committed.
