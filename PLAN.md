# SiteSight — Plan (v1, pre-build)

Grounded construction-site understanding from drone images: region descriptions, a box per described object, and a structured site summary, served on CPU with monitoring and a review tool.

Status: **plan only, nothing built.** Numbers in this file are either *measured* (from files or commands run on 2026-11-22), *quoted* (from a cited page), or labelled *estimate*.

---

## 1. Environment (measured)

| Item | Value |
|---|---|
| OS | Windows 10 Pro 10.0.19045 |
| CPU | Intel Core i7-8565U, 4 cores / 8 threads, 1.8 GHz base |
| RAM | 15.8 GB total; ~3.3 GB free at measurement time |
| GPU | NVIDIA MX250 (2 GB) + Intel UHD 620. Too small for any VLM: treated as **no GPU**, CPU-only PyTorch |
| Disk free | C: 8.5 GB, **F: 7.0 GB (before download)**, D: 3.2 GB |
| Python | `python` = 3.7.8 (unusable). 3.12 installed. **uv not installed** (decision: install uv, let it provide 3.11) |
| git | 2.38.1 |

Consequences: the laptop cannot host torch, model weights and a viewer comfortably, so **runs move to Colab** (§1b). Repo `sitesight`, data `sitesight_data` (outside the repo). Set `UV_CACHE_DIR` to F: so caches do not fill C:.

## 1b. Execution model: Colab runs, repo stays local (decision, 2026-11-22)

- **Local (laptop):** write code, ruff/mypy/pytest on the tiny fixture (numpy-level deps only: **no torch, no model weights**), the PyQt6 viewer and the label audit, git.
- **Colab:** a thin notebook per stage that `git clone`s the repo, `pip install`s the package, downloads the dataset to Colab's ephemeral disk with `scripts/download_data.py` (same SHA-256 check as local), and runs everything that needs torch or a model: embeddings/dedup/splits, baselines, fine-tuning, evaluation, active learning, ONNX export + quantisation, latency benchmark, drift demo. Notebooks contain no logic: they call `make` targets / CLI entry points so the same code is tested locally.
- **What comes back** (small, committed or attached to a release): derived COCO JSON + splits, metrics JSON, predictions JSON, LoRA adapters, benchmark JSON, HTML report. Drive is the scratch store between sessions; the repo is the source of truth.
- **Things Colab cannot do, and what replaces them:** (1) **PyQt6 viewer** needs a desktop → stays local, reads the images already in `sitesight_data` plus predictions JSON from Colab; (2) **Docker build** → built in GitHub Actions or locally if Docker exists, otherwise marked untested; (3) **latency** → measured on the Colab runtime and reported as "Colab CPU, N vCPU" (hardware printed by the script); a laptop number is optional and only if disk allows; (4) the final quantised model is exported in Colab and the FastAPI service is smoke-tested locally against it only if the file fits.
- **Colab constraints to design for:** sessions reset (every notebook starts from a clean clone and is idempotent), free-tier GPU availability is not guaranteed, runtimes disconnect (checkpoint adapters each epoch to Drive), hardware varies between sessions (log `nvidia-smi`/`lscpu` into every metrics JSON).
- **Repo transport (decided):** private GitHub repo `hddaghigh/sitesight` (created 2026-11-22). A notebook clones it with a **read-only fine-grained personal access token** (scope: this repo only, Contents: read) stored in Colab **Secrets** (`GITHUB_TOKEN`) and read with `google.colab.userdata.get`. The token is never written into a notebook, the repo or an output cell. Results are pulled back to the laptop via Drive or a downloaded file, not pushed from Colab with that token.

## 2. Dataset — what is actually there (measured)

Source: Bang & Kim, Mendeley Data v1, DOI 10.17632/4h68fmktwh.1, CC BY 4.0. One file `image_with_captions.zip`, 593,575,014 bytes, SHA-256 verified against the value the Mendeley API publishes. Extracted to `sitesight_data\extracted\` (606.5 MB uncompressed: JPEGs do not compress).

Contents: `UAVdata/<n>.jpg` (1,431 files) and **`via_region_data_final.json`**, a **VGG Image Annotator (VIA 1.x)** export. No split files, no site/flight metadata, no class labels, no README.

### Format
- Top level: dict keyed by `"<filename><filesize>"` (e.g. `"1.jpg440260"`). Entry keys: `fileref` (empty), `size`, `filename`, `base64_img_data` (empty), `file_attributes` (always `{}`), `regions`.
- `regions`: dict keyed `"0","1",...` (a dict, not a list). Each region has `shape_attributes` = `{name:"rect", x, y, width, height}` (**absolute pixels, top-left origin**, all 8,428 are `rect`) and `region_attributes` = `{phrase: "<text>"}` (the only attribute).
- **Position** is the box. The caption's "located on soil" is a relation to the ground, not an image position.
- All images are **960 × 540** (1,431 / 1,431).

### Three concrete examples (verbatim from the file)
1. `1.jpg`, region 0: `{"x":543,"y":407,"width":31,"height":63}` → "A worker is located on soil"
2. `1.jpg`, region 1: `{"x":295,"y":67,"width":501,"height":336}` → "Ten foundations are installed on soil"
3. `2.jpg`, region 2: `{"x":735,"y":314,"width":72,"height":182}` → "An orange excavator is digging soil"

### Statistics (measured)
| Quantity | Value |
|---|---|
| Images / regions | 1,431 / **8,428** |
| Regions per image | min 0, median 5, mean 5.89, max 17 |
| Images with 0 regions | 19 (e.g. `46.jpg`, `66.jpg`, `169.jpg`) |
| Distinct caption strings | 795 raw / **746** after `strip()` / **735** after whitespace normalisation (13 regions have internal double spaces or a newline); vocabulary 136 words (counted on stripped text) |
| Glued captions | 9 regions hold two captions run together (e.g. "…on soilA black dust protection…"): a label-error class for the audit; the M2 parser must report them as unparsed |
| Caption length | 6–9 words typical (median 7, max 16) |
| Boxes < 1% of image area | 2,103 of 8,428 (25%); median box = 2.75% of image |
| Out-of-bounds boxes | 6 (e.g. `423.jpg` x=−41; `1185.jpg` extends past the right edge) |
| Empty caption | 1 (`268.jpg` region 1) |
| Exact duplicate box+caption within an image | 12 |
| Caption variants differing only by trailing space | present (e.g. "…installed in a line" vs "…installed in a line ") |

Most frequent captions: "Fences are installed in a line" (803), "An orange excavator is located on soil" (542), "A blue tarpaulin is located on soil" (383), "A worker is located on soil" (353), "A yellow excavator is located on soil" (312).

### Discrepancies and what the data cannot support
1. **8,428 regions in the file vs 8,601 captions stated on the Mendeley page** (173 fewer). Cause unknown (possibly regions removed after the page text was written). I will report 8,428 everywhere and say so; I will not pad or guess.
2. **Captions are highly templated**: `[quantity] [colour] [resource] [is/are] [state/relation]`, built from ~735 distinct strings. Consequence: CIDEr/BLEU will be inflated for any model that learns the template, and a zero-shot VLM will score low for style rather than for being wrong. Field-level accuracy (resource, count, colour, state) is the metric that matters; caption n-gram metrics are reported alongside with this caveat.
3. **No activity vocabulary beyond a few verbs** ("digging", "loaded with", "covered with", "installed", "located"). The brief's example `activities: ["rebar placement"]` is not supported by this data. Closest alternative: `activity` = verb/state field ∈ {digging, loaded with, covered with, installed, located, empty, …}, produced by the parser.
4. **No site/flight identifiers.** Group splits must be derived (embedding clustering). File names are numeric; adjacency may or may not mean same flight: tested in M2, not assumed.
5. **Counts are partial.** Captions describe regions; "Fences are installed in a line" has no count. Image-level summaries are aggregates of annotated regions, not ground-truth site counts, and are labelled so.
6. **Small-object regime**: 25% of boxes < 1% of a 960×540 image. Expect Florence-2's 768 px input to hurt tiny workers; the small-object breakdown will show it.
7. Reference point only: the source paper reports 43.52% mAP (its own dense-captioning metric). Not comparable to our metrics; not used as a target.

## 3. Models (sizes and licences quoted from Hugging Face API, 2026-11-22)

| Model | Licence | Params | Weights on disk | Role |
|---|---|---|---|---|
| **Florence-2-base** | MIT | 231.6 M (F16) | 463 MB | **Primary**: zero-shot, LoRA/full fine-tune (Colab), CPU serving |
| Florence-2-large | MIT | 776.7 M (F16) | 1.55 GB | Colab only: size-vs-accuracy upper bound |
| OWLv2 base-patch16-ensemble | Apache-2.0 | 155.0 M (F32) | 620 MB | Open-vocab grounding baseline (boxes + scores, no captions) |
| Grounding DINO tiny | Apache-2.0 | 172.3 M (F32) | 689 MB | Optional second grounding baseline |
| SmolVLM2-2.2B-Instruct | Apache-2.0 | 2.25 B (F32) | 9.0 GB | Colab only: zero-shot captioning comparison (2–3B class) |
| ~~Qwen2.5-VL-3B-Instruct~~ | `qwen-research` (non-commercial research licence per API metadata) | 3.75 B (BF16) | 7.5 GB | **Rejected**: licence is a poor fit for a portfolio aimed at a company |
| ~~PaliGemma 2~~ | Gemma terms | 3B-class | — | Rejected: custom licence, needs fine-tuning for grounding, too slow on this CPU |

**Choice: Florence-2-base.** Native location-token output, so grounding needs no detection head; Florence-2's documented prompts include `<CAPTION_TO_PHRASE_GROUNDING>`, `<DENSE_REGION_CAPTION>`, `<OD>` (the card does *not* list `<REGION_TO_DESCRIPTION>`; I must check the processor code in M4 before relying on it, with fallback = fine-tune region captioning through the location-token prompt). MIT licence; 463 MB; trainable with LoRA on a free Colab T4. 

Open questions to resolve at the start of M4, not assumed now:
- **`trust_remote_code`**: the card still shows `trust_remote_code=True`, and says nothing about a native `transformers` class. I will pin a `transformers` version and read the remote code before running it (supply-chain risk) or use a native class if one exists.
- **CPU latency**: no published number found; measured in M7 on the Colab CPU runtime (hardware logged). It will not equal a laptop number, and the README says so.
- The model landscape was checked through API metadata, not benchmarks; if a newer sub-0.5B grounded model appears I will add it as a baseline, not swap the plan.

## 4. Repo structure

```
sitesight/
  pyproject.toml  uv.lock  Makefile  Dockerfile  CONTRIBUTING.md  PLAN.md
  README.md  TECHNICAL_REPORT.md  DATASHEET.md  DATA_LICENSES.md
  .github/workflows/ci.yml
  configs/                (yaml: data, train, eval, serve, monitor)
  notebooks/              (colab_baselines.ipynb, colab_finetune.ipynb)
  scripts/                (download_data.py ...)
  tests/  tests/fixtures/ (tiny synthetic VIA file + 10 generated images, <5 MB)
  sitesight/
    data/      via_parse.py schema.py validate.py dedup.py splits.py caption_fields.py
    baselines/ owlv2.py florence_zeroshot.py
    train/     dataset.py lora.py
    eval/      grounding.py captioning.py summary.py calibration.py gallery.py
    active/    ranking.py simulate.py
    serve/     export.py quantize.py api.py cli.py bench.py
    monitor/   quality.py drift.py request_log.py report.py
    viewer/    app.py widgets/          (PyQt6)
    geometry/  (optional M9)
sitesight_data\  raw/  extracted/  derived/   (outside the repo, gitignored)
```

## 5. Schema (derived from the real format)

COCO-style JSON written by `via_parse.py`:
- `images[]`: `id`, `file_name`, `width`, `height` (all 960×540), `group_id`, `split`.
- `annotations[]`: `id`, `image_id`, `bbox` `[x,y,w,h]` absolute px (clipped to the image; original kept as `bbox_raw` when clipping changed it), `category_id`, `caption` (whitespace-normalised), `caption_raw`, `fields` `{resource, count, colour, state, relation}`, `flags[]` (`clipped`, `empty_caption`, `duplicate`, `unparsed`), `source` `{via_key, region_id}`.
- `categories[]`: resource types produced by the parser (e.g. worker, excavator, dump truck, tarpaulin, fence, concrete pillar, H-beam, rebar…); the list is data-driven from the 746 strings and reviewed by hand.

Validation rules (each drop/clip logged with a reason code, never silent): box inside image (6 clipped), non-empty caption (1 flagged), no exact duplicate within an image (12 flagged), caption normalisation (trailing spaces), `0`-region images kept as negatives (19).

## 6. Caption field parser (design)

Deterministic grammar, no ML: `^(count)? (colour)* (resource) (is|are) (state) (relation)?$`, where count ∈ {a, an, a pile of, one…twenty, numerals}, colour and resource from a small lexicon built from the 136-word vocabulary, state ∈ {located, installed, digging, loaded with, covered with, empty, …}. Coverage = fraction of the 8,428 regions that parse fully; unparsed captions are listed, never guessed. Tests: hand-labelled table of ≥ 50 captions.

## 7. Split strategy

1. pHash and a self-supervised image embedding (DINOv2-small class; size reported before install) per image.
2. Agglomerative clustering; threshold chosen by looking at cluster purity on a hand-checked sample, recorded in config.
3. `GroupShuffleSplit` 70/15/15, fixed seed, by cluster.
4. Report: number of near-duplicate pairs that cross train/test under a *naive random* split of the same sizes vs 0 under the grouped split; sensitivity to the threshold.
5. The test split is read only by final evaluation scripts; model selection uses val.

## 8. Milestones

| # | Deliverable | Done when… | Effort (estimate) | Runs | Local disk added (estimate) |
|---|---|---|---|---|---|
| M1 | uv project (py3.11), ruff/mypy/pytest, VIA parser → COCO JSON, validator, fixture, `download_data.py` | `pytest` < 1 min; all 8,428 regions parsed or listed with reasons; 6/1/12/19 findings reproduced by tests | 1 session | Local code + tests (no torch); parse run locally (stdlib only) | ~0.3 GB |
| M2 | pHash+embedding dedup, grouped split, field parser, DATASHEET.md | leak count + parser coverage reported by committed scripts; parser tests hand-checked | 1–2 sessions | Code/tests local; **embeddings + split run in Colab**; field parser runs locally | ~0 (torch only in Colab) |
| M3 | PyQt6 viewer + 100-image audit | you can load, edit, export labels; audit agreement computed from your labels (IoU, label κ) | 1–2 sessions + your 1–2 h | **Local only** (desktop) | ~0.1 GB |
| M4 | OWLv2 + Florence-2 zero-shot; eval harness (mAP@0.5, recall@IoU, CIDEr/METEOR/BLEU, field accuracy, ECE) | every metric matches a hand-computed example in a test; zero-shot table on val | 2 sessions | Metrics code/tests local (numpy); **model runs in Colab** | ~0 locally (predictions JSON only) |
| M5 | LoRA fine-tune, group-split eval, failure gallery | seeds + configs + experiment log committed; zero-shot vs fine-tuned table; ≥ 1 diagnosed failure per type | 2 sessions | **Colab** train + eval; gallery browsed in the local viewer | adapters, tens of MB |
| M6 | Active-learning simulation | ≥ 3 seeds; uncertainty+diversity vs random; labels needed for N% of full-data score stated | 1 session | Colab (needs model scores/embeddings) | ~0 |
| M7 | ONNX/int8, FastAPI, batch CLI, latency benchmark, Dockerfile | fp32 vs int8 metrics side by side; p50/p95 on the Colab CPU (hardware logged); Docker build in CI or marked untested | 2 sessions | Export + benchmark in Colab; API unit tests local with a stub model | ~0.2 GB (fastapi, onnxruntime) |
| M8 | Request logging, quality checks, drift, alerts, HTML report | blurred / darkened / rescaled batches alert; clean held-out batch does not | 1–2 sessions | Quality-check/stat code local (numpy/Pillow); embedding drift demo in Colab | ~0 |
| M9 | *(optional; ask first)* TUM RGB-D fr1/desk geometry | ATE vs mocap via `evo` | 3+ sessions | Colab or local (CPU-light) | +0.36 GB data if local (quoted by you) |
| M10 | CI, README, technical report, CONTRIBUTING.md | CI green; every README number traces to script+config+seed | 1 session | Local | ~0 |

Colab notebooks (`notebooks/`): `01_data_splits.ipynb` (M2), `02_baselines.ipynb` (M4), `03_finetune_eval.ipynb` (M5), `04_active_learning.ipynb` (M6), `05_export_bench.ipynb` (M7), `06_drift_demo.ipynb` (M8). Each is a few cells: clone → install → run CLI → save outputs.

Adjustments from the data: M1 also produces a *data-findings* table (8,428 vs 8,601 etc.); M4 reports caption metrics with the template caveat from §2; M6's "performance" is field-level accuracy, not CIDEr.

## 9. Disk budget (F:)

Measured: F: free was 7.04 GB before the download and is 5.92 GB now (the zip, 0.59 GB, and the extracted folder, 0.61 GB, take ~1.2 GB). Local budget therefore stays within the ~3 GB you set, but the floor of 3 GB free is approached by M7.

| Stage | Change (estimate unless noted) | Running F: free (GB) |
|---|---|---|
| Now (zip + extracted) | measured | **5.92** (measured) |
| Delete zip after SHA verified (needs your OK) | +0.59 (measured) | 6.51 |
| M1 venv + deps (no torch) | −0.3 | 6.2 |
| M3 PyQt6 | −0.1 | 6.1 |
| M2/M4/M5/M6/M8 outputs brought back (JSON, adapters) | −0.1 | 6.0 |
| M7 fastapi/onnxruntime for local API tests | −0.2 | 5.8 |
| M9 (optional, local) | −0.36 data | 5.4 |

Estimated total local add ≈ 0.7 GB (M1–M7) beyond the dataset, leaving roughly **5.8 GB free on F: after deleting the zip** (5.4 GB with M9), well above the 3 GB floor. Torch and all model weights live only in Colab. Space is still checked before every local install. Colab: the free tier's ephemeral disk (re-downloading the 593 MB zip each session takes under a minute on this connection's measured ~12 MB/s) is not a constraint; only adapters, metrics JSON and predictions come back (tens of MB).

## 10. Risks and fallbacks

| Risk | Fallback |
|---|---|
| Templated captions make n-gram metrics meaningless | Lead with field-level accuracy and grounding; show both; say so in README |
| Phrase grounding has few distinct phrases (746) | Evaluate grounding per resource class (parser categories); report class frequencies; rare classes flagged, not averaged away |
| Parser coverage < 100% | Report coverage; unparsed regions excluded from field metrics only |
| 25% tiny boxes hurt Florence-2 at 768 px | Report small-object breakdown; optional tiled inference as an ablation |
| Florence-2 too slow on a Colab CPU (few vCPUs) | int8, smaller input, batch 1; fallback serving path = OWLv2 + parser-based template text, labelled as such |
| ONNX export of Florence-2 fails (custom code, encoder-decoder, generation loop) | PyTorch dynamic int8 on the decoder; accuracy drop still measured |
| `trust_remote_code` risk / `transformers` version drift | Pin versions, read the remote code, keep a lockfile |
| Colab session limits, resets, no GPU on a given day | Checkpoint adapters every epoch to Drive; idempotent notebooks; CPU-only fallback for evaluation (slow, stated) |
| Results differ between Colab sessions (hardware varies) | Fixed seeds; log GPU/CPU model into every metrics JSON; report latency with the hardware named |
| Colab cannot reach the repo | Private GitHub repo (needs your OK) or a zip on Drive |
| Viewer needs images + predictions that now come from Colab | Images stay in `sitesight_data`; predictions JSON copied back; schema is shared so no conversion |
| Label audit not done by you | Replace with model-vs-label disagreement review; mark it clearly as *not* a human audit |
| Free disk < 3 GB | Stop and ask; move heavy models to Colab |

## 10b. Progress log

- **M1 done** (2026-11-22): parser/validator/schema/downloader; real-data findings reproduced.
- **M2 done** (2026-11-22): caption-field parser (99.63% coverage), pHash + DINOv2-small dedup, grouped split (428 groups; naive split leaks a mean 1,210 train-test pairs vs 0), DATASHEET.md. Embeddings were computed on the laptop (710 MB venv `.venv-embed`) rather than Colab, by agreement, so results could be verified; the Colab notebook is written but not yet run.
- **M3 built** (2026-11-23): PyQt6 review tool, shared predictions format, audit sampling/matching/agreement/taxonomy code, 100-image sample (`reports/m3/audit_sample.json`, seed 0), protocol in AUDIT.md. **Not done: the audit itself (needs your 2 to 3 h in blind mode, 45 to 90 min in review mode, both rough estimates) and therefore no agreement numbers or error-taxonomy results exist yet.**
- **M4 built** (2026-11-23): evaluation harness and baseline runners (OWLv2, Florence-2-base). Resolved: Florence-2's `trust_remote_code` risk (a native `transformers` conversion exists, MIT). New measured constraint: Florence-2-base's vision encoder takes ~45 s per image on the laptop CPU (OWLv2 ~11 s), so Florence on the full validation split is a Colab job; the local run uses a seeded 30-image subset and says so.
- **M4 results** (2026-11-23, Colab T4, validation split, 215 images): OWLv2-base `given_phrases` mAP@0.5 24.1 / recall@0.5 50.8; Florence-2-large 21.5 / 33.8; Florence-2-base 16.3 / 27.5; OWLv2 `detection` protocol 7.9. Zero-shot Florence-2 region descriptions: loose resource accuracy 14.4% to 18.1%, 0% fit the template. Details and caveats in `reports/m4/README.md`. These are the numbers M5 fine-tuning must beat (same split, same harness).
- **New risk found in M2:** residual same-site leakage under looser similarity (297 train-test pairs at pHash <= 16 / cosine >= 0.85). Mitigation options for M5: report val/test scores with this caveat; try average-linkage clustering (needs scipy) or manual site labels from the M3 audit set; never describe the test set as "unseen sites".

## 11. Open items for you

1. Okay to delete the 0.59 GB zip now that the SHA-256 matched and the content is extracted (frees space; the original stays re-downloadable)?
2. Approve installing **uv** (not yet installed; I will report package sizes first) for the M1 session. Only light dependencies are installed locally (no torch).
3. M9 (TUM RGB-D geometry): decide at the end of M8; not part of the commitment yet.
4. ~~How should Colab get the repo?~~ Decided: private GitHub repo (§1b). You still need to create the read-only token and add it to Colab Secrets before the first notebook runs (I will give the exact steps in M1).
5. Colab account: the free tier is assumed (T4 when available, ~12 GB RAM). If you have Pro, the plan does not need it but runs faster.
