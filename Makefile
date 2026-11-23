# SiteSight tasks. On Windows without make, run the same commands with `uv run ...`.
# Data lives outside the repo; override with:  make data DATA_DIR=/content/sitesight_data
DATA_DIR ?= sitesight_data
RUN = uv run

.PHONY: sync lint format typecheck test check download data parse fields embed curate view audit audit-report check-real baseline-owlv2 baseline-florence eval-zero-shot

sync:
	uv sync

lint:
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format:
	$(RUN) ruff format .

typecheck:
	$(RUN) mypy

test:
	$(RUN) pytest

# What CI runs.
check: lint typecheck test

# Download + verify (SHA-256) + extract the dataset (idempotent).
download:
	$(RUN) python scripts/download_data.py --data-dir $(DATA_DIR)

# Parse VIA -> COCO-style annotations.json + validation_report.json
parse: download
	$(RUN) python -m sitesight.data.via_parse \
	  --via $(DATA_DIR)/extracted/via_region_data_final.json \
	  --images $(DATA_DIR)/extracted/UAVdata \
	  --out $(DATA_DIR)/derived/annotations.json \
	  --report $(DATA_DIR)/derived/validation_report.json

data: parse

# M2: caption fields -> embeddings (needs the embed venv / Colab GPU) -> groups + split + report
fields: parse
	$(RUN) python -m sitesight.data.caption_fields \
	  --annotations $(DATA_DIR)/derived/annotations.json \
	  --out $(DATA_DIR)/derived/annotations_fields.json \
	  --report $(DATA_DIR)/derived/coverage_report.json

embed: fields
	python scripts/embed_images.py --annotations $(DATA_DIR)/derived/annotations_fields.json \
	  --images $(DATA_DIR)/extracted/UAVdata --out $(DATA_DIR)/derived/dinov2_small.npy

curate:
	$(RUN) python -m sitesight.data.curate --config configs/data.toml \
	  --annotations $(DATA_DIR)/derived/annotations_fields.json \
	  --images $(DATA_DIR)/extracted/UAVdata \
	  --embeddings $(DATA_DIR)/derived/dinov2_small.npy \
	  --out-dir $(DATA_DIR)/derived/curated

# M3: review/debug viewer (needs `uv sync --group viewer`); PRED=file1.json file2.json to compare models
view:
	$(RUN) python -m sitesight.viewer.app \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --images $(DATA_DIR)/extracted/UAVdata $(if $(PRED),--predictions $(PRED),)

# M4: zero-shot baselines. The full validation run was done in Colab (see reports/m4/README.md);
# these local targets exist for small smoke runs only (torch venv: `uv sync --group embed`).
#   On the dev laptop CPU: OWLv2 ~11 s/image, Florence-2 ~140 s/image: use --limit.
baseline-owlv2:
	python -m sitesight.baselines.run --model owlv2 \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --images $(DATA_DIR)/extracted/UAVdata --split val \
	  --out $(DATA_DIR)/derived/predictions/owlv2_val.json

baseline-florence:
	python -m sitesight.baselines.run --model florence --limit 30 --seed 0 \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --images $(DATA_DIR)/extracted/UAVdata --split val \
	  --out $(DATA_DIR)/derived/predictions/florence2_base_val30.json

# Score prediction files on val; --common-images compares models on the images ALL of them cover
eval-zero-shot:
	$(RUN) python -m sitesight.eval.run --split val --common-images \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --predictions $(DATA_DIR)/derived/predictions/owlv2_val.json \
	    $(DATA_DIR)/derived/predictions/florence2_base_val30_grounding.json \
	    $(DATA_DIR)/derived/predictions/florence2_base_val30_regions.json \
	  --out reports/m4/zero_shot_val30.json

# Blind label audit of the 100-image sample (see AUDIT.md); resumes if audit.json exists
audit:
	$(RUN) python -m sitesight.viewer.app \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --images $(DATA_DIR)/extracted/UAVdata \
	  --audit-sample reports/m3/audit_sample.json \
	  --audit-file $(DATA_DIR)/audit/audit.json --audit-mode $(or $(MODE),blind)

audit-report:
	$(RUN) python -m sitesight.data.audit report \
	  --annotations $(DATA_DIR)/derived/curated/annotations_curated.json \
	  --audit $(DATA_DIR)/audit/audit.json --out reports/m3/audit_report.json

# Assert the parser reproduces the independently measured findings on the real data.
check-real:
	SITESIGHT_DATA_DIR=$(DATA_DIR) $(RUN) pytest -m real_data
