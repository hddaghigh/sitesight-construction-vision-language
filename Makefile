# SiteSight tasks. On Windows without make, run the same commands with `uv run ...`.
# Data lives outside the repo; override with:  make data DATA_DIR=/content/sitesight_data
DATA_DIR ?= sitesight_data
RUN = uv run

.PHONY: sync lint format typecheck test check download data parse check-real

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

# Assert the parser reproduces the independently measured findings on the real data.
check-real:
	SITESIGHT_DATA_DIR=$(DATA_DIR) $(RUN) pytest -m real_data
