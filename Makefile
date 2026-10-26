# VLM Scenario Mining
# Day 1 live targets: install, data, windows, test, lint.
# mine / eval / report / assets stay thin until later days.

ifeq ($(OS),Windows_NT)
SHELL := cmd.exe
.SHELLFLAGS := /C
endif

.PHONY: install data windows mine index eval report label assets test lint tags

install:
	uv sync --frozen

data:
	uv run python scripts/download_data.py
	uv run python -c "from vsm.ingest.convert import build_frames_table; from vsm.config import load_named, REPO_ROOT; from vsm.ingest.integrity import run_ingestion_gate; from vsm.ingest.conditions import add_condition_columns; cfg=load_named('ingest'); df=add_condition_columns(build_frames_table(REPO_ROOT/'data'/'raw'/'v1.0-mini', REPO_ROOT/'data'/'raw'/'maps', cfg)); print(run_ingestion_gate(df))"

windows:
	uv run python -c "from pathlib import Path; import polars as pl; from vsm.config import load_named, REPO_ROOT; from vsm.ingest.windows import build_grid_windows; frames=pl.read_parquet(REPO_ROOT/'data'/'frames'/'frames.parquet'); w=build_grid_windows(frames, load_named('ingest')); Path(REPO_ROOT/'data'/'frames').mkdir(parents=True, exist_ok=True); w.write_parquet(REPO_ROOT/'data'/'frames'/'windows.parquet'); print('windows', w.height)"

mine:
	uv run python scripts/run_mining.py

index:
	uv run python scripts/run_index.py

eval:
	uv run python scripts/run_eval.py

report:
	uv run python -c "print('report: day 12 (viz + README numbers). Day 1 writes docs/NEXT_STEPS.md counts.')"

label:
	uv run python -m vsm.labeling.tool --smoke 5

assets:
	uv run python scripts/make_assets.py

test:
	uv run pytest -q -m "not slow"

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

tags:
	uv run python -c "from pathlib import Path; print('\n'.join(sorted(p.name for p in Path('.git/refs/tags').glob('*'))) if Path('.git/refs/tags').exists() else 'no tags')"
