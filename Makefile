.PHONY: install ingest index eval sweep test lint clean

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

ingest:
	.venv/bin/python -m compliance_mcp.build_index ingest

index:
	.venv/bin/python -m compliance_mcp.build_index index --all

eval:
	.venv/bin/python -m compliance_mcp.eval.ablation --split test

sweep:
	.venv/bin/python -m compliance_mcp.eval.sweep

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check src tests

clean:
	rm -rf data/processed data/index
