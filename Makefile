.PHONY: install install-serve doctor serve ingest index eval eval-generation sweep test lint clean

install:
	uv venv --python 3.11
	uv pip install -e ".[dev]"

install-serve:
	uv pip install -e ".[dense,serve,dev]"

# Preflight sin red: corpus, indice fresco, golden set, extras y credenciales.
doctor:
	.venv/bin/python -m compliance_mcp.doctor

# Arranca el servidor MCP por stdio. Pasa antes el preflight: un servidor stdio
# que falla al arrancar se ve desde el cliente como "no conecta", sin mas.
serve: doctor
	.venv/bin/python -m compliance_mcp.server

ingest:
	.venv/bin/python -m compliance_mcp.build_index ingest

index:
	.venv/bin/python -m compliance_mcp.build_index index --all

eval:
	.venv/bin/python -m compliance_mcp.eval.ablation --split test

# Evaluacion de la generacion. Gasta API: usa la cadena real de proveedores.
# Para el suelo sin red: PROVIDER=extractive make eval-generation
PROVIDER ?= chain
eval-generation:
	.venv/bin/python -m compliance_mcp.eval.generation --split test --provider $(PROVIDER)

sweep:
	.venv/bin/python -m compliance_mcp.eval.sweep

test:
	.venv/bin/python -m pytest

lint:
	.venv/bin/python -m ruff check src tests

clean:
	rm -rf data/processed data/index
