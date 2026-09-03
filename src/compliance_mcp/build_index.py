"""CLI: ingest del catalogo y construccion de indices.

    python -m compliance_mcp.build_index ingest
    python -m compliance_mcp.build_index index --strategy C
    python -m compliance_mcp.build_index index --all
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import numpy as np

from .chunking import chunk_records
from .config import Config, load_config
from .index_manifest import build_entry, write_entry
from .ingest import build_records, read_records, write_records
from .observability import configure_logging, log_event, trace_context
from .retrieval.dense import DenseRetriever
from .retrieval.search import embeddings_path


def do_ingest(config: Config) -> None:
    start = time.perf_counter()
    records = build_records(config)
    path = config.path("corpus.records_path")
    write_records(records, path)
    withdrawn = sum(1 for r in records if r.status == "withdrawn")
    log_event(
        "ingest.completed",
        records=len(records),
        controls=sum(1 for r in records if r.kind == "control"),
        enhancements=sum(1 for r in records if r.kind == "enhancement"),
        withdrawn=withdrawn,
        with_baseline=sum(1 for r in records if r.baselines),
        unresolved_params=sum(1 for r in records if "{{ insert" in r.statement + r.guidance),
        path=str(path),
        elapsed_ms=round((time.perf_counter() - start) * 1000, 1),
    )


def do_index(config: Config, strategies: list[str]) -> None:
    records = read_records(config.path("corpus.records_path"))
    chunks_dir = config.path("corpus.chunks_dir")
    index_dir = config.path("corpus.index_dir")
    chunks_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    for strategy in strategies:
        chunks = chunk_records(records, strategy, config)
        chunk_path = chunks_dir / f"chunks_{strategy}.jsonl"
        with chunk_path.open("w", encoding="utf-8") as fh:
            for chunk in chunks:
                fh.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")

        start = time.perf_counter()
        embeddings = DenseRetriever.encode_corpus([c.text for c in chunks], config)
        out = embeddings_path(config, strategy)
        np.save(out, embeddings)
        # El manifiesto se escribe DESPUES del .npy: si el encode falla a medias,
        # queda un indice sin manifiesto (que se rechaza al cargar) en vez de un
        # manifiesto que certifica un indice que no existe.
        entry = build_entry(config, strategy, chunks, int(embeddings.shape[1]), out)
        manifest = write_entry(config, entry)
        log_event(
            "index.built",
            strategy=strategy,
            chunks=len(chunks),
            unique_controls=len({c.control_id for c in chunks}),
            dim=int(embeddings.shape[1]),
            model=config.get("retrieval.dense.model"),
            chunks_digest=entry.chunks_digest,
            encode_ms=round((time.perf_counter() - start) * 1000, 1),
            path=str(out),
            manifest=str(manifest),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance-mcp-index")
    parser.add_argument("command", choices=["ingest", "index", "all"])
    parser.add_argument("--strategy", help="Estrategia de chunking (por defecto: chunking.active)")
    parser.add_argument("--all", action="store_true", help="Todas las estrategias de la ablacion")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config)

    if args.all:
        strategies = config.get("evaluation.ablation.chunking_strategies")
    elif args.strategy:
        strategies = [args.strategy]
    else:
        strategies = [config.get("chunking.active")]

    with trace_context():
        if args.command in ("ingest", "all"):
            do_ingest(config)
        if args.command in ("index", "all"):
            do_index(config, strategies)
    return 0


if __name__ == "__main__":
    sys.exit(main())
