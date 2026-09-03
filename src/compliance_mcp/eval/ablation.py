"""Ablacion de recuperacion: 3 estrategias de chunking x 3 metodos.

Se reporta el split de TEST. El barrido de hiperparametros vive en sweep.py y
solo mira TRAIN.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Any

from ..config import Config, load_config
from ..ingest import read_records
from ..observability import configure_logging, log_event, trace_context
from ..provenance import DirtyTreeError, provenance_block, require_clean_tree
from ..retrieval.search import METHODS, Retriever
from .golden import GoldenCase, load_golden_set, split_cases, validate_against_corpus
from .metrics import evaluate_retrieval


def build_retriever(config: Config, strategy: str) -> Retriever:
    return Retriever.build(config, strategy, with_dense=True)


def evaluate_config(
    retriever: Retriever,
    cases: list[GoldenCase],
    config: Config,
    method: str,
    *,
    limit: int,
    weight_bm25: float | None = None,
    rrf_k: int | None = None,
    rollup_alpha: float | None = None,
) -> dict[str, Any]:
    latencies_ms: list[float] = []

    def rank(case: GoldenCase) -> list[str]:
        started = time.perf_counter()
        result = retriever.rank_control_ids(
            case.question,
            method=method,
            limit=limit,
            filters=None,
            weight_bm25=weight_bm25,
            rrf_k=rrf_k,
            rollup_alpha=rollup_alpha,
        )
        latencies_ms.append((time.perf_counter() - started) * 1000)
        return result

    # Calentamiento: el modelo de embeddings se carga de forma perezosa en la
    # primera consulta. Sin esta llamada previa, el primer metodo evaluado carga
    # el modelo dentro de la medicion y su latencia sale inflada un orden de
    # magnitud. Las latencias tampoco pueden incluir el bootstrap, asi que se
    # cronometra solo la llamada de ranking.
    if cases:
        rank(cases[0])
        latencies_ms.clear()

    metrics = evaluate_retrieval(cases, rank, config)
    out = metrics.to_dict()
    if latencies_ms:
        ordered = sorted(latencies_ms)
        out["latency_ms"] = {
            "p50": round(ordered[len(ordered) // 2], 2),
            "p95": round(ordered[min(int(len(ordered) * 0.95), len(ordered) - 1)], 2),
            "mean": round(sum(ordered) / len(ordered), 2),
        }
    return out


def by_style(
    retriever: Retriever, cases: list[GoldenCase], config: Config, method: str, limit: int
) -> dict[str, Any]:
    """Metricas por estrato de estilo. Con n=10 por estrato son direccionales."""
    out: dict[str, Any] = {}
    styles = sorted({c.style for c in cases if c.style})
    for style in styles:
        subset = [c for c in cases if c.style == style]
        out[style] = evaluate_config(retriever, subset, config, method, limit=limit)
    return out


def run(config: Config, *, split: str = "test") -> dict[str, Any]:
    cases = load_golden_set(config)
    train, test = split_cases(cases, config)
    selected = {"train": train, "test": test, "all": cases}[split]

    records = read_records(config.path("corpus.records_path"))
    errors = validate_against_corpus(cases, {r.control_id for r in records})
    if errors:
        raise ValueError("El golden set es inconsistente con el corpus:\n  " + "\n  ".join(errors))

    strategies: list[str] = config.get("evaluation.ablation.chunking_strategies")
    methods: list[str] = config.get("evaluation.ablation.retrieval_methods")
    limit = max(config.get("evaluation.metrics.recall_at_k"))

    results: dict[str, Any] = {
        "split": split,
        # Sin procedencia, una tabla de resultados es un numero sin sujeto: no
        # se sabe de que corpus, que config ni que commit salio.
        "provenance": provenance_block(config),
        "n_cases_total": len(selected),
        "n_cases_scorable": sum(1 for c in selected if c.scorable_for_retrieval),
        "config": {
            "embedding_model": config.get("retrieval.dense.model"),
            "rrf_k": config.get("retrieval.fusion.rrf_k"),
            "weight_bm25": config.get("retrieval.fusion.weight_bm25"),
            "weight_dense": config.get("retrieval.fusion.weight_dense"),
            "parent_rollup_alpha": config.get("retrieval.parent_rollup.alpha"),
            "candidate_pool": config.get("retrieval.candidate_pool"),
        },
        "grid": {},
        "by_style": {},
        "rollup_effect": {},
    }

    for strategy in strategies:
        retriever = build_retriever(config, strategy)
        results["grid"][strategy] = {}
        for method in methods:
            if method not in METHODS:
                raise ValueError(f"Metodo desconocido en config: {method}")
            metrics = evaluate_config(retriever, selected, config, method, limit=limit)
            results["grid"][strategy][method] = metrics
            log_event(
                "ablation.cell",
                strategy=strategy,
                method=method,
                recall_at_5=metrics.get("recall@5"),
                mrr=metrics.get("mrr"),
                n=metrics["n"],
            )

        # Efecto aislado del roll-up al padre, con el resto de parametros fijos.
        results["rollup_effect"][strategy] = {
            "alpha_0.0": evaluate_config(
                retriever, selected, config, "hybrid", limit=limit, rollup_alpha=0.0
            ),
            "alpha_active": evaluate_config(retriever, selected, config, "hybrid", limit=limit),
        }

        # Estratos de estilo solo para la estrategia activa: es lo que se sirve.
        if strategy == config.get("chunking.active"):
            results["by_style"] = {
                method: by_style(retriever, selected, config, method, limit) for method in methods
            }

    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance-mcp-ablation")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Producir el artefacto aunque el arbol tenga cambios. Queda marcado "
             "con dirty=true en la procedencia y el gate de CI lo rechazara.",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config)
    # Antes de gastar una corrida entera: si el arbol esta sucio, estos numeros
    # no se podran atribuir a ningun codigo concreto.
    try:
        require_clean_tree(allow_dirty=args.allow_dirty)
    except DirtyTreeError as exc:
        # Sin traza: esto no es una caida, es una negativa deliberada, y una
        # traza invita a leerlo como un bug del programa.
        print(f"FALLO: {exc}", file=sys.stderr)
        return 1
    with trace_context():
        results = run(config, split=args.split)

    out_path = config.path("evaluation.ablation.output_path")
    if args.out:
        from pathlib import Path

        out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    log_event("ablation.written", path=str(out_path), split=args.split)
    print(json.dumps(results["grid"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
