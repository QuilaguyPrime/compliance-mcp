"""Barrido de hiperparametros de fusion. SOLO sobre el split de train.

Reemplaza al `hill_climb.py` de la version anterior, que optimizaba
directamente contra el mismo conjunto con el que se reportaban resultados.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Any

from ..config import Config, load_config
from ..observability import configure_logging, log_event, trace_context
from ..retrieval.search import Retriever
from .ablation import evaluate_config
from .golden import load_golden_set, split_cases

OBJECTIVE = "recall@5"


def run(config: Config) -> dict[str, Any]:
    cases = load_golden_set(config)
    train, _ = split_cases(cases, config)
    strategy = config.get("chunking.active")
    retriever = Retriever.build(config, strategy, with_dense=True)
    limit = max(config.get("evaluation.metrics.recall_at_k"))

    grid = list(
        itertools.product(
            config.get("evaluation.sweep.weight_bm25"),
            config.get("evaluation.sweep.rrf_k"),
            config.get("evaluation.sweep.parent_rollup_alpha"),
        )
    )
    rows: list[dict[str, Any]] = []
    for weight_bm25, rrf_k, alpha in grid:
        metrics = evaluate_config(
            retriever,
            train,
            config,
            "hybrid",
            limit=limit,
            weight_bm25=weight_bm25,
            rrf_k=rrf_k,
            rollup_alpha=alpha,
        )
        rows.append(
            {
                "weight_bm25": weight_bm25,
                "rrf_k": rrf_k,
                "parent_rollup_alpha": alpha,
                **{k: v for k, v in metrics.items() if k != "ci95"},
            }
        )

    rows.sort(key=lambda r: (-r[OBJECTIVE], -r["mrr"]))
    log_event(
        "sweep.completed",
        strategy=strategy,
        combinations=len(rows),
        split="train",
        n=rows[0]["n"] if rows else 0,
        best=rows[0] if rows else None,
    )
    return {"split": "train", "strategy": strategy, "objective": OBJECTIVE, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance-mcp-sweep")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    configure_logging(config)
    with trace_context():
        results = run(config)
    out = config.path("evaluation.sweep.output_path")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    header = f"{'w_bm25':>7}{'rrf_k':>7}{'alpha':>7}{'r@1':>7}{'r@5':>7}{'r@10':>7}{'MRR':>7}"
    print(f"\nBarrido sobre TRAIN (n={results['rows'][0]['n']}), objetivo {OBJECTIVE}\n")
    print(header)
    print("-" * len(header))
    for row in results["rows"][:15]:
        print(
            f"{row['weight_bm25']:>7}{row['rrf_k']:>7}{row['parent_rollup_alpha']:>7}"
            f"{row['recall@1']:>7.3f}{row['recall@5']:>7.3f}{row['recall@10']:>7.3f}{row['mrr']:>7.3f}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
