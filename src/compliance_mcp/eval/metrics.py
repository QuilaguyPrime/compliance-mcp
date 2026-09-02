"""Metricas de recuperacion con intervalos de confianza bootstrap.

Dos reglas que aqui no se negocian:

1. Las metricas se calculan sobre control_id DEDUPLICADO. Sin esto, una
   estrategia que emite varios chunks por control acumula ganancia repetida y
   produce nDCG mayor que 1.0.
2. Todo punto va acompanado de su IC bootstrap. Con n=30 casos respondibles el
   IC95 de recall@5 mide ~0.30 de ancho: publicar el punto solo invita a leer
   como diferencia real lo que es ruido de muestreo.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from math import log2
from typing import Any

import numpy as np

from ..config import Config
from .golden import GoldenCase


def _unique(retrieved: Sequence[str]) -> list[str]:
    """Colapsa repeticiones conservando el rango de la primera aparicion.

    Las metricas deduplican por su cuenta ademas de hacerlo el retriever. Un
    nDCG que puede superar 1.0 si le llegan duplicados es una trampa: asi es
    exactamente como la version anterior de este repo publico ndcg@10 = 1.206.
    """
    seen: set[str] = set()
    out: list[str] = []
    for control_id in retrieved:
        if control_id not in seen:
            seen.add(control_id)
            out.append(control_id)
    return out


def recall_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    return float(bool(set(_unique(retrieved)[:k]) & relevant))


def reciprocal_rank(retrieved: Sequence[str], relevant: set[str]) -> float:
    for position, control_id in enumerate(_unique(retrieved), start=1):
        if control_id in relevant:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: set[str], k: int) -> float:
    """nDCG binario. El IDCG usa min(|relevantes|, k) para que el maximo sea 1.0."""
    dcg = sum(
        1.0 / log2(position + 1)
        for position, control_id in enumerate(_unique(retrieved)[:k], start=1)
        if control_id in relevant
    )
    idcg = sum(1.0 / log2(position + 1) for position in range(1, min(len(relevant), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass(slots=True)
class RetrievalMetrics:
    n: int
    point: dict[str, float]
    ci: dict[str, tuple[float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n": self.n,
            **{k: round(v, 4) for k, v in self.point.items()},
            "ci95": {k: [round(lo, 4), round(hi, 4)] for k, (lo, hi) in self.ci.items()},
        }


def bootstrap_ci(
    per_case: Sequence[float], resamples: int, confidence: float, seed: int
) -> tuple[float, float]:
    """IC percentil por bootstrap sobre los valores por caso."""
    values = np.asarray(per_case, dtype=float)
    if values.size == 0:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True).mean(axis=1)
    tail = (1.0 - confidence) / 2.0 * 100.0
    return float(np.percentile(draws, tail)), float(np.percentile(draws, 100.0 - tail))


def evaluate_retrieval(
    cases: Sequence[GoldenCase],
    rank_fn: Callable[[GoldenCase], list[str]],
    config: Config,
) -> RetrievalMetrics:
    """Evalua `rank_fn` sobre los casos que tienen ground truth."""
    scorable = [c for c in cases if c.scorable_for_retrieval]
    ks: list[int] = config.get("evaluation.metrics.recall_at_k")
    ndcg_k: int = config.get("evaluation.metrics.ndcg_at_k")

    per_case: dict[str, list[float]] = {f"recall@{k}": [] for k in ks}
    per_case["mrr"] = []
    per_case[f"ndcg@{ndcg_k}"] = []

    for case in scorable:
        retrieved = rank_fn(case)
        relevant = case.relevant
        for k in ks:
            per_case[f"recall@{k}"].append(recall_at_k(retrieved, relevant, k))
        per_case["mrr"].append(reciprocal_rank(retrieved, relevant))
        per_case[f"ndcg@{ndcg_k}"].append(ndcg_at_k(retrieved, relevant, ndcg_k))

    resamples = config.get("evaluation.bootstrap.resamples")
    confidence = config.get("evaluation.bootstrap.confidence")
    seed = config.get("evaluation.bootstrap.seed")

    point = {name: float(np.mean(vals)) if vals else 0.0 for name, vals in per_case.items()}
    ci = {
        name: bootstrap_ci(vals, resamples, confidence, seed) for name, vals in per_case.items()
    }
    return RetrievalMetrics(n=len(scorable), point=point, ci=ci)
