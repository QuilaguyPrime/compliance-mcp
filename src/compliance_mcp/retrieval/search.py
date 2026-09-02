"""Buscador: orquesta BM25 + denso + RRF + roll-up + filtros de metadatos."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..chunking import Chunk, chunk_records
from ..config import Config
from ..ingest import ControlRecord, read_records
from ..observability import StageTimings, stage
from .dense import DenseRetriever
from .fusion import dedupe_by_control, parent_rollup, reciprocal_rank_fusion
from .lexical import BM25Retriever

METHODS = ("bm25", "dense", "hybrid")


@dataclass(slots=True)
class SearchFilters:
    """Filtrado por metadatos. Se aplica ANTES de fusionar para no gastar el
    presupuesto de top_k en documentos que el usuario ya excluyo."""

    family: str | None = None
    baseline: str | None = None
    kind: str | None = None            # "control" | "enhancement"
    include_withdrawn: bool = False

    def matches(self, record: ControlRecord) -> bool:
        if not self.include_withdrawn and record.status == "withdrawn":
            return False
        if self.family and record.family_id != self.family.lower():
            return False
        if self.baseline and self.baseline.lower() not in record.baselines:
            return False
        if self.kind and record.kind != self.kind:
            return False
        return True


@dataclass(slots=True)
class SearchHit:
    control_id: str
    label: str
    title: str
    family_id: str
    kind: str
    status: str
    score: float
    rank: int
    snippet: str
    baselines: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "label": self.label,
            "title": self.title,
            "family_id": self.family_id,
            "kind": self.kind,
            "status": self.status,
            "score": round(self.score, 6),
            "rank": self.rank,
            "snippet": self.snippet,
            "baselines": self.baselines,
        }


class Retriever:
    """Indice hibrido en memoria sobre una estrategia de chunking.

    Se construye una vez y se reutiliza. Reconstruirlo por peticion (como hacia
    la version anterior) implica re-embeber todo el corpus en cada consulta.
    """

    def __init__(
        self,
        records: list[ControlRecord],
        chunks: list[Chunk],
        config: Config,
        embeddings: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.records = {r.control_id: r for r in records}
        self.chunks = chunks
        self.chunk_control_ids = [c.control_id for c in chunks]
        self.parent_of = {r.control_id: r.parent_id for r in records}
        self._pool = config.get("retrieval.candidate_pool")
        self._rrf_k = config.get("retrieval.fusion.rrf_k")
        self._w_bm25 = config.get("retrieval.fusion.weight_bm25")
        self._w_dense = config.get("retrieval.fusion.weight_dense")
        self._rollup_enabled = config.get("retrieval.parent_rollup.enabled")
        self._rollup_alpha = config.get("retrieval.parent_rollup.alpha")
        self.bm25 = BM25Retriever([c.text for c in chunks], config)
        self.dense = DenseRetriever(embeddings, config) if embeddings is not None else None

    # ---------------------------------------------------------------- fabrica
    @classmethod
    def build(
        cls,
        config: Config,
        strategy: str | None = None,
        *,
        with_dense: bool = True,
    ) -> Retriever:
        strategy = strategy or config.get("chunking.active")
        records = read_records(config.path("corpus.records_path"))
        chunks = chunk_records(records, strategy, config)
        embeddings = None
        if with_dense:
            embeddings = DenseRetriever.from_file(
                embeddings_path(config, strategy), config
            )._embeddings
        return cls(records, chunks, config, embeddings)

    # --------------------------------------------------------------- busqueda
    def rank_control_ids(
        self,
        query: str,
        *,
        method: str = "hybrid",
        limit: int = 10,
        filters: SearchFilters | None = None,
        weight_bm25: float | None = None,
        rrf_k: int | None = None,
        rollup_alpha: float | None = None,
        timings: StageTimings | None = None,
    ) -> list[str]:
        """Devuelve control_ids unicos ordenados. Nucleo compartido por la
        herramienta MCP y por el arnes de evaluacion, para que ambos midan
        exactamente el mismo camino de codigo."""
        if method not in METHODS:
            raise ValueError(f"Metodo desconocido: {method}. Opciones: {METHODS}")

        w_bm25 = self._w_bm25 if weight_bm25 is None else weight_bm25
        alpha = self._rollup_alpha if rollup_alpha is None else rollup_alpha
        k = self._rrf_k if rrf_k is None else rrf_k

        rankings: list[list[int]] = []
        weights: list[float] = []

        if method in ("bm25", "hybrid"):
            with _maybe_stage(timings, "retrieval.bm25"):
                rankings.append(self.bm25.rank(query, self._pool))
            weights.append(w_bm25 if method == "hybrid" else 1.0)

        if method in ("dense", "hybrid"):
            if self.dense is None:
                raise RuntimeError("El indice denso no esta cargado; construyelo con `make index`.")
            with _maybe_stage(timings, "retrieval.dense"):
                rankings.append(self.dense.rank(query, self._pool))
            weights.append(self._w_dense if method == "hybrid" else 1.0)

        with _maybe_stage(timings, "retrieval.fusion"):
            fused = reciprocal_rank_fusion(rankings, k, weights)[: self._pool]
            control_ids = [self.chunk_control_ids[i] for i in fused]

            if filters is not None:
                control_ids = [c for c in control_ids if filters.matches(self.records[c])]

            if self._rollup_enabled and alpha > 0:
                control_ids = parent_rollup(control_ids, self.parent_of, alpha)
                if filters is not None:
                    # El roll-up puede introducir un padre que el filtro excluye.
                    control_ids = [c for c in control_ids if filters.matches(self.records[c])]

            return dedupe_by_control(control_ids, limit)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: SearchFilters | None = None,
        method: str = "hybrid",
        timings: StageTimings | None = None,
    ) -> list[SearchHit]:
        top_k = self.config.get("retrieval.top_k") if top_k is None else top_k
        max_top_k = self.config.get("retrieval.max_top_k")
        if not 1 <= top_k <= max_top_k:
            raise ValueError(f"top_k debe estar entre 1 y {max_top_k}, recibido {top_k}")
        if filters is None:
            filters = SearchFilters(
                include_withdrawn=self.config.get("retrieval.default_include_withdrawn")
            )
        control_ids = self.rank_control_ids(
            query, method=method, limit=top_k, filters=filters, timings=timings
        )
        hits: list[SearchHit] = []
        for rank, control_id in enumerate(control_ids, start=1):
            record = self.records[control_id]
            hits.append(
                SearchHit(
                    control_id=record.control_id,
                    label=record.label,
                    title=record.title,
                    family_id=record.family_id,
                    kind=record.kind,
                    status=record.status,
                    score=1.0 / rank,
                    rank=rank,
                    snippet=self.snippet(record),
                    baselines=record.baselines,
                )
            )
        return hits

    def snippet(self, record: ControlRecord) -> str:
        if record.status == "withdrawn":
            targets = ", ".join(t.upper() for t in record.incorporated_into) or "n/a"
            return f"Retirado. Incorporado a: {targets}"
        return record.statement or record.guidance

    def control_ids(self) -> set[str]:
        return set(self.records)


def embeddings_path(config: Config, strategy: str) -> Path:
    model_slug = config.get("retrieval.dense.model").split("/")[-1]
    return config.path("corpus.index_dir") / f"emb_{strategy}_{model_slug}.npy"


class _NullStage:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


def _maybe_stage(timings: StageTimings | None, name: str):
    return stage(timings, name) if timings is not None else _NullStage()
