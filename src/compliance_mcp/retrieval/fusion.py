"""Fusion RRF y roll-up al control padre."""
from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]], rrf_k: int, weights: Sequence[float]
) -> list[int]:
    """RRF ponderado: score(d) = sum_i w_i / (rrf_k + rank_i(d)).

    Los pesos importan. Con pesos 1:1 el hibrido quedo por debajo del denso solo
    en las 30 configuraciones barridas en fase 1, porque BM25 tenia voto igual
    pese a ser mucho mas debil sobre consultas parafraseadas.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("Hacen falta tantos pesos como listas de ranking")
    scores: dict[int, float] = {}
    for weight, ranking in zip(weights, ranked_lists):
        if weight == 0:
            continue
        for position, doc_index in enumerate(ranking, start=1):
            scores[doc_index] = scores.get(doc_index, 0.0) + weight / (rrf_k + position)
    return [doc for doc, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def parent_rollup(
    control_ids: Sequence[str], parent_of: dict[str, str | None], alpha: float
) -> list[str]:
    """Si un enhancement entra en el ranking, su control padre recibe alpha*credito.

    Motivo: el catalogo tiene 872 enhancements contra 324 controles base. Los
    enhancements son cortos y especificos, asi que copan el top-k y desplazan al
    padre (ejemplo real: AU-11(1) entra y AU-11 no aparece).

    Es un mecanismo de recall, no de precision: solo puede insertar padres por
    debajo de un acierto que ya existia, nunca cambia la posicion 1.
    """
    scores: dict[str, float] = {}
    for position, control_id in enumerate(control_ids, start=1):
        credit = 1.0 / position
        scores[control_id] = max(scores.get(control_id, 0.0), credit)
        parent = parent_of.get(control_id)
        if parent:
            scores[parent] = max(scores.get(parent, 0.0), alpha * credit)
    return [cid for cid, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def dedupe_by_control(control_ids: Sequence[str], limit: int) -> list[str]:
    """Colapsa a control_id unico; la primera aparicion fija el rango.

    Imprescindible cuando un control genera varios chunks: sin esto una
    estrategia con mas chunks por control acumula ganancia repetida y produce
    nDCG por encima de 1.0, que es exactamente el sintoma que tenia el eval
    anterior de este repo.
    """
    seen: set[str] = set()
    out: list[str] = []
    for control_id in control_ids:
        if control_id in seen:
            continue
        seen.add(control_id)
        out.append(control_id)
        if len(out) >= limit:
            break
    return out
