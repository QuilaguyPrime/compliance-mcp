"""Recuperacion: fusion, roll-up, dedup, filtros y el bug de tokenizacion."""
from __future__ import annotations

import pytest

from compliance_mcp.chunking import chunk_records
from compliance_mcp.retrieval.fusion import (
    dedupe_by_control,
    parent_rollup,
    reciprocal_rank_fusion,
)
from compliance_mcp.retrieval.lexical import BM25Retriever
from compliance_mcp.retrieval.search import Retriever, SearchFilters


# ------------------------------------------------------------------ fusion RRF
def test_rrf_prefers_a_document_ranked_first_by_both_retrievers():
    fused = reciprocal_rank_fusion([[5, 1, 2], [5, 3, 4]], rrf_k=60, weights=[1.0, 1.0])
    assert fused[0] == 5


def test_rrf_is_convex_so_spread_ranks_beat_a_consistent_middle():
    """Propiedad real y poco intuitiva de RRF: 1/(k+r) es convexa, asi que un
    documento en posiciones 1 y 3 puntua por encima de uno que sale 2 y 2.
    Se fija en un test porque invita a escribir aserciones con premisa falsa.
    """
    fused = reciprocal_rank_fusion([[1, 2, 3], [3, 2, 1]], rrf_k=60, weights=[1.0, 1.0])
    assert fused[0] in (1, 3)   # los extremos
    assert fused[-1] == 2       # el consistentemente intermedio queda ultimo


def test_rrf_weight_zero_ignores_that_ranking():
    fused = reciprocal_rank_fusion([[9, 9, 9], [1, 2, 3]], rrf_k=60, weights=[0.0, 1.0])
    assert fused == [1, 2, 3]


def test_rrf_weights_shift_the_order():
    lists = [[10, 11], [11, 10]]
    assert reciprocal_rank_fusion(lists, 60, [1.0, 0.1])[0] == 10
    assert reciprocal_rank_fusion(lists, 60, [0.1, 1.0])[0] == 11


def test_rrf_rejects_mismatched_weights():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1], [2]], rrf_k=60, weights=[1.0])


# --------------------------------------------------------------- roll-up padre
def test_parent_rollup_surfaces_the_parent_of_a_retrieved_enhancement():
    parents = {"au-11.1": "au-11", "au-11": None, "au-4": None}
    out = parent_rollup(["au-11.1", "au-4"], parents, alpha=0.8)
    assert "au-11" in out


def test_parent_rollup_never_changes_the_top_hit():
    """Es un mecanismo de recall, no de precision: solo inserta por debajo."""
    parents = {"ac-2.3": "ac-2", "ac-2": None, "pe-8": None}
    ranked = ["ac-2.3", "pe-8"]
    assert parent_rollup(ranked, parents, alpha=0.8)[0] == ranked[0]


def test_parent_rollup_alpha_zero_is_a_noop_on_order():
    parents = {"ac-2.3": "ac-2", "ac-2": None}
    assert parent_rollup(["ac-2.3"], parents, alpha=0.0)[0] == "ac-2.3"


# ----------------------------------------------------------------------- dedup
def test_dedupe_keeps_first_occurrence_rank():
    assert dedupe_by_control(["ac-2", "ac-2", "ac-3", "ac-2", "ac-4"], 3) == ["ac-2", "ac-3", "ac-4"]


def test_dedupe_respects_limit():
    assert len(dedupe_by_control([f"c-{i}" for i in range(50)], 5)) == 5


# --------------------------------------------------------------- tokenizacion
def test_query_and_corpus_share_one_tokenizer(config):
    """El bug que hundio BM25 en la version anterior: el corpus se tokenizaba
    con regex y la consulta con str.split(), asi que 'Information'?' nunca
    hacia match. recall@5 caia de 0.711 a 0.111."""
    retriever = BM25Retriever(["access control policy and procedures"], config)
    query_tokens = retriever.tokenize("Which control covers 'Access Control'?")
    # La puntuacion y las mayusculas se normalizan igual en consulta y documento.
    assert "access" in query_tokens and "control" in query_tokens
    assert not any("'" in t or "?" in t for t in query_tokens)
    doc_tokens = retriever.tokenize("Access Control policy and procedures")
    assert set(query_tokens) & set(doc_tokens) == {"access", "control"}
    # Las stopwords de config.yaml se aplican a ambos lados.
    assert "and" not in doc_tokens


def test_bm25_finds_a_lexically_matching_document(config):
    texts = ["media sanitization of digital media", "emergency lighting in facilities"]
    retriever = BM25Retriever(texts, config)
    assert retriever.rank("media sanitization", pool=2)[0] == 0


# -------------------------------------------------------------------- filtros
def test_filters_exclude_withdrawn_by_default(records_by_id):
    filters = SearchFilters(include_withdrawn=False)
    assert filters.matches(records_by_id["ac-2"]) is True
    assert filters.matches(records_by_id["ac-3.1"]) is False


def test_filters_by_family_and_baseline_and_kind(records_by_id):
    ac2 = records_by_id["ac-2"]
    assert SearchFilters(family="ac").matches(ac2) is True
    assert SearchFilters(family="sc").matches(ac2) is False
    assert SearchFilters(baseline="low").matches(ac2) is True
    assert SearchFilters(kind="enhancement").matches(ac2) is False
    assert SearchFilters(kind="enhancement").matches(records_by_id["ac-2.1"]) is True


def test_baseline_filter_excludes_controls_outside_the_baseline(records_by_id):
    # CM-4(1) Separate Test Environments solo esta en HIGH.
    cm41 = records_by_id["cm-4.1"]
    assert SearchFilters(baseline="high").matches(cm41) is True
    assert SearchFilters(baseline="low").matches(cm41) is False


# ------------------------------------------------------- integracion sin denso
def test_bm25_only_retriever_end_to_end(records, config):
    """Se puede recuperar sin el indice denso, para que los tests corran en CI
    sin descargar un modelo de embeddings."""
    chunks = chunk_records(records, "A", config)
    retriever = Retriever(records, chunks, config, embeddings=None)
    hits = retriever.rank_control_ids("media sanitization", method="bm25", limit=5)
    assert "mp-6" in hits


def test_hybrid_requires_a_dense_index(records, config):
    chunks = chunk_records(records, "A", config)
    retriever = Retriever(records, chunks, config, embeddings=None)
    with pytest.raises(RuntimeError, match="denso"):
        retriever.rank_control_ids("anything", method="hybrid", limit=5)


def test_unknown_method_is_rejected(records, config):
    chunks = chunk_records(records, "A", config)
    retriever = Retriever(records, chunks, config, embeddings=None)
    with pytest.raises(ValueError, match="Metodo desconocido"):
        retriever.rank_control_ids("q", method="magic", limit=5)


# ----------------------------------------------- search() end-to-end sin denso
def test_search_returns_ranked_hits_with_metadata(records, config):
    from compliance_mcp.chunking import chunk_records

    retriever = Retriever(records, chunk_records(records, "A", config), config, embeddings=None)
    hits = retriever.search("media sanitization", top_k=5, method="bm25")
    assert len(hits) == 5
    assert [h.rank for h in hits] == [1, 2, 3, 4, 5]
    assert all(h.score > 0 for h in hits)

    by_id = {h.control_id: h for h in hits}
    assert "mp-6" in by_id, f"MP-6 deberia estar en el top-5, salio {list(by_id)}"
    mp6 = by_id["mp-6"]
    assert mp6.label == "MP-6"
    assert mp6.family_id == "mp"
    assert mp6.kind == "control"
    assert mp6.snippet
    assert "moderate" in mp6.baselines


def test_search_rejects_top_k_outside_the_configured_range(records, config):
    from compliance_mcp.chunking import chunk_records

    retriever = Retriever(records, chunk_records(records, "A", config), config, embeddings=None)
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("q", top_k=config.get("retrieval.max_top_k") + 1, method="bm25")
    with pytest.raises(ValueError, match="top_k"):
        retriever.search("q", top_k=0, method="bm25")


def test_search_hides_withdrawn_controls_by_default(records, config):
    from compliance_mcp.chunking import chunk_records

    retriever = Retriever(records, chunk_records(records, "A", config), config, embeddings=None)
    hits = retriever.search("shared and group account credential change", top_k=10, method="bm25")
    assert all(h.status != "withdrawn" for h in hits)


def test_search_can_be_restricted_to_a_family(records, config):
    from compliance_mcp.chunking import chunk_records

    retriever = Retriever(records, chunk_records(records, "A", config), config, embeddings=None)
    hits = retriever.search(
        "policy and procedures", top_k=5, method="bm25", filters=SearchFilters(family="ir")
    )
    assert hits and all(h.family_id == "ir" for h in hits)


def test_search_can_be_restricted_to_a_baseline(records, config):
    from compliance_mcp.chunking import chunk_records

    retriever = Retriever(records, chunk_records(records, "A", config), config, embeddings=None)
    hits = retriever.search(
        "separate test environments", top_k=5, method="bm25",
        filters=SearchFilters(baseline="low"),
    )
    assert all("low" in h.baselines for h in hits)
