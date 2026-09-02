"""Chunking: contexto heredado y cobertura, medidos sobre el corpus real."""
from __future__ import annotations

from compliance_mcp.chunking import build_header, chunk_records


def test_strategy_c_covers_every_control_including_enhancements(records, config):
    """La estrategia B del repo anterior no recorria control['controls'], asi que
    ninguno de los 872 enhancements llegaba al indice y su recall estaba
    acotado a cero para cualquier pregunta sobre un enhancement."""
    chunks = chunk_records(records, "C", config)
    covered = {c.control_id for c in chunks}
    assert covered == {r.control_id for r in records}
    assert any("." in cid for cid in covered)  # hay enhancements


def test_every_strategy_covers_every_control(records, config):
    expected = {r.control_id for r in records}
    for strategy in ("A", "B", "C"):
        chunks = chunk_records(records, strategy, config)
        assert {c.control_id for c in chunks} == expected, strategy


def test_split_strategy_produces_more_chunks_than_control_level(records, config):
    a = chunk_records(records, "A", config)
    c = chunk_records(records, "C", config)
    assert len(a) == len(records)
    assert len(c) > len(a)


def test_enhancement_chunk_inherits_parent_context(records, config):
    chunks = {c.chunk_id: c for c in chunk_records(records, "C", config)}
    chunk = chunks["ac-2.1::statement"]
    assert "Account Management (AC-2)" in chunk.text   # padre
    assert "Access Control (AC)" in chunk.text          # familia
    assert "AC-2(1)" in chunk.text                      # etiqueta propia


def test_header_of_a_base_control_has_no_parent_segment(records_by_id, config):
    header = build_header(records_by_id["ac-2"], config)
    assert ">" in header
    assert "AC-2)" not in header.split(">")[0]


def test_strategy_b_includes_assessment_content(records, config):
    a = {c.control_id: c for c in chunk_records(records, "A", config)}
    b = {c.control_id: c for c in chunk_records(records, "B", config)}
    assert len(b["ac-2"].text) > len(a["ac-2"].text)


def test_withdrawn_control_is_chunked_with_its_destination(records, config):
    chunks = {c.chunk_id: c for c in chunk_records(records, "C", config)}
    chunk = chunks["ac-3.1::withdrawn"]
    assert "withdrawn" in chunk.text.lower()
    assert "AC-6" in chunk.text  # AC-3(1) se incorporo a AC-6
