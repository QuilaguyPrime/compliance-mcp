"""Ingest: las propiedades que la fase 1 demostro que hay que garantizar."""
from __future__ import annotations

import re

from compliance_mcp.ingest import PARAM_PATTERN, resolve_params

EXPECTED_TOTAL = 1196
EXPECTED_BASE = 324
EXPECTED_ENHANCEMENTS = 872
EXPECTED_WITHDRAWN = 182


def test_corpus_shape(records):
    assert len(records) == EXPECTED_TOTAL
    assert sum(1 for r in records if r.kind == "control") == EXPECTED_BASE
    assert sum(1 for r in records if r.kind == "enhancement") == EXPECTED_ENHANCEMENTS
    assert sum(1 for r in records if r.status == "withdrawn") == EXPECTED_WITHDRAWN


def test_control_ids_are_unique(records):
    ids = [r.control_id for r in records]
    assert len(set(ids)) == len(ids)


def test_no_unresolved_parameter_placeholders(records):
    """Los placeholders anidan dentro de select.choice. Con una sola pasada de
    sustitucion quedan 44 controles rotos; el ingest debe iterar a punto fijo."""
    offenders = [
        r.control_id for r in records if PARAM_PATTERN.search(r.statement + r.guidance)
    ]
    assert offenders == [], f"placeholders sin resolver en: {offenders[:10]}"


def test_nested_select_placeholder_is_resolved():
    """Reproduce el anidamiento real de AC-7: la etiqueta de un parametro
    contiene a su vez un placeholder, dentro de un select.choice."""
    param_map = {
        "p1": "lock the account for {{ insert: param, p2 }}",
        "p2": "time period",
    }
    text = "Automatically {{ insert: param, p1 }} when exceeded."
    out = resolve_params(text, param_map, passes=4, template="[{label}]")
    assert "{{ insert" not in out
    assert "time period" in out


def test_single_pass_would_leave_nesting():
    """Documenta por que param_resolution_passes > 1: con una pasada quedaban
    44 controles del catalogo real con placeholders crudos en el indice."""
    param_map = {"p1": "lock for {{ insert: param, p2 }}", "p2": "time period"}
    once = resolve_params("{{ insert: param, p1 }}", param_map, passes=1, template="[{label}]")
    assert "{{ insert" in once


def test_statement_hierarchy_labels_are_preserved(records_by_id):
    ac2 = records_by_id["ac-2"]
    assert re.search(r"^a\. ", ac2.statement, flags=re.MULTILINE)
    assert re.search(r"^b\. ", ac2.statement, flags=re.MULTILINE)


def test_enhancement_links_to_parent(records_by_id):
    enh = records_by_id["ac-2.1"]
    assert enh.kind == "enhancement"
    assert enh.parent_id == "ac-2"
    assert enh.parent_title == "Account Management"


def test_withdrawn_control_keeps_its_destination(records_by_id):
    wd = records_by_id["ac-3.1"]
    assert wd.status == "withdrawn"
    assert wd.incorporated_into  # sabe a donde se incorporo
    assert wd.statement == ""


def test_baselines_are_loaded_and_nested(records):
    by_baseline = {b: {r.control_id for r in records if b in r.baselines}
                   for b in ("low", "moderate", "high")}
    assert len(by_baseline["low"]) == 149
    assert len(by_baseline["moderate"]) == 287
    assert len(by_baseline["high"]) == 370
    assert by_baseline["low"] <= by_baseline["moderate"] <= by_baseline["high"]


def test_assessment_content_is_separated_from_normative_text(records_by_id):
    """El contenido SP 800-53A no debe contaminar statement/guidance: la
    ablacion decide si se indexa, y no se puede decidir si esta mezclado."""
    ac2 = records_by_id["ac-2"]
    assert ac2.assessment
    assert ac2.assessment not in ac2.statement
    assert ac2.assessment not in ac2.guidance
