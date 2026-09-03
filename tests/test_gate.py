"""El gate debe fallar de verdad cuando la calidad cae."""
from __future__ import annotations

from compliance_mcp.eval.gate import check


def _results(recall: float, strategy: str = "C") -> dict:
    """Resultados SIN procedencia. Los tests de umbral la anaden aparte, para
    que el fallo por procedencia y el fallo por calidad no se confundan."""
    return {"split": "test", "grid": {strategy: {"hybrid": {"recall@5": recall, "n": 30}}}}


def _with_provenance(results: dict, config) -> dict:
    from compliance_mcp.provenance import provenance_block

    results["provenance"] = provenance_block(config)
    return results


def test_gate_passes_above_threshold(config):
    threshold = config.get("gates.min_recall_at_5")
    results = _with_provenance(_results(threshold + 0.05, config.get("chunking.active")), config)
    assert check(results, config) == []


def test_gate_fails_below_threshold(config):
    threshold = config.get("gates.min_recall_at_5")
    results = _with_provenance(_results(threshold - 0.05, config.get("chunking.active")), config)
    failures = check(results, config)
    assert len(failures) == 1
    assert "recall@5" in failures[0]


def test_gate_fails_when_the_evaluated_cell_is_missing(config):
    assert check(_with_provenance({"grid": {}}, config), config) != []


def test_gate_flags_hallucinated_citations(config):
    results = _with_provenance(_results(1.0, config.get("chunking.active")), config)
    results["generation"] = {"citation_precision": 1.0, "hallucinated_citation_rate": 0.02}
    failures = check(results, config)
    assert any("hallucinated" in f for f in failures)


def test_gate_flags_low_citation_precision(config):
    results = _with_provenance(_results(1.0, config.get("chunking.active")), config)
    results["generation"] = {"citation_precision": 0.5, "hallucinated_citation_rate": 0.0}
    failures = check(results, config)
    assert any("citation_precision" in f for f in failures)


def test_gate_rejects_a_generation_block_produced_by_the_baseline(config):
    """El baseline extractivo copia: 1.0 de precision y 0.0 de alucinacion son
    propiedades de copiar, no evidencia sobre el generador que se sirve."""
    results = _results(1.0, config.get("chunking.active"))
    results["generation"] = {
        "provider": config.get("generation.baseline_provider"),
        "citation_precision": 1.0,
        "hallucinated_citation_rate": 0.0,
    }
    failures = check(results, config)
    assert any("baseline" in f for f in failures)


def test_gate_accepts_a_generation_block_from_a_real_provider(config):
    results = _with_provenance(_results(1.0, config.get("chunking.active")), config)
    results["generation"] = {
        "provider": "anthropic",
        "citation_precision": 1.0,
        "hallucinated_citation_rate": 0.0,
    }
    assert check(results, config) == []


def _provenanced(config, recall: float = 1.0) -> dict:
    """Resultados con procedencia coherente con el arbol actual."""
    return _with_provenance(_results(recall, config.get("chunking.active")), config)


def test_gate_rejects_results_without_provenance(config):
    """Sin procedencia no se sabe de que corpus salieron: podrian ser de una
    corrida anterior commiteada, y el gate pasaria sin medir nada."""
    failures = check(_results(1.0, config.get("chunking.active")), config)
    assert any("procedencia" in f for f in failures)


def test_gate_rejects_results_from_another_corpus(config):
    results = _provenanced(config)
    results["provenance"]["corpus_digest"] = "sha256:" + "0" * 64
    failures = check(results, config)
    assert any("otro corpus" in f for f in failures)


def test_gate_rejects_results_from_another_configuration(config):
    results = _provenanced(config)
    results["provenance"]["config_digest"] = "sha256:" + "0" * 64
    failures = check(results, config)
    assert any("otra configuracion" in f for f in failures)


def test_gate_accepts_freshly_produced_results(config):
    assert check(_provenanced(config), config) == []
