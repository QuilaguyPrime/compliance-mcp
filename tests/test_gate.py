"""El gate debe fallar de verdad cuando la calidad cae."""
from __future__ import annotations

from compliance_mcp.eval.gate import check


def _results(recall: float, strategy: str = "C") -> dict:
    return {"split": "test", "grid": {strategy: {"hybrid": {"recall@5": recall, "n": 30}}}}


def test_gate_passes_above_threshold(config):
    threshold = config.get("gates.min_recall_at_5")
    assert check(_results(threshold + 0.05, config.get("chunking.active")), config) == []


def test_gate_fails_below_threshold(config):
    threshold = config.get("gates.min_recall_at_5")
    failures = check(_results(threshold - 0.05, config.get("chunking.active")), config)
    assert len(failures) == 1
    assert "recall@5" in failures[0]


def test_gate_fails_when_the_evaluated_cell_is_missing(config):
    assert check({"grid": {}}, config) != []


def test_gate_flags_hallucinated_citations(config):
    results = _results(1.0, config.get("chunking.active"))
    results["generation"] = {"citation_precision": 1.0, "hallucinated_citation_rate": 0.02}
    failures = check(results, config)
    assert any("hallucinated" in f for f in failures)


def test_gate_flags_low_citation_precision(config):
    results = _results(1.0, config.get("chunking.active"))
    results["generation"] = {"citation_precision": 0.5, "hallucinated_citation_rate": 0.0}
    failures = check(results, config)
    assert any("citation_precision" in f for f in failures)
