"""Metricas y golden set. Cubre el bug de nDCG>1 que tenia el repo anterior."""
from __future__ import annotations

import pytest

from compliance_mcp.eval.golden import (
    load_golden_set,
    split_cases,
    validate_against_corpus,
)
from compliance_mcp.eval.metrics import (
    bootstrap_ci,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


# --------------------------------------------------------------------- metricas
def test_recall_is_binary_hit_at_k():
    assert recall_at_k(["a", "b", "c"], {"c"}, 3) == 1.0
    assert recall_at_k(["a", "b", "c"], {"c"}, 2) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["x", "y", "a"], {"a", "y"}) == pytest.approx(1 / 2)
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_never_exceeds_one():
    """El eval anterior publicaba ndcg@10 = 1.206, que es imposible: sumaba
    ganancia varias veces porque no deduplicaba por control_id."""
    perfect = ndcg_at_k(["a", "b"], {"a", "b"}, 10)
    assert perfect == pytest.approx(1.0)
    for retrieved, relevant in [
        (["a", "a", "a"], {"a"}),
        (["a", "b", "c", "d"], {"a", "b", "c", "d"}),
        (["a"] * 10, {"a", "b"}),
    ]:
        assert 0.0 <= ndcg_at_k(retrieved, relevant, 10) <= 1.0


def test_ndcg_rewards_higher_positions():
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 10) > ndcg_at_k(["x", "y", "a"], {"a"}, 10)


def test_ndcg_with_no_relevant_documents_is_zero():
    assert ndcg_at_k(["a", "b"], set(), 10) == 0.0


def test_bootstrap_ci_brackets_the_mean_and_is_deterministic():
    values = [1.0] * 17 + [0.0] * 5
    lo, hi = bootstrap_ci(values, resamples=2000, confidence=0.95, seed=1)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    assert (lo, hi) == bootstrap_ci(values, resamples=2000, confidence=0.95, seed=1)


def test_bootstrap_ci_narrows_with_more_samples():
    small = bootstrap_ci([1.0] * 17 + [0.0] * 5, 5000, 0.95, 1)
    large = bootstrap_ci([1.0] * 170 + [0.0] * 50, 5000, 0.95, 1)
    assert (large[1] - large[0]) < (small[1] - small[0])


# ------------------------------------------------------------------ golden set
def test_golden_set_composition(config):
    cases = load_golden_set(config)
    assert len(cases) == 60
    assert sum(1 for c in cases if c.type == "answerable") == 30
    assert sum(1 for c in cases if c.type == "unanswerable") == 15
    assert sum(1 for c in cases if c.type == "adversarial") == 15


def test_answerable_cases_are_stratified_by_style(config):
    cases = [c for c in load_golden_set(config) if c.type == "answerable"]
    styles = {}
    for case in cases:
        styles[case.style] = styles.get(case.style, 0) + 1
    assert styles == {"paraphrase": 10, "lexical": 10, "multi_control": 10}


def test_every_ground_truth_control_exists_in_the_corpus(config, records):
    """Un ground truth inventado es un fallo del golden set, no del sistema."""
    cases = load_golden_set(config)
    errors = validate_against_corpus(cases, {r.control_id for r in records})
    assert errors == []


def test_unanswerable_cases_have_no_ground_truth(config):
    for case in load_golden_set(config):
        if case.must_refuse:
            assert case.expected_control_ids == []


def test_refusal_cases_are_excluded_from_retrieval_scoring(config):
    """Meter los casos de rehuso en el denominador de recall castigaria al
    sistema justo por hacer lo correcto."""
    cases = load_golden_set(config)
    scorable = [c for c in cases if c.scorable_for_retrieval]
    assert all(not c.must_refuse for c in scorable)
    assert len(scorable) == sum(1 for c in cases if c.expected_control_ids)


def test_split_is_deterministic_and_disjoint(config):
    cases = load_golden_set(config)
    train_a, test_a = split_cases(cases, config)
    train_b, _ = split_cases(list(reversed(cases)), config)
    assert {c.id for c in train_a} == {c.id for c in train_b}
    assert {c.id for c in train_a} & {c.id for c in test_a} == set()
    assert len(train_a) + len(test_a) == len(cases)


def test_split_puts_cases_on_both_sides(config):
    train, test = split_cases(load_golden_set(config), config)
    assert len(train) > 5 and len(test) > 5
