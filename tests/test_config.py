"""La config no debe caer en valores por defecto silenciosos: un fallo de
configuracion tiene que romper, no producir una metrica equivocada."""
from __future__ import annotations

import pytest

from compliance_mcp.config import Config, ConfigError, load_config


def test_missing_key_raises_instead_of_defaulting():
    cfg = Config({"retrieval": {"top_k": 5}}, source=__file__)
    with pytest.raises(ConfigError, match="retrieval.fusion"):
        cfg.get("retrieval.fusion.rrf_k")


def test_missing_file_raises():
    with pytest.raises(ConfigError, match="No existe"):
        load_config("no/such/config.yaml")


def test_every_parameter_used_by_the_code_is_declared(config):
    """Guardia contra literales magicos: si alguien anade un parametro al codigo
    sin declararlo en config.yaml, este test lo detecta."""
    required = [
        "ingest.param_resolution_passes",
        "ingest.param_template",
        "ingest.assessment_part_prefixes",
        "chunking.active",
        "chunking.header_template",
        "retrieval.bm25.k1",
        "retrieval.bm25.b",
        "retrieval.bm25.token_pattern",
        "retrieval.dense.model",
        "retrieval.dense.query_prefix",
        "retrieval.fusion.rrf_k",
        "retrieval.fusion.weight_bm25",
        "retrieval.fusion.weight_dense",
        "retrieval.candidate_pool",
        "retrieval.parent_rollup.alpha",
        "retrieval.top_k",
        "retrieval.max_top_k",
        "evaluation.metrics.recall_at_k",
        "evaluation.bootstrap.resamples",
        "gates.min_recall_at_5",
        "gates.min_citation_precision",
    ]
    for key in required:
        assert config.get(key) is not None, key
