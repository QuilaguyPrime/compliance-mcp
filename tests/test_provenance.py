"""Fingerprints: que cambia el digest y que no.

Un fingerprint que cambia con cualquier cosa es ruido y se acaba ignorando; uno
que no cambia cuando deberia es peor, porque certifica algo falso.
"""
from __future__ import annotations

import copy

from compliance_mcp.config import Config
from compliance_mcp.provenance import (
    INDEX_CONFIG_KEYS,
    digest_config,
    digest_texts,
    provenance_block,
)


def variant(config, mutate) -> Config:
    data = copy.deepcopy(config.as_dict())
    mutate(data)
    return Config(data, config.source)


def test_el_digest_de_textos_es_estable(config):
    assert digest_texts(["a", "b"]) == digest_texts(["a", "b"])


def test_textos_distintos_dan_digests_distintos():
    assert digest_texts(["a", "b"]) != digest_texts(["a", "c"])


def test_la_frontera_entre_textos_cuenta():
    """Sin meter la longitud de cada texto en el hash, ['ab','c'] y ['a','bc']
    colisionarian: dos corpus distintos con el mismo fingerprint."""
    assert digest_texts(["ab", "c"]) != digest_texts(["a", "bc"])


def test_reordenar_el_yaml_no_cuenta_como_cambio(config):
    """Se comparan valores, no el orden en el fichero: si no, cualquier
    reformateo invalidaria el indice sin motivo."""
    reordered = variant(config, lambda d: d.__setitem__("chunking", dict(reversed(list(d["chunking"].items())))))
    assert digest_config(reordered, INDEX_CONFIG_KEYS) == digest_config(config, INDEX_CONFIG_KEYS)


def test_cambiar_el_chunking_cambia_el_digest(config):
    mutated = variant(config, lambda d: d["chunking"].__setitem__("header_template", "otra {label}"))
    assert digest_config(mutated, INDEX_CONFIG_KEYS) != digest_config(config, INDEX_CONFIG_KEYS)


def test_cambiar_un_umbral_del_gate_no_invalida_el_indice(config):
    """`gates:` es criterio de aceptacion, no productor de artefactos: tocarlo
    no debe obligar a reindexar."""
    mutated = variant(config, lambda d: d["gates"].__setitem__("min_recall_at_5", 0.99))
    assert digest_config(mutated, INDEX_CONFIG_KEYS) == digest_config(config, INDEX_CONFIG_KEYS)


def test_el_bloque_de_procedencia_identifica_la_corrida(config):
    block = provenance_block(config)
    assert block["corpus_digest"].startswith("sha256:")
    assert block["embedding_model"] == config.get("retrieval.dense.model")
    assert block["version"] and block["generated_at"].endswith("Z")
