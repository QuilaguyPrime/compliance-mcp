"""El manifiesto del indice y el fallo silencioso que existe para evitar.

El caso que motiva todo esto: cambiar una plantilla de chunking cambia el TEXTO
de los chunks pero no su NUMERO, asi que el .npy viejo sigue cargando con las
formas correctas y cada vector pasa a corresponder a otro texto. Sin esta
comprobacion no hay excepcion, no hay aviso y la evaluacion publica numeros de
un indice que no es el que se sirve.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from compliance_mcp.chunking import chunk_records
from compliance_mcp.config import Config
from compliance_mcp.index_manifest import (
    StaleIndexError,
    build_entry,
    check_entry,
    read_manifest,
    verify,
)


@pytest.fixture(scope="module")
def strategy(config):
    return config.get("chunking.active")


@pytest.fixture(scope="module")
def chunks(config, records, strategy):
    return chunk_records(records, strategy, config)


def variant(config, mutate) -> Config:
    data = copy.deepcopy(config.as_dict())
    mutate(data)
    return Config(data, config.source)


@pytest.fixture
def indexed(config, chunks, strategy, tmp_path):
    """Config cuyo indice vive en tmp_path, con manifiesto recien escrito.

    Hermetico a proposito: no depende de que el repo tenga el .npy construido,
    que es justo lo que el job rapido de CI no hace.
    """
    from compliance_mcp.index_manifest import write_entry

    scoped = variant(config, lambda d: d["corpus"].__setitem__("index_dir", str(tmp_path)))
    entry = build_entry(scoped, strategy, chunks, DIM, tmp_path / "emb.npy")
    write_entry(scoped, entry)
    return scoped, entry


DIM = 768


def test_el_indice_construido_corresponde_al_corpus_actual(config, chunks, strategy, indexed_repo):
    """Sobre el indice real del repo: si falla, esta caducado (`make ingest index`)."""
    entry = read_manifest(config)["entries"][strategy]
    assert check_entry(config, strategy, chunks, rows=entry["chunks"], dim=entry["dim"]) == []


def test_cambiar_el_texto_de_los_chunks_caduca_el_indice(indexed, records, strategy):
    """El numero de chunks no cambia, solo su texto: es justo el caso que las
    comprobaciones de forma no detectan."""
    scoped, entry = indexed
    mutated = variant(
        scoped, lambda d: d["chunking"].__setitem__("header_template", "TEXTO DISTINTO {label}")
    )
    stale = chunk_records(records, strategy, mutated)
    assert len(stale) == entry.chunks  # misma forma...

    problems = check_entry(mutated, strategy, stale, rows=entry.chunks, dim=entry.dim)
    assert any("texto de los chunks cambio" in p for p in problems)


def test_cambiar_el_modelo_de_embeddings_caduca_el_indice(indexed, chunks, strategy):
    scoped, entry = indexed
    mutated = variant(scoped, lambda d: d["retrieval"]["dense"].__setitem__("model", "otro/modelo"))
    problems = check_entry(mutated, strategy, chunks, rows=entry.chunks, dim=entry.dim)
    assert any("otro/modelo" in p for p in problems)


def test_un_numero_de_vectores_distinto_se_detecta(indexed, chunks, strategy):
    scoped, entry = indexed
    problems = check_entry(scoped, strategy, chunks, rows=entry.chunks - 1, dim=entry.dim)
    assert any("vectores" in p for p in problems)


def test_sin_manifiesto_no_se_confia_en_el_indice(config, chunks, tmp_path):
    """Un indice sin manifiesto puede ser de cualquier version anterior. No se
    puede saber, asi que no se usa."""
    mutated = variant(config, lambda d: d["corpus"].__setitem__("index_dir", str(tmp_path)))
    problems = check_entry(mutated, "C", chunks, rows=len(chunks), dim=768)
    assert any("No hay entrada de manifiesto" in p for p in problems)


def test_verify_lanza_con_un_mensaje_accionable(indexed, records, strategy):
    scoped, entry = indexed
    mutated = variant(
        scoped, lambda d: d["chunking"].__setitem__("header_template", "TEXTO DISTINTO {label}")
    )
    stale = chunk_records(records, strategy, mutated)
    fake = np.zeros((entry.chunks, entry.dim), dtype="float32")
    with pytest.raises(StaleIndexError, match="make ingest index"):
        verify(mutated, strategy, stale, fake)


def test_la_entrada_registra_de_donde_salio(config, chunks, strategy, tmp_path):
    entry = build_entry(config, strategy, chunks, DIM, tmp_path / "emb.npy")
    payload = json.loads(json.dumps(entry.to_dict()))  # serializable
    assert payload["chunks"] == len(chunks)
    assert payload["model"] == config.get("retrieval.dense.model")
    assert payload["chunks_digest"].startswith("sha256:")
