from __future__ import annotations

import pytest

from compliance_mcp.config import load_config
from compliance_mcp.ingest import build_records


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def records(config):
    return build_records(config)


@pytest.fixture(scope="session")
def records_by_id(records):
    return {r.control_id: r for r in records}


@pytest.fixture(scope="session")
def retriever(config):
    """Indice lexico sobre el corpus real, sin modelo de embeddings.

    Los tests que ejercitan generacion, politica y servidor no necesitan el
    indice denso: lo que verifican es el camino de datos, no la calidad del
    ranking. Asi la suite corre sin torch y sin descargar un modelo.
    """
    from compliance_mcp.retrieval.search import Retriever

    return Retriever.build(config, with_dense=False)


@pytest.fixture(scope="session")
def lexical_config(config):
    """Config identica a la real salvo que sirve con BM25.

    Permite ejercitar el camino completo (servidor incluido) sin indice denso ni
    torch. Se sobreescribe el parametro en vez de hacer que el buscador caiga a
    lexico por su cuenta: un fallback silencioso convertiria un indice que falta
    en produccion en un resultado peor sin que nadie se entere.
    """
    import copy

    from compliance_mcp.config import Config

    data = copy.deepcopy(config.as_dict())
    data["retrieval"]["method"] = "bm25"
    return Config(data, config.source)


@pytest.fixture
def empty_context():
    """Contexto vacio: los tests de proveedor no ejercitan la recuperacion."""
    from compliance_mcp.generation.context import AnswerContext

    return AnswerContext()


@pytest.fixture(scope="session")
def indexed_repo(config):
    """Salta si el repo no tiene indice denso construido.

    El job rapido de CI corre `ingest` pero no `index` a proposito, para no
    descargar torch. Las comprobaciones que necesitan el .npy real se saltan
    ahi y se ejercitan en el job de evaluacion, que si lo construye.
    """
    from compliance_mcp.index_manifest import read_manifest

    strategy = config.get("chunking.active")
    if strategy not in read_manifest(config).get("entries", {}):
        pytest.skip("no hay indice denso construido (`make index`)")
    return strategy
