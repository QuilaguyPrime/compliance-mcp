"""Herramientas MCP: contrato de entrada y salida de cada una."""
from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("mcp", reason="el servidor vive en el extra `serve`")

from compliance_mcp.generation.engine import AnswerEngine
from compliance_mcp.generation.providers import ExtractiveProvider, ProviderChain
from compliance_mcp.server import build_server


@pytest.fixture(scope="module")
def server(lexical_config, retriever):
    """Servidor real con el baseline extractivo: ejercita el cableado completo
    sin red, sin claves de API y sin indice denso."""
    engine = AnswerEngine(
        retriever, ProviderChain([ExtractiveProvider(lexical_config)]), lexical_config
    )
    return build_server(lexical_config, engine=engine)


def call(server, name: str, args: dict):
    result = asyncio.run(server.call_tool(name, args))
    assert not result.is_error, result.content
    return result.structured_content


def test_se_exponen_las_tres_herramientas(server):
    names = {tool.name for tool in asyncio.run(server.list_tools())}
    assert names == {"search_controls", "get_control", "answer_question"}


def test_search_controls_respeta_top_k(server):
    payload = call(server, "search_controls", {"query": "account management", "top_k": 3})
    assert payload["n_hits"] == 3
    assert [hit["rank"] for hit in payload["hits"]] == [1, 2, 3]


def test_search_controls_filtra_por_familia(server):
    payload = call(server, "search_controls", {"query": "incident", "family": "ir", "top_k": 5})
    assert payload["hits"]
    assert {hit["family_id"] for hit in payload["hits"]} == {"ir"}


def test_search_controls_oculta_los_retirados_por_defecto(server):
    payload = call(server, "search_controls", {"query": "access enforcement", "top_k": 10})
    assert all(hit["status"] == "active" for hit in payload["hits"])


def test_get_control_acepta_la_forma_humana_del_id(server):
    payload = call(server, "get_control", {"control_id": "AC-2(1)"})
    assert payload["found"]
    assert payload["control"]["control_id"] == "ac-2.1"
    assert payload["control"]["label"] == "AC-2(1)"


def test_get_control_lista_los_enhancements_del_padre(server):
    payload = call(server, "get_control", {"control_id": "ac-2"})
    labels = {e["label"] for e in payload["control"]["enhancements"]}
    assert "AC-2(1)" in labels


def test_get_control_devuelve_los_retirados_con_su_destino(server):
    """Rehusar con 'no existe' ante un control retirado es peor que explicar a
    donde se incorporo: es la pregunta tipica de quien viene de Rev 4."""
    payload = call(server, "get_control", {"control_id": "AC-3(1)"})
    assert payload["found"]
    assert payload["control"]["status"] == "withdrawn"
    assert payload["control"]["incorporated_into"]


def test_get_control_no_inventa_un_control_inexistente(server):
    payload = call(server, "get_control", {"control_id": "zz-99"})
    assert payload["found"] is False
    assert "no existe" in payload["error"]


def test_answer_question_devuelve_citas_verificadas(server):
    payload = call(server, "answer_question", {"question": "How long are audit records kept?"})
    assert payload["verification"]["verified"] == len(payload["citations"])
    for citation in payload["citations"]:
        assert citation["control_id"] and citation["label"] and citation["quote"]
    assert payload["trace_id"]


def call_expecting_error(server, name: str, args: dict) -> str:
    """mcp distingue el error que una herramienta lanza a proposito (ToolError,
    su mensaje llega a quien llamo) del que se le escapa (UnexpectedToolError,
    mensaje generico). Un argumento invalido tiene que ser lo primero: el
    mensaje ES la respuesta util."""
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError

    with pytest.raises(ToolError) as exc:
        asyncio.run(server.call_tool(name, args))
    assert not isinstance(exc.value, UnexpectedToolError)
    return str(exc.value)


def test_una_familia_invalida_devuelve_un_error_util_no_una_lista_vacia(server):
    """Cero resultados le dice al modelo 'el catalogo no cubre esto'. Un error
    que enumera las familias validas le deja corregir en el mismo turno."""
    message = call_expecting_error(
        server, "search_controls", {"query": "incidentes", "family": "access control"}
    )
    assert "no existe" in message
    assert "ir" in message


def test_un_baseline_invalido_sugiere_el_correcto(server):
    message = call_expecting_error(
        server, "search_controls", {"query": "cuentas", "baseline": "medium"}
    )
    assert "moderate" in message


def test_top_k_fuera_de_rango_se_rechaza(server, lexical_config):
    limit = lexical_config.get("retrieval.max_top_k")
    message = call_expecting_error(server, "search_controls", {"query": "x", "top_k": limit + 1})
    assert str(limit) in message


def test_una_pregunta_vacia_se_rechaza(server):
    assert "vacia" in call_expecting_error(server, "answer_question", {"question": "  "})
