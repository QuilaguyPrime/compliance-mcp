"""La cadena de proveedores tiene que degradar tambien cuando falla al CONSTRUIR.

`_get_client()` solo envolvia el ImportError. El constructor del SDK de OpenAI
lanza OpenAIError cuando falta la credencial, y eso ocurria fuera del try de
`generate`, asi que escapaba de ProviderChain -que solo captura ProviderError-.
Tres consecuencias: la degradacion al siguiente proveedor no ocurria, el
`ProviderError("Todos los proveedores fallaron")` era inalcanzable por ese
camino, y al cliente MCP le llegaba una excepcion cruda de un SDK. Un fallo
transitorio construyendo el cliente del primario tenia el mismo efecto.
"""
from __future__ import annotations

import pytest

from compliance_mcp.generation.context import AnswerContext
from compliance_mcp.generation.providers import (
    AnthropicProvider,
    Completion,
    OpenAIProvider,
    ProviderChain,
    ProviderError,
)

SPEC_ANTHROPIC = {
    "name": "anthropic",
    "model": "claude-opus-5",
    "max_tokens": 4096,
    "effort": "medium",
    "timeout_s": 60,
    "max_retries": 2,
}
SPEC_OPENAI = {
    "name": "openai",
    "model": "gpt-4o-2024-08-06",
    "max_tokens": 4096,
    "timeout_s": 60,
    "max_retries": 2,
}


class Boom(Exception):
    """Excepcion de un SDK cualquiera: no comparte base con ProviderError."""


class FakeProvider:
    """Proveedor que responde, para comprobar que la degradacion llega a el."""

    name = "fake"
    model = "fake-1"

    def generate(self, question: str, context: AnswerContext) -> Completion:
        return Completion(draft=None, usage={"input_tokens": 1, "output_tokens": 1})


def break_construction(monkeypatch, provider, exc: Exception) -> None:
    def _get_client(self=provider):
        raise exc

    monkeypatch.setattr(provider, "_get_client", _get_client)


# ----------------------------------------------- envoltura en _get_client


def test_el_fallo_al_construir_anthropic_sale_como_provider_error(monkeypatch):
    provider = AnthropicProvider(SPEC_ANTHROPIC)
    monkeypatch.setattr(
        "anthropic.Anthropic", lambda **kw: (_ for _ in ()).throw(Boom("sin credencial"))
    )
    with pytest.raises(ProviderError) as excinfo:
        provider._get_client()
    assert "anthropic" in str(excinfo.value)
    assert "Boom" in str(excinfo.value)


def test_el_fallo_al_construir_openai_sale_como_provider_error(monkeypatch):
    provider = OpenAIProvider(SPEC_OPENAI)
    monkeypatch.setattr(
        "openai.OpenAI", lambda **kw: (_ for _ in ()).throw(Boom("sin credencial"))
    )
    with pytest.raises(ProviderError) as excinfo:
        provider._get_client()
    assert "openai" in str(excinfo.value)


# ------------------------------------------------------- degradacion real


def test_si_el_primario_no_construye_se_degrada_al_secundario(monkeypatch, empty_context):
    """El caso que tumbaria una corrida de evaluacion a mitad de camino."""
    primary = AnthropicProvider(SPEC_ANTHROPIC)
    break_construction(monkeypatch, primary, ProviderError("no se pudo construir"))
    chain = ProviderChain([primary, FakeProvider()])
    _, info = chain.generate("pregunta", empty_context)
    assert info.name == "fake"
    assert info.degraded_from == ["anthropic"]


def test_si_ninguno_construye_el_error_es_el_de_la_cadena(monkeypatch, empty_context):
    """Antes este mensaje era inalcanzable por este camino: el segundo proveedor
    reventaba con una excepcion que la cadena no capturaba."""
    primary = AnthropicProvider(SPEC_ANTHROPIC)
    secondary = OpenAIProvider(SPEC_OPENAI)
    break_construction(monkeypatch, primary, ProviderError("anthropic sin credencial"))
    break_construction(monkeypatch, secondary, ProviderError("openai sin credencial"))
    chain = ProviderChain([primary, secondary])
    with pytest.raises(ProviderError) as excinfo:
        chain.generate("pregunta", empty_context)
    assert "Todos los proveedores fallaron" in str(excinfo.value)


# ------------------------------------------------------------ servidor MCP


def test_el_cliente_mcp_recibe_un_toolerror_con_mensaje_util(lexical_config, monkeypatch):
    """Sin proveedor, answer_question tiene que explicar que pasa.

    Sin la traduccion, la ProviderError se escapa del handler y mcp la trata
    como fallo interno: el mensaje se queda en el servidor y quien llama ve algo
    generico. La causa habitual -falta una credencial- es justo lo que quien
    llama puede arreglar.
    """
    from mcp.server.mcpserver.exceptions import ToolError

    from compliance_mcp.generation.engine import AnswerEngine
    from compliance_mcp.server import build_server

    engine = AnswerEngine.build(lexical_config, provider="extractive", with_dense=False)
    monkeypatch.setattr(
        engine.chain,
        "generate",
        lambda *a, **k: (_ for _ in ()).throw(ProviderError("Todos los proveedores fallaron")),
    )
    server = build_server(lexical_config, engine=engine)
    tool = server._tool_manager.get_tool("answer_question")
    with pytest.raises(ToolError) as excinfo:
        tool.fn(question="how often must audit records be reviewed?")
    message = str(excinfo.value)
    assert "answer_question no puede responder" in message
    # Y que dice que hacer: las otras dos herramientas siguen sirviendo.
    assert "search_controls" in message
