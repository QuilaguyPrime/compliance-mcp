"""Cableado de los proveedores con red.

No se llama a ninguna API: eso gastaria dinero y ademas mediria la API, no este
codigo. Lo que se fija aqui es lo que si es responsabilidad del repo: que la
peticion lleva el modelo y los limites que dice config.yaml, que se le impone el
esquema como formato de salida, y que una respuesta rara se convierte en
degradacion al siguiente proveedor en vez de en una excepcion suelta.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from compliance_mcp.generation.providers import (
    AnthropicProvider,
    OpenAIProvider,
    ProviderError,
    available_providers,
    build_chain,
)
from compliance_mcp.generation.schema import AnswerDraft

DRAFT = {
    "refused": False,
    "refusal_reason": None,
    "answer": "Respuesta.",
    "citations": [{"control_id": "AC-2", "part": "statement", "quote": "una cita cualquiera"}],
}


class FakeAnthropic:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.captured: dict = {}
        self._text = text
        self._stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text=self._text)],
            stop_reason=self._stop_reason,
            usage=SimpleNamespace(input_tokens=11, output_tokens=22),
        )


class FakeOpenAI:
    def __init__(self, text: str) -> None:
        self.captured: dict = {}
        self._text = text
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.captured = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self._text))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=22),
        )


@pytest.fixture
def specs(config):
    return {spec["name"]: spec for spec in config.get("generation.providers")}


def make(cls, spec, client):
    provider = cls(spec)
    provider._client = client
    return provider


def test_anthropic_manda_lo_que_dice_config(config, specs, empty_context):
    client = FakeAnthropic(json.dumps(DRAFT))
    provider = make(AnthropicProvider, specs["anthropic"], client)
    completion = provider.generate("pregunta", empty_context)

    sent = client.captured
    assert sent["model"] == specs["anthropic"]["model"]
    assert sent["max_tokens"] == specs["anthropic"]["max_tokens"]
    assert sent["output_config"]["effort"] == specs["anthropic"]["effort"]
    assert sent["output_config"]["format"]["type"] == "json_schema"
    # El esquema impuesto es el mismo contrato que luego valida la respuesta.
    schema = sent["output_config"]["format"]["schema"]
    assert set(schema["properties"]) == {"refused", "refusal_reason", "answer", "citations"}
    assert schema["additionalProperties"] is False
    assert "pregunta" in sent["messages"][0]["content"]

    assert isinstance(completion.draft, AnswerDraft)
    assert completion.draft.citations[0].control_id == "ac-2"
    assert completion.usage == {"input_tokens": 11, "output_tokens": 22}


def test_anthropic_trata_un_decline_como_fallo_del_proveedor(specs, empty_context):
    """El clasificador puede declinar con HTTP 200. Sin mirar stop_reason antes
    que content, eso se leeria como una respuesta vacia en vez de degradar."""
    client = FakeAnthropic("", stop_reason="refusal")
    provider = make(AnthropicProvider, specs["anthropic"], client)
    with pytest.raises(ProviderError, match="refusal"):
        provider.generate("pregunta", empty_context)


def test_openai_impone_el_esquema_en_modo_estricto(specs, empty_context):
    client = FakeOpenAI(json.dumps(DRAFT))
    provider = make(OpenAIProvider, specs["openai"], client)
    provider.generate("pregunta", empty_context)

    fmt = client.captured["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    assert client.captured["max_completion_tokens"] == specs["openai"]["max_tokens"]
    assert client.captured["messages"][0]["role"] == "system"


def test_una_respuesta_fuera_de_contrato_degrada(specs, empty_context):
    client = FakeAnthropic("Claro, te lo explico: AC-2 dice que...")
    provider = make(AnthropicProvider, specs["anthropic"], client)
    with pytest.raises(ProviderError, match="esquema"):
        provider.generate("pregunta", empty_context)


def test_la_cadena_de_config_se_monta_en_orden(config):
    chain = build_chain(config)
    declared = [spec["name"] for spec in config.get("generation.providers")]
    assert [p.name for p in chain.providers] == declared


def test_el_baseline_se_monta_solo(config):
    chain = build_chain(config, provider=config.get("generation.baseline_provider"))
    assert len(chain.providers) == 1
    assert chain.providers[0].name == config.get("generation.baseline_provider")


def test_se_reportan_los_proveedores_con_credencial(config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert available_providers(config) == ["anthropic"]
