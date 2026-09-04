"""Proveedores de generacion y cadena de degradacion.

Tres implementaciones tras la misma interfaz:

* `anthropic`  — primario.
* `openai`     — fallback. Existe para que una caida del primario degrade el
                 servicio en vez de tumbarlo; la degradacion se loguea, de modo
                 que en los logs se distingue quien respondio cada peticion.
* `extractive` — baseline determinista y sin red. No es un sustituto del LLM:
                 es el suelo contra el que se compara en la evaluacion y el
                 doble que permite que la suite de tests corra sin claves.

Los tres devuelven el MISMO objeto validado, y a los dos con red se les impone
el esquema como formato de salida en la propia llamada. Parsear JSON a mano de
una respuesta en prosa es una fuente de fallos que no hace falta tener.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import Config
from ..observability import log_event
from .context import AnswerContext
from .prompt import SYSTEM_PROMPT, build_user_message
from .schema import AnswerDraft, ProviderInfo, strict_json_schema

SCHEMA_NAME = "answer_draft"


class ProviderError(RuntimeError):
    """Fallo de un proveedor. Provoca la caida al siguiente de la cadena."""


@dataclass(slots=True)
class Completion:
    draft: AnswerDraft
    usage: dict[str, int] = field(default_factory=dict)


class Provider(Protocol):
    name: str
    model: str

    def generate(self, question: str, context: AnswerContext) -> Completion: ...


# --------------------------------------------------------------------------- #
# Anthropic (primario)
# --------------------------------------------------------------------------- #

class AnthropicProvider:
    name = "anthropic"

    def __init__(self, spec: dict[str, Any]) -> None:
        self.model: str = spec["model"]
        self._max_tokens: int = spec["max_tokens"]
        self._effort: str = spec["effort"]
        self._timeout: float = spec["timeout_s"]
        self._max_retries: int = spec["max_retries"]
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - depende del extra `serve`
                raise ProviderError(
                    "Falta el paquete `anthropic`. Instala el extra: pip install -e '.[serve]'"
                ) from exc
            try:
                self._client = anthropic.Anthropic(
                timeout=self._timeout, max_retries=self._max_retries
            )
            except Exception as exc:
                # Excepcion ancha a proposito: cada SDK lanza su propio tipo al
                # construir sin credencial (OpenAIError, y lo que traiga la
                # proxima version), y ninguno comparte base con ProviderError. Si
                # escapa de aqui, ProviderChain no lo captura, la degradacion al
                # siguiente proveedor no ocurre y el cliente MCP recibe un error
                # crudo de un SDK que ni sabia que existia.
                raise ProviderError(
                    f"No se pudo construir el cliente de {self.name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    def generate(self, question: str, context: AnswerContext) -> Completion:
        client = self._get_client()
        try:
            response = client.messages.create(
                model=self.model,
                max_tokens=self._max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_message(question, context)}],
                output_config={
                    "effort": self._effort,
                    "format": {
                        "type": "json_schema",
                        "schema": strict_json_schema(AnswerDraft),
                    },
                },
            )
        except Exception as exc:  # el SDK ya reintenta; aqui se degrada
            raise ProviderError(f"anthropic fallo: {type(exc).__name__}: {exc}") from exc

        # El clasificador de seguridad puede declinar con HTTP 200. Sin mirar
        # stop_reason antes que content, esto se lee como respuesta vacia.
        if getattr(response, "stop_reason", None) == "refusal":
            raise ProviderError("anthropic declino la peticion (stop_reason=refusal)")

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = {
            "input_tokens": getattr(response.usage, "input_tokens", 0),
            "output_tokens": getattr(response.usage, "output_tokens", 0),
        }
        return Completion(draft=parse_draft(text, self.name), usage=usage)


# --------------------------------------------------------------------------- #
# OpenAI (fallback)
# --------------------------------------------------------------------------- #

class OpenAIProvider:
    name = "openai"

    def __init__(self, spec: dict[str, Any]) -> None:
        self.model: str = spec["model"]
        self._max_tokens: int = spec["max_tokens"]
        self._timeout: float = spec["timeout_s"]
        self._max_retries: int = spec["max_retries"]
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - depende del extra `serve`
                raise ProviderError(
                    "Falta el paquete `openai`. Instala el extra: pip install -e '.[serve]'"
                ) from exc
            try:
                self._client = openai.OpenAI(timeout=self._timeout, max_retries=self._max_retries)
            except Exception as exc:
                # Excepcion ancha a proposito: cada SDK lanza su propio tipo al
                # construir sin credencial (OpenAIError, y lo que traiga la
                # proxima version), y ninguno comparte base con ProviderError. Si
                # escapa de aqui, ProviderChain no lo captura, la degradacion al
                # siguiente proveedor no ocurre y el cliente MCP recibe un error
                # crudo de un SDK que ni sabia que existia.
                raise ProviderError(
                    f"No se pudo construir el cliente de {self.name}: "
                    f"{type(exc).__name__}: {exc}"
                ) from exc
        return self._client

    def generate(self, question: str, context: AnswerContext) -> Completion:
        client = self._get_client()
        try:
            response = client.chat.completions.create(
                model=self.model,
                max_completion_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, context)},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": SCHEMA_NAME,
                        "strict": True,
                        "schema": strict_json_schema(AnswerDraft),
                    },
                },
            )
        except Exception as exc:
            raise ProviderError(f"openai fallo: {type(exc).__name__}: {exc}") from exc

        text = response.choices[0].message.content or ""
        usage = {
            "input_tokens": getattr(response.usage, "prompt_tokens", 0),
            "output_tokens": getattr(response.usage, "completion_tokens", 0),
        }
        return Completion(draft=parse_draft(text, self.name), usage=usage)


# --------------------------------------------------------------------------- #
# Extractivo (baseline determinista, sin red)
# --------------------------------------------------------------------------- #

class ExtractiveProvider:
    """Cita el primer control recuperado, literal, sin redactar nada.

    Es el suelo de la comparacion: por construccion su precision de citacion es
    1.0 y su tasa de alucinacion 0.0, porque copia. Lo que NO sabe hacer es
    rehusar por criterio semantico (solo rehusa si no hay contexto) ni sintetizar
    varios controles, que es justo lo que el LLM tiene que ganar para justificar
    su coste.
    """

    name = "extractive"

    def __init__(self, config: Config) -> None:
        self.model = "none"
        self._min_chars: int = config.get("generation.citations.min_quote_chars")
        self._max_chars: int = config.get("generation.citations.max_quote_chars")

    def generate(self, question: str, context: AnswerContext) -> Completion:
        entry = context.entries[0] if context.entries else None
        if entry is None or not entry.parts:
            return Completion(
                draft=AnswerDraft(
                    refused=True,
                    refusal_reason="no_relevant_control",
                    answer="No catalog passages were retrieved for this question.",
                    citations=[],
                )
            )
        part, text = next(iter(entry.parts.items()))
        quote = self._quote(text)
        return Completion(
            draft=AnswerDraft(
                refused=False,
                refusal_reason=None,
                answer=f"[{entry.label}] {entry.title}. {quote}",
                citations=[{"control_id": entry.control_id, "part": part, "quote": quote}],
            )
        )

    def _quote(self, text: str) -> str:
        """Lineas completas hasta cubrir el minimo, sin pasarse del maximo."""
        quote = ""
        for line in text.splitlines():
            candidate = f"{quote}\n{line}" if quote else line
            if len(candidate) > self._max_chars:
                break
            quote = candidate
            if len(quote) >= self._min_chars:
                break
        return quote or text[: self._max_chars]


# --------------------------------------------------------------------------- #
# Cadena
# --------------------------------------------------------------------------- #

class ProviderChain:
    """Intenta los proveedores en orden y registra la degradacion."""

    def __init__(self, providers: list[Provider]) -> None:
        if not providers:
            raise ValueError("La cadena de proveedores esta vacia")
        self.providers = providers

    def generate(self, question: str, context: AnswerContext) -> tuple[Completion, ProviderInfo]:
        degraded_from: list[str] = []
        last_error: Exception | None = None
        for provider in self.providers:
            try:
                completion = provider.generate(question, context)
            except ProviderError as exc:
                last_error = exc
                degraded_from.append(provider.name)
                log_event(
                    "generation.provider.failed",
                    provider=provider.name,
                    model=provider.model,
                    error=str(exc),
                )
                continue
            if degraded_from:
                log_event(
                    "generation.provider.degraded",
                    served_by=provider.name,
                    degraded_from=degraded_from,
                )
            return completion, ProviderInfo(
                name=provider.name, model=provider.model, degraded_from=degraded_from
            )
        raise ProviderError(f"Todos los proveedores fallaron. Ultimo: {last_error}")


def parse_draft(text: str, provider_name: str) -> AnswerDraft:
    """Valida la respuesta contra el contrato.

    Se impone el esquema en la llamada, asi que llegar aqui con JSON invalido
    significa que el proveedor incumplio su propia garantia: se trata como fallo
    del proveedor y se degrada, no se intenta reparar la cadena a mano.
    """
    try:
        return AnswerDraft.model_validate_json(text)
    except Exception as exc:
        preview = text[:200].replace("\n", " ")
        raise ProviderError(
            f"{provider_name} devolvio una respuesta que no cumple el esquema: "
            f"{type(exc).__name__}. Empieza por: {preview!r}"
        ) from exc


BUILDERS = {
    AnthropicProvider.name: AnthropicProvider,
    OpenAIProvider.name: OpenAIProvider,
}

# Variable de entorno que debe existir para que el proveedor sea utilizable.
API_KEY_ENV = {
    AnthropicProvider.name: "ANTHROPIC_API_KEY",
    OpenAIProvider.name: "OPENAI_API_KEY",
}


def available_providers(config: Config) -> list[str]:
    """Proveedores de la cadena que tienen credencial en el entorno."""
    return [
        spec["name"]
        for spec in config.get("generation.providers")
        if os.environ.get(API_KEY_ENV.get(spec["name"], ""), "")
    ]


def build_chain(config: Config, *, provider: str = "chain") -> ProviderChain:
    """`chain` monta la cadena de config.yaml; cualquier otro nombre monta ese
    proveedor a secas (usado por la evaluacion para medir el baseline)."""
    baseline = config.get("generation.baseline_provider")
    if provider == baseline:
        return ProviderChain([ExtractiveProvider(config)])
    if provider != "chain":
        specs = [s for s in config.get("generation.providers") if s["name"] == provider]
        if not specs:
            raise ValueError(f"Proveedor desconocido: {provider}")
        return ProviderChain([BUILDERS[provider](specs[0])])
    return ProviderChain(
        [BUILDERS[spec["name"]](spec) for spec in config.get("generation.providers")]
    )


def draft_to_json(draft: AnswerDraft) -> str:
    """Serializacion estable; la usan los tests y los logs."""
    return json.dumps(draft.model_dump(), ensure_ascii=False, sort_keys=True)
