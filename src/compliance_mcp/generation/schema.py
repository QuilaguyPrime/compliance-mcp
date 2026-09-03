"""Contrato de salida del generador.

El modelo no devuelve prosa libre: devuelve un objeto con citaciones
estructuradas. Una citacion escrita en prosa ("segun AC-2, la organizacion
debe...") no se puede verificar contra el corpus sin adivinar donde empieza y
donde acaba lo citado; una cita con `control_id`, `part` y `quote` se verifica
por igualdad de cadena.

Estos modelos cumplen dos papeles a la vez: validan lo que vuelve del proveedor
y, convertidos a JSON Schema, son lo que se le impone al proveedor como formato
de salida. Un solo sitio donde cambiar el contrato.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

# Motivos de rehuso. Cerrado a proposito: "no lo se" sin taxonomia no se puede
# medir, y el golden set distingue pregunta de otro marco de pregunta fuera de
# corpus.
REFUSAL_REASONS = (
    "other_framework",       # pregunta por GDPR, PCI-DSS, ISO 27001, HIPAA...
    "not_in_corpus",         # el dato no esta en el catalogo (costes, CVEs, historico)
    "no_relevant_control",   # ningun control recuperado responde la pregunta
    "unsupported_by_context",  # el modelo respondio pero nada quedo verificado
)

# Estados de una citacion tras verificarla.
VERIFIED = "verified"
NOT_IN_CORPUS = "not_in_corpus"        # el control_id no existe: alucinacion dura
NOT_IN_CONTEXT = "not_in_context"      # existe, pero no se le paso al modelo
QUOTE_NOT_FOUND = "quote_not_found"    # el control es real, la cita no esta en el
BAD_PART = "bad_part"                  # parte inexistente o no expuesta
QUOTE_TOO_SHORT = "quote_too_short"    # no ancla nada


class Citation(BaseModel):
    """Una cita al corpus. `quote` debe ser texto copiado literalmente del
    contexto que se le mostro al modelo."""

    model_config = ConfigDict(extra="forbid")

    control_id: str
    part: str
    quote: str

    @field_validator("control_id")
    @classmethod
    def _canonical(cls, v: str) -> str:
        # El modelo escribe "AC-2(1)"; la clave del corpus es "ac-2.1".
        return normalize_control_id(v)


class AnswerDraft(BaseModel):
    """Lo que el proveedor debe emitir, antes de verificar nada.

    Se llama draft y no Answer porque hasta que la verificacion no pasa no es
    una respuesta: es una propuesta de respuesta.
    """

    model_config = ConfigDict(extra="forbid")

    refused: bool
    refusal_reason: str | None
    answer: str
    citations: list[Citation]

    @field_validator("refusal_reason")
    @classmethod
    def _known_reason(cls, v: str | None) -> str | None:
        if v is None or v in REFUSAL_REASONS:
            return v
        raise ValueError(f"Motivo de rehuso desconocido: {v}. Opciones: {REFUSAL_REASONS}")


def normalize_control_id(raw: str) -> str:
    """'AC-2(1)' | 'ac-2.1' | ' AC-2 ' -> 'ac-2.1' | 'ac-2'.

    El modelo cita como escribe un humano; el corpus indexa como escribe OSCAL.
    Normalizar aqui evita marcar como inexistente un control que si existe.
    """
    out = raw.strip().lower().replace(" ", "")
    out = out.replace("(", ".").replace(")", "")
    return out.rstrip(".")


@dataclass(slots=True)
class CitationVerdict:
    """Resultado de verificar una cita. Se conserva tambien para las que fallan:
    lo que el modelo intento citar y no pudo es justo lo que hay que medir."""

    control_id: str
    part: str
    quote: str
    status: str
    label: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "control_id": self.control_id,
            "label": self.label,
            "part": self.part,
            "quote": self.quote,
            "status": self.status,
        }


@dataclass(slots=True)
class Verification:
    """Contabilidad completa de la verificacion de una respuesta."""

    verdicts: list[CitationVerdict] = field(default_factory=list)
    unsupported_inline_refs: list[str] = field(default_factory=list)
    forced_refusal: bool = False

    @property
    def verified(self) -> list[CitationVerdict]:
        return [v for v in self.verdicts if v.ok]

    @property
    def rejected(self) -> list[CitationVerdict]:
        return [v for v in self.verdicts if not v.ok]

    @property
    def emitted(self) -> int:
        return len(self.verdicts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "emitted": self.emitted,
            "verified": len(self.verified),
            "rejected": [v.to_dict() for v in self.rejected],
            "unsupported_inline_refs": self.unsupported_inline_refs,
            "forced_refusal": self.forced_refusal,
        }


@dataclass(slots=True)
class ProviderInfo:
    name: str
    model: str
    # Proveedores que fallaron antes de este. Vacio en el camino feliz.
    degraded_from: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "model": self.model, "degraded_from": self.degraded_from}


@dataclass(slots=True)
class VerifiedAnswer:
    """Lo que sale del servidor. Toda cita aqui ya paso la verificacion."""

    question: str
    refused: bool
    refusal_reason: str | None
    answer: str
    citations: list[CitationVerdict]
    verification: Verification
    retrieved: list[dict[str, Any]]
    provider: ProviderInfo
    timings: dict[str, float]
    trace_id: str | None = None
    usage: dict[str, int] = field(default_factory=dict)
    # None cuando el modelo no tiene precio declarado en config.yaml.
    cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "answer": self.answer,
            "citations": [
                {"control_id": c.control_id, "label": c.label, "part": c.part, "quote": c.quote}
                for c in self.citations
            ],
            "verification": self.verification.to_dict(),
            "retrieved": self.retrieved,
            "provider": self.provider.to_dict(),
            "timings_ms": self.timings,
            "usage": self.usage,
            "cost_usd": self.cost_usd,
            "trace_id": self.trace_id,
        }


def strict_json_schema(model: type[BaseModel]) -> dict[str, Any]:
    """JSON Schema del modelo en modo estricto.

    Los proveedores con salida estructurada exigen `additionalProperties: false`
    y que toda propiedad este en `required`; pydantic no emite ni una cosa ni la
    otra por defecto. Se recorre el esquema entero, `$defs` incluidos.
    """
    schema = model.model_json_schema()

    def tighten(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "object" and "properties" in node:
                node["additionalProperties"] = False
                node["required"] = list(node["properties"])
            for value in node.values():
                tighten(value)
        elif isinstance(node, list):
            for item in node:
                tighten(item)

    tighten(schema)
    return schema
