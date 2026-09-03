"""Verificacion de citaciones contra el contexto mostrado al modelo.

Es la pieza que convierte "el modelo dice que cita AC-2" en "esta cadena existe
literalmente en el AC-2 que le ensenamos". Todo lo que no pase por aqui no se
sirve.

La normalizacion es imprescindible y esta parametrizada en config.yaml: el
statement se aplana con saltos de linea e indentacion, asi que una cita
perfectamente correcta copiada de la pantalla del modelo difiere del texto
fuente en espacios. Sin colapsarlos, la tasa de "citas inventadas" mediria
formateo, no alucinaciones.
"""
from __future__ import annotations

import re

from ..config import Config
from .context import AnswerContext, ContextEntry
from .schema import (
    BAD_PART,
    NOT_IN_CONTEXT,
    NOT_IN_CORPUS,
    QUOTE_NOT_FOUND,
    QUOTE_TOO_SHORT,
    VERIFIED,
    AnswerDraft,
    Citation,
    CitationVerdict,
    Verification,
    normalize_control_id,
)

# Etiquetas citadas en la prosa: [AC-2], [AC-2(1)], [PM-31].
INLINE_REF = re.compile(r"\[([A-Za-z]{2}-\d+(?:\(\d+\))?)\]")
_WHITESPACE = re.compile(r"\s+")


class QuoteNormalizer:
    """Normalizador simetrico: se aplica igual a la cita y al texto fuente."""

    def __init__(self, config: Config) -> None:
        self._collapse = config.get("generation.quote_match.collapse_whitespace")
        self._casefold = config.get("generation.quote_match.casefold")
        self._strip = config.get("generation.quote_match.strip_chars")

    def __call__(self, text: str) -> str:
        out = text
        if self._collapse:
            out = _WHITESPACE.sub(" ", out)
        if self._casefold:
            out = out.casefold()
        return out.strip(self._strip)


def verify_citations(
    draft: AnswerDraft, context: AnswerContext, config: Config
) -> Verification:
    """Emite un veredicto por cita, conservando tambien las que fallan."""
    normalize = QuoteNormalizer(config)
    min_chars: int = config.get("generation.citations.min_quote_chars")

    verdicts: list[CitationVerdict] = []
    for citation in draft.citations:
        control_id = citation.control_id
        entry = context.entry(control_id)
        label = entry.label if entry else None
        status = _status_for(citation, entry, context, normalize, min_chars)
        verdicts.append(
            CitationVerdict(
                control_id=control_id,
                part=citation.part,
                quote=citation.quote,
                status=status,
                label=label,
            )
        )

    verification = Verification(verdicts=verdicts)
    if config.get("generation.citations.require_inline_refs_verified"):
        verification.unsupported_inline_refs = unsupported_inline_refs(
            draft.answer, {v.control_id for v in verification.verified}
        )
    return verification


def _status_for(
    citation: Citation,
    entry: ContextEntry | None,
    context: AnswerContext,
    normalize: QuoteNormalizer,
    min_chars: int,
) -> str:
    if entry is None:
        return NOT_IN_CORPUS if citation.control_id not in context.corpus_ids else NOT_IN_CONTEXT
    if citation.part not in entry.parts:
        return BAD_PART
    # El minimo se mide sobre el texto normalizado: una cita de 30 caracteres
    # de los cuales 12 son espacios no ancla mas que una de 18.
    quote = normalize(citation.quote)
    if len(quote) < min_chars:
        return QUOTE_TOO_SHORT
    if quote not in normalize(entry.parts[citation.part]):
        return QUOTE_NOT_FOUND
    return VERIFIED


def unsupported_inline_refs(answer: str, verified_ids: set[str]) -> list[str]:
    """Etiquetas citadas en la prosa que no respalda ninguna cita verificada.

    Sin esta comprobacion, un modelo puede escribir "[AC-2] exige revisiones
    anuales" y adjuntar una cita verificada de otro control: la cita pasa y la
    afirmacion sigue sin respaldo.
    """
    out: list[str] = []
    for raw in INLINE_REF.findall(answer):
        if normalize_control_id(raw) not in verified_ids and raw not in out:
            out.append(raw)
    return out
