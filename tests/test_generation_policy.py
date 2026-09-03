"""Politica del motor: que sale servido y que se retiene.

La regla de la casa es que lo que no verifica no se sirve, aunque eso convierta
una respuesta en un rehuso. Estos tests fijan esa regla con un proveedor
guionizado, para poder provocar a voluntad los fallos que un modelo real comete
de vez en cuando.
"""
from __future__ import annotations

import pytest

from compliance_mcp.generation.engine import AnswerEngine
from compliance_mcp.generation.providers import (
    Completion,
    ProviderChain,
    ProviderError,
    parse_draft,
)
from compliance_mcp.generation.schema import NOT_IN_CORPUS, AnswerDraft

QUESTION = "How long must audit records be retained?"


class ScriptedProvider:
    """Devuelve el borrador que fabrique `factory` a partir del contexto real."""

    name = "scripted"
    model = "test"

    def __init__(self, factory) -> None:
        self.factory = factory
        self.calls = 0

    def generate(self, question, context):
        self.calls += 1
        return Completion(draft=self.factory(question, context))


class BrokenProvider:
    name = "broken"
    model = "test"

    def __init__(self) -> None:
        self.calls = 0

    def generate(self, question, context):
        self.calls += 1
        raise ProviderError("caida simulada")


def engine_with(config, retriever, factory) -> AnswerEngine:
    return AnswerEngine(retriever, ProviderChain([ScriptedProvider(factory)]), config)


def first_quote(context) -> tuple[str, str, str]:
    entry = context.entries[0]
    part, text = next(iter(entry.parts.items()))
    return entry.control_id, part, text.splitlines()[0]


def answer(engine, **kwargs):
    return engine.answer(QUESTION, method="bm25", **kwargs)


def test_una_cita_literal_del_contexto_se_sirve(config, retriever):
    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="Respuesta respaldada.",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    result = answer(engine_with(config, retriever, factory))
    assert not result.refused
    assert len(result.citations) == 1
    assert result.citations[0].label


def test_una_cita_inventada_convierte_la_respuesta_en_rehuso(config, retriever):
    """Sin citas verificadas no hay respuesta: es lo que hace que la tasa de
    alucinacion servida sea cero por construccion."""

    def factory(question, context):
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="El catalogo exige conservar los registros 90 dias.",
            citations=[
                {
                    "control_id": "zz-99",
                    "part": "statement",
                    "quote": "Retain audit records for ninety days without exception.",
                }
            ],
        )

    result = answer(engine_with(config, retriever, factory))
    assert result.refused
    assert result.refusal_reason == "unsupported_by_context"
    assert result.citations == []
    assert result.verification.forced_refusal
    # La cita rechazada se conserva: es justo lo que mide el arnes de evaluacion.
    assert result.verification.rejected[0].status == NOT_IN_CORPUS
    assert "90 dias" not in result.answer


def test_la_cita_buena_sobrevive_a_la_mala(config, retriever):
    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer=f"Respuesta con dos fuentes [{context.entries[0].label}].",
            citations=[
                {"control_id": control_id, "part": part, "quote": quote},
                {"control_id": "zz-99", "part": "statement", "quote": "Texto que no existe en parte alguna."},
            ],
        )

    result = answer(engine_with(config, retriever, factory))
    assert not result.refused
    assert len(result.citations) == 1
    assert result.verification.emitted == 2


def test_etiqueta_en_prosa_sin_respaldo_retiene_la_respuesta(config, retriever):
    """Citar bien AC-2 y afirmar cosas de [ZZ-99] en la prosa deja la afirmacion
    sin respaldo aunque la cita adjunta sea valida."""

    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="Segun [ZZ-99] hay que conservarlos indefinidamente.",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    result = answer(engine_with(config, retriever, factory))
    assert result.refused
    assert result.verification.unsupported_inline_refs == ["ZZ-99"]


def test_el_rehuso_del_modelo_se_respeta_y_no_lleva_citas(config, retriever):
    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=True,
            refusal_reason="other_framework",
            answer="Esa pregunta es de GDPR, no de SP 800-53.",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    result = answer(engine_with(config, retriever, factory))
    assert result.refused
    assert result.refusal_reason == "other_framework"
    assert result.citations == []


def test_el_top_k_limita_el_contexto(config, retriever):
    seen: list[int] = []

    def factory(question, context):
        seen.append(len(context.entries))
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="ok",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    answer(engine_with(config, retriever, factory), top_k=2)
    assert seen == [2]


def test_la_cadena_cae_al_siguiente_proveedor_y_lo_registra(config, retriever):
    broken = BrokenProvider()

    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="ok",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    working = ScriptedProvider(factory)
    engine = AnswerEngine(retriever, ProviderChain([broken, working]), config)
    result = engine.answer(QUESTION, method="bm25")
    assert broken.calls == 1 and working.calls == 1
    assert result.provider.name == "scripted"
    assert result.provider.degraded_from == ["broken"]


def test_si_caen_todos_los_proveedores_se_propaga_el_error(config, retriever):
    engine = AnswerEngine(retriever, ProviderChain([BrokenProvider(), BrokenProvider()]), config)
    with pytest.raises(ProviderError):
        engine.answer(QUESTION, method="bm25")


def test_una_respuesta_que_no_cumple_el_esquema_es_fallo_del_proveedor():
    """Se le impone el esquema en la llamada: JSON invalido significa que el
    proveedor incumplio su garantia. Se degrada, no se repara a mano."""
    with pytest.raises(ProviderError):
        parse_draft("Claro, aqui tienes la respuesta: AC-2 dice que...", "scripted")


def test_los_tiempos_por_etapa_se_reportan(config, retriever):
    def factory(question, context):
        control_id, part, quote = first_quote(context)
        return AnswerDraft(
            refused=False,
            refusal_reason=None,
            answer="ok",
            citations=[{"control_id": control_id, "part": part, "quote": quote}],
        )

    result = answer(engine_with(config, retriever, factory))
    assert {"retrieval.bm25", "generation.provider", "generation.verify", "total"} <= set(
        result.timings
    )
