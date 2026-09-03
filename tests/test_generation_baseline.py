"""El baseline extractivo: que hace y, sobre todo, que no hace.

Existe para que la evaluacion tenga un suelo y para que la suite corra sin
claves. Estos tests fijan sus dos propiedades: cita literalmente (asi que su
precision es 1.0 por construccion, no por merito) y no sabe rehusar por
criterio semantico, que es justo lo que el LLM tiene que ganar.
"""
from __future__ import annotations

from compliance_mcp.generation.context import build_context
from compliance_mcp.generation.providers import ExtractiveProvider
from compliance_mcp.generation.verify import verify_citations


def context_for(config, records_by_id, control_ids):
    records = [records_by_id[c] for c in control_ids]
    return build_context(records, set(records_by_id), config)


def test_cita_literalmente_asi_que_siempre_verifica(config, records_by_id):
    context = context_for(config, records_by_id, ["au-11", "au-11.1"])
    completion = ExtractiveProvider(config).generate("cuanto tiempo", context)
    verification = verify_citations(completion.draft, context, config)
    assert verification.emitted == 1
    assert len(verification.verified) == 1
    assert verification.unsupported_inline_refs == []


def test_la_cita_alcanza_el_minimo_configurado(config, records_by_id):
    context = context_for(config, records_by_id, ["ac-2"])
    completion = ExtractiveProvider(config).generate("cuentas", context)
    quote = completion.draft.citations[0].quote
    assert config.get("generation.citations.min_quote_chars") <= len(quote)
    assert len(quote) <= config.get("generation.citations.max_quote_chars")


def test_sin_contexto_rehusa(config):
    from compliance_mcp.generation.context import AnswerContext

    completion = ExtractiveProvider(config).generate("lo que sea", AnswerContext())
    assert completion.draft.refused
    assert completion.draft.refusal_reason == "no_relevant_control"


def test_no_rehusa_una_pregunta_de_otro_marco(config, records_by_id):
    """Con contexto recuperado responde siempre. Es el limite del baseline y la
    razon de que refusal_recall sea su metrica mas baja: no distingue una
    pregunta de GDPR de una del catalogo."""
    context = context_for(config, records_by_id, ["pm-19"])
    completion = ExtractiveProvider(config).generate("What does GDPR require?", context)
    assert not completion.draft.refused
