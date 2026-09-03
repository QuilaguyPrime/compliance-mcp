"""Verificacion de citaciones: la pieza que decide que se sirve y que no."""
from __future__ import annotations

import pytest

from compliance_mcp.generation.context import AnswerContext, ContextEntry
from compliance_mcp.generation.schema import (
    BAD_PART,
    NOT_IN_CONTEXT,
    NOT_IN_CORPUS,
    QUOTE_NOT_FOUND,
    QUOTE_TOO_SHORT,
    VERIFIED,
    AnswerDraft,
)
from compliance_mcp.generation.verify import unsupported_inline_refs, verify_citations

STATEMENT = (
    "a. Define and document the types of accounts allowed and specifically prohibited;\n"
    "b. Assign account managers;\n"
    "c. Require [approvals] for requests to create accounts;"
)


@pytest.fixture
def context():
    return AnswerContext(
        entries=[
            ContextEntry(
                control_id="ac-2",
                label="AC-2",
                title="Account Management",
                family_title="Access Control",
                status="active",
                baselines=["low", "moderate", "high"],
                parts={"statement": STATEMENT},
            )
        ],
        # ac-3 existe en el corpus pero no se le paso al modelo.
        corpus_ids={"ac-2", "ac-3"},
    )


def draft(control_id: str, part: str, quote: str, answer: str = "texto") -> AnswerDraft:
    return AnswerDraft(
        refused=False,
        refusal_reason=None,
        answer=answer,
        citations=[{"control_id": control_id, "part": part, "quote": quote}],
    )


def status_of(d: AnswerDraft, context, config) -> str:
    return verify_citations(d, context, config).verdicts[0].status


def test_cita_literal_verifica(context, config):
    quote = "Require [approvals] for requests to create accounts"
    assert status_of(draft("ac-2", "statement", quote), context, config) == VERIFIED


def test_cita_con_espacios_distintos_verifica(context, config):
    """El statement se aplana con saltos de linea e indentacion. Si la
    normalizacion no colapsara espacios, esta cita correcta contaria como
    inventada y la metrica mediria formateo, no alucinaciones."""
    quote = "Define and document the types of accounts   allowed\n and specifically prohibited"
    assert status_of(draft("ac-2", "statement", quote), context, config) == VERIFIED


def test_cita_parafraseada_no_verifica(context, config):
    quote = "The organization must define which account types are permitted"
    assert status_of(draft("ac-2", "statement", quote), context, config) == QUOTE_NOT_FOUND


def test_control_inexistente_es_alucinacion_dura(context, config):
    assert status_of(draft("zz-99", "statement", "algo largo que no importa"), context, config) == (
        NOT_IN_CORPUS
    )


def test_control_real_que_no_se_mostro_se_distingue_del_inexistente(context, config):
    """Citar de memoria parametrica un control que existe pero no se paso no es
    lo mismo que inventarse un control: se cuentan aparte."""
    assert status_of(draft("ac-3", "statement", "Enforce approved authorizations"), context, config) == (
        NOT_IN_CONTEXT
    )


def test_parte_no_expuesta(context, config):
    assert status_of(draft("ac-2", "guidance", "cualquier texto suficientemente largo"), context, config) == (
        BAD_PART
    )


def test_cita_demasiado_corta_no_ancla(context, config):
    """'account managers' aparece literalmente, pero una cita asi de corta no
    respalda nada: coincide en decenas de controles."""
    assert status_of(draft("ac-2", "statement", "account managers"), context, config) == (
        QUOTE_TOO_SHORT
    )


def test_etiqueta_en_prosa_sin_cita_verificada_se_detecta():
    refs = unsupported_inline_refs("Segun [AC-2] y [AC-6], hay que revisar cuentas.", {"ac-2"})
    assert refs == ["AC-6"]


def test_etiqueta_en_prosa_con_enhancement_se_normaliza():
    assert unsupported_inline_refs("Ver [AC-2(1)].", {"ac-2.1"}) == []
