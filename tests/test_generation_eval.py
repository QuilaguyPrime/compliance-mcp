"""Definicion de las metricas de generacion.

Se testean sobre respuestas fabricadas a mano y no corriendo el arnes entero:
lo que hay que fijar aqui es que cada metrica cuenta lo que dice contar. Una
metrica mal definida no falla, solo publica un numero equivocado.
"""
from __future__ import annotations

import json

import pytest

from compliance_mcp.eval.generation import (
    gate_block,
    merge_into_ablation,
    quantity_pattern,
    score_case,
    summarize,
)
from compliance_mcp.eval.golden import GoldenCase
from compliance_mcp.generation.schema import (
    NOT_IN_CORPUS,
    QUOTE_NOT_FOUND,
    VERIFIED,
    CitationVerdict,
    ProviderInfo,
    Verification,
    VerifiedAnswer,
)


@pytest.fixture
def quantities(config):
    return quantity_pattern(config.get("evaluation.generation.quantity_units"))


def make_answer(
    *, refused=False, reason=None, answer="texto", verdicts=None, served=None
) -> VerifiedAnswer:
    verdicts = verdicts or []
    verification = Verification(verdicts=verdicts)
    return VerifiedAnswer(
        question="q",
        refused=refused,
        refusal_reason=reason,
        answer=answer,
        citations=served if served is not None else [v for v in verdicts if v.ok],
        verification=verification,
        retrieved=[],
        provider=ProviderInfo(name="scripted", model="test"),
        timings={},
    )


def verdict(control_id: str, status: str = VERIFIED, quote: str = "cita") -> CitationVerdict:
    return CitationVerdict(
        control_id=control_id, part="statement", quote=quote, status=status, label=control_id.upper()
    )


ANSWERABLE = GoldenCase(
    id="gs-x", type="answerable", question="q", expected_control_ids=["au-11"],
    acceptable_control_ids=["si-12"],
)
MUST_REFUSE = GoldenCase(id="gs-y", type="unanswerable", question="q", must_refuse=True)


def test_un_caso_respondible_acertando_el_control_cuenta_como_grounded(quantities):
    outcome = score_case(ANSWERABLE, make_answer(verdicts=[verdict("au-11")]), quantities)
    assert outcome.grounded is True
    assert outcome.correct_refusal is None


def test_citar_un_control_verificado_pero_irrelevante_no_es_grounded(quantities):
    """La cita verifica (el texto existe) y aun asi la respuesta no responde la
    pregunta: verificar no es acertar."""
    outcome = score_case(ANSWERABLE, make_answer(verdicts=[verdict("ac-2")]), quantities)
    assert outcome.grounded is False


def test_rehusar_un_caso_respondible_no_es_grounded(quantities):
    outcome = score_case(ANSWERABLE, make_answer(refused=True, reason="not_in_corpus"), quantities)
    assert outcome.grounded is False


def test_un_caso_de_rehuso_solo_puntua_el_rehuso(quantities):
    outcome = score_case(MUST_REFUSE, make_answer(refused=True, reason="other_framework"), quantities)
    assert outcome.correct_refusal is True
    assert outcome.grounded is None


def test_responder_un_caso_de_rehuso_es_fallo(quantities):
    outcome = score_case(MUST_REFUSE, make_answer(verdicts=[verdict("ac-2")]), quantities)
    assert outcome.correct_refusal is False


def test_una_cifra_que_no_esta_en_ninguna_cita_se_marca(quantities):
    """En Rev 5 los periodos son parametros definidos por la organizacion: una
    cifra concreta sin fuente es la alucinacion tipica de este dominio."""
    answer = make_answer(
        answer="Los registros se conservan 90 days segun el estandar.",
        verdicts=[verdict("au-11", quote="Retain audit records for [time period]")],
    )
    assert score_case(ANSWERABLE, answer, quantities).unsourced_quantities == ["90 days"]


def test_una_cifra_presente_en_la_cita_no_se_marca(quantities):
    answer = make_answer(
        answer="El estandar fija 30 days.",
        verdicts=[verdict("au-11", quote="Retain audit records for 30 days")],
    )
    assert score_case(ANSWERABLE, answer, quantities).unsourced_quantities == []


def test_un_identificador_con_numero_no_es_una_cantidad(quantities):
    """'AC-2' y 'SP 800-53' llevan numero y no afirman ninguna cantidad."""
    answer = make_answer(
        answer="Ver AC-2 y SP 800-53 Rev 5.",
        verdicts=[verdict("ac-2", quote="Define and document the types of accounts")],
    )
    assert score_case(ANSWERABLE, answer, quantities).unsourced_quantities == []


def test_la_precision_de_citacion_se_mide_en_bruto(config, quantities):
    """Se cuentan las citas EMITIDAS por el modelo, no las servidas. Medir solo
    lo servido daria 1.0 siempre, porque la politica descarta lo que falla."""
    outcomes = [
        score_case(
            ANSWERABLE,
            make_answer(
                verdicts=[verdict("au-11"), verdict("zz-99", NOT_IN_CORPUS)], served=[verdict("au-11")]
            ),
            quantities,
        )
    ]
    citations = summarize(outcomes, config)["citations"]
    assert citations["emitted"] == 2
    assert citations["verified"] == 1
    assert citations["citation_precision"] == 0.5
    assert citations["hallucinated_citation_rate"] == 0.5
    assert citations["served_hallucinated_citation_rate"] == 0.0


def test_una_cita_que_no_verifica_no_es_lo_mismo_que_un_control_inventado(config, quantities):
    outcomes = [
        score_case(ANSWERABLE, make_answer(verdicts=[verdict("au-11", QUOTE_NOT_FOUND)]), quantities)
    ]
    citations = summarize(outcomes, config)["citations"]
    assert citations["citation_precision"] == 0.0
    assert citations["hallucinated_citation_rate"] == 0.0
    assert citations["by_status"] == {QUOTE_NOT_FOUND: 1}


def test_sin_citas_emitidas_la_precision_es_indefinida_no_perfecta(config, quantities):
    """Un sistema que rehusa siempre no tiene precision 1.0: no tiene precision."""
    outcomes = [score_case(MUST_REFUSE, make_answer(refused=True, reason="not_in_corpus"), quantities)]
    citations = summarize(outcomes, config)["citations"]
    assert citations["citation_precision"] is None
    assert citations["hallucinated_citation_rate"] is None


def test_las_tasas_llevan_su_intervalo_de_confianza(config, quantities):
    outcomes = [
        score_case(MUST_REFUSE, make_answer(refused=True, reason="not_in_corpus"), quantities),
        score_case(MUST_REFUSE, make_answer(verdicts=[verdict("ac-2")]), quantities),
    ]
    refusal = summarize(outcomes, config)["refusal"]["refusal_recall"]
    assert refusal["n"] == 2
    assert refusal["rate"] == 0.5
    lo, hi = refusal["ci95"]
    assert lo <= refusal["rate"] <= hi


def test_el_baseline_nunca_siembra_el_bloque_del_gate(config, tmp_path, monkeypatch):
    """El baseline copia: su precision es 1.0 y su alucinacion 0.0 por
    construccion. Inyectarlo dejaria el gate en verde sin medir nada."""
    ablation = config.path("evaluation.ablation.output_path")
    original = ablation.read_text(encoding="utf-8") if ablation.exists() else None
    try:
        results = {"provider": config.get("generation.baseline_provider"), "model": "none"}
        assert merge_into_ablation(results, config) is None
    finally:
        if original is not None:
            assert ablation.read_text(encoding="utf-8") == original


def test_el_bloque_del_gate_lleva_lo_que_el_gate_lee(config, quantities):
    from compliance_mcp.cost import aggregate, compute
    from compliance_mcp.provenance import provenance_block

    outcomes = [score_case(ANSWERABLE, make_answer(verdicts=[verdict("au-11")]), quantities)]
    summary = summarize(outcomes, config)
    meta = {
        "provider": "anthropic",
        "model": "claude-opus-5",
        "split": "test",
        "n": 1,
        "provenance": provenance_block(config),
        "cost": aggregate(
            [compute(config, "claude-opus-5", {"input_tokens": 4000, "output_tokens": 200})], config
        ),
    }
    block = gate_block({**summary, **meta})
    assert set(block) >= {
        "provider",
        "citation_precision",
        "hallucinated_citation_rate",
        "usd_per_query",
        "provenance",
    }
    # El gate compara la procedencia del bloque con el arbol actual, asi que
    # tiene que viajar con las metricas y no en otro sitio.
    assert block["provenance"]["corpus_digest"].startswith("sha256:")
    assert json.dumps(block)  # serializable: acaba en un fichero JSON
