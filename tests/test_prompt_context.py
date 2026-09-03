"""Contexto y prompt: lo que el modelo ve es lo que puede citar."""
from __future__ import annotations

from compliance_mcp.generation.context import build_context
from compliance_mcp.generation.prompt import build_user_message, render_context
from compliance_mcp.generation.schema import REFUSAL_REASONS
from compliance_mcp.generation.verify import QuoteNormalizer


def context_for(config, records_by_id, ids):
    return build_context([records_by_id[i] for i in ids], set(records_by_id), config)


def test_el_contexto_expone_solo_las_partes_configuradas(config, records_by_id):
    context = context_for(config, records_by_id, ["ac-2"])
    assert set(context.entries[0].parts) <= set(config.get("generation.context.parts"))


def test_el_contexto_lleva_el_id_canonico_y_la_etiqueta_humana(config, records_by_id):
    """El modelo debe devolver el id canonico; la etiqueta es para la prosa.
    Ensenarle solo una de las dos formas garantiza que devuelva la equivocada."""
    rendered = render_context(context_for(config, records_by_id, ["ac-2.1"]))
    assert "control_id: ac-2.1" in rendered
    assert "label: AC-2(1)" in rendered


def test_las_partes_van_delimitadas(config, records_by_id):
    rendered = render_context(context_for(config, records_by_id, ["ac-2"]))
    assert "<statement>" in rendered and "</statement>" in rendered


def test_un_control_retirado_expone_su_destino(config, records_by_id):
    """Un retirado no tiene statement. Si no se expusiera nada, no habria de
    donde citar y la unica salida seria rehusar con 'no existe'."""
    context = context_for(config, records_by_id, ["ac-3.1"])
    text = context.entries[0].parts["statement"]
    assert "Withdrawn" in text and "Incorporated into" in text


def test_el_corte_por_longitud_respeta_la_frontera_de_linea(config, records_by_id):
    """Cortar un item numerado del statement por la mitad produce citas
    imposibles de verificar por culpa del corte, no del modelo."""
    import copy

    from compliance_mcp.config import Config

    data = copy.deepcopy(config.as_dict())
    data["generation"]["context"]["max_part_chars"] = 120
    tight = Config(data, config.source)

    entry = context_for(tight, records_by_id, ["ac-2"]).entries[0]
    text = entry.parts["statement"]
    assert len(text) <= 120
    assert not text.endswith("\n")
    # Lo mostrado sigue siendo un prefijo literal del original: lo que se cite
    # de aqui tiene que verificar.
    normalize = QuoteNormalizer(tight)
    assert normalize(text) in normalize(records_by_id["ac-2"].statement)


def test_sin_contexto_se_pide_el_rehuso_explicitamente(config):
    from compliance_mcp.generation.context import AnswerContext

    message = build_user_message("cualquier cosa", AnswerContext())
    assert "refuse" in message.lower()
    assert all(reason in message for reason in REFUSAL_REASONS)


def test_el_mensaje_lleva_la_pregunta_y_los_pasajes(config, records_by_id):
    message = build_user_message("How are accounts managed?", context_for(config, records_by_id, ["ac-2"]))
    assert "How are accounts managed?" in message
    assert "control_id: ac-2" in message
