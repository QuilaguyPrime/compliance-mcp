"""Contrato de salida: lo que el modelo puede y no puede devolver."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from compliance_mcp.generation.schema import (
    AnswerDraft,
    Citation,
    normalize_control_id,
    strict_json_schema,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AC-2", "ac-2"),
        ("AC-2(1)", "ac-2.1"),
        ("ac-2.1", "ac-2.1"),
        (" AC-2 ", "ac-2"),
        ("AC-2.", "ac-2"),
        ("PM-31", "pm-31"),
    ],
)
def test_control_id_se_normaliza_a_la_clave_del_corpus(raw, expected):
    """El modelo cita como escribe un humano y el corpus indexa como OSCAL.
    Sin normalizar, 'AC-2(1)' se marcaria como control inexistente."""
    assert normalize_control_id(raw) == expected


def test_la_citacion_normaliza_su_control_id():
    assert Citation(control_id="AC-2(1)", part="statement", quote="x").control_id == "ac-2.1"


def test_se_rechaza_un_motivo_de_rehuso_desconocido():
    with pytest.raises(ValidationError):
        AnswerDraft(refused=True, refusal_reason="porque si", answer="no", citations=[])


def test_se_rechaza_un_campo_extra():
    """extra='forbid': un campo que no esta en el contrato es senal de que el
    proveedor no respeto el esquema, no algo que ignorar en silencio."""
    with pytest.raises(ValidationError):
        AnswerDraft.model_validate(
            {
                "refused": False,
                "refusal_reason": None,
                "answer": "a",
                "citations": [],
                "confidence": 0.9,
            }
        )


def test_el_esquema_estricto_cierra_todos_los_objetos():
    """Los proveedores con salida estructurada exigen additionalProperties=false
    y required completo en cada objeto, tambien dentro de $defs."""
    schema = strict_json_schema(AnswerDraft)
    objects = [schema, *schema.get("$defs", {}).values()]
    for node in objects:
        assert node["additionalProperties"] is False
        assert set(node["required"]) == set(node["properties"])
    assert set(schema["properties"]) == {"refused", "refusal_reason", "answer", "citations"}
