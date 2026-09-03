"""Validacion de argumentos: fallar diciendo que valores existen.

Quien llama es un modelo. Un filtro mal escrito que devuelve cero resultados le
hace concluir que el catalogo no cubre el tema; un error que enumera lo valido
le deja arreglarlo en el mismo turno.
"""
from __future__ import annotations

import pytest

from compliance_mcp.validation import ToolInputError, ToolInputValidator


@pytest.fixture(scope="module")
def validate(config, records_by_id):
    return ToolInputValidator(records_by_id, config)


def test_una_familia_valida_se_normaliza(validate):
    assert validate.family("AC") == "ac"


def test_una_familia_inexistente_enumera_las_validas(validate):
    with pytest.raises(ToolInputError) as exc:
        validate.family("access control")
    message = str(exc.value)
    assert "no existe" in message
    assert "ac" in message and "ir" in message


def test_un_baseline_plausible_pero_falso_se_rechaza(validate):
    """'medium' es lo que escribe cualquiera; el catalogo dice 'moderate'."""
    with pytest.raises(ToolInputError) as exc:
        validate.baseline("medium")
    assert "moderate" in str(exc.value)


def test_se_sugiere_lo_parecido(validate):
    with pytest.raises(ToolInputError) as exc:
        validate.baseline("mod")
    assert "Quiza querias: moderate" in str(exc.value)


def test_un_kind_invalido_se_rechaza(validate):
    with pytest.raises(ToolInputError):
        validate.kind("enhancements")


def test_top_k_fuera_de_rango(validate, config):
    limit = config.get("retrieval.max_top_k")
    assert validate.top_k(limit) == limit
    with pytest.raises(ToolInputError):
        validate.top_k(limit + 1)
    with pytest.raises(ToolInputError):
        validate.top_k(0)


def test_una_consulta_vacia_se_rechaza(validate):
    with pytest.raises(ToolInputError):
        validate.query("   ")


def test_los_argumentos_omitidos_pasan(validate):
    assert validate.family(None) is None
    assert validate.top_k(None) is None
