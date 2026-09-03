"""Preflight. Comprueba sin llamar a ninguna API."""
from __future__ import annotations

import copy

from compliance_mcp.config import Config
from compliance_mcp.doctor import (
    CORPUS,
    FAIL,
    GOLDEN,
    INDEX,
    OK,
    WARN,
    check_index,
    check_pricing,
    check_providers,
    run,
)


def variant(config, mutate) -> Config:
    data = copy.deepcopy(config.as_dict())
    mutate(data)
    return Config(data, config.source)


def status_of(checks, name):
    return next(c.status for c in checks if c.name == name)


def test_el_repo_pasa_corpus_y_golden_set(config):
    checks = run(config)
    assert status_of(checks, CORPUS) == OK
    assert status_of(checks, GOLDEN) == OK


def test_el_indice_construido_pasa_el_preflight(config, indexed_repo):
    assert status_of(run(config), INDEX) == OK


def test_detecta_un_indice_caducado(config, records, indexed_repo):
    mutated = variant(
        config, lambda d: d["chunking"].__setitem__("header_template", "TEXTO DISTINTO {label}")
    )
    check = check_index(mutated, records)
    assert check.status == FAIL
    assert "texto de los chunks cambio" in check.detail


def test_detecta_que_falta_el_indice(config, records, tmp_path):
    mutated = variant(config, lambda d: d["corpus"].__setitem__("index_dir", str(tmp_path)))
    check = check_index(mutated, records)
    assert check.status == FAIL
    assert "make index" in check.detail


def test_sin_credenciales_lo_dice_sin_llamar_a_nadie(config, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    check = check_providers(config)
    assert check.status == FAIL
    # Y precisa que search_controls y get_control siguen funcionando: no todo
    # el servidor depende de un proveedor.
    assert "get_control" in check.detail


def test_con_solo_el_primario_avisa_de_que_no_hay_fallback(config, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    check = check_providers(config)
    assert check.status == WARN
    assert "openai" in check.detail
    # Solo se comprueba que la credencial existe; validarla cuesta una llamada.
    assert "existencia, no validez" in check.detail


def test_avisa_de_un_modelo_servido_sin_precio(config):
    mutated = variant(
        config, lambda d: d["generation"]["pricing"]["usd_per_mtok"].pop("claude-opus-5")
    )
    check = check_pricing(mutated)
    assert check.status == WARN
    assert "claude-opus-5" in check.detail


def test_solo_deciden_las_comprobaciones_exigidas(config, monkeypatch, capsys):
    """Un job que solo evalua recuperacion no tiene credenciales y no debe
    fallar por eso; en un PR desde un fork no hay secreto que valga."""
    from compliance_mcp.doctor import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["--require", "corpus,golden_set"]) == 0
    # Sigue apareciendo en el informe, marcado como informativo.
    out = capsys.readouterr().out
    assert "providers" in out and "(informativo)" in out


def test_sin_credenciales_el_preflight_completo_falla(config, monkeypatch):
    from compliance_mcp.doctor import main

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main([]) == 1


def test_una_comprobacion_inventada_es_error_de_uso(config):
    import pytest

    from compliance_mcp.doctor import main

    with pytest.raises(SystemExit):
        main(["--require", "no_existe"])
