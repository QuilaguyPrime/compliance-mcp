"""Los digests que identifican de que salio un resultado.

Antes de esto el gate comparaba corpus y una parte de la config, y nada mas. Se
podia editar `retrieval/fusion.py`, o cambiar `evaluation.bootstrap.seed`, o
reescribir una pregunta del golden set, no volver a correr la evaluacion,
commitear, y el gate pasaba: ninguno de esos cambios tocaba lo que se
comprobaba. Los numeros publicados eran de otro sistema y nada lo decia.
"""
from __future__ import annotations

import copy

import pytest

from compliance_mcp import provenance
from compliance_mcp.config import Config
from compliance_mcp.eval.gate import check_provenance
from compliance_mcp.provenance import (
    RESULT_CONFIG_KEYS,
    code_digest,
    digest_config,
    golden_digest,
    provenance_block,
)


def fake_tree(monkeypatch, root, files: dict[str, bytes]) -> None:
    """Arbol de fuentes de mentira, para no depender del repo real."""
    for name, content in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    monkeypatch.setattr(provenance, "project_root", lambda: root)


# ------------------------------------------------------------------ code_digest


def test_cambiar_un_fichero_de_src_mueve_el_digest(monkeypatch, tmp_path):
    fake_tree(monkeypatch, tmp_path, {"src/pkg/fusion.py": b"def rank():\n    return 1\n"})
    before = code_digest()
    (tmp_path / "src/pkg/fusion.py").write_bytes(b"def rank():\n    return 2\n")
    assert code_digest() != before


def test_commitear_el_artefacto_no_mueve_el_digest(monkeypatch, tmp_path):
    """La propiedad entera por la que existe esto.

    El `git_sha` cambia al commitear el artefacto -el commit es nuevo- y por eso
    el fichero acaba nombrando a su commit padre. El digest de codigo no: crear
    un JSON en data/derived no toca ningun `.py`, asi que el artefacto sigue
    identificando el codigo que lo produjo con independencia de donde aterrice.
    """
    fake_tree(monkeypatch, tmp_path, {"src/pkg/eval.py": b"X = 1\n"})
    before = code_digest()
    (tmp_path / "data/derived").mkdir(parents=True)
    (tmp_path / "data/derived/ablation.json").write_text('{"recall@5": 0.8636}')
    assert code_digest() == before


def test_los_finales_de_linea_no_cuentan(monkeypatch, tmp_path):
    """Un clon en Windows con autocrlf entrega los mismos ficheros con otros
    bytes. Sin normalizar, el digest diria que el codigo cambio sin que nadie
    haya tocado una linea, y la comprobacion se volveria ruido."""
    unix = tmp_path / "unix"
    windows = tmp_path / "windows"
    source = b"def rank():\n    return 1\n"
    fake_tree(monkeypatch, unix, {"src/pkg/fusion.py": source})
    digest_unix = code_digest()
    fake_tree(monkeypatch, windows, {"src/pkg/fusion.py": source.replace(b"\n", b"\r\n")})
    assert code_digest() == digest_unix


def test_renombrar_un_modulo_mueve_el_digest(monkeypatch, tmp_path):
    """La ruta entra en el hash: mismo contenido en otro sitio es otro codigo."""
    antes = tmp_path / "antes"
    despues = tmp_path / "despues"
    fake_tree(monkeypatch, antes, {"src/pkg/fusion.py": b"X = 1\n"})
    digest_antes = code_digest()
    fake_tree(monkeypatch, despues, {"src/pkg/merge.py": b"X = 1\n"})
    assert code_digest() != digest_antes


def test_anadir_un_modulo_mueve_el_digest(monkeypatch, tmp_path):
    fake_tree(monkeypatch, tmp_path, {"src/pkg/a.py": b"X = 1\n"})
    before = code_digest()
    (tmp_path / "src/pkg/b.py").write_bytes(b"Y = 2\n")
    assert code_digest() != before


def test_dos_arboles_identicos_dan_el_mismo_digest(monkeypatch, tmp_path):
    """El orden es por ruta POSIX, no por el que devuelva el sistema de
    ficheros, que varia entre maquinas."""
    files = {"src/pkg/z.py": b"Z = 1\n", "src/pkg/a.py": b"A = 1\n", "src/m.py": b"M = 1\n"}
    fake_tree(monkeypatch, tmp_path / "uno", files)
    primero = code_digest()
    fake_tree(monkeypatch, tmp_path / "dos", dict(reversed(list(files.items()))))
    assert code_digest() == primero


def test_el_bytecode_no_cuenta(monkeypatch, tmp_path):
    fake_tree(monkeypatch, tmp_path, {"src/pkg/a.py": b"X = 1\n"})
    before = code_digest()
    (tmp_path / "src/pkg/__pycache__").mkdir()
    (tmp_path / "src/pkg/__pycache__/a.cpython-311.pyc").write_bytes(b"\x00basura")
    assert code_digest() == before


def test_sin_arbol_de_fuentes_rompe_en_vez_de_hashear_el_vacio(monkeypatch, tmp_path):
    """Un digest de cero ficheros seria estable y no significaria nada: un
    resultado se publicaria con procedencia de codigo valida y vacia."""
    monkeypatch.setattr(provenance, "project_root", lambda: tmp_path)
    with pytest.raises(FileNotFoundError):
        code_digest()


# ---------------------------------------------------------------- golden_digest


def test_cambiar_el_golden_set_mueve_su_digest(config, tmp_path):
    data = copy.deepcopy(config.as_dict())
    original = golden_digest(config)
    copia = tmp_path / "golden.yaml"
    copia.write_text(
        config.path("evaluation.golden_set_path").read_text(encoding="utf-8") + "\n# tocado\n",
        encoding="utf-8",
    )
    data["evaluation"]["golden_set_path"] = str(copia)
    assert golden_digest(Config(data, config.source)) != original


# ---------------------------------------------------------------- config_digest


def test_la_semilla_de_bootstrap_cuenta(config):
    """El agujero concreto que esto cierra: cambiar una semilla mueve cada IC
    de la tabla y antes no tocaba ningun digest."""
    data = copy.deepcopy(config.as_dict())
    data["evaluation"]["bootstrap"]["seed"] += 1
    assert digest_config(Config(data, config.source), RESULT_CONFIG_KEYS) != digest_config(
        config, RESULT_CONFIG_KEYS
    )


def test_la_semilla_de_split_cuenta(config):
    data = copy.deepcopy(config.as_dict())
    data["evaluation"]["split"]["seed"] += 1
    assert digest_config(Config(data, config.source), RESULT_CONFIG_KEYS) != digest_config(
        config, RESULT_CONFIG_KEYS
    )


def test_los_umbrales_del_gate_no_cuentan(config):
    """`gates` queda fuera a proposito: mover un umbral cambia el veredicto,
    no la medicion. Invalidar el artefacto ahi seria pedir una corrida para
    nada."""
    data = copy.deepcopy(config.as_dict())
    data["gates"]["min_recall_at_5"] = 0.99
    assert digest_config(Config(data, config.source), RESULT_CONFIG_KEYS) == digest_config(
        config, RESULT_CONFIG_KEYS
    )


# ------------------------------------------------------------------------ gate


def test_la_procedencia_completa_pasa_el_gate(config):
    results = {"provenance": {**provenance_block(config), "dirty": False}}
    results["provenance"]["git_sha"] = "0" * 40
    assert check_provenance(results, config) == []


@pytest.mark.parametrize(
    ("field", "esperado"),
    [
        ("corpus_digest", "el corpus ingerido es otro"),
        ("config_digest", "cambio la configuracion"),
        ("golden_digest", "el golden set es otro"),
        ("code_digest", "el codigo de src/ es otro"),
    ],
)
def test_el_gate_nombra_el_digest_que_no_cuadra(config, field, esperado):
    """Un fallo que solo dijera "procedencia incoherente" obliga a investigar."""
    block = {**provenance_block(config), "dirty": False, "git_sha": "0" * 40}
    block[field] = "sha256:" + "f" * 64
    failures = check_provenance({"provenance": block}, config)
    assert len(failures) == 1
    assert failures[0].startswith(field)
    assert esperado in failures[0]


def test_el_gate_rechaza_procedencia_sin_los_digests_nuevos(config):
    """Un artefacto anterior a este bloque no se puede comprobar, y no poder
    comprobar no es lo mismo que estar bien."""
    block = {**provenance_block(config), "dirty": False, "git_sha": "0" * 40}
    del block["code_digest"]
    failures = check_provenance({"provenance": block}, config)
    assert len(failures) == 1
    assert "no llevan code_digest" in failures[0]
