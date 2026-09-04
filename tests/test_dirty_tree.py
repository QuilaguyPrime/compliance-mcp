"""Producir resultados desde un arbol sucio: falla, o queda marcado.

Antes esto era un aviso: el sufijo `-dirty` se escribia en el propio fichero
publicado. Sobrevivio a un commit y a un push sin que nadie lo leyera, que es lo
que le pasa a un aviso. Ahora el proceso no arranca, y si se fuerza con
--allow-dirty el artefacto queda marcado y el gate lo rechaza.
"""
from __future__ import annotations

import copy

import pytest

from compliance_mcp import provenance
from compliance_mcp.config import Config
from compliance_mcp.eval.gate import check_provenance
from compliance_mcp.provenance import (
    DirtyTreeError,
    GitState,
    UnresolvedProvenanceError,
    git_state,
    provenance_block,
    require_clean_tree,
)

SHA = "0123456789abcdef0123456789abcdef01234567"


def fake_git(monkeypatch, status: str) -> None:
    """Sustituye la llamada a git para no depender del arbol de quien ejecuta."""

    def _git(*args: str) -> str:
        return SHA + "\n" if args[0] == "rev-parse" else status

    monkeypatch.setattr(provenance, "_git", _git)


# ------------------------------------------------------------------ git_state


def test_un_arbol_limpio_no_esta_sucio(monkeypatch):
    fake_git(monkeypatch, "")
    state = git_state()
    assert state == GitState(SHA, False, ())
    assert state.labelled_sha == SHA


def test_se_conservan_las_rutas_culpables(monkeypatch):
    fake_git(monkeypatch, " M src/compliance_mcp/server.py\n?? nuevo.py\n")
    state = git_state()
    assert state.dirty
    assert state.dirty_paths == ("src/compliance_mcp/server.py", "nuevo.py")


def test_la_primera_ruta_no_se_recorta(monkeypatch):
    """El formato porcelain es "XY ruta" y la primera linea empieza por espacio.
    Recortar el bloque entero se comia un caracter del primer nombre."""
    fake_git(monkeypatch, " M src/compliance_mcp/provenance.py\n")
    assert git_state().dirty_paths == ("src/compliance_mcp/provenance.py",)


def test_el_sha_lleva_sufijo_cuando_hay_suciedad(monkeypatch):
    fake_git(monkeypatch, " M config.yaml\n")
    assert git_state().labelled_sha == f"{SHA}-dirty"


def test_los_directorios_de_salida_se_excluyen_de_la_consulta(monkeypatch):
    """Escribir un artefacto no puede contar como ensuciar el arbol.

    Si contara, el segundo `make eval` seguido se declararia sucio por culpa del
    primero, y en CI el paso de generacion saldria siempre sucio porque el de
    ablacion acaba de reescribir un fichero versionado.
    """
    seen: list[tuple[str, ...]] = []

    def _git(*args: str) -> str:
        seen.append(args)
        return SHA + "\n" if args[0] == "rev-parse" else ""

    monkeypatch.setattr(provenance, "_git", _git)
    git_state()
    status_args = next(a for a in seen if a[0] == "status")
    for directory in provenance.ARTIFACT_DIRS:
        assert f":(exclude){directory}" in status_args


def test_sin_git_no_se_inventa_un_sha(monkeypatch):
    def _git(*args: str) -> str:
        raise OSError("git no esta instalado")

    monkeypatch.setattr(provenance, "_git", _git)
    assert git_state() == GitState(None, False)
    assert git_state().labelled_sha is None


# ----------------------------------------------------------- require_clean_tree


def test_el_arbol_sucio_rompe_por_defecto(monkeypatch):
    fake_git(monkeypatch, " M src/compliance_mcp/retrieval/fusion.py\n")
    with pytest.raises(DirtyTreeError) as excinfo:
        require_clean_tree()
    # El fallo tiene que nombrar al culpable: uno que solo diga "esta sucio" se
    # sortea con --allow-dirty sin mirar que habia cambiado.
    assert "src/compliance_mcp/retrieval/fusion.py" in str(excinfo.value)
    assert "--allow-dirty" in str(excinfo.value)


def test_allow_dirty_deja_pasar(monkeypatch):
    fake_git(monkeypatch, " M src/compliance_mcp/retrieval/fusion.py\n")
    assert require_clean_tree(allow_dirty=True).dirty


def test_el_arbol_limpio_pasa_sin_flag(monkeypatch):
    fake_git(monkeypatch, "")
    assert require_clean_tree().dirty is False


# -------------------------------------------- procedencia irresoluble


def no_git(monkeypatch) -> None:
    """Sin repositorio resoluble: es lo que ve el proceso dentro de la imagen,
    donde la etapa runtime no instala git."""

    def _git(*args: str) -> str:
        raise OSError("git no esta instalado")

    monkeypatch.setattr(provenance, "_git", _git)


def test_sin_commit_resoluble_no_se_produce_artefacto(monkeypatch):
    """La imagen sirve, no evalua, y esto lo hace cumplir en vez de confiar en
    que alguien lo recuerde."""
    no_git(monkeypatch)
    with pytest.raises(UnresolvedProvenanceError) as excinfo:
        require_clean_tree()
    assert "no evalua" in str(excinfo.value)
    assert "--allow-dirty" in str(excinfo.value)


def test_allow_dirty_tambien_cubre_la_procedencia_irresoluble(monkeypatch):
    no_git(monkeypatch)
    assert require_clean_tree(allow_dirty=True).sha is None


def test_los_main_existentes_manejan_el_error_sin_cambiar(monkeypatch):
    """La razon de que sea subclase de DirtyTreeError.

    Los `main()` de ablacion y generacion capturan DirtyTreeError para salir con
    un mensaje y codigo 1 en vez de una traza. Si esto fuera un error hermano,
    se escaparia de ese except y volveria a salir como traza sin que nadie lo
    notara hasta verlo en un log.
    """
    assert issubclass(UnresolvedProvenanceError, DirtyTreeError)
    no_git(monkeypatch)
    with pytest.raises(DirtyTreeError):
        require_clean_tree()


def test_el_gate_rechaza_un_artefacto_sin_git_sha(config):
    """El flag local se puede sortear; el gate no."""
    provenance_data = clean_provenance(config)
    provenance_data["git_sha"] = None
    failures = check_provenance({"provenance": provenance_data}, config)
    assert len(failures) == 1
    assert "no llevan git_sha" in failures[0]


def test_el_gate_rechaza_un_artefacto_con_git_sha_ausente(config):
    provenance_data = clean_provenance(config)
    del provenance_data["git_sha"]
    failures = check_provenance({"provenance": provenance_data}, config)
    assert len(failures) == 1
    assert "no llevan git_sha" in failures[0]


# ------------------------------------------------------------------ procedencia


def test_la_procedencia_marca_la_suciedad(monkeypatch, config):
    fake_git(monkeypatch, " M config.yaml\n")
    block = provenance_block(config)
    assert block["dirty"] is True
    assert block["git_sha"].endswith("-dirty")


def test_la_procedencia_de_un_arbol_limpio_no_la_marca(monkeypatch, config):
    fake_git(monkeypatch, "")
    block = provenance_block(config)
    assert block["dirty"] is False
    assert not block["git_sha"].endswith("-dirty")


# ------------------------------------------------------------------------ gate


def clean_provenance(config: Config) -> dict:
    """Procedencia real del arbol, forzada a limpia.

    Se parte del bloque de verdad y no de un diccionario a mano: asi anadir un
    digest nuevo no deja estos tests pasando contra una procedencia incompleta,
    que es como se colaria un hueco de cobertura sin que nadie lo note.
    """
    from compliance_mcp.provenance import provenance_block

    return {**provenance_block(config), "git_sha": SHA, "dirty": False}


def test_el_gate_acepta_procedencia_limpia(config):
    assert check_provenance({"provenance": clean_provenance(config)}, config) == []


def test_el_gate_rechaza_el_campo_dirty(config):
    provenance_data = clean_provenance(config)
    provenance_data["dirty"] = True
    failures = check_provenance({"provenance": provenance_data}, config)
    assert len(failures) == 1
    assert "sin commitear" in failures[0]


def test_el_gate_rechaza_el_sufijo_heredado(config):
    """Un artefacto anterior al campo booleano solo lleva el sufijo en el sha.

    Es el caso real del `ablation.json` que se publico con
    `24e578c...-dirty`: si el gate solo mirase el campo nuevo, ese fichero
    pasaria.
    """
    provenance_data = clean_provenance(config)
    del provenance_data["dirty"]
    provenance_data["git_sha"] = f"{SHA}-dirty"
    failures = check_provenance({"provenance": provenance_data}, config)
    assert len(failures) == 1
    assert "sin commitear" in failures[0]


def test_el_gate_sigue_rechazando_otro_corpus(config):
    """La comprobacion nueva no puede haber tapado las que ya habia."""
    provenance_data = clean_provenance(config)
    provenance_data["corpus_digest"] = "sha256:" + "0" * 64
    failures = check_provenance({"provenance": provenance_data}, config)
    assert any(f.startswith("corpus_digest") for f in failures)


def test_un_arbol_sucio_no_contamina_una_config_valida(config):
    """Suciedad y digests son fallos independientes: uno no debe enmascarar al
    otro ni contarse dos veces."""
    provenance_data = clean_provenance(config)
    provenance_data["dirty"] = True
    provenance_data["config_digest"] = "sha256:" + "1" * 64
    failures = check_provenance({"provenance": provenance_data}, config)
    assert len(failures) == 2


def test_la_config_de_prueba_no_altera_la_real(config):
    """Guardia del propio fixture: los tests copian la config, no la mutan."""
    data = copy.deepcopy(config.as_dict())
    data["gates"]["min_recall_at_5"] = 0.99
    assert Config(data, config.source).get("gates.min_recall_at_5") == 0.99
    assert config.get("gates.min_recall_at_5") != 0.99
