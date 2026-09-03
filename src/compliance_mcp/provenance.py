"""Procedencia: de que corpus, que configuracion y que commit salio cada cosa.

Existe por un fallo concreto y silencioso. Los embeddings se guardan en un .npy
cuyo nombre solo codifica estrategia y modelo. Si alguien cambia
`ingest.param_resolution_passes` o una plantilla de chunking y vuelve a hacer
ingest, el TEXTO de los chunks cambia pero su NUMERO no: el .npy viejo sigue
cargando, las formas siguen cuadrando y cada vector pasa a corresponder a un
texto distinto del que dice. No hay excepcion, no hay aviso, y la evaluacion
publica numeros de un indice que no es el que se esta sirviendo.

La defensa es fingerprint del contenido, no del nombre del fichero: se hashean
los textos exactos que se embebieron. Y se aplica la misma regla que el resto
del proyecto: romper antes que adivinar.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import __version__
from .config import Config, project_root

DIGEST_PREFIX = "sha256:"


def _digest(chunks: list[bytes]) -> str:
    h = hashlib.sha256()
    for part in chunks:
        # Se mete la longitud para que ["ab","c"] y ["a","bc"] no colisionen.
        h.update(len(part).to_bytes(8, "big"))
        h.update(part)
    return DIGEST_PREFIX + h.hexdigest()


def digest_texts(texts: list[str]) -> str:
    """Fingerprint del contenido exacto que se indexo."""
    return _digest([t.encode("utf-8") for t in texts])


def digest_file(path: Path) -> str:
    return _digest([path.read_bytes()])


def digest_config(config: Config, keys: list[str]) -> str:
    """Fingerprint de las secciones de config que afectan a un artefacto.

    Se serializa con `sort_keys` para que reordenar el YAML no cuente como
    cambio: lo que se compara son los valores, no su orden en el fichero.
    """
    payload = {key: config.get(key) for key in keys}
    return _digest([json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")])


# Directorios cuyo contenido es SALIDA de la evaluacion. Escribir un artefacto
# no cuenta como ensuciar el arbol: los numeros salen de src/, config.yaml y
# data/raw, no de aqui. Sin esta exclusion el segundo `make eval` seguido se
# declararia sucio por culpa del primero, y en CI el paso de generacion saldria
# siempre sucio porque el de ablacion acaba de reescribir un fichero versionado.
ARTIFACT_DIRS = ("data/derived",)


class DirtyTreeError(RuntimeError):
    """Se pidio producir un artefacto publicable desde un arbol con cambios."""


@dataclass(frozen=True)
class GitState:
    """Commit actual y si el arbol tiene cambios que afecten al resultado."""

    sha: str | None
    dirty: bool
    dirty_paths: tuple[str, ...] = ()

    @property
    def labelled_sha(self) -> str | None:
        """El sha con sufijo `-dirty`, que es como se ha venido publicando."""
        if self.sha is None:
            return None
        return f"{self.sha}-dirty" if self.dirty else self.sha


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=project_root(),
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    ).stdout


def git_state() -> GitState:
    """Commit actual y suciedad del arbol, excluidos los directorios de salida.

    Se devuelven tambien las rutas culpables: un fallo que dice "el arbol esta
    sucio" y no dice por que se acaba sorteando con --allow-dirty sin mirar.
    """
    try:
        sha = _git("rev-parse", "HEAD").strip()
        excludes = [f":(exclude){d}" for d in ARTIFACT_DIRS]
        status = _git("status", "--porcelain", "--", ".", *excludes)
    except (subprocess.SubprocessError, OSError):
        return GitState(None, False)
    # Nada de .strip() sobre la salida entera: el formato porcelain es "XY ruta"
    # y el primer caracter puede ser un espacio significativo (" M ruta"),
    # asi que recortar el bloque desplaza la primera ruta un caracter.
    paths = tuple(line[3:] for line in status.splitlines() if line.strip())
    return GitState(sha, bool(paths), paths)


def git_sha() -> str | None:
    """Commit actual, con sufijo `-dirty` si hay cambios sin commitear.

    Un numero producido sobre un arbol sucio no es reproducible, y decirlo es
    mas util que omitirlo.
    """
    return git_state().labelled_sha


def require_clean_tree(*, allow_dirty: bool = False) -> GitState:
    """Puerta de los generadores de artefactos publicables.

    Falla por defecto en vez de avisar. Un aviso ya existia —el sufijo `-dirty`
    estaba escrito en el propio fichero publicado— y sobrevivio a un commit y a
    un push sin que nadie lo leyera. Lo que no se puede ignorar es un proceso
    que no arranca.
    """
    state = git_state()
    if state.dirty and not allow_dirty:
        listed = "\n  ".join(state.dirty_paths[:20])
        extra = "" if len(state.dirty_paths) <= 20 else f"\n  ... y {len(state.dirty_paths) - 20} mas"
        raise DirtyTreeError(
            "El arbol tiene cambios sin commitear, asi que no se puede saber que codigo "
            "produjo estos numeros:\n  " + listed + extra +
            "\n\nCommitea o descarta lo pendiente, o repite con --allow-dirty si sabes "
            "lo que haces: el artefacto quedara marcado y el gate de CI lo rechazara."
        )
    return state


def now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def corpus_digest(config: Config) -> str:
    """Fingerprint del corpus servido: los registros ya ingeridos."""
    return digest_file(config.path("corpus.records_path"))


# Secciones de config que determinan el texto de un chunk. Si cambia
# cualquiera, el indice construido antes ya no corresponde.
INDEX_CONFIG_KEYS = ["ingest", "chunking"]


def provenance_block(config: Config) -> dict[str, Any]:
    """Bloque que se adjunta a todo resultado publicable.

    Sin esto, una tabla de resultados es un numero sin sujeto: no se sabe de que
    corpus, que configuracion ni que commit salio.
    """
    state = git_state()
    return {
        "version": __version__,
        "git_sha": state.labelled_sha,
        # Booleano explicito ademas del sufijo del sha: el gate compara un campo,
        # no analiza una cadena, y un artefacto viejo sin este campo se sigue
        # detectando por el sufijo.
        "dirty": state.dirty,
        "generated_at": now_iso(),
        "corpus_digest": corpus_digest(config),
        "config_digest": digest_config(config, INDEX_CONFIG_KEYS + ["retrieval"]),
        "embedding_model": config.get("retrieval.dense.model"),
    }
