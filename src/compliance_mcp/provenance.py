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


def git_sha() -> str | None:
    """Commit actual, con sufijo `-dirty` si hay cambios sin commitear.

    Un numero producido sobre un arbol sucio no es reproducible, y decirlo es
    mas util que omitirlo.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=project_root(),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except (subprocess.SubprocessError, OSError):
        return None


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
    return {
        "version": __version__,
        "git_sha": git_sha(),
        "generated_at": now_iso(),
        "corpus_digest": corpus_digest(config),
        "config_digest": digest_config(config, INDEX_CONFIG_KEYS + ["retrieval"]),
        "embedding_model": config.get("retrieval.dense.model"),
    }
