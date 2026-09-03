"""Manifiesto del indice: que texto exacto hay detras de cada .npy.

El nombre del fichero de embeddings solo codifica estrategia y modelo, asi que
por si solo no distingue un indice fresco de uno construido antes de cambiar el
ingest o el chunking. El manifiesto guarda el fingerprint del texto que se
embebio, y cargar el indice lo comprueba.

La comprobacion falla ruidosamente. Servir con un indice caducado no da un
error: da resultados peores de forma invisible, que es el fallo mas caro de
todos porque no deja rastro.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .chunking import Chunk
from .config import Config
from .provenance import (
    INDEX_CONFIG_KEYS,
    corpus_digest,
    digest_config,
    digest_texts,
    git_sha,
    now_iso,
)

MANIFEST_NAME = "manifest.json"


class StaleIndexError(RuntimeError):
    """El indice en disco no corresponde al corpus o a la config actuales."""


@dataclass(slots=True)
class IndexEntry:
    strategy: str
    chunks: int
    dim: int
    model: str
    normalize: bool
    chunks_digest: str
    corpus_digest: str
    config_digest: str
    built_at: str
    git_sha: str | None
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_path(config: Config) -> Path:
    return config.path("corpus.index_dir") / MANIFEST_NAME


def read_manifest(config: Config) -> dict[str, Any]:
    path = manifest_path(config)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_entry(config: Config, entry: IndexEntry) -> Path:
    """Actualiza una entrada sin tocar las demas: `index --strategy A` no debe
    invalidar el manifiesto de B y C."""
    path = manifest_path(config)
    manifest = read_manifest(config)
    manifest.setdefault("entries", {})[entry.strategy] = entry.to_dict()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def build_entry(
    config: Config, strategy: str, chunks: list[Chunk], dim: int, path: Path
) -> IndexEntry:
    return IndexEntry(
        strategy=strategy,
        chunks=len(chunks),
        dim=dim,
        model=config.get("retrieval.dense.model"),
        normalize=config.get("retrieval.dense.normalize"),
        chunks_digest=digest_texts([c.text for c in chunks]),
        corpus_digest=corpus_digest(config),
        config_digest=digest_config(config, INDEX_CONFIG_KEYS),
        built_at=now_iso(),
        git_sha=git_sha(),
        path=path.name,
    )


def check_entry(
    config: Config, strategy: str, chunks: list[Chunk], rows: int, dim: int
) -> list[str]:
    """Devuelve los desajustes encontrados. Lista vacia = indice fresco."""
    entry = read_manifest(config).get("entries", {}).get(strategy)
    if entry is None:
        return [
            (
                f"No hay entrada de manifiesto para la estrategia {strategy} en "
                f"{manifest_path(config)}. El indice puede ser de una version anterior "
                f"del corpus o de la configuracion; no se puede saber."
            )
        ]

    problems: list[str] = []
    if entry["chunks"] != rows:
        problems.append(
            f"el indice tiene {rows} vectores y el manifiesto declara {entry['chunks']}"
        )
    if len(chunks) != rows:
        problems.append(
            f"el corpus produce {len(chunks)} chunks y el indice tiene {rows} vectores"
        )
    if entry["dim"] != dim:
        problems.append(f"dimension {dim} frente a {entry['dim']} declarada")
    if entry["model"] != config.get("retrieval.dense.model"):
        problems.append(
            f"el indice se construyo con {entry['model']} y config pide "
            f"{config.get('retrieval.dense.model')}"
        )
    if entry["normalize"] != config.get("retrieval.dense.normalize"):
        problems.append("cambio retrieval.dense.normalize desde que se construyo el indice")

    # La comprobacion que de verdad importa: el texto embebido. Las anteriores
    # solo detectan cambios que ademas mueven una forma.
    current = digest_texts([c.text for c in chunks])
    if entry["chunks_digest"] != current:
        problems.append(
            "el texto de los chunks cambio desde que se construyo el indice "
            f"({entry['chunks_digest'][:19]}... -> {current[:19]}...). "
            "Cambiar ingest o chunking sin reindexar deja cada vector apuntando "
            "a un texto distinto del que dice"
        )
    return problems


def verify(config: Config, strategy: str, chunks: list[Chunk], embeddings) -> None:
    """Lanza StaleIndexError si el indice no corresponde. No hay modo laxo."""
    problems = check_entry(
        config, strategy, chunks, rows=int(embeddings.shape[0]), dim=int(embeddings.shape[1])
    )
    if problems:
        detail = "\n  - ".join(problems)
        raise StaleIndexError(
            f"El indice denso de la estrategia {strategy} esta caducado:\n  - {detail}\n"
            f"Reconstruyelo con `make ingest index`."
        )
