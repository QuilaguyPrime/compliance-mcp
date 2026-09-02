"""Carga de configuracion.

Regla del proyecto: todo parametro vive en config.yaml. Este modulo es el unico
punto que lee ese fichero y falla ruidosamente si algo falta, en vez de caer a
valores por defecto silenciosos (un fallback silencioso convierte un error de
configuracion en un resultado de evaluacion equivocado).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ENV_CONFIG_PATH = "COMPLIANCE_MCP_CONFIG"
DEFAULT_CONFIG_NAME = "config.yaml"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ConfigError(RuntimeError):
    """La configuracion falta, no parsea, o le falta una clave requerida."""


class Config:
    """Acceso a config.yaml por ruta con puntos, sin valores por defecto ocultos."""

    def __init__(self, data: dict[str, Any], source: Path) -> None:
        self._data = data
        self.source = source

    def get(self, path: str) -> Any:
        """Devuelve el valor en `path` (p.ej. "retrieval.fusion.rrf_k").

        Lanza ConfigError si la clave no existe: preferimos romper a adivinar.
        """
        node: Any = self._data
        walked: list[str] = []
        for part in path.split("."):
            walked.append(part)
            if not isinstance(node, dict) or part not in node:
                raise ConfigError(
                    f"Falta la clave '{'.'.join(walked)}' en {self.source}. "
                    f"Todo parametro debe estar declarado en config.yaml."
                )
            node = node[part]
        return node

    def path(self, key: str) -> Path:
        """Resuelve una ruta de config relativa a la raiz del proyecto."""
        value = self.get(key)
        p = Path(value)
        return p if p.is_absolute() else project_root() / p

    def section(self, path: str) -> dict[str, Any]:
        value = self.get(path)
        if not isinstance(value, dict):
            raise ConfigError(f"'{path}' no es una seccion en {self.source}")
        return value

    def as_dict(self) -> dict[str, Any]:
        return self._data


def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    candidate = path or os.environ.get(ENV_CONFIG_PATH) or (project_root() / DEFAULT_CONFIG_NAME)
    p = Path(candidate)
    if not p.is_absolute():
        p = project_root() / p
    if not p.exists():
        raise ConfigError(f"No existe el fichero de configuracion: {p}")
    try:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"YAML invalido en {p}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{p} no contiene un mapeo en la raiz")
    return Config(data, p)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Config del proceso. Usa load_config() directamente en tests."""
    return load_config()
