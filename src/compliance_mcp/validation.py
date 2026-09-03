"""Validacion de los argumentos de las herramientas MCP.

Quien llama a estas herramientas es un modelo, y un modelo se equivoca de forma
plausible: pide `family="access control"` en vez de `"ac"`, o
`baseline="medium"` en vez de `"moderate"`. Sin validar, el filtro no casa con
nada y la herramienta devuelve cero resultados; el modelo concluye que el
catalogo no cubre el tema y responde que no existe.

Un error que enumera los valores validos es recuperable en el mismo turno; una
lista vacia no lo es. Los valores permitidos se derivan del corpus y de
config.yaml, no de una lista escrita a mano que se queda vieja.
"""
from __future__ import annotations

from .config import Config
from .ingest import ControlRecord

KINDS = ("control", "enhancement")


class ToolInputError(ValueError):
    """Argumento invalido. Su mensaje va destinado a quien llamo: dice que
    valores existen, no solo que el dado no vale."""


def _closest(value: str, options: set[str]) -> list[str]:
    """Sugerencias por prefijo o subcadena. Sin distancia de edicion: con
    familias de dos letras, cualquier umbral razonable sugiere media docena."""
    value = value.strip().lower()
    return sorted(o for o in options if o.startswith(value) or value in o)[:5]


def _enumerate(value: str, options: set[str], field: str) -> str:
    suggestions = _closest(value, options)
    listing = ", ".join(sorted(options))
    message = f"{field}={value!r} no existe. Valores validos: {listing}."
    if suggestions:
        message += f" Quiza querias: {', '.join(suggestions)}."
    return message


class ToolInputValidator:
    """Valores validos derivados del corpus cargado."""

    def __init__(self, records: dict[str, ControlRecord], config: Config) -> None:
        self.families = {r.family_id for r in records.values()}
        self.baselines = set(config.section("corpus.baseline_profiles"))
        self.max_top_k: int = config.get("retrieval.max_top_k")

    def query(self, value: str, field: str = "query") -> str:
        if not value or not value.strip():
            raise ToolInputError(f"{field} esta vacia. Escribe la pregunta o los terminos.")
        return value

    def top_k(self, value: int | None) -> int | None:
        if value is None:
            return None
        if not 1 <= value <= self.max_top_k:
            raise ToolInputError(
                f"top_k={value} fuera de rango. Debe estar entre 1 y {self.max_top_k}."
            )
        return value

    def family(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in self.families:
            raise ToolInputError(_enumerate(value, self.families, "family"))
        return normalized

    def baseline(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in self.baselines:
            raise ToolInputError(_enumerate(value, self.baselines, "baseline"))
        return normalized

    def kind(self, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if normalized not in KINDS:
            raise ToolInputError(_enumerate(value, set(KINDS), "kind"))
        return normalized
