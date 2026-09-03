"""Coste por consulta a partir del uso real de tokens.

El coste no se estima con reglas del pulgar: se calcula con los tokens que
reporta el proveedor en cada respuesta y con los precios de config.yaml. Dos
consecuencias deliberadas:

* Si un modelo no tiene precio declarado, el coste es `None`, no cero. Cero es
  un numero y se acaba sumando; `None` obliga a mirar.
* El baseline extractivo no llama a nadie, asi que su coste es 0.0 de verdad, y
  eso es informacion: es el suelo con el que comparar lo que cuesta el LLM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config

PER_MTOK = 1_000_000


@dataclass(slots=True)
class Cost:
    usd: float | None
    input_tokens: int
    output_tokens: int
    model: str
    # None cuando el modelo no tiene precio declarado en config.yaml.
    priced: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "usd": round(self.usd, 6) if self.usd is not None else None,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "model": self.model,
            "priced": self.priced,
        }


def price_for(config: Config, model: str) -> dict[str, float] | None:
    return config.get("generation.pricing.usd_per_mtok").get(model)


def compute(config: Config, model: str, usage: dict[str, int]) -> Cost:
    input_tokens = int(usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("output_tokens") or 0)
    prices = price_for(config, model)
    if prices is None:
        # El baseline no consume tokens ni llama a nadie: su coste es cero de
        # verdad, no desconocido.
        if input_tokens == 0 and output_tokens == 0:
            return Cost(0.0, 0, 0, model)
        return Cost(None, input_tokens, output_tokens, model, priced=False)
    usd = (input_tokens * prices["input"] + output_tokens * prices["output"]) / PER_MTOK
    return Cost(usd, input_tokens, output_tokens, model)


def aggregate(costs: list[Cost], config: Config) -> dict[str, Any]:
    """Resumen para el informe. Si algo quedo sin precio, se dice cuanto."""
    priced = [c for c in costs if c.usd is not None]
    unpriced = [c for c in costs if c.usd is None]
    total = sum(c.usd for c in priced) if priced else 0.0
    return {
        "n_queries": len(costs),
        "n_unpriced": len(unpriced),
        "unpriced_models": sorted({c.model for c in unpriced}),
        "total_usd": round(total, 6),
        "usd_per_query": round(total / len(priced), 6) if priced else None,
        "input_tokens_total": sum(c.input_tokens for c in costs),
        "output_tokens_total": sum(c.output_tokens for c in costs),
        "mean_input_tokens": (
            round(sum(c.input_tokens for c in costs) / len(costs), 1) if costs else None
        ),
        "prices_checked_at": config.get("generation.pricing.checked_at"),
    }
