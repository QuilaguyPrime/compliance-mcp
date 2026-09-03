"""Coste por consulta: se calcula con tokens reales y precios versionados."""
from __future__ import annotations

from compliance_mcp.cost import aggregate, compute


def test_el_coste_sale_de_los_tokens_y_del_precio_de_config(config):
    model = "claude-opus-5"
    prices = config.get("generation.pricing.usd_per_mtok")[model]
    cost = compute(config, model, {"input_tokens": 1_000_000, "output_tokens": 0})
    assert cost.usd == prices["input"]


def test_un_modelo_sin_precio_da_desconocido_no_cero(config):
    """Cero es un numero y acaba sumandose a un total que parece medido.
    None obliga a mirar."""
    cost = compute(config, "modelo-nuevo-sin-precio", {"input_tokens": 100, "output_tokens": 50})
    assert cost.usd is None
    assert cost.priced is False


def test_el_baseline_cuesta_cero_de_verdad(config):
    """No llama a nadie: su coste es cero medido, no desconocido. Es el suelo
    contra el que se compara lo que cuesta el LLM."""
    cost = compute(config, "none", {})
    assert cost.usd == 0.0
    assert cost.priced is True


def test_el_agregado_declara_lo_que_no_pudo_valorar(config):
    costs = [
        compute(config, "claude-opus-5", {"input_tokens": 1000, "output_tokens": 100}),
        compute(config, "modelo-sin-precio", {"input_tokens": 1000, "output_tokens": 100}),
    ]
    summary = aggregate(costs, config)
    assert summary["n_queries"] == 2
    assert summary["n_unpriced"] == 1
    assert summary["unpriced_models"] == ["modelo-sin-precio"]
    # El coste por consulta se promedia solo sobre lo que si tiene precio.
    assert summary["usd_per_query"] == summary["total_usd"]


def test_el_agregado_fecha_los_precios(config):
    """Un precio de hace un ano convierte el coste publicado en ficcion. La
    fecha viaja con el numero."""
    summary = aggregate([compute(config, "claude-opus-5", {"input_tokens": 1, "output_tokens": 1})], config)
    assert summary["prices_checked_at"] == config.get("generation.pricing.checked_at")
