"""Arnes de evaluacion de la generacion.

Mide tres cosas distintas que se suelen confundir en una sola:

1. **Fidelidad de las citas** — de todo lo que el modelo intento citar, cuanto
   se verifica contra el texto que se le mostro. Se mide EN BRUTO, antes de que
   la politica del servidor descarte nada. Medir solo lo servido daria
   precision 1.0 y alucinacion 0.0 siempre, por construccion, y el gate seria
   vacuo.
2. **Comportamiento de rehuso** — recall de rehuso sobre los casos donde
   rehusar es la unica respuesta correcta, y tasa de rehuso falso sobre los
   respondibles. Un sistema que rehusa siempre saca 1.0 en el primero: hay que
   leer los dos juntos.
3. **Cantidades sin fuente** — cifras afirmadas en la prosa que no aparecen en
   ninguna cita verificada. En Rev 5 casi todo periodo y umbral es un parametro
   definido por la organizacion, asi que una cifra sin fuente es exactamente la
   alucinacion que este dominio produce.

Lo que este arnes NO mide es `must_not_invent` del golden set: ese campo esta
escrito como criterio en prosa para un humano ("no puede confirmar que exista
un requisito de rotacion de 90 dias"), no como cadena comparable. Fingir que se
comprueba con un `in` seria peor que no comprobarlo, asi que esos casos se
vuelcan en el bloque `manual_review` para adjudicacion humana.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import Config, load_config
from ..cost import Cost
from ..cost import aggregate as aggregate_cost
from ..cost import compute as compute_cost
from ..generation.engine import AnswerEngine
from ..generation.schema import NOT_IN_CORPUS, VerifiedAnswer
from ..observability import configure_logging, log_event, trace_context
from ..provenance import provenance_block
from .golden import GoldenCase, load_golden_set, split_cases
from .metrics import bootstrap_ci


# "90 days", "12 characters", "3 attempts". El numero solo no cuenta: "AC-2" y
# "SP 800-53" llevan numero y no afirman ninguna cantidad.
def quantity_pattern(units: list[str]) -> re.Pattern[str]:
    joined = "|".join(sorted(units, key=len, reverse=True))
    return re.compile(rf"\b\d[\d,.]*\s+(?:{joined})\b", re.IGNORECASE)


@dataclass(slots=True)
class CaseOutcome:
    case_id: str
    case_type: str
    style: str | None
    refused: bool
    refusal_reason: str | None
    emitted: int
    verified: int
    statuses: dict[str, int]
    served_control_ids: list[str]
    answer: str
    unsourced_quantities: list[str] = field(default_factory=list)
    # None cuando la metrica no aplica a este tipo de caso.
    correct_refusal: bool | None = None
    grounded: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "type": self.case_type,
            "style": self.style,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "citations_emitted": self.emitted,
            "citations_verified": self.verified,
            "statuses": self.statuses,
            "served_control_ids": self.served_control_ids,
            "unsourced_quantities": self.unsourced_quantities,
            "correct_refusal": self.correct_refusal,
            "grounded": self.grounded,
            "answer": self.answer,
        }


def score_case(case: GoldenCase, answer: VerifiedAnswer, quantities: re.Pattern[str]) -> CaseOutcome:
    statuses = Counter(v.status for v in answer.verification.verdicts)
    served = [c.control_id for c in answer.citations]

    outcome = CaseOutcome(
        case_id=case.id,
        case_type=case.type,
        style=case.style,
        refused=answer.refused,
        refusal_reason=answer.refusal_reason,
        emitted=answer.verification.emitted,
        verified=len(answer.verification.verified),
        statuses=dict(statuses),
        served_control_ids=served,
        answer=answer.answer,
    )

    if case.must_refuse:
        outcome.correct_refusal = answer.refused
    elif case.scorable_for_retrieval:
        # En un caso respondible, responder citando un control relevante es el
        # exito; rehusar es un fallo aunque sea un fallo seguro.
        outcome.grounded = (not answer.refused) and bool(set(served) & case.relevant)

    if not answer.refused:
        sourced = " ".join(c.quote for c in answer.citations).casefold()
        outcome.unsourced_quantities = [
            q for q in quantities.findall(answer.answer) if q.casefold() not in sourced
        ]
    return outcome


def _rate(values: list[float], config: Config) -> dict[str, Any]:
    """Punto + IC bootstrap. Los n por estrato son 15-30: sin IC, cualquier
    comparacion entre corridas lee ruido como senal."""
    if not values:
        return {"n": 0, "rate": None, "ci95": None}
    lo, hi = bootstrap_ci(
        values,
        config.get("evaluation.bootstrap.resamples"),
        config.get("evaluation.bootstrap.confidence"),
        config.get("evaluation.bootstrap.seed"),
    )
    return {
        "n": len(values),
        "rate": round(sum(values) / len(values), 4),
        "ci95": [round(lo, 4), round(hi, 4)],
    }


def summarize(outcomes: list[CaseOutcome], config: Config) -> dict[str, Any]:
    emitted = sum(o.emitted for o in outcomes)
    verified = sum(o.verified for o in outcomes)
    statuses: Counter[str] = Counter()
    for outcome in outcomes:
        statuses.update(outcome.statuses)

    hallucinated = statuses.get(NOT_IN_CORPUS, 0)
    citations: dict[str, Any] = {
        "emitted": emitted,
        "verified": verified,
        # None y no 1.0 cuando no se emitio ninguna cita: un sistema que rehusa
        # siempre no tiene precision perfecta, tiene precision indefinida.
        "citation_precision": round(verified / emitted, 4) if emitted else None,
        "hallucinated_citation_rate": round(hallucinated / emitted, 4) if emitted else None,
        # Cero por construccion, no por merito: la politica del servidor no deja
        # salir una cita sin verificar. Se reporta para no confundirlo con lo de
        # arriba, que si mide al modelo.
        "served_hallucinated_citation_rate": 0.0,
        "by_status": dict(statuses),
    }

    refusals = [o for o in outcomes if o.correct_refusal is not None]
    answerable = [o for o in outcomes if o.grounded is not None]
    with_quantities = [o for o in outcomes if not o.refused]

    return {
        "citations": citations,
        "refusal": {
            "refusal_recall": _rate([float(o.correct_refusal) for o in refusals], config),
            "false_refusal_rate": _rate([float(o.refused) for o in answerable], config),
        },
        "answers": {
            "grounded_answer_rate": _rate([float(o.grounded) for o in answerable], config),
        },
        "quantities": {
            "unsourced_quantity_rate": _rate(
                [float(bool(o.unsourced_quantities)) for o in with_quantities], config
            ),
            "examples": sorted(
                {q for o in outcomes for q in o.unsourced_quantities}
            ),
        },
    }


def gate_block(results: dict[str, Any]) -> dict[str, Any]:
    """Bloque plano que consume el gate de CI (ver eval/gate.py).

    Lleva `provider` a proposito: el gate rechaza un bloque que venga del
    baseline, y sin este campo no podria distinguirlo.
    """
    return {
        "provider": results["provider"],
        "model": results["model"],
        "split": results["split"],
        "n": results["n"],
        "citation_precision": results["citations"]["citation_precision"],
        "hallucinated_citation_rate": results["citations"]["hallucinated_citation_rate"],
        "refusal_recall": results["refusal"]["refusal_recall"]["rate"],
        "false_refusal_rate": results["refusal"]["false_refusal_rate"]["rate"],
        "grounded_answer_rate": results["answers"]["grounded_answer_rate"]["rate"],
        "usd_per_query": results["cost"]["usd_per_query"],
        "provenance": results["provenance"],
    }


def run(config: Config, *, split: str = "test", provider: str | None = None) -> dict[str, Any]:
    provider = provider or config.get("evaluation.generation.provider")
    engine = AnswerEngine.build(config, provider=provider)
    quantities = quantity_pattern(config.get("evaluation.generation.quantity_units"))

    cases = load_golden_set(config)
    train, test = split_cases(cases, config)
    selected = {"train": train, "test": test, "all": cases}[split]

    outcomes: list[CaseOutcome] = []
    manual_review: list[dict[str, Any]] = []
    costs: list[Cost] = []
    for case in selected:
        with trace_context():
            answer = engine.answer(case.question)
        outcome = score_case(case, answer, quantities)
        outcomes.append(outcome)
        costs.append(compute_cost(config, answer.provider.model, answer.usage))
        log_event(
            "generation.case",
            case_id=case.id,
            type=case.type,
            refused=outcome.refused,
            verified=outcome.verified,
            emitted=outcome.emitted,
        )
        if case.must_not_invent:
            # Criterio en prosa: lo adjudica un humano, no este arnes.
            manual_review.append(
                {
                    "case_id": case.id,
                    "question": case.question,
                    "must_not_invent": case.must_not_invent,
                    "refused": outcome.refused,
                    "answer": outcome.answer,
                    "served_control_ids": outcome.served_control_ids,
                }
            )

    served = engine.chain.providers[0]
    meta = {
        "split": split,
        "provider": provider,
        "model": served.model,
        "n": len(selected),
        "top_k": config.get("generation.context.top_k"),
        "retrieval_method": config.get("retrieval.method"),
        "chunking_strategy": config.get("chunking.active"),
    }
    summary = summarize(outcomes, config)
    return {
        **meta,
        "provenance": provenance_block(config),
        **summary,
        "cost": aggregate_cost(costs, config),
        "manual_review": manual_review,
        "manual_review_note": (
            "`must_not_invent` es un criterio en prosa: estos casos requieren "
            "adjudicacion humana y no entran en ninguna metrica automatica."
        ),
        "per_case": [o.to_dict() for o in outcomes],
    }


def merge_into_ablation(results: dict[str, Any], config: Config) -> Path | None:
    """Inyecta el bloque `generation` en el fichero de la ablacion, que es donde
    el gate de CI lo busca. Si la ablacion no se ha corrido, no hay nada que
    hacer: el gate ya trata el bloque como opcional.

    El baseline extractivo NUNCA se inyecta: copia, asi que su precision es 1.0
    y su alucinacion 0.0 por construccion. Sembrar el gate con esas cifras lo
    dejaria en verde permanente sin haber medido nada.
    """
    if results["provider"] == config.get("generation.baseline_provider"):
        log_event("generation.eval.merge_skipped", reason="baseline", provider=results["provider"])
        return None
    path = config.path("evaluation.ablation.output_path")
    if not path.exists():
        return None
    ablation = json.loads(path.read_text(encoding="utf-8"))
    ablation["generation"] = gate_block(results)
    path.write_text(json.dumps(ablation, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance-mcp-generation-eval")
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--provider", default=None, help="chain | anthropic | openai | extractive")
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config)
    results = run(config, split=args.split, provider=args.provider)

    out_path = Path(args.out) if args.out else config.path("evaluation.generation.output_path")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    merged = None
    if config.get("evaluation.generation.merge_into_ablation"):
        merged = merge_into_ablation(results, config)
    log_event(
        "generation.eval.written",
        path=str(out_path),
        merged_into=str(merged) if merged else None,
        split=args.split,
        provider=results["provider"],
    )
    print(json.dumps(gate_block(results), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
