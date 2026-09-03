"""Gate de evaluacion para CI.

Lee los resultados de la ablacion y los compara con los umbrales de config.yaml.
Falla con codigo distinto de cero si el sistema se degrada por debajo del
umbral. Los umbrales viven en `gates:` en config.yaml, no en el YAML del
workflow: asi el criterio de aceptacion se versiona junto al codigo que evalua.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from ..config import Config, load_config
from ..observability import configure_logging, log_event
from ..provenance import INDEX_CONFIG_KEYS, corpus_digest, digest_config


def check_provenance(results: dict[str, Any], config: Config) -> list[str]:
    """Los resultados tienen que venir del corpus y la config de ahora.

    Un fichero de resultados commiteado de una corrida anterior pasa el gate sin
    haber medido nada de lo que hay en el arbol. Es la misma familia de verde
    vacuo que sembrar el gate con el baseline.
    """
    provenance = results.get("provenance")
    if provenance is None:
        return [
            (
                "Los resultados no llevan bloque de procedencia: no se puede saber de que "
                "corpus ni de que configuracion salieron. Vuelve a correr la evaluacion."
            )
        ]
    failures: list[str] = []
    # Dos senales de suciedad, porque hay dos generaciones de artefactos: el
    # campo booleano lo estampan los generadores actuales, y el sufijo del sha
    # es como se marcaba antes de que existiera el campo. Un artefacto viejo
    # tiene que seguir siendo rechazado.
    sha = str(provenance.get("git_sha") or "")
    if provenance.get("dirty") is True or sha.endswith("-dirty"):
        failures.append(
            "Los resultados se produjeron sobre un arbol con cambios sin commitear "
            f"(git_sha={sha or 'desconocido'}), asi que no se puede saber que codigo "
            "los produjo. Vuelve a correr la evaluacion desde un arbol limpio."
        )
    current_corpus = corpus_digest(config)
    if provenance.get("corpus_digest") != current_corpus:
        failures.append(
            "Los resultados se produjeron con otro corpus "
            f"({str(provenance.get('corpus_digest'))[:19]}... frente al actual "
            f"{current_corpus[:19]}...). Vuelve a correr la evaluacion."
        )
    current_config = digest_config(config, INDEX_CONFIG_KEYS + ["retrieval"])
    if provenance.get("config_digest") != current_config:
        failures.append(
            "Los resultados se produjeron con otra configuracion de ingest, chunking o "
            "recuperacion. Vuelve a correr la evaluacion."
        )
    return failures


def check(results: dict[str, Any], config: Config) -> list[str]:
    failures: list[str] = check_provenance(results, config)

    strategy = config.get("chunking.active")
    method = config.get("retrieval.method")
    cell = results.get("grid", {}).get(strategy, {}).get(method)
    if cell is None:
        return [f"La ablacion no contiene la celda {strategy}/{method}"]

    min_recall = config.get("gates.min_recall_at_5")
    observed = cell.get("recall@5")
    if observed is None:
        failures.append("La celda evaluada no reporta recall@5")
    elif observed < min_recall:
        failures.append(
            f"recall@5 = {observed:.3f} por debajo del umbral {min_recall:.3f} "
            f"({strategy}/{method}, n={cell.get('n')})"
        )

    generation = results.get("generation")
    if generation:
        # El baseline extractivo saca precision 1.0 y alucinacion 0.0 porque
        # copia, no porque acierte. Aceptarlo como evidencia dejaria el gate en
        # verde sin haber medido al generador que se sirve.
        if generation.get("provider") == config.get("generation.baseline_provider"):
            failures.append(
                "El bloque de generacion procede del baseline "
                f"'{generation['provider']}'; no es evidencia sobre el sistema servido"
            )
        precision = generation.get("citation_precision")
        min_precision = config.get("gates.min_citation_precision")
        if precision is not None and precision < min_precision:
            failures.append(
                f"citation_precision = {precision:.3f} por debajo de {min_precision:.3f}"
            )
        hallucinated = generation.get("hallucinated_citation_rate")
        max_hallucinated = config.get("gates.max_hallucinated_citation_rate")
        if hallucinated is not None and hallucinated > max_hallucinated:
            failures.append(
                f"hallucinated_citation_rate = {hallucinated:.3f} por encima de "
                f"{max_hallucinated:.3f}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="compliance-mcp-gate")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    configure_logging(config)

    path = config.path("evaluation.ablation.output_path")
    if args.results:
        from pathlib import Path

        path = Path(args.results)
    if not path.exists():
        print(f"FALLO: no hay resultados de evaluacion en {path}", file=sys.stderr)
        return 1

    results = json.loads(path.read_text(encoding="utf-8"))
    failures = check(results, config)

    strategy = config.get("chunking.active")
    cell = results.get("grid", {}).get(strategy, {}).get(config.get("retrieval.method"), {})
    log_event(
        "gate.evaluated",
        passed=not failures,
        strategy=strategy,
        recall_at_5=cell.get("recall@5"),
        threshold=config.get("gates.min_recall_at_5"),
        split=results.get("split"),
        failures=failures,
    )

    if failures:
        print("GATE FALLIDO:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"GATE OK: recall@5 = {cell.get('recall@5')} (umbral {config.get('gates.min_recall_at_5')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
