"""Preflight: comprueba, sin llamar a ninguna API, que el sistema puede servir.

Un servidor MCP arranca por stdio dentro de otro proceso. Cuando algo falta
—el corpus, el indice, una clave— el sintoma que ve el usuario es un cliente
que no conecta, o peor, respuestas degradadas sin aviso. Este comando convierte
todos esos fallos en una lista legible antes de arrancar.

No se llama a ningun proveedor: comprobar que una clave existe es gratis,
comprobar que es valida cuesta dinero. Se dice cual de las dos cosas se ha
comprobado.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

from .chunking import chunk_records
from .config import Config, ConfigError, load_config
from .eval.golden import load_golden_set, validate_against_corpus
from .index_manifest import check_entry, read_manifest
from .ingest import read_records
from .observability import configure_logging, log_event
from .provenance import provenance_block
from .retrieval.search import embeddings_path

OK = "ok"
WARN = "warn"
FAIL = "fail"

# Nombres de comprobacion. Son parte de la interfaz: `--require` los usa, y CI
# exige subconjuntos distintos segun lo que ese job vaya a hacer de verdad.
CORPUS = "corpus"
INDEX = "index"
GOLDEN = "golden_set"
SERVE_EXTRA = "serve_extra"
PROVIDERS = "providers"
PRICING = "pricing"
ALL_CHECKS = (CORPUS, INDEX, GOLDEN, SERVE_EXTRA, PROVIDERS, PRICING)


@dataclass(slots=True)
class Check:
    name: str
    status: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def check_corpus(config: Config) -> tuple[Check, list | None]:
    path = config.path("corpus.records_path")
    if not path.exists():
        return Check(CORPUS, FAIL, f"No hay corpus en {path}. Ejecuta `make ingest`."), None
    records = read_records(path)
    withdrawn = sum(1 for r in records if r.status == "withdrawn")
    unresolved = sum(1 for r in records if "{{ insert" in r.statement + r.guidance)
    status = FAIL if unresolved else OK
    detail = f"{len(records)} registros ({withdrawn} retirados)"
    if unresolved:
        detail += f"; {unresolved} con placeholders sin resolver"
    return Check(CORPUS, status, detail), records


def check_index(config: Config, records: list | None) -> Check:
    strategy = config.get("chunking.active")
    path = embeddings_path(config, strategy)
    if not path.exists():
        return Check(INDEX, FAIL, f"Faltan los embeddings en {path}. Ejecuta `make index`.")
    if records is None:
        return Check(INDEX, FAIL, "No se puede verificar sin corpus")

    import numpy as np

    embeddings = np.load(path, mmap_mode="r")
    chunks = chunk_records(records, strategy, config)
    problems = check_entry(
        config, strategy, chunks, rows=int(embeddings.shape[0]), dim=int(embeddings.shape[1])
    )
    if problems:
        return Check(INDEX, FAIL, "; ".join(problems))
    return Check(
        INDEX,
        OK,
        f"estrategia {strategy}: {embeddings.shape[0]} vectores de dim {embeddings.shape[1]}, "
        f"corresponde al corpus actual",
    )


def check_golden_set(config: Config, records: list | None) -> Check:
    try:
        cases = load_golden_set(config)
    except (OSError, KeyError, ValueError) as exc:
        return Check(GOLDEN, FAIL, f"No se pudo cargar: {exc}")
    if records is None:
        return Check(GOLDEN, WARN, f"{len(cases)} casos; sin corpus no se puede validar")
    errors = validate_against_corpus(cases, {r.control_id for r in records})
    if errors:
        return Check(GOLDEN, FAIL, "; ".join(errors[:3]))
    return Check(GOLDEN, OK, f"{len(cases)} casos, coherentes con el corpus")


def check_providers(config: Config) -> Check:
    """Solo mira que exista la credencial. Validarla cuesta una llamada."""
    from .generation.providers import API_KEY_ENV, available_providers

    declared = [spec["name"] for spec in config.get("generation.providers")]
    usable = available_providers(config)
    missing = [n for n in declared if n not in usable]
    if not usable:
        envs = ", ".join(API_KEY_ENV[n] for n in declared)
        return Check(
            PROVIDERS,
            FAIL,
            f"Ninguno tiene credencial en el entorno ({envs}). answer_question no puede "
            f"responder; search_controls y get_control si funcionan.",
        )
    detail = f"con credencial: {', '.join(usable)} (existencia, no validez)"
    if missing:
        return Check(PROVIDERS, WARN, detail + f"; sin credencial: {', '.join(missing)}")
    return Check(PROVIDERS, OK, detail)


def check_pricing(config: Config) -> Check:
    """Un modelo servido sin precio declarado hace que el coste por consulta
    salga como desconocido en el informe."""
    priced = config.get("generation.pricing.usd_per_mtok")
    missing = [
        spec["model"] for spec in config.get("generation.providers") if spec["model"] not in priced
    ]
    if missing:
        return Check(PRICING, WARN, f"sin precio declarado: {', '.join(missing)}")
    return Check(
        PRICING, OK, f"declarados y contrastados el {config.get('generation.pricing.checked_at')}"
    )


def check_serve_extra() -> Check:
    missing = []
    for module, package in (("mcp", "mcp"), ("anthropic", "anthropic"), ("openai", "openai")):
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    if missing:
        return Check(
            SERVE_EXTRA,
            FAIL,
            f"faltan {', '.join(missing)}. Instala: pip install -e '.[serve]'",
        )
    return Check(SERVE_EXTRA, OK, "mcp, anthropic y openai importables")


def run(config: Config) -> list[Check]:
    corpus_check, records = check_corpus(config)
    return [
        corpus_check,
        check_index(config, records),
        check_golden_set(config, records),
        check_serve_extra(),
        check_providers(config),
        check_pricing(config),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compliance-mcp-doctor",
        description="Comprueba que el sistema puede servir. No llama a ninguna API.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--strict", action="store_true", help="Tratar los avisos como fallos"
    )
    parser.add_argument(
        "--require",
        default=None,
        help=(
            "Comprobaciones que deben pasar, separadas por comas "
            f"({', '.join(ALL_CHECKS)}). Por defecto todas. Las demas se "
            "informan pero no deciden el codigo de salida: un job que solo "
            "evalua recuperacion no necesita credenciales de proveedor."
        ),
    )
    args = parser.parse_args(argv)

    required = set(ALL_CHECKS)
    if args.require:
        required = {name.strip() for name in args.require.split(",") if name.strip()}
        unknown = required - set(ALL_CHECKS)
        if unknown:
            parser.error(
                f"comprobacion desconocida: {', '.join(sorted(unknown))}. "
                f"Opciones: {', '.join(ALL_CHECKS)}"
            )

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"FALLO config: {exc}", file=sys.stderr)
        return 1
    configure_logging(config)

    checks = run(config)
    manifest = read_manifest(config)
    symbols = {OK: "ok   ", WARN: "aviso", FAIL: "FALLO"}
    for check in checks:
        scope = "" if check.name in required else "  (informativo)"
        print(f"[{symbols[check.status]}] {check.name}: {check.detail}{scope}")

    provenance = provenance_block(config)
    print(f"\ncommit: {provenance['git_sha']}")
    print(f"corpus: {provenance['corpus_digest'][:26]}...")
    print(f"indices con manifiesto: {', '.join(sorted(manifest.get('entries', {}))) or 'ninguno'}")

    failed = [c for c in checks if c.status == FAIL and c.name in required]
    warned = [c for c in checks if c.status == WARN and c.name in required]
    log_event(
        "doctor.completed",
        required=sorted(required),
        failed=[c.name for c in failed],
        warned=[c.name for c in warned],
        ignored=[c.name for c in checks if c.status == FAIL and c.name not in required],
        strict=args.strict,
    )
    if failed or (args.strict and warned):
        names = ", ".join(c.name for c in (failed or warned))
        print(f"\nPreflight fallido: {names}.", file=sys.stderr)
        return 1
    print(f"\nPreflight OK ({', '.join(sorted(required))}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
