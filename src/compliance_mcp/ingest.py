"""Ingest OSCAL -> registros canonicos de control.

Decisiones tomadas en fase 1, todas medidas sobre el corpus real:

* Los placeholders `{{ insert: param, X }}` se resuelven hasta punto fijo. Anidan
  dentro de `select.choice`, asi que una sola pasada deja 44 controles con
  placeholders crudos y cuatro pasadas dejan cero.
* El statement conserva su jerarquia (`a.`, `b.`, `1.`) al aplanarse; sin las
  etiquetas un statement de ocho sub-items es ilegible y no se puede citar
  "AC-2 a.".
* Los 182 controles retirados se conservan marcados con su destino
  `incorporated-into`, para poder responder "AC-3(1) se incorporo a AC-3" en vez
  de "no existe".
* El contenido de evaluacion (SP 800-53A) se guarda en un campo aparte, no
  mezclado con el normativo, porque la ablacion decide si se indexa o no.
"""
from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .config import Config

PARAM_PATTERN = re.compile(r"\{\{\s*insert:\s*param,\s*([^\s}]+)\s*\}\}")
# Deja la puntuacion pegada tras sustituir un placeholder ("... [time period] ;" -> "...;")
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([;.,])")


@dataclass(slots=True)
class Reference:
    title: str
    url: str


@dataclass(slots=True)
class ControlRecord:
    """Un nodo del catalogo: control base o enhancement."""

    control_id: str          # "ac-2.1" — clave canonica en minusculas
    label: str               # "AC-2(1)" — forma citable para humanos
    family_id: str
    family_title: str
    title: str
    kind: str                # "control" | "enhancement"
    status: str              # "active" | "withdrawn"
    parent_id: str | None
    parent_title: str | None
    statement: str
    guidance: str
    assessment: str
    related: list[str] = field(default_factory=list)
    required: list[str] = field(default_factory=list)
    incorporated_into: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    baselines: list[str] = field(default_factory=list)
    enhancement_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ControlRecord:
        d = dict(d)
        d["references"] = [Reference(**r) for r in d.get("references", [])]
        return cls(**d)


# --------------------------------------------------------------------------- #
# Parametros
# --------------------------------------------------------------------------- #

def _param_label(param: dict[str, Any]) -> str:
    """Etiqueta legible de un parametro OSCAL, en orden de preferencia."""
    if param.get("label"):
        return param["label"]
    select = param.get("select")
    if select and select.get("choice"):
        return " or ".join(str(c) for c in select["choice"])
    guidelines = param.get("guidelines")
    if guidelines and guidelines[0].get("prose"):
        return guidelines[0]["prose"].rstrip(";")
    return param["id"]


def build_param_map(catalog: dict[str, Any]) -> dict[str, str]:
    """id de parametro -> etiqueta. Los parametros son visibles en todo el
    catalogo, asi que se construye un mapa global."""
    out: dict[str, str] = {}

    def walk(control: dict[str, Any]) -> None:
        for param in control.get("params") or []:
            out[param["id"]] = _param_label(param)
        for child in control.get("controls") or []:
            walk(child)

    for group in catalog["groups"]:
        for control in group.get("controls") or []:
            walk(control)
    return out


def resolve_params(text: str, param_map: dict[str, str], passes: int, template: str) -> str:
    """Sustituye placeholders hasta punto fijo (max `passes` iteraciones)."""
    if not text:
        return ""
    for _ in range(passes):
        replaced = PARAM_PATTERN.sub(
            lambda m: template.format(label=param_map.get(m.group(1), m.group(1))), text
        )
        if replaced == text:
            break
        text = replaced
    return _SPACE_BEFORE_PUNCT.sub(r"\1", text).strip()


# --------------------------------------------------------------------------- #
# Partes
# --------------------------------------------------------------------------- #

def _prop(props: Iterable[dict[str, Any]], name: str, *, without_class: bool = False) -> str | None:
    for p in props or []:
        if p.get("name") != name:
            continue
        if without_class and "class" in p:
            continue
        return p.get("value")
    return None


def flatten_part(
    part: dict[str, Any],
    param_map: dict[str, str],
    passes: int,
    template: str,
    depth: int = 0,
    out: list[str] | None = None,
) -> list[str]:
    """Aplana una parte OSCAL conservando etiquetas e indentacion jerarquicas."""
    if out is None:
        out = []
    label = _prop(part.get("props", []), "label") or ""
    prose = resolve_params(part.get("prose", ""), param_map, passes, template)
    if prose:
        indent = "  " * depth
        out.append(f"{indent}{label} {prose}".strip() if label else f"{indent}{prose}")
    child_depth = depth + 1 if (prose or label) else depth
    for sub in part.get("parts") or []:
        flatten_part(sub, param_map, passes, template, child_depth, out)
    return out


def build_reference_map(catalog: dict[str, Any]) -> dict[str, Reference]:
    out: dict[str, Reference] = {}
    for res in catalog.get("back-matter", {}).get("resources", []):
        url = next((rl.get("href", "") for rl in res.get("rlinks") or []), "")
        title = res.get("title") or res.get("citation", {}).get("text", "")
        out[res["uuid"]] = Reference(title=title, url=url)
    return out


def load_baselines(config: Config) -> dict[str, set[str]]:
    """control_id -> baselines a los que pertenece, desde los perfiles OSCAL.

    El catalogo por si solo no contiene el mapeo low/moderate/high; vive en
    perfiles separados que se versionan junto al catalogo en data/raw/.
    """
    out: dict[str, set[str]] = {}
    for name in config.section("corpus.baseline_profiles"):
        path = config.path(f"corpus.baseline_profiles.{name}")
        profile = json.loads(path.read_text(encoding="utf-8"))["profile"]
        ids: set[str] = set()
        for imp in profile.get("imports", []):
            for inc in imp.get("include-controls", []):
                ids.update(inc.get("with-ids", []))
        out[name] = ids
    return out


# --------------------------------------------------------------------------- #
# Construccion de registros
# --------------------------------------------------------------------------- #

def build_records(config: Config) -> list[ControlRecord]:
    catalog = json.loads(config.path("corpus.catalog_path").read_text(encoding="utf-8"))["catalog"]
    param_map = build_param_map(catalog)
    ref_map = build_reference_map(catalog)
    baselines = load_baselines(config)

    passes = config.get("ingest.param_resolution_passes")
    template = config.get("ingest.param_template")
    assessment_prefixes = tuple(config.get("ingest.assessment_part_prefixes"))
    include_withdrawn = config.get("ingest.include_withdrawn")

    records: list[ControlRecord] = []

    def emit(control: dict[str, Any], family: dict[str, Any], parent: dict[str, Any] | None) -> None:
        props = control.get("props", [])
        status = "withdrawn" if _prop(props, "status") == "withdrawn" else "active"
        if status == "withdrawn" and not include_withdrawn:
            return

        buckets: dict[str, list[str]] = {"statement": [], "guidance": [], "assessment": []}
        for part in control.get("parts") or []:
            name = part.get("name", "")
            key = "assessment" if name.startswith(assessment_prefixes) else name
            if key in buckets:
                buckets[key].extend(flatten_part(part, param_map, passes, template))

        related, required, incorporated, refs = [], [], [], []
        for link in control.get("links") or []:
            href = (link.get("href") or "").lstrip("#")
            rel = link.get("rel")
            if rel == "related":
                related.append(href)
            elif rel == "required":
                required.append(href)
            elif rel in ("incorporated-into", "moved-to"):
                incorporated.append(href)
            elif rel == "reference" and href in ref_map:
                refs.append(ref_map[href])

        records.append(
            ControlRecord(
                control_id=control["id"],
                label=_prop(props, "label", without_class=True) or control["id"].upper(),
                family_id=family["id"],
                family_title=family["title"],
                title=control.get("title", ""),
                kind="enhancement" if parent else "control",
                status=status,
                parent_id=parent["id"] if parent else None,
                parent_title=parent.get("title") if parent else None,
                statement="\n".join(buckets["statement"]),
                guidance="\n".join(buckets["guidance"]),
                assessment="\n".join(buckets["assessment"]),
                related=related,
                required=required,
                incorporated_into=incorporated,
                references=refs,
                baselines=sorted(b for b, ids in baselines.items() if control["id"] in ids),
                enhancement_ids=[c["id"] for c in control.get("controls") or []],
            )
        )
        for child in control.get("controls") or []:
            emit(child, family, parent=control)

    for group in catalog["groups"]:
        for control in group.get("controls") or []:
            emit(control, group, parent=None)
    return records


def write_records(records: list[ControlRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")


def read_records(path: Path) -> list[ControlRecord]:
    if not path.exists():
        raise FileNotFoundError(f"No hay registros en {path}. Ejecuta `make ingest` primero.")
    with path.open(encoding="utf-8") as fh:
        return [ControlRecord.from_dict(json.loads(line)) for line in fh if line.strip()]
