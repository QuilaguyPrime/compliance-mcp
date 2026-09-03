"""El contexto que ve el modelo, y que es a la vez el universo de lo citable.

La verificacion no compara contra el registro completo del control sino contra
este objeto. La diferencia importa: si un modelo cita, palabra por palabra, un
trozo real de AC-2 que nunca se le enseno, eso no es una cita, es memoria
parametrica que resulto acertar. Verificar contra el contexto lo detecta;
verificar contra el corpus lo premia.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import Config
from ..ingest import ControlRecord


@dataclass(slots=True)
class ContextEntry:
    control_id: str
    label: str
    title: str
    family_title: str
    status: str
    baselines: list[str]
    # part -> texto tal y como se le muestra al modelo (ya truncado).
    parts: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class AnswerContext:
    """Contexto de una sola pregunta."""

    entries: list[ContextEntry] = field(default_factory=list)
    # Ids de TODO el corpus, no solo del contexto: distingue "control que no
    # existe" (alucinacion dura) de "control real que no se le paso".
    corpus_ids: set[str] = field(default_factory=set)

    def entry(self, control_id: str) -> ContextEntry | None:
        for e in self.entries:
            if e.control_id == control_id:
                return e
        return None

    def label_for(self, control_id: str) -> str | None:
        entry = self.entry(control_id)
        return entry.label if entry else None

    @property
    def control_ids(self) -> list[str]:
        return [e.control_id for e in self.entries]


def _truncate_on_line(text: str, max_chars: int) -> str:
    """Corta en frontera de linea. Cortar a medias un item numerado del
    statement produce citas que no se pueden verificar por culpa del corte."""
    if len(text) <= max_chars:
        return text
    head = text[:max_chars]
    cut = head.rfind("\n")
    return head[:cut] if cut > 0 else head


def build_context(
    records: list[ControlRecord], corpus_ids: set[str], config: Config
) -> AnswerContext:
    parts_wanted: list[str] = config.get("generation.context.parts")
    max_chars: int = config.get("generation.context.max_part_chars")

    entries: list[ContextEntry] = []
    for record in records:
        available = {
            "statement": record.statement,
            "guidance": record.guidance,
            "assessment": record.assessment,
        }
        parts = {
            name: _truncate_on_line(available[name], max_chars)
            for name in parts_wanted
            if available.get(name)
        }
        if not parts and record.status == "withdrawn":
            # Un retirado no tiene statement. Se expone su destino para poder
            # explicar la retirada citandola en vez de rehusar.
            targets = ", ".join(t.upper() for t in record.incorporated_into) or "n/a"
            parts = {"statement": f"Withdrawn. Incorporated into: {targets}"}
        entries.append(
            ContextEntry(
                control_id=record.control_id,
                label=record.label,
                title=record.title,
                family_title=record.family_title,
                status=record.status,
                baselines=record.baselines,
                parts=parts,
            )
        )
    return AnswerContext(entries=entries, corpus_ids=corpus_ids)
