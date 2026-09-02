"""Estrategias de chunking.

Ningun control se parte por ventana de tokens: la estructura del documento ya
define los limites semanticos (el p90 del statement es 62 palabras y el de la
guidance 197). Lo que cambia entre estrategias es que partes entran al chunk y
si cada parte va en su propio chunk.

Medido en fase 1 con el tokenizer real del embedder (limite 512 tokens):

    A  1196 chunks  p50=130  p90=317  >512: 26  (2.2%)
    B  1196 chunks  p50=256  p90=539  >512: 138 (11.5%)   <- trunca el 11.5%
    C  2210 chunks  p50=71   p90=207  >512: 12  (0.5%)

Todo chunk arrastra su contexto jerarquico (FAMILIA > CONTROL PADRE > CONTROL),
de modo que un enhancement se puede interpretar sin su padre delante.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .ingest import ControlRecord


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    control_id: str
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"chunk_id": self.chunk_id, "control_id": self.control_id, "text": self.text}


def build_header(record: ControlRecord, config: Config) -> str:
    """FAMILIA (ID) [> PADRE (ID)] > ETIQUETA Titulo"""
    parent_segment = ""
    if record.parent_id:
        parent_segment = config.get("chunking.parent_segment_template").format(
            parent_title=record.parent_title, parent_id=record.parent_id.upper()
        )
    return config.get("chunking.header_template").format(
        family_title=record.family_title,
        family_id=record.family_id.upper(),
        parent_segment=parent_segment,
        label=record.label,
        title=record.title,
    )


def _part_texts(record: ControlRecord, include_parts: list[str]) -> list[tuple[str, str]]:
    available = {
        "statement": record.statement,
        "guidance": record.guidance,
        "assessment": record.assessment,
    }
    return [(name, available[name]) for name in include_parts if available.get(name)]


def chunk_records(records: list[ControlRecord], strategy: str, config: Config) -> list[Chunk]:
    spec = config.section(f"chunking.strategies.{strategy}")
    include_parts: list[str] = spec["include_parts"]
    split_by_part: bool = spec["split_by_part"]
    part_marker = config.get("chunking.part_marker_template")
    withdrawn_marker = config.get("chunking.withdrawn_marker_template")

    chunks: list[Chunk] = []
    for record in records:
        header = build_header(record, config)
        parts = _part_texts(record, include_parts)

        if not parts:
            # Controles retirados: no tienen statement. Se indexan con su destino
            # para poder explicar la retirada en vez de decir "no existe".
            targets = ", ".join(t.upper() for t in record.incorporated_into) or "n/a"
            body = withdrawn_marker.format(targets=targets)
            chunks.append(
                Chunk(f"{record.control_id}::withdrawn", record.control_id, f"{header}\n{body}")
            )
            continue

        if split_by_part:
            for name, text in parts:
                marker = part_marker.format(part_name=name)
                chunks.append(
                    Chunk(
                        f"{record.control_id}::{name}",
                        record.control_id,
                        f"{header}\n{marker}\n{text}",
                    )
                )
        else:
            body = "\n".join(text for _, text in parts)
            chunks.append(
                Chunk(f"{record.control_id}::full", record.control_id, f"{header}\n{body}")
            )
    return chunks
