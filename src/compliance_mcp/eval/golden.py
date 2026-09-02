"""Carga y particion del golden set."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import Config

ANSWERABLE = "answerable"
UNANSWERABLE = "unanswerable"
ADVERSARIAL = "adversarial"


@dataclass(slots=True)
class GoldenCase:
    id: str
    type: str
    question: str
    expected_control_ids: list[str] = field(default_factory=list)
    acceptable_control_ids: list[str] = field(default_factory=list)
    style: str | None = None
    subtype: str | None = None
    must_refuse: bool = False
    must_not_invent: str | None = None
    notes: str | None = None

    @property
    def scorable_for_retrieval(self) -> bool:
        """Solo los casos con ground truth entran en recall/MRR/nDCG.

        Los casos de rehuso no tienen control correcto, asi que meterlos en el
        denominador de recall castigaria al sistema por hacer lo correcto.
        """
        return bool(self.expected_control_ids)

    @property
    def relevant(self) -> set[str]:
        return set(self.expected_control_ids) | set(self.acceptable_control_ids)


def load_golden_set(config: Config, path: Path | None = None) -> list[GoldenCase]:
    p = path or config.path("evaluation.golden_set_path")
    data: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    cases = [GoldenCase(**case) for case in data["cases"]]
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("El golden set tiene ids duplicados")
    return cases


def split_cases(cases: list[GoldenCase], config: Config) -> tuple[list[GoldenCase], list[GoldenCase]]:
    """Particion train/test determinista por hash del id del caso.

    Es estable frente a reordenaciones y a anadir casos nuevos, y evita el
    pecado de ajustar hiperparametros sobre el mismo conjunto que se reporta.
    """
    fraction = config.get("evaluation.split.train_fraction")
    seed = config.get("evaluation.split.seed")
    train, test = [], []
    for case in cases:
        digest = hashlib.sha256(f"{seed}:{case.id}".encode()).digest()
        bucket = int.from_bytes(digest[:8], "big") / 2**64
        (train if bucket < fraction else test).append(case)
    return train, test


def validate_against_corpus(cases: list[GoldenCase], control_ids: set[str]) -> list[str]:
    """Devuelve los errores encontrados. Un ground truth que no existe en el
    corpus es un fallo del golden set, no del sistema."""
    errors: list[str] = []
    for case in cases:
        for cid in case.expected_control_ids + case.acceptable_control_ids:
            if cid not in control_ids:
                errors.append(f"{case.id}: el control '{cid}' no existe en el corpus")
        if case.must_refuse and case.expected_control_ids:
            errors.append(f"{case.id}: must_refuse=true pero declara expected_control_ids")
        if case.type == ANSWERABLE and not case.expected_control_ids:
            errors.append(f"{case.id}: caso respondible sin ground truth")
    return errors
