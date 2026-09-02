"""Logging JSON estructurado, trace id por request y cronometraje por etapa.

Todo evento sale como una linea JSON con el mismo trace_id, para poder
reconstruir una peticion completa (recuperacion -> generacion -> validacion)
desde los logs, incluida la degradacion a proveedor de fallback.
"""
from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from .config import Config

_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("trace_id", default=None)

LOGGER_NAME = "compliance_mcp"


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def current_trace_id() -> str | None:
    return _trace_id.get()


@contextmanager
def trace_context(trace_id: str | None = None) -> Iterator[str]:
    """Fija el trace id para todo lo que se loguee dentro del bloque."""
    tid = trace_id or new_trace_id()
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


class JsonFormatter(logging.Formatter):
    def __init__(self, trace_id_field: str) -> None:
        super().__init__()
        self.trace_id_field = trace_id_field

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "event": record.getMessage(),
            self.trace_id_field: current_trace_id(),
        }
        extra = getattr(record, "fields", None)
        if extra:
            payload.update(extra)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(config: Config) -> logging.Logger:
    """Configura el logger del paquete. Los logs van a stderr porque el
    transporte MCP por stdio usa stdout para el protocolo."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(config.get("logging.level"))
    logger.handlers.clear()
    handler = logging.StreamHandler(stream=sys.stderr)
    if config.get("logging.format") == "json":
        handler.setFormatter(JsonFormatter(config.get("logging.trace_id_field")))
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, level: int = logging.INFO, **fields: Any) -> None:
    get_logger().log(level, event, extra={"fields": fields})


@dataclass
class StageTimings:
    """Latencia acumulada por etapa, en milisegundos."""

    stages: dict[str, float] = field(default_factory=dict)

    def record(self, stage: str, ms: float) -> None:
        self.stages[stage] = round(self.stages.get(stage, 0.0) + ms, 3)

    def total_ms(self) -> float:
        return round(sum(self.stages.values()), 3)

    def as_dict(self) -> dict[str, float]:
        return {**self.stages, "total": self.total_ms()}


@contextmanager
def stage(timings: StageTimings, name: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        timings.record(name, (time.perf_counter() - start) * 1000.0)
