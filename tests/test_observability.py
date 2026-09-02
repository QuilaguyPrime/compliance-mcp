"""Logging estructurado, trace id y cronometraje por etapa."""
from __future__ import annotations

import json
import logging

from compliance_mcp.observability import (
    JsonFormatter,
    StageTimings,
    configure_logging,
    current_trace_id,
    log_event,
    new_trace_id,
    stage,
    trace_context,
)


def test_trace_context_sets_and_restores():
    assert current_trace_id() is None
    with trace_context("abc123") as tid:
        assert tid == "abc123"
        assert current_trace_id() == "abc123"
    assert current_trace_id() is None


def test_nested_trace_contexts_restore_the_outer_id():
    with trace_context("outer"):
        with trace_context("inner"):
            assert current_trace_id() == "inner"
        assert current_trace_id() == "outer"


def test_new_trace_ids_are_distinct():
    assert new_trace_id() != new_trace_id()


def test_log_line_is_json_and_carries_the_trace_id(config):
    formatter = JsonFormatter(config.get("logging.trace_id_field"))
    record = logging.LogRecord(
        name="compliance_mcp", level=logging.INFO, pathname=__file__, lineno=1,
        msg="retrieval.completed", args=(), exc_info=None,
    )
    record.fields = {"hits": 5, "method": "hybrid"}
    with trace_context("trace-1"):
        payload = json.loads(formatter.format(record))
    assert payload["event"] == "retrieval.completed"
    assert payload[config.get("logging.trace_id_field")] == "trace-1"
    assert payload["hits"] == 5
    assert payload["method"] == "hybrid"
    assert payload["level"] == "INFO"


def test_logger_writes_to_stderr_not_stdout(config):
    """El transporte MCP por stdio usa stdout para el protocolo; un log en
    stdout corromperia la sesion."""
    import sys

    logger = configure_logging(config)
    assert logger.handlers
    for handler in logger.handlers:
        assert handler.stream is sys.stderr


def test_log_event_emits_json_with_its_fields(config):
    """La degradacion a proveedor de fallback tiene que ser observable en los
    logs, no solo en el valor de retorno."""
    import io

    logger = configure_logging(config)
    buffer = io.StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter(config.get("logging.trace_id_field")))
    logger.addHandler(handler)
    try:
        with trace_context("t-42"):
            log_event("provider.fallback", **{"from": "anthropic", "to": "openai"})
    finally:
        logger.removeHandler(handler)

    payload = json.loads(buffer.getvalue().strip())
    assert payload["event"] == "provider.fallback"
    assert payload["from"] == "anthropic"
    assert payload["to"] == "openai"
    assert payload[config.get("logging.trace_id_field")] == "t-42"


# ------------------------------------------------------------------ latencias
def test_stage_timings_accumulate_per_stage():
    timings = StageTimings()
    with stage(timings, "retrieval"):
        pass
    with stage(timings, "retrieval"):
        pass
    with stage(timings, "generation"):
        pass
    assert set(timings.stages) == {"retrieval", "generation"}
    assert timings.total_ms() == sum(timings.stages.values())


def test_stage_records_even_when_the_body_raises():
    timings = StageTimings()
    try:
        with stage(timings, "generation"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "generation" in timings.stages


def test_timings_as_dict_includes_a_total():
    timings = StageTimings()
    timings.record("retrieval", 5.0)
    timings.record("validation", 1.0)
    assert timings.as_dict()["total"] == 6.0
