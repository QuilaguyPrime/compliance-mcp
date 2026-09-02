from __future__ import annotations

import pytest

from compliance_mcp.config import load_config
from compliance_mcp.ingest import build_records


@pytest.fixture(scope="session")
def config():
    return load_config()


@pytest.fixture(scope="session")
def records(config):
    return build_records(config)


@pytest.fixture(scope="session")
def records_by_id(records):
    return {r.control_id: r for r in records}
