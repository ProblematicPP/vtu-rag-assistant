"""Keep unit tests isolated from a developer's local .env and running services."""

import os
import tempfile

import pytest

_ENV_PREFIXES = ("LANGFUSE_", "JINA_", "OLLAMA_", "LLM_", "POSTGRES_", "OPENSEARCH_", "REDIS_")


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    for key in list(os.environ):
        if key.startswith(_ENV_PREFIXES):
            monkeypatch.delenv(key, raising=False)
    # pydantic-settings would otherwise read the project's .env
    monkeypatch.chdir(tempfile.mkdtemp())
