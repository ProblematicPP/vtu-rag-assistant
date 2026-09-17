"""Langfuse tracing that degrades to a no-op when keys are missing or Langfuse is down.

Usage:
    trace = tracer.start_trace("agentic-ask", input={...})
    span = trace.span("retrieve", input={...})
    span.end(output={...})
    trace.generation("generate", response, input=messages)
    trace.end(output={...})
"""

import logging
from typing import Any

from vtu_rag.config import LangfuseSettings
from vtu_rag.services.llm.base import LLMResponse

logger = logging.getLogger(__name__)


class Span:
    def __init__(self, client: Any = None):
        self._client = client

    def end(self, output: Any = None, **metadata: Any) -> None:
        if self._client is None:
            return
        try:
            self._client.end(output=output, metadata=metadata or None)
        except Exception as exc:  # tracing must never break a request
            logger.debug("Langfuse span end failed: %s", exc)


class Trace:
    def __init__(self, client: Any = None):
        self._client = client

    @property
    def id(self) -> str | None:
        return getattr(self._client, "id", None)

    def span(self, name: str, input: Any = None, **metadata: Any) -> Span:
        if self._client is None:
            return Span()
        try:
            return Span(self._client.span(name=name, input=input, metadata=metadata or None))
        except Exception as exc:
            logger.debug("Langfuse span failed: %s", exc)
            return Span()

    def generation(self, name: str, response: LLMResponse, input: Any = None) -> None:
        if self._client is None:
            return
        try:
            usage = None
            if response.prompt_tokens is not None or response.completion_tokens is not None:
                usage = {
                    "input": response.prompt_tokens or 0,
                    "output": response.completion_tokens or 0,
                }
            self._client.generation(
                name=name,
                model=response.model,
                input=input,
                output=response.content,
                usage_details=usage,
                metadata={"latency_ms": response.latency_ms},
            )
        except Exception as exc:
            logger.debug("Langfuse generation failed: %s", exc)

    def end(self, output: Any = None, **metadata: Any) -> None:
        if self._client is None:
            return
        try:
            self._client.update(output=output, metadata=metadata or None)
        except Exception as exc:
            logger.debug("Langfuse trace update failed: %s", exc)


NOOP_TRACE = Trace()


class Tracer:
    def __init__(self, settings: LangfuseSettings):
        self._langfuse = None
        if not settings.enabled:
            logger.info("Langfuse keys not set: tracing disabled")
            return
        try:
            from langfuse import Langfuse

            self._langfuse = Langfuse(
                public_key=settings.public_key,
                secret_key=settings.secret_key,
                host=settings.host,
            )
            logger.info("Langfuse tracing enabled (%s)", settings.host)
        except Exception as exc:
            logger.warning("Could not initialise Langfuse, tracing disabled: %s", exc)

    @property
    def enabled(self) -> bool:
        return self._langfuse is not None

    def start_trace(self, name: str, input: Any = None, **metadata: Any) -> Trace:
        if self._langfuse is None:
            return NOOP_TRACE
        try:
            return Trace(self._langfuse.trace(name=name, input=input, metadata=metadata or None))
        except Exception as exc:
            logger.debug("Langfuse trace failed: %s", exc)
            return NOOP_TRACE

    def flush(self) -> None:
        if self._langfuse is not None:
            try:
                self._langfuse.flush()
            except Exception as exc:
                logger.debug("Langfuse flush failed: %s", exc)
