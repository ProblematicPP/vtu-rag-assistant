import operator
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

from vtu_rag.schemas.ask import Source
from vtu_rag.services.search import SearchFilters, SearchHit
from vtu_rag.services.tracing import NOOP_TRACE, Trace


class AgentState(TypedDict, total=False):
    question: str
    # Query used for the next retrieval; starts as the question, replaced on rewrite
    query: str
    rewritten_queries: Annotated[list[str], operator.add]
    attempts: int

    in_scope: bool
    guardrail_score: int | None
    guardrail_reason: str | None

    hits: list[SearchHit]
    relevant_hits: list[SearchHit]
    search_mode: str | None

    answer: str
    sources: list[Source]

    # Append-only log of what each node did, returned to the client
    steps: Annotated[list[dict[str, Any]], operator.add]


@dataclass
class AgentContext:
    """Per-request, non-state inputs available to every node."""

    filters: SearchFilters
    top_k: int = 5
    trace: Trace = NOOP_TRACE
