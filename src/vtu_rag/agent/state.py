import operator
from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from vtu_rag.schemas.ask import Source, Turn
from vtu_rag.services.search import SearchFilters, SearchHit
from vtu_rag.services.tracing import NOOP_TRACE, Trace


class AgentState(TypedDict, total=False):
    question: str
    # The question as it will be searched and answered: the same text, unless it
    # was a follow-up ("explain them briefly") resolved against the thread
    standalone: str
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
    # Earlier turns of the conversation, oldest first
    history: list[Turn] = field(default_factory=list)
    # Marks on a paper, when known; the answer is sized to them
    marks: int | None = None
