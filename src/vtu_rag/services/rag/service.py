"""Single-pass RAG: retrieve → generate with citations."""

import time
from collections.abc import Awaitable, Callable

from vtu_rag.schemas.ask import AskResponse, Source, Turn
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.llm import ChatMessage, LLMProvider
from vtu_rag.services.rag import prompts
from vtu_rag.services.rag.context import (
    drop_diagram_narration,
    format_context,
    mark_cited,
    scope_description,
    strip_citation_markers,
)
from vtu_rag.services.rag.followup import FollowUpResolver
from vtu_rag.services.search import SearchFilters, SearchHit, SearchService
from vtu_rag.services.tracing import NOOP_TRACE, Trace, Tracer


async def no_diagram(hits: list[SearchHit], question: str) -> bool:
    """Default probe: assume nothing will be shown, so a sketch is allowed."""
    return False


# Given the retrieved excerpts and the question, is a diagram going to be shown?
DiagramProbe = Callable[[list[SearchHit], str], Awaitable[bool]]


class AnswerGenerator:
    """Shared by the simple RAG endpoint and the agent's generate node."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def generate(
        self,
        question: str,
        hits: list[SearchHit],
        filters: SearchFilters,
        trace: Trace = NOOP_TRACE,
        diagram_shown: bool = False,
        marks: int | None = None,
    ) -> tuple[str, list[Source]]:
        if not hits:
            return prompts.NO_CONTEXT_ANSWER.format(scope=scope_description(filters)), []

        context, sources = format_context(hits)
        system = prompts.ANSWER_SYSTEM
        system += prompts.DIAGRAM_SHOWN if diagram_shown else prompts.DIAGRAM_MISSING
        if marks:
            # One well-explained point per two marks is about right for VTU
            system += prompts.MARKS_HINT.format(marks=marks, points=max(3, round(marks / 2)))
        messages = [
            ChatMessage("system", system),
            ChatMessage("user", prompts.ANSWER_USER.format(question=question, context=context)),
        ]
        response = await self.llm.generate(messages)
        trace.generation("generate-answer", response, input=[m.as_dict() for m in messages])
        answer = response.content.strip()
        cleaned = strip_citation_markers(answer)
        if diagram_shown:
            cleaned = drop_diagram_narration(cleaned)
        # Keep the cited flags (they order the diagrams) but drop the markers
        return cleaned, mark_cited(answer, sources)


class RAGService:
    def __init__(
        self,
        search: SearchService,
        llm: LLMProvider,
        cache: ResponseCache,
        tracer: Tracer,
        diagram_probe: DiagramProbe | None = None,
    ):
        self.search = search
        self.llm = llm
        self.cache = cache
        self.tracer = tracer
        self.generator = AnswerGenerator(llm)
        self.followup = FollowUpResolver(llm)
        # Answers the question "will a diagram from the notes be shown?"
        self.diagram_probe = diagram_probe or no_diagram

    def cache_key(
        self,
        kind: str,
        question: str,
        filters: SearchFilters,
        top_k: int,
        marks: int | None = None,
    ) -> str:
        return ResponseCache.make_key(
            kind,
            question=" ".join(question.lower().split()),
            filters=filters.cache_key(),
            top_k=top_k,
            marks=marks or 0,
            provider=self.llm.name,
            model=self.llm.model,
        )

    async def ask(
        self,
        question: str,
        filters: SearchFilters,
        top_k: int = 5,
        use_cache: bool = True,
        marks: int | None = None,
        history: list[Turn] | None = None,
    ) -> AskResponse:
        started = time.perf_counter()
        # Resolved before the cache is consulted, so "explain them briefly" after
        # two different questions doesn't hit the same entry
        asked = await self.followup.resolve(question, history)
        key = self.cache_key("ask", asked, filters, top_k, marks)
        if use_cache and (cached := await self.cache.get(key)):
            return AskResponse.model_validate(cached | {"cached": True})

        trace = self.tracer.start_trace(
            "ask", input={"question": asked, "filters": filters.cache_key(), "top_k": top_k}
        )
        span = trace.span("retrieve", input={"query": asked})
        result = await self.search.search(asked, filters, size=top_k)
        span.end(output={"hits": [h.doc_id for h in result.hits]}, mode=result.mode.value)

        diagram_shown = await self.diagram_probe(result.hits, asked)
        answer, sources = await self.generator.generate(
            asked, result.hits, filters, trace, diagram_shown=diagram_shown, marks=marks
        )
        response = AskResponse(
            question=question,
            resolved_question=asked if asked != question else None,
            answer=answer,
            sources=sources,
            search_mode=result.mode,
            model=self.llm.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            trace_id=trace.id,
        )
        trace.end(output={"answer": answer, "sources": len(sources)})

        if sources:  # don't cache "nothing found" — notes may be indexed shortly
            await self.cache.set(key, response.model_dump(mode="json"))
        return response
