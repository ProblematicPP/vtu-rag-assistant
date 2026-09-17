"""Single-pass RAG: retrieve → generate with citations."""

import time

from vtu_rag.schemas.ask import AskResponse, Source
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.llm import ChatMessage, LLMProvider
from vtu_rag.services.rag import prompts
from vtu_rag.services.rag.context import format_context, mark_cited, scope_description
from vtu_rag.services.search import SearchFilters, SearchHit, SearchService
from vtu_rag.services.tracing import NOOP_TRACE, Trace, Tracer


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
    ) -> tuple[str, list[Source]]:
        if not hits:
            return prompts.NO_CONTEXT_ANSWER.format(scope=scope_description(filters)), []

        context, sources = format_context(hits)
        messages = [
            ChatMessage("system", prompts.ANSWER_SYSTEM),
            ChatMessage("user", prompts.ANSWER_USER.format(question=question, context=context)),
        ]
        response = await self.llm.generate(messages)
        trace.generation("generate-answer", response, input=[m.as_dict() for m in messages])
        answer = response.content.strip()
        return answer, mark_cited(answer, sources)


class RAGService:
    def __init__(
        self,
        search: SearchService,
        llm: LLMProvider,
        cache: ResponseCache,
        tracer: Tracer,
    ):
        self.search = search
        self.llm = llm
        self.cache = cache
        self.tracer = tracer
        self.generator = AnswerGenerator(llm)

    def cache_key(self, kind: str, question: str, filters: SearchFilters, top_k: int) -> str:
        return ResponseCache.make_key(
            kind,
            question=" ".join(question.lower().split()),
            filters=filters.cache_key(),
            top_k=top_k,
            provider=self.llm.name,
            model=self.llm.model,
        )

    async def ask(
        self, question: str, filters: SearchFilters, top_k: int = 5, use_cache: bool = True
    ) -> AskResponse:
        started = time.perf_counter()
        key = self.cache_key("ask", question, filters, top_k)
        if use_cache and (cached := await self.cache.get(key)):
            return AskResponse.model_validate(cached | {"cached": True})

        trace = self.tracer.start_trace(
            "ask", input={"question": question, "filters": filters.cache_key(), "top_k": top_k}
        )
        span = trace.span("retrieve", input={"query": question})
        result = await self.search.search(question, filters, size=top_k)
        span.end(output={"hits": [h.doc_id for h in result.hits]}, mode=result.mode.value)

        answer, sources = await self.generator.generate(question, result.hits, filters, trace)
        response = AskResponse(
            question=question,
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
