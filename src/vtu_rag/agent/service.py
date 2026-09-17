import time

from vtu_rag.agent.graph import AgentGraph
from vtu_rag.agent.state import AgentContext, AgentState
from vtu_rag.config import AgentSettings
from vtu_rag.schemas.ask import AgenticAskResponse, AgentStep
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.llm import LLMProvider
from vtu_rag.services.search import SearchFilters, SearchMode, SearchService
from vtu_rag.services.tracing import Tracer


class AgentService:
    def __init__(
        self,
        search: SearchService,
        llm: LLMProvider,
        cache: ResponseCache,
        tracer: Tracer,
        settings: AgentSettings,
    ):
        self.llm = llm
        self.cache = cache
        self.tracer = tracer
        self.settings = settings
        self.agent = AgentGraph(search, llm, settings)

    async def ask(
        self,
        question: str,
        filters: SearchFilters,
        top_k: int | None = None,
        use_cache: bool = True,
    ) -> AgenticAskResponse:
        started = time.perf_counter()
        top_k = top_k or self.settings.top_k
        key = ResponseCache.make_key(
            "agentic-ask",
            question=" ".join(question.lower().split()),
            filters=filters.cache_key(),
            top_k=top_k,
            provider=self.llm.name,
            model=self.llm.model,
            max_rewrites=self.settings.max_rewrites,
        )
        if use_cache and (cached := await self.cache.get(key)):
            return AgenticAskResponse.model_validate(cached | {"cached": True})

        trace = self.tracer.start_trace(
            "agentic-ask",
            input={"question": question, "filters": filters.cache_key(), "top_k": top_k},
        )
        initial: AgentState = {"question": question, "rewritten_queries": [], "steps": []}
        final: AgentState = await self.agent.graph.ainvoke(
            initial, context=AgentContext(filters=filters, top_k=top_k, trace=trace)
        )

        search_mode = final.get("search_mode")
        response = AgenticAskResponse(
            question=question,
            answer=final["answer"],
            sources=final.get("sources", []),
            search_mode=SearchMode(search_mode) if search_mode else None,
            model=self.llm.model,
            latency_ms=round((time.perf_counter() - started) * 1000, 1),
            trace_id=trace.id,
            in_scope=final.get("in_scope", True),
            guardrail_score=final.get("guardrail_score"),
            guardrail_reason=final.get("guardrail_reason"),
            rewritten_queries=final.get("rewritten_queries", []),
            retrieval_attempts=final.get("attempts", 0),
            steps=[AgentStep(**s) for s in final.get("steps", [])],
        )
        trace.end(
            output={"answer": response.answer, "sources": len(response.sources)},
            in_scope=response.in_scope,
            attempts=response.retrieval_attempts,
        )

        # Cache real answers and refusals; skip "nothing found" so new notes are picked up
        if response.sources or not response.in_scope:
            await self.cache.set(key, response.model_dump(mode="json"))
        return response
