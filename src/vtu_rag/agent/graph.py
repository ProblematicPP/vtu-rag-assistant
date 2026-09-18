"""Agentic RAG workflow.

    START → contextualize → guardrail ─┬─ in scope ─→ retrieve → grade ─┬─ relevant ────→ generate
                                       │                  ↑             ├─ retries left → rewrite ┐
                                       │                  └─────────────┼─────────────────────────┘
                                       │                                └─ no retries ─→ generate
                                       └─ out of scope → decline

Contextualize resolves a follow-up against the thread, so every node after it
works on a question that stands on its own.

Grading drives query rewriting. When retries are exhausted, generation still
runs on the best retrieved excerpts and the answer prompt tells the model to
say so if they don't contain the answer — small local graders are often
stricter than they should be.
"""

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.runtime import Runtime

from vtu_rag.agent.state import AgentContext, AgentState
from vtu_rag.config import AgentSettings
from vtu_rag.services.llm import ChatMessage, LLMError, LLMProvider, parse_json_object
from vtu_rag.services.rag import prompts
from vtu_rag.services.rag.context import format_context, scope_description
from vtu_rag.services.rag.followup import FollowUpResolver
from vtu_rag.services.rag.service import AnswerGenerator, DiagramProbe, no_diagram
from vtu_rag.services.search import SearchService

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 300


def _step(node: str, **detail) -> dict:
    return {"steps": [{"node": node, "detail": detail}]}


class AgentGraph:
    def __init__(
        self,
        search: SearchService,
        llm: LLMProvider,
        settings: AgentSettings,
        diagram_probe: DiagramProbe | None = None,
    ):
        self.search = search
        self.llm = llm
        self.settings = settings
        self.generator = AnswerGenerator(llm)
        self.followup = FollowUpResolver(llm)
        self.diagram_probe = diagram_probe or no_diagram
        self.graph: CompiledStateGraph = self._build()

    def _build(self) -> CompiledStateGraph:
        builder = StateGraph(AgentState, context_schema=AgentContext)
        builder.add_node("contextualize", self.contextualize)
        builder.add_node("guardrail", self.guardrail)
        builder.add_node("decline", self.decline)
        builder.add_node("retrieve", self.retrieve)
        builder.add_node("grade", self.grade)
        builder.add_node("rewrite", self.rewrite)
        builder.add_node("generate", self.generate)

        builder.add_edge(START, "contextualize")
        builder.add_edge("contextualize", "guardrail")
        builder.add_conditional_edges("guardrail", self.route_after_guardrail)
        builder.add_edge("decline", END)
        builder.add_edge("retrieve", "grade")
        builder.add_conditional_edges("grade", self.route_after_grade)
        builder.add_edge("rewrite", "retrieve")
        builder.add_edge("generate", END)
        return builder.compile()

    # ------------------------------------------------------------------ nodes
    async def contextualize(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        """Resolves "explain them briefly" into a question that can be searched."""
        ctx = runtime.context
        question = state["question"]
        standalone = await self.followup.resolve(question, ctx.history, ctx.trace)
        resolved = standalone != question
        return {
            "standalone": standalone,
            "query": standalone,
            "attempts": 0,
            **_step("contextualize", resolved=resolved, standalone=standalone if resolved else ""),
        }

    async def guardrail(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        span = ctx.trace.span("guardrail", input={"question": state["standalone"]})
        scope = scope_description(ctx.filters)
        hint = f"The student is studying{scope}.\n" if scope else ""
        messages = [
            ChatMessage("system", prompts.GUARDRAIL_SYSTEM),
            ChatMessage(
                "user", prompts.GUARDRAIL_USER.format(scope_hint=hint, question=state["standalone"])
            ),
        ]
        try:
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=120, json_mode=True
            )
            ctx.trace.generation("guardrail-llm", response, input=[m.as_dict() for m in messages])
            verdict = parse_json_object(response.content)
            score = max(0, min(100, int(verdict.get("score", 100))))
            reason = str(verdict.get("reason", ""))[:300]
        except (LLMError, ValueError, TypeError) as exc:
            # Fail open: a broken classifier shouldn't block legitimate questions
            logger.warning("Guardrail failed, allowing question: %s", exc)
            score, reason = None, f"guardrail unavailable: {exc}"[:300]

        in_scope = score is None or score >= self.settings.guardrail_threshold
        span.end(output={"score": score, "in_scope": in_scope, "reason": reason})
        return {
            "in_scope": in_scope,
            "guardrail_score": score,
            "guardrail_reason": reason,
            **_step("guardrail", score=score, in_scope=in_scope, reason=reason),
        }

    async def decline(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        return {"answer": prompts.OUT_OF_SCOPE_ANSWER, "sources": [], **_step("decline")}

    async def retrieve(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        query = state["query"]
        span = ctx.trace.span("retrieve", input={"query": query, "attempt": state["attempts"] + 1})
        result = await self.search.search(query, ctx.filters, size=ctx.top_k)
        span.end(output={"hits": [h.doc_id for h in result.hits]}, mode=result.mode.value)
        return {
            "hits": result.hits,
            "search_mode": result.mode.value,
            "attempts": state["attempts"] + 1,
            **_step("retrieve", query=query, hits=len(result.hits), mode=result.mode.value),
        }

    async def grade(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        hits = state["hits"]
        if not hits:
            return {"relevant_hits": [], **_step("grade", relevant=0, retrieved=0)}

        span = ctx.trace.span("grade", input={"hits": len(hits)})
        context, sources = format_context(hits)
        messages = [
            ChatMessage("system", prompts.GRADE_SYSTEM),
            ChatMessage(
                "user", prompts.GRADE_USER.format(question=state["standalone"], context=context)
            ),
        ]
        try:
            response = await self.llm.generate(
                messages, temperature=0.0, max_tokens=150, json_mode=True
            )
            ctx.trace.generation("grade-llm", response, input=[m.as_dict() for m in messages])
            verdict = parse_json_object(response.content)
            numbers = {int(n) for n in verdict.get("relevant", []) if str(n).strip().isdigit()}
            reason = str(verdict.get("reason", ""))[:300]
            # sources may be shorter than hits when the context word budget ran out
            relevant = [hit for s, hit in zip(sources, hits, strict=False) if s.index in numbers]
        except (LLMError, ValueError, TypeError) as exc:
            logger.warning("Grading failed, keeping all hits: %s", exc)
            relevant, reason = hits[: len(sources)], f"grader unavailable: {exc}"[:300]

        span.end(output={"relevant": len(relevant)}, reason=reason)
        return {
            "relevant_hits": relevant,
            **_step("grade", relevant=len(relevant), retrieved=len(hits), reason=reason),
        }

    async def rewrite(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        span = ctx.trace.span("rewrite", input={"query": state["query"]})
        messages = [
            ChatMessage("system", prompts.REWRITE_SYSTEM),
            ChatMessage(
                "user",
                prompts.REWRITE_USER.format(question=state["standalone"], query=state["query"]),
            ),
        ]
        try:
            response = await self.llm.generate(messages, temperature=0.3, max_tokens=80)
            ctx.trace.generation("rewrite-llm", response, input=[m.as_dict() for m in messages])
            new_query = _clean_query(response.content)
        except LLMError as exc:
            logger.warning("Rewrite failed: %s", exc)
            new_query = ""
        if not new_query or new_query.lower() == state["query"].lower():
            # Fall back to a broader keyword query so the retry isn't identical
            new_query = f"{state['standalone']} definition explanation concepts"[:MAX_QUERY_CHARS]

        span.end(output={"query": new_query})
        return {
            "query": new_query,
            "rewritten_queries": [new_query],
            **_step("rewrite", query=new_query),
        }

    async def generate(self, state: AgentState, runtime: Runtime[AgentContext]) -> dict:
        ctx = runtime.context
        hits = state.get("relevant_hits") or state.get("hits") or []
        span = ctx.trace.span("generate", input={"hits": len(hits)})
        diagram_shown = await self.diagram_probe(hits, state["standalone"])
        answer, sources = await self.generator.generate(
            state["standalone"], hits, ctx.filters, ctx.trace, diagram_shown=diagram_shown
        )
        span.end(output={"answer_chars": len(answer), "sources": len(sources)})
        return {
            "answer": answer,
            "sources": sources,
            **_step("generate", sources=len(sources), cited=sum(s.cited for s in sources)),
        }

    # ---------------------------------------------------------------- routing
    def route_after_guardrail(self, state: AgentState) -> Literal["retrieve", "decline"]:
        return "retrieve" if state["in_scope"] else "decline"

    def route_after_grade(self, state: AgentState) -> Literal["generate", "rewrite"]:
        if state.get("relevant_hits"):
            return "generate"
        # attempts counts retrievals; the first one isn't a rewrite
        if state["attempts"] - 1 < self.settings.max_rewrites:
            return "rewrite"
        return "generate"


def _clean_query(text: str) -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = line.strip().strip("\"'`").removeprefix("Query:").removeprefix("query:").strip()
    return line[:MAX_QUERY_CHARS]
