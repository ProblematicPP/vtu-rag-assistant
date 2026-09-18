"""Agent routing tests with a scripted LLM and fake search — no services needed."""

import json

from tests.fakes import FakeSearch, MemoryCache, ScriptedLLM, make_hit
from vtu_rag.agent import AgentService
from vtu_rag.config import AgentSettings, LangfuseSettings
from vtu_rag.services.llm import ChatMessage
from vtu_rag.services.rag import prompts
from vtu_rag.services.search import SearchFilters
from vtu_rag.services.tracing import Tracer

PAGING_HITS = [make_hit(0, "Paging splits memory into fixed-size frames."), make_hit(1, "TLB.")]


def llm_for(
    *,
    guard_score: int = 90,
    relevant: list[list[int]] | None = None,
    rewrite: str = "paging memory management frames",
    answer: str = "Paging divides memory into frames [1].",
) -> ScriptedLLM:
    grades = iter(relevant if relevant is not None else [[1]])

    def handler(messages: list[ChatMessage]) -> str:
        system = messages[0].content
        if system == prompts.GUARDRAIL_SYSTEM:
            return json.dumps({"score": guard_score, "reason": "test"})
        if system == prompts.GRADE_SYSTEM:
            return json.dumps({"relevant": next(grades), "reason": "test"})
        if system == prompts.REWRITE_SYSTEM:
            return rewrite
        # the answer prompt carries a diagram rule appended to it
        if system.startswith(prompts.ANSWER_SYSTEM):
            return answer
        raise AssertionError(f"unexpected prompt: {system[:40]}")

    return ScriptedLLM(handler)


def make_agent(llm, search, max_rewrites: int = 2) -> AgentService:
    settings = AgentSettings(max_rewrites=max_rewrites, guardrail_threshold=50, top_k=5)
    tracer = Tracer(LangfuseSettings(public_key="", secret_key=""))
    return AgentService(search, llm, MemoryCache(), tracer, settings)


def nodes(response) -> list[str]:
    return [s.node for s in response.steps]


async def test_happy_path_answers_with_citations():
    search = FakeSearch(lambda q: PAGING_HITS)
    agent = make_agent(llm_for(), search)
    res = await agent.ask("What is paging?", SearchFilters(subject_code="BCS303"))

    assert nodes(res) == ["contextualize", "guardrail", "retrieve", "grade", "generate"]
    assert res.in_scope and res.guardrail_score == 90
    assert res.answer.startswith("Paging divides memory")
    # only the relevant hit is passed to generation
    assert [s.cited for s in res.sources] == [True]
    assert res.retrieval_attempts == 1 and res.rewritten_queries == []


async def test_out_of_scope_question_is_declined_without_retrieval():
    search = FakeSearch(lambda q: PAGING_HITS)
    agent = make_agent(llm_for(guard_score=5), search)
    res = await agent.ask("Who won the cricket match yesterday?", SearchFilters())

    assert nodes(res) == ["contextualize", "guardrail", "decline"]
    assert not res.in_scope
    assert res.answer == prompts.OUT_OF_SCOPE_ANSWER
    assert search.queries == []


async def test_weak_retrieval_triggers_rewrite_then_succeeds():
    search = FakeSearch(lambda q: PAGING_HITS)
    llm = llm_for(relevant=[[], [2]])
    res = await make_agent(llm, search).ask("wat is pagin", SearchFilters())

    assert nodes(res) == [
        "contextualize",
        "guardrail",
        "retrieve",
        "grade",
        "rewrite",
        "retrieve",
        "grade",
        "generate",
    ]
    assert search.queries == ["wat is pagin", "paging memory management frames"]
    assert res.rewritten_queries == ["paging memory management frames"]
    assert res.retrieval_attempts == 2
    assert [s.snippet for s in res.sources] == ["TLB."]


async def test_rewrites_are_capped_then_generation_uses_best_effort_hits():
    search = FakeSearch(lambda q: PAGING_HITS)
    llm = llm_for(relevant=[[], [], []])
    res = await make_agent(llm, search, max_rewrites=2).ask("obscure question", SearchFilters())

    assert nodes(res).count("rewrite") == 2
    assert nodes(res).count("retrieve") == 3
    assert nodes(res)[-1] == "generate"
    assert len(res.sources) == 2


async def test_no_hits_at_all_returns_not_found_message_without_answer_llm_call():
    search = FakeSearch(lambda q: [])
    llm = llm_for()
    res = await make_agent(llm, search, max_rewrites=1).ask("What is paging?", SearchFilters())

    assert res.sources == []
    assert "couldn't find" in res.answer
    systems = [call[0].content for call in llm.calls]
    assert prompts.ANSWER_SYSTEM not in systems


async def test_broken_guardrail_output_fails_open():
    search = FakeSearch(lambda q: PAGING_HITS)
    llm = llm_for()
    original = llm.handler
    llm.handler = lambda m: "not json" if m[0].content == prompts.GUARDRAIL_SYSTEM else original(m)
    res = await make_agent(llm, search).ask("What is paging?", SearchFilters())
    assert res.in_scope and res.guardrail_score is None
    assert nodes(res)[-1] == "generate"


async def test_identical_rewrite_falls_back_to_broadened_query():
    search = FakeSearch(lambda q: PAGING_HITS)
    llm = llm_for(relevant=[[], [1]], rewrite="What is paging?")
    await make_agent(llm, search).ask("What is paging?", SearchFilters())
    assert search.queries[1] != search.queries[0]


async def test_answers_are_cached():
    search = FakeSearch(lambda q: PAGING_HITS)
    llm = llm_for(relevant=[[1]])
    agent = make_agent(llm, search)
    first = await agent.ask("What is paging?", SearchFilters())
    second = await agent.ask("  what is PAGING? ", SearchFilters())
    assert not first.cached and second.cached
    assert len(search.queries) == 1
