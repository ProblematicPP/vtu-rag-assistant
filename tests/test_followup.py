import pytest

from tests.fakes import ScriptedLLM
from vtu_rag.schemas.ask import Turn
from vtu_rag.services.llm import LLMError
from vtu_rag.services.rag.followup import (
    FollowUpResolver,
    format_history,
    needs_context,
)


@pytest.mark.parametrize(
    "question",
    [
        "explain them briefly",
        "Explain them briefly.",
        "elaborate",
        "more",
        "give an example",
        "what about the second one?",
        "explain each of them",
        "why is that so?",
        "compare those two",
    ],
)
def test_dependent_questions_need_the_thread(question):
    assert needs_context(question)


@pytest.mark.parametrize(
    "question",
    [
        "What is a system call? List its types",
        "Explain dual-mode operation with a neat diagram",
        "Difference between monolithic and microkernel structures",
        # Long enough to name its own topic, despite the stray pronoun
        "Explain how the operating system manages memory and describe how it "
        "decides which pages to evict from main memory",
    ],
)
def test_self_contained_questions_are_searched_as_typed(question):
    assert not needs_context(question)


def test_history_keeps_recent_turns_and_trims_long_answers():
    history = [Turn(question=f"q{i}", answer="word " * 200) for i in range(5)]
    text = format_history(history)
    assert "q0" not in text and "q4" in text  # only the last three turns
    assert "…" in text


HISTORY = [Turn(question="List the different operating modes", answer="User mode and kernel mode.")]


@pytest.mark.asyncio
async def test_follow_up_is_rewritten_to_stand_on_its_own():
    llm = ScriptedLLM(lambda messages: "Explain user mode and kernel mode briefly")
    resolved = await FollowUpResolver(llm).resolve("explain them briefly", HISTORY)
    assert resolved == "Explain user mode and kernel mode briefly"
    # The rewriter was shown the turn the question leans on
    assert "List the different operating modes" in llm.calls[0][1].content


@pytest.mark.asyncio
async def test_self_contained_question_skips_the_extra_call():
    llm = ScriptedLLM(lambda messages: "should not be used")
    question = "What is a system call? List its types"
    assert await FollowUpResolver(llm).resolve(question, HISTORY) == question
    assert llm.calls == []


@pytest.mark.asyncio
async def test_first_question_in_a_thread_skips_the_extra_call():
    llm = ScriptedLLM(lambda messages: "should not be used")
    assert await FollowUpResolver(llm).resolve("explain them briefly", []) == "explain them briefly"
    assert llm.calls == []


@pytest.mark.asyncio
async def test_question_stands_when_the_rewrite_fails():
    def fail(messages):
        raise LLMError("model down")

    llm = ScriptedLLM(fail)
    assert await FollowUpResolver(llm).resolve("explain them", HISTORY) == "explain them"


@pytest.mark.asyncio
async def test_rewrite_that_resolved_nothing_is_discarded():
    llm = ScriptedLLM(lambda messages: "Explain them in detail")
    assert await FollowUpResolver(llm).resolve("explain them", HISTORY) == "explain them"


@pytest.mark.asyncio
async def test_labels_and_quotes_are_stripped_from_the_rewrite():
    llm = ScriptedLLM(
        lambda messages: 'Standalone question: "Explain user mode and kernel mode"\nnotes: ...'
    )
    resolved = await FollowUpResolver(llm).resolve("explain them", HISTORY)
    assert resolved == "Explain user mode and kernel mode"
