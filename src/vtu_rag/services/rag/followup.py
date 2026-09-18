"""Resolves follow-up questions against the conversation before retrieval.

Retrieval is stateless: "explain them briefly" names nothing, so searching it
returns noise. When a message looks like it leans on the previous turn, an LLM
rewrites it into a question that stands on its own, and that is what gets
searched, graded and answered.

The heuristic in `needs_context` keeps the extra LLM call off the common path —
a question that already names its topic is searched exactly as it was typed.
"""

import logging
import re

from vtu_rag.schemas.ask import Turn
from vtu_rag.services.llm import ChatMessage, LLMError, LLMProvider
from vtu_rag.services.rag import prompts
from vtu_rag.services.tracing import NOOP_TRACE, Trace

logger = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 300
# Enough to resolve "them" without burying the rewriter in old topics
HISTORY_TURNS = 3
ANSWER_EXCERPT_CHARS = 400
# Kept alongside the opening so a plural reference can reach every topic
MAX_ITEMS = 12
ITEM_CHARS = 70

# "1. Kernel mode", "- User mode", "* Protection", "• Memory management"
_ITEM = re.compile(r"^\s*(?:[-*+•▪]|\d+[.)])\s+\S")
_ITEM_LABEL = re.compile(r"^\s*(?:[-*+•▪]|\d+[.)])\s+")

# Words that never name a topic: references to an earlier turn, the verbs a
# student uses to ask for one, and ordinary glue. A message built only from
# these has nothing to search — "explain them briefly" says nothing about what.
# Anything left over is a topic, so "List its types" is searched as it stands.
_EMPTY_WORDS = frozenset(
    (  # noqa: SIM905 - grouped prose reads better than a list literal
        # glue
        "a an and the of in on for to with by from as at or if is are was were be "
        "so then but not no yes ok okay thanks well any some i you me my your we us our "
        # references to something said earlier
        "it its them they their those these that this there each all both other others "
        "same above below latter former one ones two three second third last next another "
        # ways of asking, with no topic of their own
        "explain describe elaborate expand define discuss list give tell show write "
        "compare continue go on more most much again also too please now about "
        "what why how when where which who whom do does did can could would should "
        # how much detail, and what shape the answer should take
        "briefly brief short shortly detail details detailed depth simple simply "
        "example examples instance point points line lines word words "
        "diagram neat labelled labeled sketch draw note notes answer question mark marks"
    ).split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'\-]*")


def needs_context(question: str) -> bool:
    """True when the message names no topic of its own, only a reference to one."""
    words = _WORD_RE.findall(question.lower())
    if not words:
        return False
    return not any(word not in _EMPTY_WORDS for word in words)


def summarize_answer(answer: str) -> str:
    """An answer cut down to what a follow-up might be pointing at.

    Plain truncation loses the tail of a list, and "explain them" then resolves
    to whichever items survived — the rewrite that drops half the topics. Every
    list item is kept, headline first, so a plural reference can still find all
    of them.
    """
    text = (answer or "").strip()
    if not text:
        return ""

    # Only the outermost level: nested bullets are detail about a topic, and
    # listing them alongside buries the topics themselves
    matched = [line for line in text.splitlines() if _ITEM.match(line)]
    if matched:
        outermost = min(len(line) - len(line.lstrip()) for line in matched)
        matched = [line for line in matched if len(line) - len(line.lstrip()) == outermost]
    items = [_ITEM_LABEL.sub("", line).strip(" .:*") for line in matched]
    items = [item[:ITEM_CHARS] for item in items if item][:MAX_ITEMS]

    opening = " ".join(text.split())
    if len(opening) > ANSWER_EXCERPT_CHARS:
        opening = opening[:ANSWER_EXCERPT_CHARS].rstrip() + "…"
    if not items:
        return opening
    return f"{opening}\nTopics named: {'; '.join(items)}"


def format_history(history: list[Turn]) -> str:
    """The recent turns, with answers cut down to the topics they name."""
    lines: list[str] = []
    for turn in history[-HISTORY_TURNS:]:
        lines.append(f"Student: {turn.question.strip()}")
        summary = summarize_answer(turn.answer)
        if summary:
            lines.append(f"Assistant: {summary}")
    return "\n".join(lines)


def _clean(text: str) -> str:
    line = next((ln for ln in text.strip().splitlines() if ln.strip()), "")
    line = line.strip()
    for prefix in ("Standalone question:", "Rewritten question:", "Question:"):
        if line.lower().startswith(prefix.lower()):
            line = line[len(prefix) :].strip()
    line = line.strip("\"'`")
    if len(line) <= MAX_QUESTION_CHARS:
        return line
    # Cut at a word boundary: the resolved question is shown to the student
    cut = line[:MAX_QUESTION_CHARS].rsplit(" ", 1)[0]
    return cut.rstrip(" ,;")


class FollowUpResolver:
    """Turns a dependent message into a standalone question, or leaves it alone."""

    def __init__(self, llm: LLMProvider):
        self.llm = llm

    async def resolve(
        self, question: str, history: list[Turn] | None, trace: Trace = NOOP_TRACE
    ) -> str:
        if not history or not needs_context(question):
            return question

        messages = [
            ChatMessage("system", prompts.CONDENSE_SYSTEM),
            ChatMessage(
                "user",
                prompts.CONDENSE_USER.format(
                    history=format_history(history), question=question.strip()
                ),
            ),
        ]
        span = trace.span("condense", input={"question": question, "turns": len(history)})
        try:
            response = await self.llm.generate(messages, temperature=0.0, max_tokens=80)
            trace.generation("condense-llm", response, input=[m.as_dict() for m in messages])
            standalone = _clean(response.content)
        except LLMError as exc:
            # Searching the question as typed beats failing the whole request
            logger.warning("Follow-up rewrite failed, using the question as asked: %s", exc)
            span.end(output={"standalone": question, "error": str(exc)[:200]})
            return question

        # A rewrite that dropped the topic again is no better than the original
        if len(standalone.split()) < 2 or needs_context(standalone):
            standalone = question
        span.end(output={"standalone": standalone})
        return standalone
