from collections.abc import Awaitable, Callable

from fastapi import APIRouter

from vtu_rag.dependencies import AgentDep, ContainerDep, SessionDep
from vtu_rag.repositories import FigureRepository
from vtu_rag.schemas.ask import (
    AgenticAskResponse,
    AskRequest,
    AskResponse,
    notes_from_sources,
)
from vtu_rag.services.rag.context import drop_text_diagrams
from vtu_rag.services.rag.figures import page_spans, select_figures
from vtu_rag.services.rag.parts import combine, marks_per_part, split_parts

router = APIRouter(prefix="/api/v1", tags=["ask"])

# Answers one standalone question with the given marks
AnswerOne = Callable[[str, int | None], Awaitable[AskResponse]]


async def _attach_sources(response: AskResponse, session: SessionDep, container: ContainerDep):
    """Links the whole source notes, and the diagrams on the pages used."""
    response.notes = notes_from_sources(response.sources)

    settings = container.settings.figures
    spans = page_spans(response.sources) if settings.enabled else []
    if spans:
        figures = await FigureRepository(session).find_on_pages(spans)
        response.figures = select_figures(
            figures,
            response.sources,
            settings.max_per_answer,
            # Figures are matched against what was actually answered, which
            # for a follow-up is the resolved question, not "explain them"
            question=response.resolved_question or response.question,
            answer=response.answer,
        )
    if response.figures:
        # The notes' own diagram is on screen; a text sketch beside it is noise
        response.answer = drop_text_diagrams(response.answer)
    return response


async def answer_question(
    question: str,
    marks: int | None,
    answer_one: AnswerOne,
    session: SessionDep,
    container: ContainerDep,
) -> AskResponse:
    """Answers a question whole, or part by part when it has (i), (ii)… parts.

    Each part gets its own retrieval, grading and diagrams, so a section only
    one part needs isn't judged irrelevant to the whole, and each diagram sits
    under the part it illustrates.
    """
    parts = split_parts(question)
    if not parts:
        return await _attach_sources(await answer_one(question, marks), session, container)

    share = marks_per_part(marks, len(parts))
    answered = []
    # One after another: a single local GPU serves every request anyway
    for part in parts:
        response = await answer_one(part.question, share)
        answered.append((part, await _attach_sources(response, session, container)))
    return combine(question, answered)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Quick answer (single retrieval + generation)",
)
async def ask(request: AskRequest, container: ContainerDep, session: SessionDep) -> AskResponse:
    async def answer_one(question: str, marks: int | None) -> AskResponse:
        return await container.rag.ask(
            question,
            request.to_filters(),
            top_k=request.top_k,
            use_cache=request.use_cache,
            marks=marks,
            history=request.history,
        )

    return await answer_question(request.question, request.marks, answer_one, session, container)


@router.post(
    "/agentic-ask",
    response_model=AgenticAskResponse,
    summary="Agentic answer (guardrail → retrieve → grade → rewrite → generate)",
)
async def agentic_ask(
    request: AskRequest, agent: AgentDep, container: ContainerDep, session: SessionDep
) -> AgenticAskResponse:
    async def answer_one(question: str, marks: int | None) -> AgenticAskResponse:
        return await agent.ask(
            question,
            request.to_filters(),
            top_k=request.top_k,
            use_cache=request.use_cache,
            history=request.history,
            marks=marks,
        )

    return await answer_question(request.question, request.marks, answer_one, session, container)
