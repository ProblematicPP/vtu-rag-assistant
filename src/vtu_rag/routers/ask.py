from fastapi import APIRouter

from vtu_rag.dependencies import AgentDep, ContainerDep, SessionDep
from vtu_rag.repositories import FigureRepository
from vtu_rag.schemas.ask import (
    AgenticAskResponse,
    AskRequest,
    AskResponse,
    notes_from_sources,
)
from vtu_rag.services.rag.figures import page_spans, select_figures

router = APIRouter(prefix="/api/v1", tags=["ask"])


async def _attach_sources(response: AskResponse, session: SessionDep, container: ContainerDep):
    """Links the whole source notes, and the diagrams on the pages used."""
    response.notes = notes_from_sources(response.sources)

    settings = container.settings.figures
    spans = page_spans(response.sources) if settings.enabled else []
    if spans:
        figures = await FigureRepository(session).find_on_pages(spans)
        response.figures = select_figures(figures, response.sources, settings.max_per_answer)
    return response


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Quick answer (single retrieval + generation)",
)
async def ask(request: AskRequest, container: ContainerDep, session: SessionDep) -> AskResponse:
    response = await container.rag.ask(
        request.question, request.to_filters(), top_k=request.top_k, use_cache=request.use_cache
    )
    return await _attach_sources(response, session, container)


@router.post(
    "/agentic-ask",
    response_model=AgenticAskResponse,
    summary="Agentic answer (guardrail → retrieve → grade → rewrite → generate)",
)
async def agentic_ask(
    request: AskRequest, agent: AgentDep, container: ContainerDep, session: SessionDep
) -> AgenticAskResponse:
    response = await agent.ask(
        request.question, request.to_filters(), top_k=request.top_k, use_cache=request.use_cache
    )
    return await _attach_sources(response, session, container)
