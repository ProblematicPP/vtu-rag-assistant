from fastapi import APIRouter

from vtu_rag.dependencies import AgentDep, ContainerDep
from vtu_rag.schemas.ask import AgenticAskResponse, AskRequest, AskResponse

router = APIRouter(prefix="/api/v1", tags=["ask"])


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Quick answer (single retrieval + generation)",
)
async def ask(request: AskRequest, container: ContainerDep) -> AskResponse:
    return await container.rag.ask(
        request.question, request.to_filters(), top_k=request.top_k, use_cache=request.use_cache
    )


@router.post(
    "/agentic-ask",
    response_model=AgenticAskResponse,
    summary="Agentic answer (guardrail → retrieve → grade → rewrite → generate)",
)
async def agentic_ask(request: AskRequest, agent: AgentDep) -> AgenticAskResponse:
    return await agent.ask(
        request.question, request.to_filters(), top_k=request.top_k, use_cache=request.use_cache
    )
