from fastapi import APIRouter

from vtu_rag.dependencies import ContainerDep
from vtu_rag.schemas.search import ChunkResult, SearchRequest, SearchResponse

router = APIRouter(prefix="/api/v1", tags=["search"])


@router.post("/search", response_model=SearchResponse, summary="Search note chunks")
async def search(request: SearchRequest, container: ContainerDep) -> SearchResponse:
    result = await container.search.search(
        request.query, request.to_filters(), size=request.size, mode=request.mode
    )
    return SearchResponse(
        query=request.query,
        mode=result.mode,
        total=result.total,
        results=[ChunkResult.from_hit(h) for h in result.hits],
    )
