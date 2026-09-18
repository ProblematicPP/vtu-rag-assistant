from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse

from vtu_rag.dependencies import ContainerDep, SessionDep
from vtu_rag.repositories import FigureRepository
from vtu_rag.schemas.ask import FigureOut

router = APIRouter(prefix="/api/v1", tags=["figures"])


@router.get(
    "/figures/{figure_id}",
    summary="Fetch a diagram image extracted from a note",
    response_class=FileResponse,
    responses={200: {"content": {"image/png": {}}}, 404: {"description": "Unknown figure"}},
)
async def get_figure(figure_id: int, session: SessionDep, container: ContainerDep) -> Response:
    figure = await FigureRepository(session).get(figure_id)
    if figure is None:
        raise HTTPException(404, "Figure not found")

    path = container.ingestion.figure_store.resolve(figure.path)
    if path is None:
        raise HTTPException(404, "Figure file is missing; re-index the note")
    return FileResponse(
        path,
        media_type=figure.media_type,
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get(
    "/notes/{note_id}/figures",
    response_model=list[FigureOut],
    summary="List the diagrams extracted from a note",
)
async def list_note_figures(note_id: int, session: SessionDep) -> list[FigureOut]:
    figures = await FigureRepository(session).list_for_note(note_id)
    return [FigureOut.from_figure(f) for f in figures]
