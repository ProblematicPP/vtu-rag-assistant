from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from vtu_rag.dependencies import SessionDep
from vtu_rag.repositories import CatalogRepository
from vtu_rag.schemas.catalog import ModuleOut, SubjectModulesOut, SubjectOut

router = APIRouter(prefix="/api/v1/subjects", tags=["catalog"])


@router.get("", response_model=list[SubjectOut], summary="List subjects")
async def list_subjects(
    session: SessionDep,
    branch: Annotated[str | None, Query(examples=["cse"])] = None,
    scheme: Annotated[str | None, Query(examples=["2022"])] = None,
    semester: Annotated[int | None, Query(ge=1, le=8)] = None,
) -> list[SubjectOut]:
    subjects = await CatalogRepository(session).list_subjects(branch, scheme, semester)
    return [SubjectOut.model_validate(s) for s in subjects]


@router.get(
    "/{code}/modules",
    response_model=SubjectModulesOut,
    summary="List a subject's modules with note counts",
)
async def list_modules(code: str, session: SessionDep) -> SubjectModulesOut:
    repo = CatalogRepository(session)
    subject = await repo.get_subject(code)
    if subject is None:
        raise HTTPException(status_code=404, detail=f"Subject {code.upper()} not found")
    stats = await repo.module_stats(subject.id)
    modules = [
        ModuleOut(
            number=m.number,
            title=m.title,
            notes=stats.get(m.id, (0, 0))[0],
            indexed_notes=stats.get(m.id, (0, 0))[1],
        )
        for m in subject.modules
    ]
    return SubjectModulesOut(subject=SubjectOut.model_validate(subject), modules=modules)
