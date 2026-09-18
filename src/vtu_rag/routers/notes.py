import asyncio
import re
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select

from vtu_rag.dependencies import ContainerDep, SessionDep
from vtu_rag.ingestion.path_parser import (
    SUPPORTED_EXTENSIONS,
    canonical_note_path,
    parse_note_path,
)
from vtu_rag.ingestion.sources import DiscoveredNote, LocalFolderSource
from vtu_rag.models import Chunk, NoteStatus, SourceType
from vtu_rag.repositories import CatalogRepository, NoteRepository
from vtu_rag.schemas.notes import IngestResult, NoteOut, SyncResponse, UploadResponse
from vtu_rag.services.data_files import media_type_for, resolve_in_data_dir

router = APIRouter(prefix="/api/v1/notes", tags=["notes"])

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


def _safe_stem(filename: str) -> str:
    stem = Path(filename).stem.lower()
    stem = re.sub(r"^mod(?:ule)?[\s_-]*\d+", "", stem)  # we prefix the module ourselves
    stem = re.sub(r"[^a-z0-9]+", "-", stem).strip("-")
    return stem[:60]


@router.post(
    "",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and index a note file",
    description=(
        "Stores the file under DATA_DIR/<branch>/<scheme>/sem<N>/<SUBJECT>/module<N>-*.ext "
        "and indexes it. For subjects already in the catalogue only subject_code and "
        "module_number are needed."
    ),
)
async def upload_note(
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="PDF, Markdown or text file")],
    subject_code: Annotated[str, Form(min_length=4, max_length=12, pattern=r"^[A-Za-z0-9]+$")],
    module_number: Annotated[int, Form(ge=1, le=9)],
    subject_name: Annotated[str | None, Form(max_length=255)] = None,
    module_title: Annotated[str | None, Form(max_length=255)] = None,
    branch: Annotated[str | None, Form(pattern=r"^[A-Za-z]{2,10}$")] = None,
    scheme: Annotated[str | None, Form(pattern=r"^20\d{2}$")] = None,
    semester: Annotated[int | None, Form(ge=1, le=8)] = None,
) -> UploadResponse:
    settings = container.settings
    extension = Path(file.filename or "").suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            400, f"Unsupported file type {extension!r}; use {sorted(SUPPORTED_EXTENSIONS)}"
        )
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 50 MB")
    if not content:
        raise HTTPException(400, "File is empty")

    async with container.db.session() as session:
        repo = CatalogRepository(session)
        subject = await repo.get_subject(subject_code)
        if subject is None:
            if semester is None:
                raise HTTPException(
                    422,
                    f"Subject {subject_code.upper()} is not in the catalogue; "
                    "provide semester (and optionally branch, scheme, subject_name)",
                )
            subject = await repo.upsert_subject(
                code=subject_code,
                name=subject_name,
                branch=branch or settings.default_branch,
                scheme=scheme or settings.default_scheme,
                semester=semester,
            )
        elif subject_name:
            subject.name = subject_name
        await repo.upsert_module(subject, module_number, module_title)
        target_branch, target_scheme, target_sem = subject.branch, subject.scheme, subject.semester

    stem = _safe_stem(file.filename or "")
    filename = f"module{module_number}{'-' + stem if stem else ''}{extension}"
    path = canonical_note_path(
        settings.data_dir, target_branch, target_scheme, target_sem, subject_code, filename
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(path.write_bytes, content)

    location = parse_note_path(path, settings.data_dir)
    outcome = await container.ingestion.ingest(
        DiscoveredNote(
            location=location,
            source_uri=location.relative_path,
            source_type=SourceType.UPLOAD,
            extension=extension,
            content=content,
            title=location.title,
        ),
        force=True,
    )

    note_out = None
    if outcome.note_id is not None:
        async with container.db.session() as session:
            note = await NoteRepository(session).get(outcome.note_id)
            if note is not None:
                note_out = NoteOut.from_note(note, chunk_count=outcome.chunks)
    return UploadResponse(**IngestResult.from_outcome(outcome).model_dump(), note=note_out)


@router.post(
    "/sync",
    response_model=SyncResponse,
    summary="Index new or changed notes from the data folder",
)
async def sync_notes(
    container: ContainerDep,
    force: Annotated[bool, Query(description="Re-index unchanged notes too")] = False,
) -> SyncResponse:
    if container.ingestion.sync_in_progress:
        raise HTTPException(409, "A sync is already running")
    report = await container.ingestion.sync(
        LocalFolderSource(container.settings.data_dir), force=force
    )
    return SyncResponse.from_report(report)


@router.get("", response_model=list[NoteOut], summary="List notes")
async def list_notes(
    session: SessionDep,
    subject_code: str | None = None,
    note_status: Annotated[NoteStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[NoteOut]:
    notes = await NoteRepository(session).list_notes(subject_code, note_status, limit, offset)
    return [NoteOut.from_note(n) for n in notes]


@router.get(
    "/{note_id}/file",
    summary="Open the original note file (PDF/markdown)",
    response_class=FileResponse,
    responses={200: {"content": {"application/pdf": {}}}, 404: {"description": "Unknown note"}},
)
async def get_note_file(note_id: int, session: SessionDep, container: ContainerDep) -> Response:
    note = await NoteRepository(session).get(note_id)
    if note is None:
        raise HTTPException(404, "Note not found")

    path = resolve_in_data_dir(container.settings.data_dir, note.source_uri)
    if path is None:
        raise HTTPException(404, "The note file is no longer in the data folder")
    # inline, so the browser opens the PDF (honouring #page=N) instead of downloading it
    return FileResponse(
        path,
        media_type=media_type_for(path),
        headers={
            "Content-Disposition": f'inline; filename="{path.name}"',
            "Cache-Control": "public, max-age=3600",
        },
    )


@router.get("/{note_id}", response_model=NoteOut, summary="Get a note")
async def get_note(note_id: int, session: SessionDep) -> NoteOut:
    note = await NoteRepository(session).get(note_id)
    if note is None:
        raise HTTPException(404, "Note not found")
    count = await session.scalar(select(func.count(Chunk.id)).where(Chunk.note_id == note_id))
    return NoteOut.from_note(note, chunk_count=count)


@router.delete(
    "/{note_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Remove a note from the index (the file on disk is kept)",
)
async def delete_note(note_id: int, container: ContainerDep) -> None:
    if not await container.ingestion.delete_note(note_id):
        raise HTTPException(404, "Note not found")
