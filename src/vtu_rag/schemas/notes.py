from datetime import datetime

from pydantic import BaseModel

from vtu_rag.ingestion.pipeline import IngestOutcome, IngestStatus, SyncReport
from vtu_rag.models import Note, NoteStatus, SourceType


class NoteOut(BaseModel):
    id: int
    title: str
    subject_code: str
    subject_name: str
    semester: int
    module_number: int
    module_title: str | None
    source_uri: str
    source_type: SourceType
    status: NoteStatus
    error: str | None
    page_count: int | None
    chunk_count: int | None = None
    embedding_model: str | None
    indexed_at: datetime | None

    @classmethod
    def from_note(cls, note: Note, chunk_count: int | None = None) -> "NoteOut":
        module = note.module
        return cls(
            id=note.id,
            title=note.title,
            subject_code=module.subject.code,
            subject_name=module.subject.name,
            semester=module.subject.semester,
            module_number=module.number,
            module_title=module.title,
            source_uri=note.source_uri,
            source_type=note.source_type,
            status=note.status,
            error=note.error,
            page_count=note.page_count,
            chunk_count=chunk_count,
            embedding_model=note.embedding_model,
            indexed_at=note.indexed_at,
        )


class IngestResult(BaseModel):
    source_uri: str
    status: IngestStatus
    note_id: int | None
    chunks: int
    embedded: bool
    error: str | None

    @classmethod
    def from_outcome(cls, outcome: IngestOutcome) -> "IngestResult":
        return cls(**outcome.__dict__)


class UploadResponse(IngestResult):
    note: NoteOut | None = None


class SyncResponse(BaseModel):
    summary: dict[str, int]
    results: list[IngestResult]
    invalid_paths: list[str]

    @classmethod
    def from_report(cls, report: SyncReport) -> "SyncResponse":
        return cls(
            summary=report.summary(),
            results=[IngestResult.from_outcome(o) for o in report.outcomes],
            invalid_paths=report.invalid_paths,
        )
