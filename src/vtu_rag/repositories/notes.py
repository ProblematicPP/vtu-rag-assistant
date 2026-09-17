from datetime import UTC, datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vtu_rag.models import Chunk, Module, Note, NoteStatus, SourceType, Subject


class NoteRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, note_id: int) -> Note | None:
        stmt = (
            select(Note)
            .where(Note.id == note_id)
            .options(selectinload(Note.module).selectinload(Module.subject))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def get_by_source(self, source_uri: str) -> Note | None:
        stmt = select(Note).where(Note.source_uri == source_uri)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_notes(
        self,
        subject_code: str | None = None,
        status: NoteStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Note]:
        stmt = (
            select(Note)
            .join(Module)
            .join(Subject)
            .options(selectinload(Note.module).selectinload(Module.subject))
            .order_by(Subject.code, Module.number, Note.id)
            .limit(limit)
            .offset(offset)
        )
        if subject_code:
            stmt = stmt.where(Subject.code == subject_code.upper())
        if status:
            stmt = stmt.where(Note.status == status)
        return list((await self.session.execute(stmt)).scalars())

    async def upsert(
        self,
        *,
        module: Module,
        title: str,
        source_uri: str,
        source_type: SourceType,
        content_hash: str,
    ) -> Note:
        note = await self.get_by_source(source_uri)
        if note is None:
            note = Note(
                module_id=module.id,
                title=title,
                source_uri=source_uri,
                source_type=source_type,
                content_hash=content_hash,
                status=NoteStatus.PENDING,
            )
            self.session.add(note)
        else:
            note.module_id = module.id
            note.title = title
            note.content_hash = content_hash
            note.status = NoteStatus.PENDING
            note.error = None
        await self.session.flush()
        return note

    async def chunk_doc_ids(self, note_id: int) -> list[str]:
        stmt = select(Chunk.opensearch_doc_id).where(Chunk.note_id == note_id)
        return list((await self.session.execute(stmt)).scalars())

    async def replace_chunks(self, note: Note, chunks: list[Chunk]) -> None:
        await self.session.execute(delete(Chunk).where(Chunk.note_id == note.id))
        for chunk in chunks:
            chunk.note_id = note.id
        self.session.add_all(chunks)
        await self.session.flush()

    async def mark_indexed(
        self, note: Note, page_count: int | None, embedding_model: str | None
    ) -> None:
        note.status = NoteStatus.INDEXED
        note.error = None
        note.page_count = page_count
        note.embedding_model = embedding_model
        note.indexed_at = datetime.now(UTC)
        await self.session.flush()

    async def mark_failed(self, note: Note, error: str) -> None:
        note.status = NoteStatus.FAILED
        note.error = error[:2000]
        await self.session.flush()

    async def delete(self, note: Note) -> None:
        await self.session.delete(note)
        await self.session.flush()
