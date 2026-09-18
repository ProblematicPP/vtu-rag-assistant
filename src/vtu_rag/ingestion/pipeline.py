"""Ingestion pipeline: discovered note → parse → chunk → embed → index → persist."""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from vtu_rag.config import Settings
from vtu_rag.db import Database
from vtu_rag.ingestion.catalog_loader import CatalogSubject, load_catalog
from vtu_rag.ingestion.chunker import SectionChunker, TextChunk
from vtu_rag.ingestion.ocr import OcrConfig
from vtu_rag.ingestion.parsers import parse_document
from vtu_rag.ingestion.sources import DiscoveredNote, NoteSource
from vtu_rag.models import Chunk, Module, Note, NoteStatus, Subject
from vtu_rag.repositories import CatalogRepository, NoteRepository
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.embeddings import EmbeddingError, EmbeddingProvider
from vtu_rag.services.search import SearchService

logger = logging.getLogger(__name__)


class IngestStatus(StrEnum):
    INDEXED = "indexed"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class IngestOutcome:
    source_uri: str
    status: IngestStatus
    note_id: int | None = None
    chunks: int = 0
    embedded: bool = False
    ocr: bool = False
    error: str | None = None


@dataclass
class SyncReport:
    outcomes: list[IngestOutcome] = field(default_factory=list)
    invalid_paths: list[str] = field(default_factory=list)

    def count(self, status: IngestStatus) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    def summary(self) -> dict[str, int]:
        return {s.value: self.count(s) for s in IngestStatus} | {
            "invalid_paths": len(self.invalid_paths)
        }


def chunk_doc_id(note_id: int, chunk_index: int) -> str:
    return f"note{note_id}-chunk{chunk_index}"


def embedding_input(subject: Subject, module: Module, chunk: TextChunk) -> str:
    """Prefixes chunk text with its syllabus context so vectors carry topic information."""
    header = f"{subject.code} {subject.name} | {module.display_title}"
    if chunk.section_heading:
        header += f" | {chunk.section_heading}"
    return f"{header}\n\n{chunk.text}"


class IngestionService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        search: SearchService,
        embeddings: EmbeddingProvider,
        cache: ResponseCache | None = None,
    ):
        self.settings = settings
        self.cache = cache
        self._sync_lock = asyncio.Lock()
        self.db = db
        self.search = search
        self.embeddings = embeddings
        self.chunker = SectionChunker(
            target_words=settings.chunking.target_words,
            overlap_words=settings.chunking.overlap_words,
            min_words=settings.chunking.min_words,
        )

    @property
    def data_dir(self) -> Path:
        return self.settings.data_dir

    @property
    def embedding_model(self) -> str | None:
        return self.settings.jina.model if self.embeddings.enabled else None

    # ------------------------------------------------------------------ batch
    @property
    def sync_in_progress(self) -> bool:
        return self._sync_lock.locked()

    async def sync(self, source: NoteSource, force: bool = False) -> SyncReport:
        async with self._sync_lock:
            catalog = load_catalog(self.data_dir)
            report = SyncReport()
            async for discovered in source.discover():
                report.outcomes.append(await self.ingest(discovered, force=force, catalog=catalog))
            report.invalid_paths = list(getattr(source, "skipped", []))
        if report.count(IngestStatus.INDEXED):
            await self._invalidate_cache()
        logger.info("Sync finished: %s", report.summary())
        return report

    # ----------------------------------------------------------------- single
    async def ingest(
        self,
        discovered: DiscoveredNote,
        force: bool = False,
        catalog: dict[str, CatalogSubject] | None = None,
    ) -> IngestOutcome:
        # Within sync() the catalogue is passed in and the cache is invalidated once at the end
        batch = catalog is not None
        catalog = catalog if batch else load_catalog(self.data_dir)
        uri = discovered.source_uri

        async with self.db.session() as session:
            catalog_repo = CatalogRepository(session)
            note_repo = NoteRepository(session)
            existing = await note_repo.get_by_source(uri)
            if not force and self._is_up_to_date(existing, discovered):
                return IngestOutcome(uri, IngestStatus.SKIPPED, note_id=existing.id)

            subject, module = await self._resolve_catalog(catalog_repo, discovered, catalog)
            old_doc_ids = await note_repo.chunk_doc_ids(existing.id) if existing else []
            note = await note_repo.upsert(
                module=module,
                title=discovered.title,
                source_uri=uri,
                source_type=discovered.source_type,
                content_hash=discovered.content_hash,
            )
            note_id = note.id

        try:
            outcome = await self._index_note(note_id, discovered, subject, module, old_doc_ids)
        except Exception as exc:
            logger.exception("Failed to ingest %s", uri)
            async with self.db.session() as session:
                repo = NoteRepository(session)
                if failed := await repo.get(note_id):
                    await repo.mark_failed(failed, str(exc))
            return IngestOutcome(uri, IngestStatus.FAILED, note_id=note_id, error=str(exc))
        if not batch:
            await self._invalidate_cache()
        return outcome

    async def delete_note(self, note_id: int) -> bool:
        async with self.db.session() as session:
            repo = NoteRepository(session)
            note = await repo.get(note_id)
            if note is None:
                return False
            await self.search.delete_documents(await repo.chunk_doc_ids(note_id))
            await repo.delete(note)
        await self._invalidate_cache()
        return True

    # ---------------------------------------------------------------- helpers
    async def _invalidate_cache(self) -> None:
        if self.cache is not None:
            await self.cache.invalidate()

    def _is_up_to_date(self, note: Note | None, discovered: DiscoveredNote) -> bool:
        return (
            note is not None
            and note.status == NoteStatus.INDEXED
            and note.content_hash == discovered.content_hash
            # Re-embed notes indexed before an embedding key was configured
            and note.embedding_model == self.embedding_model
        )

    async def _resolve_catalog(
        self,
        repo: CatalogRepository,
        discovered: DiscoveredNote,
        catalog: dict[str, CatalogSubject],
    ) -> tuple[Subject, Module]:
        loc = discovered.location
        known = catalog.get(loc.subject_code)
        subject = await repo.upsert_subject(
            code=loc.subject_code,
            name=known.name if known else None,
            branch=loc.branch,
            scheme=loc.scheme,
            semester=loc.semester,
        )
        module_title = known.modules.get(loc.module_number) if known else None
        module = await repo.upsert_module(subject, loc.module_number, module_title)
        return subject, module

    async def _index_note(
        self,
        note_id: int,
        discovered: DiscoveredNote,
        subject: Subject,
        module: Module,
        old_doc_ids: list[str],
    ) -> IngestOutcome:
        # Parsing (and OCR especially) is CPU-bound; keep the event loop free
        parsed = await asyncio.to_thread(
            parse_document,
            discovered.content,
            discovered.extension,
            OcrConfig.from_settings(self.settings),
        )
        text_chunks = self.chunker.chunk(parsed)
        if not text_chunks:
            raise ValueError("No text chunks produced from note")

        vectors = await self._embed(subject, module, text_chunks, discovered.source_uri)
        embedding_model = self.embedding_model if vectors else None

        async with self.db.session() as session:
            repo = NoteRepository(session)
            note = await repo.get(note_id)
            rows = [
                Chunk(
                    chunk_index=c.chunk_index,
                    section_heading=c.section_heading[:512] if c.section_heading else None,
                    text=c.text,
                    word_count=c.word_count,
                    page_start=c.page_start,
                    page_end=c.page_end,
                    opensearch_doc_id=chunk_doc_id(note_id, c.chunk_index),
                )
                for c in text_chunks
            ]
            await repo.replace_chunks(note, rows)

            now = datetime.now(UTC).isoformat()
            docs = []
            for row, vector in zip(rows, vectors or [None] * len(rows), strict=True):
                body = {
                    "chunk_id": row.id,
                    "note_id": note_id,
                    "chunk_index": row.chunk_index,
                    "branch": subject.branch,
                    "scheme": subject.scheme,
                    "semester": subject.semester,
                    "subject_code": subject.code,
                    "subject_name": subject.name,
                    "module_number": module.number,
                    "module_title": module.title,
                    "note_title": note.title,
                    "source_uri": note.source_uri,
                    "source_type": note.source_type.value,
                    "section_heading": row.section_heading,
                    "text": row.text,
                    "page_start": row.page_start,
                    "page_end": row.page_end,
                    "word_count": row.word_count,
                    "indexed_at": now,
                }
                if vector is not None:
                    body["embedding"] = vector
                docs.append((row.opensearch_doc_id, body))

            await self.search.index_documents(docs)
            new_ids = {doc_id for doc_id, _ in docs}
            await self.search.delete_documents([i for i in old_doc_ids if i not in new_ids])
            await repo.mark_indexed(note, parsed.page_count, embedding_model)

        logger.info(
            "Indexed %s: %d chunks (%s%s)",
            discovered.source_uri,
            len(docs),
            "hybrid" if vectors else "bm25-only",
            ", ocr" if parsed.ocr_applied else "",
        )
        return IngestOutcome(
            discovered.source_uri,
            IngestStatus.INDEXED,
            note_id=note_id,
            chunks=len(docs),
            embedded=bool(vectors),
            ocr=parsed.ocr_applied,
        )

    async def _embed(
        self, subject: Subject, module: Module, chunks: list[TextChunk], uri: str
    ) -> list[list[float]] | None:
        if not self.embeddings.enabled:
            return None
        try:
            return await self.embeddings.embed_documents(
                [embedding_input(subject, module, c) for c in chunks]
            )
        except EmbeddingError as exc:
            # Index for BM25 now; the next sync retries embedding (embedding_model stays None)
            logger.warning("Embedding failed for %s, indexing BM25-only: %s", uri, exc)
            return None
