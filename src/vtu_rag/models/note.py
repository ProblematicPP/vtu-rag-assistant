"""Source notes and the chunks indexed from them."""

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from vtu_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from vtu_rag.models.catalog import Module


class SourceType(enum.StrEnum):
    UPLOAD = "upload"
    SCRAPED = "scraped"


class NoteStatus(enum.StrEnum):
    PENDING = "pending"
    INDEXED = "indexed"
    FAILED = "failed"


class Note(TimestampMixin, Base):
    __tablename__ = "notes"

    id: Mapped[int] = mapped_column(primary_key=True)
    module_id: Mapped[int] = mapped_column(ForeignKey("modules.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(255))
    # Path relative to DATA_DIR for local files, or the origin URL for scraped notes
    source_uri: Mapped[str] = mapped_column(String(1024), unique=True)
    source_type: Mapped[SourceType] = mapped_column(
        Enum(SourceType, name="source_type", values_callable=lambda e: [m.value for m in e])
    )
    content_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[NoteStatus] = mapped_column(
        Enum(NoteStatus, name="note_status", values_callable=lambda e: [m.value for m in e]),
        default=NoteStatus.PENDING,
    )
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    # Model used for the chunk vectors; None means indexed for BM25 only
    embedding_model: Mapped[str | None] = mapped_column(String(128))
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    module: Mapped["Module"] = relationship(back_populates="notes")
    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="note", cascade="all, delete-orphan", order_by="Chunk.chunk_index"
    )


class Chunk(Base):
    __tablename__ = "chunks"
    __table_args__ = (UniqueConstraint("note_id", "chunk_index", name="uq_chunk_note_index"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer)
    section_heading: Mapped[str | None] = mapped_column(String(512))
    text: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int] = mapped_column(Integer)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    opensearch_doc_id: Mapped[str] = mapped_column(String(128), unique=True)

    note: Mapped[Note] = relationship(back_populates="chunks")
