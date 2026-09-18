"""Diagrams extracted from a note, addressed by the page they appear on."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from vtu_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from vtu_rag.models.note import Note


class Figure(TimestampMixin, Base):
    __tablename__ = "figures"
    __table_args__ = (UniqueConstraint("note_id", "page", "index", name="uq_figure_note_page_idx"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    note_id: Mapped[int] = mapped_column(ForeignKey("notes.id", ondelete="CASCADE"), index=True)
    page: Mapped[int] = mapped_column(Integer, index=True)
    index: Mapped[int] = mapped_column(Integer)
    # "figure" for a diagram in a text PDF, "page" for a scanned page image
    kind: Mapped[str] = mapped_column(String(16), default="figure")
    # Path relative to DATA_DIR
    path: Mapped[str] = mapped_column(String(1024), unique=True)
    media_type: Mapped[str] = mapped_column(String(64))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    size_bytes: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    caption: Mapped[str | None] = mapped_column(String(256))
    # Words read out of the image itself, used to match a diagram to a question
    label_text: Mapped[str | None] = mapped_column(String(512))

    note: Mapped["Note"] = relationship(back_populates="figures")
