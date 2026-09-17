"""Syllabus catalogue: a Subject belongs to a branch/scheme/semester and has Modules."""

from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from vtu_rag.db.base import Base, TimestampMixin

if TYPE_CHECKING:
    from vtu_rag.models.note import Note


class Subject(TimestampMixin, Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(primary_key=True)
    # VTU subject codes are scheme-specific (e.g. BCS303 only exists in the 2022 scheme)
    code: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    branch: Mapped[str] = mapped_column(String(20), index=True)  # e.g. "cse"
    scheme: Mapped[str] = mapped_column(String(10), index=True)  # e.g. "2022"
    semester: Mapped[int] = mapped_column(Integer, index=True)

    modules: Mapped[list["Module"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan", order_by="Module.number"
    )

    def __repr__(self) -> str:
        return f"<Subject {self.code} {self.branch}/{self.scheme}/sem{self.semester}>"


class Module(TimestampMixin, Base):
    __tablename__ = "modules"
    __table_args__ = (UniqueConstraint("subject_id", "number", name="uq_module_subject_number"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id", ondelete="CASCADE"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    title: Mapped[str | None] = mapped_column(String(255))

    subject: Mapped[Subject] = relationship(back_populates="modules")
    notes: Mapped[list["Note"]] = relationship(back_populates="module", cascade="all, delete-orphan")

    @property
    def display_title(self) -> str:
        return f"Module {self.number}" + (f": {self.title}" if self.title else "")
