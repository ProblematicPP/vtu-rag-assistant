from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vtu_rag.models import Module, Note, NoteStatus, Subject


class CatalogRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_subject(self, code: str) -> Subject | None:
        stmt = (
            select(Subject)
            .where(Subject.code == code.upper())
            .options(selectinload(Subject.modules))
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def list_subjects(
        self, branch: str | None = None, scheme: str | None = None, semester: int | None = None
    ) -> list[Subject]:
        stmt = select(Subject).order_by(Subject.semester, Subject.code)
        if branch:
            stmt = stmt.where(Subject.branch == branch.lower())
        if scheme:
            stmt = stmt.where(Subject.scheme == scheme)
        if semester:
            stmt = stmt.where(Subject.semester == semester)
        return list((await self.session.execute(stmt)).scalars())

    async def upsert_subject(
        self, *, code: str, name: str | None, branch: str, scheme: str, semester: int
    ) -> Subject:
        subject = await self.get_subject(code)
        if subject is None:
            subject = Subject(
                code=code.upper(),
                name=name or code.upper(),
                branch=branch.lower(),
                scheme=scheme,
                semester=semester,
            )
            self.session.add(subject)
            await self.session.flush()
        elif name and subject.name != name:
            subject.name = name
        return subject

    async def get_module(self, subject_id: int, number: int) -> Module | None:
        stmt = select(Module).where(Module.subject_id == subject_id, Module.number == number)
        return (await self.session.execute(stmt)).scalar_one_or_none()

    async def upsert_module(self, subject: Subject, number: int, title: str | None) -> Module:
        module = await self.get_module(subject.id, number)
        if module is None:
            module = Module(subject_id=subject.id, number=number, title=title)
            self.session.add(module)
            await self.session.flush()
        elif title and module.title != title:
            module.title = title
        return module

    async def module_stats(self, subject_id: int) -> dict[int, tuple[int, int]]:
        """module_id -> (note count, indexed note count)."""
        stmt = (
            select(
                Module.id,
                func.count(Note.id),
                func.count(Note.id).filter(Note.status == NoteStatus.INDEXED),
            )
            .outerjoin(Note, Note.module_id == Module.id)
            .where(Module.subject_id == subject_id)
            .group_by(Module.id)
        )
        rows = await self.session.execute(stmt)
        return {module_id: (total, indexed) for module_id, total, indexed in rows}
