from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from vtu_rag.models import Chunk, Figure, Module, Note


def _with_subject():
    """Figures are always rendered with their subject/module, so load them together."""
    return selectinload(Figure.note).selectinload(Note.module).selectinload(Module.subject)


class FigureRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get(self, figure_id: int) -> Figure | None:
        return (
            await self.session.execute(select(Figure).where(Figure.id == figure_id))
        ).scalar_one_or_none()

    async def list_for_note(self, note_id: int) -> list[Figure]:
        stmt = (
            select(Figure)
            .where(Figure.note_id == note_id)
            .options(_with_subject())
            .order_by(Figure.page, Figure.index)
        )
        return list((await self.session.execute(stmt)).scalars())

    async def replace_for_note(self, note_id: int, figures: list[Figure]) -> None:
        await self.session.execute(delete(Figure).where(Figure.note_id == note_id))
        for figure in figures:
            figure.note_id = note_id
        self.session.add_all(figures)
        await self.session.flush()

    async def passages(self, chunk_ids: list[int]) -> list[tuple[int, int, int, str]]:
        """(note_id, page_start, page_end, text) for the retrieved chunks.

        Used to tell which topic a diagram sits inside when the diagram's own
        labels don't name it.
        """
        if not chunk_ids:
            return []
        stmt = select(Chunk.note_id, Chunk.page_start, Chunk.page_end, Chunk.text).where(
            Chunk.id.in_(chunk_ids)
        )
        rows = await self.session.execute(stmt)
        return [
            (note_id, start or 0, end or start or 0, text) for note_id, start, end, text in rows
        ]

    async def find_on_pages(self, spans: list[tuple[int, int, int]]) -> list[Figure]:
        """Figures on the pages covered by (note_id, page_start, page_end) spans."""
        if not spans:
            return []
        clauses = [
            (Figure.note_id == note_id) & (Figure.page >= start) & (Figure.page <= end)
            for note_id, start, end in spans
        ]
        stmt = (
            select(Figure)
            .where(or_(*clauses))
            .options(_with_subject())
            .order_by(Figure.note_id, Figure.page, Figure.index)
        )
        return list((await self.session.execute(stmt)).scalars())
