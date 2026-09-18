"""Answers a whole question paper, one question at a time.

Papers are answered sequentially rather than in parallel: a single local GPU
serves the model, so concurrent requests would queue anyway — and answering in
order lets the caller watch progress arrive question by question.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, field

from vtu_rag.ingestion.question_paper import PaperQuestion
from vtu_rag.schemas.ask import AskResponse, FigureOut, NoteRef, PartAnswer
from vtu_rag.services.search import SearchFilters

logger = logging.getLogger(__name__)


@dataclass
class SolvedQuestion:
    number: str
    question: str
    marks: int | None
    answer: str
    figures: list[FigureOut] = field(default_factory=list)
    notes: list[NoteRef] = field(default_factory=list)
    # (i), (ii)… answered separately, each with its own diagrams
    parts: list[PartAnswer] = field(default_factory=list)
    error: str | None = None


@dataclass
class SolvedPaper:
    title: str
    filters: SearchFilters
    items: list[SolvedQuestion] = field(default_factory=list)

    @property
    def total_marks(self) -> int:
        return sum(item.marks or 0 for item in self.items)


class PaperService:
    """Runs each question through the ordinary ask path, then collects the results."""

    def __init__(self, answer):  # noqa: ANN001 - a callable returning AskResponse
        self._answer = answer

    async def solve(
        self,
        questions: list[PaperQuestion],
        filters: SearchFilters,
        title: str = "Question paper",
    ) -> AsyncIterator[tuple[SolvedPaper, int]]:
        """Yields the paper after each answer, with the number still to go."""
        paper = SolvedPaper(title=title, filters=filters)
        for index, question in enumerate(questions):
            try:
                response: AskResponse = await self._answer(question.text, filters)
                paper.items.append(
                    SolvedQuestion(
                        number=question.number,
                        question=question.text,
                        marks=question.marks,
                        answer=response.answer,
                        figures=response.figures,
                        notes=response.notes,
                        parts=response.parts,
                    )
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one bad question must not lose the rest
                logger.exception("Question %s failed", question.number)
                paper.items.append(
                    SolvedQuestion(
                        number=question.number,
                        question=question.text,
                        marks=question.marks,
                        answer="",
                        error=str(exc),
                    )
                )
            yield paper, len(questions) - index - 1
