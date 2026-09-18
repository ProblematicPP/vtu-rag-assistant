from pydantic import BaseModel, Field

from vtu_rag.schemas.ask import FigureOut, NoteRef
from vtu_rag.schemas.search import SearchFilterParams
from vtu_rag.services.papers import SolvedPaper


class QuestionOut(BaseModel):
    number: str = Field(..., examples=["1b"])
    text: str
    marks: int | None = None


class ExtractedPaper(BaseModel):
    title: str
    questions: list[QuestionOut]


class PaperAnswerRequest(SearchFilterParams):
    questions: list[QuestionOut]
    title: str = "Question paper"
    top_k: int = Field(5, ge=1, le=15)
    use_cache: bool = True
    include_figures: bool = True


class SolvedQuestionOut(BaseModel):
    number: str
    question: str
    marks: int | None
    answer: str
    figures: list[FigureOut] = Field(default_factory=list)
    notes: list[NoteRef] = Field(default_factory=list)
    error: str | None = None


class SolvedPaperOut(BaseModel):
    title: str
    total_marks: int
    items: list[SolvedQuestionOut]

    @classmethod
    def from_paper(cls, paper: SolvedPaper) -> "SolvedPaperOut":
        return cls(
            title=paper.title,
            total_marks=paper.total_marks,
            items=[
                SolvedQuestionOut(
                    number=item.number,
                    question=item.question,
                    marks=item.marks,
                    answer=item.answer,
                    figures=item.figures,
                    notes=item.notes,
                    error=item.error,
                )
                for item in paper.items
            ],
        )
