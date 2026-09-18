from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Response, UploadFile

from vtu_rag.dependencies import ContainerDep, SessionDep
from vtu_rag.ingestion.ocr import OcrConfig
from vtu_rag.ingestion.parsers import ParseError
from vtu_rag.ingestion.question_paper import PaperQuestion, parse_questions, read_paper
from vtu_rag.repositories import FigureRepository
from vtu_rag.schemas.ask import notes_from_sources
from vtu_rag.schemas.papers import (
    ExtractedPaper,
    PaperAnswerRequest,
    QuestionOut,
    SolvedPaperOut,
)
from vtu_rag.services.paper_pdf import build_pdf
from vtu_rag.services.papers import PaperService, SolvedPaper
from vtu_rag.services.rag.figures import page_spans, select_figures

router = APIRouter(prefix="/api/v1/papers", tags=["question papers"])

MAX_QUESTIONS = 30


@router.post(
    "/extract",
    response_model=ExtractedPaper,
    summary="Read the questions out of a question paper (PDF or photo)",
)
async def extract(
    container: ContainerDep,
    file: Annotated[UploadFile, File(description="A question paper as PDF or image")],
) -> ExtractedPaper:
    content = await file.read()
    if not content:
        raise HTTPException(400, "The uploaded file is empty")

    suffix = Path(file.filename or "paper.pdf").suffix or ".pdf"
    try:
        text = read_paper(content, suffix, OcrConfig.from_settings(container.settings))
    except ParseError as exc:
        raise HTTPException(422, str(exc)) from exc

    questions = parse_questions(text)
    return ExtractedPaper(
        title=Path(file.filename or "Question paper").stem.replace("_", " ")[:120],
        questions=[QuestionOut(number=q.number, text=q.text, marks=q.marks) for q in questions],
    )


async def _solve(
    request: PaperAnswerRequest, container: ContainerDep, session: SessionDep
) -> SolvedPaper:
    if not request.questions:
        raise HTTPException(400, "No questions to answer")
    questions = [
        PaperQuestion(number=q.number, text=q.text, marks=q.marks)
        for q in request.questions[:MAX_QUESTIONS]
    ]
    filters = request.to_filters()
    settings = container.settings.figures

    marks_by_text = {q.text: q.marks for q in request.questions}

    async def answer(question: str, search_filters):  # noqa: ANN001,ANN202 - local closure
        response = await container.rag.ask(
            question,
            search_filters,
            top_k=request.top_k,
            use_cache=request.use_cache,
            marks=marks_by_text.get(question),
        )
        response.notes = notes_from_sources(response.sources)
        spans = page_spans(response.sources) if settings.enabled else []
        if spans:
            figures = await FigureRepository(session).find_on_pages(spans)
            response.figures = select_figures(
                figures,
                response.sources,
                settings.max_per_answer,
                question=question,
                answer=response.answer,
            )
        return response

    # solve() yields after every answer so a UI can show progress; here we just
    # want the finished paper, so drain it
    solved = SolvedPaper(title=request.title, filters=filters)
    async for state, _remaining in PaperService(answer).solve(
        questions, filters, title=request.title
    ):
        solved = state
    return solved


@router.post(
    "/answer",
    response_model=SolvedPaperOut,
    summary="Answer every question in a paper from the indexed notes",
)
async def answer_paper(
    request: PaperAnswerRequest, container: ContainerDep, session: SessionDep
) -> SolvedPaperOut:
    paper = await _solve(request, container, session)
    return SolvedPaperOut.from_paper(paper)


@router.post(
    "/pdf",
    summary="Answer a paper and return it as a printable PDF",
    response_class=Response,
    responses={200: {"content": {"application/pdf": {}}}},
)
async def paper_pdf(
    request: PaperAnswerRequest, container: ContainerDep, session: SessionDep
) -> Response:
    paper = await _solve(request, container, session)

    # Answers carry figure ids; the PDF needs the files behind them
    store = container.ingestion.figure_store
    wanted = [f.id for item in paper.items for f in item.figures]
    rows = await FigureRepository(session).get_many(wanted)
    paths = {row.id: store.resolve(row.path) for row in rows}

    pdf = build_pdf(paper, lambda figure: paths.get(figure.id), request.include_figures)
    filename = (paper.title or "answers").replace('"', "") + ".pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
