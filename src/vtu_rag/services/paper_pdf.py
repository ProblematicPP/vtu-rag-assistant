"""Turns a solved question paper into a PDF you can print and revise from.

Built for paper, not for screen: a serif face at a size that survives printing,
each question kept with the start of its answer, diagrams placed inline at a size
you can actually copy from, and the source note named under each answer so you can
check it against the original.
"""

import html
import logging
import re
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

from vtu_rag.services.papers import SolvedPaper

logger = logging.getLogger(__name__)

INK = colors.HexColor("#111111")
SOFT = colors.HexColor("#555A62")
RULE = colors.HexColor("#C9C4B8")

_BULLET_RE = re.compile(r"^(?P<indent>\s*)[-*•+o]\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")


def _styles() -> dict[str, ParagraphStyle]:
    body = ParagraphStyle(
        "body",
        fontName="Times-Roman",
        fontSize=10.5,
        leading=15,
        textColor=INK,
        alignment=TA_LEFT,
        spaceAfter=5,
    )
    return {
        "title": ParagraphStyle(
            "title", parent=body, fontName="Helvetica-Bold", fontSize=15, leading=19, spaceAfter=2
        ),
        "subtitle": ParagraphStyle(
            "subtitle", parent=body, fontName="Helvetica", fontSize=9, textColor=SOFT, spaceAfter=12
        ),
        "question": ParagraphStyle(
            "question",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=15,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "body": body,
        "bullet": ParagraphStyle(
            "bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=3
        ),
        "subbullet": ParagraphStyle(
            "subbullet", parent=body, leftIndent=26, bulletIndent=16, spaceAfter=2, fontSize=10
        ),
        "subhead": ParagraphStyle(
            "subhead",
            parent=body,
            fontName="Helvetica-Bold",
            fontSize=10,
            spaceBefore=8,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "caption", parent=body, fontName="Helvetica-Oblique", fontSize=8.5, textColor=SOFT
        ),
        "source": ParagraphStyle(
            "source", parent=body, fontName="Helvetica", fontSize=8.5, textColor=SOFT, spaceBefore=4
        ),
    }


def _inline(text: str) -> str:
    """Markdown emphasis to ReportLab's mini-HTML, everything else escaped."""
    escaped = html.escape(text, quote=False)
    escaped = _BOLD_RE.sub(r"<b>\1</b>", escaped)
    return _ITALIC_RE.sub(r"<i>\1</i>", escaped)


def _answer_flowables(answer: str, styles: dict[str, ParagraphStyle]) -> list:
    flowables = []
    for raw_line in answer.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            continue
        if _HEADING_RE.match(line):
            flowables.append(Paragraph(_inline(_HEADING_RE.sub("", line)), styles["subhead"]))
        elif match := _BULLET_RE.match(line):
            # "+" and indented markers are the model's sub-points; step them in
            nested = line.lstrip().startswith(("+", "o")) or len(match.group("indent")) >= 2
            style = styles["subbullet"] if nested else styles["bullet"]
            flowables.append(
                Paragraph(
                    _inline(_BULLET_RE.sub("", line)), style, bulletText="–" if nested else "•"
                )
            )
        else:
            # A bold-only line is a heading in all but syntax
            stripped = line.strip()
            if stripped.startswith("**") and stripped.endswith("**") and len(stripped) > 4:
                flowables.append(Paragraph(_inline(stripped), styles["subhead"]))
            else:
                flowables.append(Paragraph(_inline(stripped), styles["body"]))
    return flowables


def _figure_flowables(figure, resolve, styles: dict[str, ParagraphStyle], max_width: float) -> list:  # noqa: ANN001 - FigureOut and a path resolver
    path: Path | None = resolve(figure)
    if path is None:
        return []
    try:
        ratio = figure.height / figure.width if figure.width else 0.6
        width = min(max_width, 105 * mm)
        image = Image(str(path), width=width, height=width * ratio)
    except Exception as exc:  # a broken image must not lose the answer
        logger.warning("Skipping figure %s in PDF: %s", figure.id, exc)
        return []

    caption = figure.caption or f"Diagram — page {figure.page}"
    return [
        Spacer(1, 6),
        image,
        Spacer(1, 3),
        Paragraph(html.escape(caption), styles["caption"]),
        Spacer(1, 4),
    ]


def build_pdf(paper: SolvedPaper, resolve_figure, include_figures: bool = True) -> bytes:
    """Renders the solved paper. `resolve_figure` maps a FigureOut to a local file."""
    styles = _styles()
    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=paper.title,
        author="ChatVTU",
    )
    usable = doc.width

    story: list = [
        Paragraph(html.escape(paper.title), styles["title"]),
        Paragraph(
            f"{len(paper.items)} questions"
            + (f" · {paper.total_marks} marks" if paper.total_marks else "")
            + f" · answered from your notes on {datetime.now(UTC):%d %b %Y}",
            styles["subtitle"],
        ),
        HRFlowable(width="100%", color=RULE, spaceAfter=4),
    ]

    for item in paper.items:
        heading = f"Q{item.number}." + (f"  [{item.marks} marks]" if item.marks else "")
        block = [
            Paragraph(heading, styles["question"]),
            Paragraph(_inline(item.question), styles["body"]),
            Spacer(1, 4),
        ]
        if item.error:
            failed = f"<i>Not answered: {html.escape(item.error)}</i>"
            block.append(Paragraph(failed, styles["body"]))
        else:
            # Keep the question with the opening of its answer across a page break
            first = _answer_flowables(item.answer, styles)
            block.extend(first[:1])
            story.append(KeepTogether(block))
            story.extend(first[1:])
            block = []

            if include_figures:
                for figure in item.figures:
                    story.extend(_figure_flowables(figure, resolve_figure, styles, usable))

            if item.notes:
                where = "; ".join(
                    f"{n.subject_code} module {n.module_number}"
                    + (f", p. {n.pages[0]}" if n.pages else "")
                    for n in item.notes
                )
                story.append(Paragraph(f"From your notes: {html.escape(where)}", styles["source"]))

        if block:
            story.append(KeepTogether(block))
        story.append(Spacer(1, 2))
        story.append(HRFlowable(width="100%", color=RULE))

    if not paper.items:
        story.append(Paragraph("No questions were found in that paper.", styles["body"]))

    doc.build(story, onLaterPages=_page_number, onFirstPage=_page_number)
    return buffer.getvalue()


def _page_number(canvas, doc) -> None:  # noqa: ANN001 - reportlab callback
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SOFT)
    canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f"{doc.page}")
    canvas.drawString(20 * mm, 10 * mm, "Answered from your own notes — check against the source.")
    canvas.restoreState()
