"""Reads a VTU question paper — a PDF or a photo of one — into separate questions.

VTU papers follow a rigid shape: numbered questions, lettered sub-parts, and a
marks column on the right.

    Q.1  a. Define operating system. Explain its two main roles.      (08 Marks)
         b. With a neat diagram, explain dual-mode operation.         (07 Marks)

Each lettered part is answerable on its own, so those are what we pull out. The
marks matter too: an 8-mark answer is a different length from a 3-mark one.
"""

import logging
import re
from dataclasses import dataclass

from vtu_rag.ingestion.ocr import OcrConfig, image_to_text
from vtu_rag.ingestion.parsers import ParseError, parse_pdf

logger = logging.getLogger(__name__)

# "(08 Marks)", "[10 marks]", "( 6 Marks )"
_MARKS_RE = re.compile(r"[(\[]\s*(\d{1,2})\s*marks?\s*[)\]]", re.IGNORECASE)
# "Q.1 a.", "1 a)", "Q1a.", "a." at the start of a line
_PART_RE = re.compile(
    r"^\s*(?:Q\s*\.?\s*(?P<q>\d{1,2})\s*\.?\s*)?(?P<part>[a-d])\s*[.)]\s+(?=\S)",
    re.IGNORECASE,
)
_QUESTION_RE = re.compile(r"^\s*(?:Q\s*\.?\s*)?(?P<q>\d{1,2})\s*[.)]\s+(?=\S)", re.IGNORECASE)
# Boilerplate around the questions themselves
_SKIP_RE = re.compile(
    r"visvesvaraya|technological university|usn|semester|examination|time:|max\.?\s*marks|"
    r"answer any|module\s*-?\s*\d|note\s*:|important note|page \d+ of|scheme|^\s*\d+\s*$",
    re.IGNORECASE,
)

MIN_QUESTION_WORDS = 4
MAX_QUESTION_CHARS = 400


@dataclass
class PaperQuestion:
    number: str  # "1a", "2b", "3"
    text: str
    marks: int | None = None

    @property
    def label(self) -> str:
        return f"Q{self.number}" + (f" ({self.marks} marks)" if self.marks else "")


def _clean_question(text: str) -> str:
    text = _MARKS_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" .;:-")
    return text[:MAX_QUESTION_CHARS]


def parse_questions(text: str) -> list[PaperQuestion]:
    """Splits paper text into answerable questions, in the order they appear."""
    questions: list[PaperQuestion] = []
    current_number: str | None = None
    current_lines: list[str] = []
    current_marks: int | None = None
    question_no = "1"

    def flush() -> None:
        nonlocal current_number, current_lines, current_marks
        if current_number and current_lines:
            body = _clean_question(" ".join(current_lines))
            if len(body.split()) >= MIN_QUESTION_WORDS:
                questions.append(PaperQuestion(current_number, body, current_marks))
        current_number, current_lines, current_marks = None, [], None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _SKIP_RE.search(line) and not _PART_RE.match(line):
            continue

        marks_match = _MARKS_RE.search(line)
        part_match = _PART_RE.match(line)
        question_match = None if part_match else _QUESTION_RE.match(line)

        if part_match:
            flush()
            if part_match.group("q"):
                question_no = part_match.group("q")
            current_number = f"{question_no}{part_match.group('part').lower()}"
            current_lines = [line[part_match.end() :]]
        elif question_match:
            flush()
            question_no = question_match.group("q")
            current_number = question_no
            current_lines = [line[question_match.end() :]]
        elif current_number:
            current_lines.append(line)
        else:
            continue

        if marks_match:
            current_marks = int(marks_match.group(1))

    flush()
    logger.info("Parsed %d question(s) from the paper", len(questions))
    return questions


def read_paper(data: bytes, extension: str, ocr: OcrConfig) -> str:
    """Text of an uploaded paper, whether it arrived as a PDF or a photo."""
    ext = extension.lower()
    if ext == ".pdf":
        try:
            document = parse_pdf(data, ocr)
        except ParseError as exc:
            raise ParseError(f"Could not read the paper: {exc}") from exc
        return "\n".join(page.text for page in document.pages)

    if ext in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}:
        # A photo of a paper is a page of prose, so read it with the layout pass
        text = image_to_text(data, language=ocr.language, psm="3", timeout=ocr.timeout_seconds)
        if not text:
            raise ParseError(
                "No text could be read from that image. A sharper, straight-on photo helps."
            )
        return text

    raise ParseError(f"Unsupported file type: {extension}. Upload a PDF or a photo.")
