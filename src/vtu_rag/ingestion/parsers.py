"""Turns note files into plain text, page by page."""

import io
import logging
import re
from dataclasses import dataclass

from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    number: int  # 1-based
    text: str


@dataclass
class ParsedDocument:
    pages: list[ParsedPage]

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def is_empty(self) -> bool:
        return not any(p.text.strip() for p in self.pages)


class ParseError(RuntimeError):
    pass


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def parse_pdf(data: bytes) -> ParsedDocument:
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:  # pypdf raises a variety of errors on broken files
        raise ParseError(f"Could not open PDF: {exc}") from exc

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception as exc:
            logger.warning("Failed to extract page %d: %s", i, exc)
            text = ""
        pages.append(ParsedPage(number=i, text=_clean(text)))

    doc = ParsedDocument(pages=pages)
    if doc.is_empty:
        raise ParseError("PDF has no extractable text (scanned images need OCR first)")
    return doc


def parse_text(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    # Markdown/plain text has no pages; treat the whole file as page 1
    doc = ParsedDocument(pages=[ParsedPage(number=1, text=_clean(text))])
    if doc.is_empty:
        raise ParseError("File is empty")
    return doc


def parse_document(data: bytes, extension: str) -> ParsedDocument:
    ext = extension.lower()
    if ext == ".pdf":
        return parse_pdf(data)
    if ext in {".md", ".markdown", ".txt"}:
        return parse_text(data)
    raise ParseError(f"Unsupported file type: {extension}")
