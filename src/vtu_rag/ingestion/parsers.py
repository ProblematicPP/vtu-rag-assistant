"""Turns note files into plain text, page by page."""

import io
import logging
import re
from dataclasses import dataclass, field

from pypdf import PdfReader

from vtu_rag.ingestion.ocr import OcrConfig, OcrError, run_ocr

logger = logging.getLogger(__name__)


@dataclass
class ParsedPage:
    number: int  # 1-based
    text: str

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ParsedDocument:
    pages: list[ParsedPage] = field(default_factory=list)
    ocr_applied: bool = False

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def word_count(self) -> int:
        return sum(p.word_count for p in self.pages)

    @property
    def is_empty(self) -> bool:
        return not any(p.text.strip() for p in self.pages)


class ParseError(RuntimeError):
    pass


_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_SPACES = re.compile(r"[ \t ]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_BREAK.sub(r"\1\2", text)
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    return _MANY_NEWLINES.sub("\n\n", text).strip()


def _extract(data: bytes) -> ParsedDocument:
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
    return ParsedDocument(pages=pages)


def needs_ocr(doc: ParsedDocument, config: OcrConfig) -> bool:
    """True when enough pages carry too little text to be anything but images."""
    if not doc.pages:
        return False
    low_text = sum(1 for p in doc.pages if p.word_count < config.min_words_per_page)
    return low_text / len(doc.pages) >= config.min_low_text_ratio


def parse_pdf(data: bytes, ocr: OcrConfig | None = None) -> ParsedDocument:
    doc = _extract(data)
    if ocr is None or not ocr.enabled or not needs_ocr(doc, ocr):
        if doc.is_empty:
            raise ParseError("PDF has no extractable text (a scanned PDF needs OCR)")
        return doc

    logger.info(
        "PDF looks scanned (%d words over %d pages); running OCR",
        doc.word_count,
        doc.page_count,
    )
    failure: str | None = None
    # Pass 1 OCRs only the pages without a text layer; pass 2 replaces junk layers.
    for force in (False, True) if ocr.force_retry else (False,):
        try:
            candidate = _extract(run_ocr(data, ocr, force=force))
        except (OcrError, ParseError) as exc:
            failure = str(exc)
            logger.warning("OCR pass (force=%s) failed: %s", force, exc)
            continue

        if candidate.word_count > doc.word_count:
            doc = ParsedDocument(pages=candidate.pages, ocr_applied=True)
            logger.info("OCR recovered %d words across %d pages", doc.word_count, doc.page_count)
        if not needs_ocr(doc, ocr):
            break

    if doc.is_empty:
        detail = f": {failure}" if failure else " (OCR produced no text)"
        raise ParseError(f"PDF has no extractable text{detail}")
    return doc


def parse_text(data: bytes) -> ParsedDocument:
    text = data.decode("utf-8", errors="replace")
    # Markdown/plain text has no pages; treat the whole file as page 1
    doc = ParsedDocument(pages=[ParsedPage(number=1, text=_clean(text))])
    if doc.is_empty:
        raise ParseError("File is empty")
    return doc


def parse_document(data: bytes, extension: str, ocr: OcrConfig | None = None) -> ParsedDocument:
    ext = extension.lower()
    if ext == ".pdf":
        return parse_pdf(data, ocr)
    if ext in {".md", ".markdown", ".txt"}:
        return parse_text(data)
    raise ParseError(f"Unsupported file type: {extension}")
