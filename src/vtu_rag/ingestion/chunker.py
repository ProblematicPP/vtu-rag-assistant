"""Section-aware chunking.

Notes are first split into sections at detected headings (markdown `#`,
numbered headings like `2.3 Paging`, `MODULE 2`, short ALL-CAPS lines).
Each section is then packed into chunks of roughly `target_words`, breaking
on paragraph and sentence boundaries, with `overlap_words` carried between
consecutive chunks of the same section. Sections too small to stand on their
own are merged into the following section.
"""

import re
from dataclasses import dataclass, field

from vtu_rag.ingestion.parsers import ParsedDocument

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*$")
_NUMBERED_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})\.?\s+([A-Z][^\n]{2,90})$")
_MODULE_HEADING = re.compile(r"^(module|unit|chapter)[\s\-–:]*\d+\b.*$", re.IGNORECASE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])")


@dataclass
class TextChunk:
    chunk_index: int
    text: str
    section_heading: str | None
    page_start: int
    page_end: int

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class _Paragraph:
    page: int
    text: str
    is_overlap: bool = False  # text carried over from the previous chunk

    @property
    def words(self) -> int:
        return len(self.text.split())


@dataclass
class _Section:
    heading: str | None
    paragraphs: list[_Paragraph] = field(default_factory=list)

    @property
    def words(self) -> int:
        return sum(p.words for p in self.paragraphs)


def _heading_level(line: str) -> int | None:
    """Returns a heading depth (1 = top level) or None if the line is body text."""
    if len(line) > 100:
        return None
    if m := _MD_HEADING.match(line):
        return len(m.group(1))
    if _MODULE_HEADING.match(line) and len(line.split()) <= 12:
        return 1
    if m := _NUMBERED_HEADING.match(line):
        title = m.group(2)
        # Body sentences that happen to start with a number end in punctuation / are long
        if title.endswith((".", ",", ";")) or len(title.split()) > 10:
            return None
        return m.group(1).count(".") + 1
    letters = [c for c in line if c.isalpha()]
    if (
        len(letters) >= 4
        and all(c.isupper() for c in letters)
        and 1 <= len(line.split()) <= 10
        and not line.endswith((".", ","))
    ):
        return 1
    return None


def _clean_heading(line: str) -> str:
    if m := _MD_HEADING.match(line):
        line = m.group(2)
    return line.strip().strip(":").strip()


def _split_sections(doc: ParsedDocument) -> list[_Section]:
    sections: list[_Section] = [_Section(heading=None)]
    stack: list[tuple[int, str]] = []  # (level, heading) — gives "Parent > Child" paths
    buffer: list[str] = []
    buffer_page = 1

    def flush() -> None:
        nonlocal buffer
        if buffer:
            sections[-1].paragraphs.append(_Paragraph(buffer_page, " ".join(buffer)))
            buffer = []

    for page in doc.pages:
        for line in page.text.split("\n"):
            line = line.strip()
            if not line:
                flush()
                continue
            level = _heading_level(line)
            if level is not None:
                flush()
                heading = _clean_heading(line)
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, heading))
                sections.append(_Section(heading=" > ".join(h for _, h in stack)))
                continue
            if not buffer:
                buffer_page = page.number
            buffer.append(line)
        # Paragraphs rarely span pages cleanly in extracted PDF text; break at page ends
        flush()

    return [s for s in sections if s.paragraphs or s.heading]


def _merge_small_sections(sections: list[_Section], min_words: int) -> list[_Section]:
    merged: list[_Section] = []
    pending: _Section | None = None
    for section in sections:
        if pending is not None:
            # Keep the more specific heading, but retain the small section's text
            section = _Section(
                heading=section.heading or pending.heading,
                paragraphs=pending.paragraphs + section.paragraphs,
            )
            pending = None
        if section.words < min_words:
            pending = section
        else:
            merged.append(section)
    if pending is not None and pending.paragraphs:
        if merged:
            merged[-1].paragraphs.extend(pending.paragraphs)
        else:
            merged.append(pending)
    return merged


def _split_long_paragraph(paragraph: _Paragraph, max_words: int) -> list[_Paragraph]:
    if paragraph.words <= max_words:
        return [paragraph]
    pieces: list[_Paragraph] = []
    for sentence in _SENTENCE_SPLIT.split(paragraph.text):
        words = sentence.split()
        # A single "sentence" longer than the limit (tables, code) is hard-split
        for start in range(0, len(words), max_words):
            pieces.append(_Paragraph(paragraph.page, " ".join(words[start : start + max_words])))
    return pieces


class SectionChunker:
    def __init__(self, target_words: int = 350, overlap_words: int = 60, min_words: int = 80):
        if overlap_words >= target_words:
            raise ValueError("overlap_words must be smaller than target_words")
        self.target_words = target_words
        self.overlap_words = overlap_words
        self.min_words = min_words

    def chunk(self, doc: ParsedDocument) -> list[TextChunk]:
        sections = _merge_small_sections(_split_sections(doc), self.min_words)
        chunks: list[TextChunk] = []
        for section in sections:
            for text, page_start, page_end in self._pack(section):
                chunks.append(
                    TextChunk(
                        chunk_index=len(chunks),
                        text=text,
                        section_heading=section.heading,
                        page_start=page_start,
                        page_end=page_end,
                    )
                )
        return chunks

    def _pack(self, section: _Section) -> list[tuple[str, int, int]]:
        units: list[_Paragraph] = []
        for paragraph in section.paragraphs:
            units.extend(_split_long_paragraph(paragraph, self.target_words))

        results: list[tuple[str, int, int]] = []
        current: list[_Paragraph] = []

        def words_in(paragraphs: list[_Paragraph]) -> int:
            return sum(p.words for p in paragraphs)

        def has_new_content() -> bool:
            return any(not p.is_overlap for p in current)

        for unit in units:
            if has_new_content() and words_in(current) + unit.words > self.target_words:
                results.append(_join(current))
                current = self._overlap_tail(current)
            current.append(unit)

        if has_new_content():
            results.append(_join(current))
        return results

    def _overlap_tail(self, paragraphs: list[_Paragraph]) -> list[_Paragraph]:
        if self.overlap_words <= 0:
            return []
        tail = " ".join(p.text for p in paragraphs).split()[-self.overlap_words :]
        return [_Paragraph(paragraphs[-1].page, " ".join(tail), is_overlap=True)]


def _join(paragraphs: list[_Paragraph]) -> tuple[str, int, int]:
    text = "\n\n".join(p.text for p in paragraphs)
    return text, paragraphs[0].page, paragraphs[-1].page
