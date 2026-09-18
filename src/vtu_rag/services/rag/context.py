"""Turns retrieved chunks into prompt context and tidies the answer that comes back."""

import re

from vtu_rag.schemas.ask import Source
from vtu_rag.services.search import SearchFilters, SearchHit

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SPACE_BEFORE_PUNCTUATION = re.compile(r"[ \t]+([.,;:!?])")
_RUNS_OF_SPACES = re.compile(r"[ \t]{2,}")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_RUNS_OF_BLANK_LINES = re.compile(r"\n{3,}")

# "[the diagram shown]", "**Diagram:** The diagram shown illustrates…" — small
# models narrate the figure even when told not to. The student can already see it.
_DIAGRAM_LINE = re.compile(
    r"^[ \t]*\[?\**[ \t]*(?:the\s+)?(?:diagram|figure)\b[^\n]*\bshown\b[^\n]*$",
    re.IGNORECASE | re.MULTILINE,
)
_DIAGRAM_BRACKET = re.compile(r"\[[^\]\n]*\bdiagram\s+shown\b[^\]\n]*\]", re.IGNORECASE)

# Keep prompts inside small local models' context windows
MAX_CONTEXT_WORDS = 3000


def build_sources(hits: list[SearchHit]) -> list[Source]:
    return [Source.from_hit(i, hit) for i, hit in enumerate(hits, start=1)]


def format_context(hits: list[SearchHit]) -> tuple[str, list[Source]]:
    """Numbers hits [1], [2], ... within a word budget.

    Returns the prompt block and the sources that actually made it in.
    """
    blocks: list[str] = []
    sources: list[Source] = []
    budget = MAX_CONTEXT_WORDS
    for source, hit in zip(build_sources(hits), hits, strict=True):
        if budget <= 0:
            break
        words = hit.text.split()
        blocks.append(f"[{source.index}] {source.label}\n{' '.join(words[:budget])}")
        sources.append(source)
        budget -= len(words)
    return "\n\n".join(blocks), sources


def cited_indices(answer: str) -> set[int]:
    found: set[int] = set()
    for group in _CITATION_RE.findall(answer):
        found.update(int(n) for n in group.split(","))
    return found


def strip_citation_markers(answer: str) -> str:
    """Removes stray [1] / [2][3] markers from an answer.

    Excerpts are numbered in the prompt, so small models sometimes cite them even
    though the prompt forbids it. Sources are presented as links to the whole
    note, which would leave those numbers pointing at nothing.
    """
    cleaned = _CITATION_RE.sub("", answer)
    # "... memory [1]." leaves a space before the stop; "a [1] b" leaves two spaces
    cleaned = _SPACE_BEFORE_PUNCTUATION.sub(r"\1", cleaned)
    cleaned = _RUNS_OF_SPACES.sub(" ", cleaned)
    return _TRAILING_SPACE.sub("", cleaned).strip()


def drop_diagram_narration(answer: str) -> str:
    """Removes chatter about the diagram displayed beneath the answer."""
    cleaned = _DIAGRAM_BRACKET.sub("", answer)
    cleaned = _DIAGRAM_LINE.sub("", cleaned)
    return _RUNS_OF_BLANK_LINES.sub("\n\n", cleaned).strip()


def mark_cited(answer: str, sources: list[Source]) -> list[Source]:
    cited = cited_indices(answer)
    return [s.model_copy(update={"cited": s.index in cited}) for s in sources]


def scope_description(filters: SearchFilters) -> str:
    parts = []
    if filters.subject_code:
        parts.append(filters.subject_code.upper())
    if filters.module_numbers:
        parts.append("module " + ", ".join(map(str, sorted(filters.module_numbers))))
    if filters.semester:
        parts.append(f"semester {filters.semester}")
    return f" for {' '.join(parts)}" if parts else ""
