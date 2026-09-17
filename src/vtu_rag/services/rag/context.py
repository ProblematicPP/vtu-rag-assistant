"""Turns retrieved chunks into prompt context and maps [n] citations back to sources."""

import re

from vtu_rag.schemas.ask import Source
from vtu_rag.services.search import SearchFilters, SearchHit

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

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
