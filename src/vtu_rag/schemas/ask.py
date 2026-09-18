from typing import Any

from pydantic import BaseModel, Field

from vtu_rag.schemas.search import SearchFilterParams
from vtu_rag.services.search import SearchHit, SearchMode


class AskRequest(SearchFilterParams):
    question: str = Field(
        ..., min_length=3, max_length=1000, examples=["Explain dual-mode operation."]
    )
    top_k: int = Field(5, ge=1, le=15)
    use_cache: bool = True


class Source(BaseModel):
    """A retrieved chunk as presented to the model, numbered for [n] citations."""

    index: int
    subject_code: str
    subject_name: str
    semester: int
    module_number: int
    module_title: str | None
    note_id: int
    note_title: str
    source_uri: str
    section_heading: str | None
    page_start: int | None
    page_end: int | None
    chunk_id: int
    score: float
    snippet: str
    cited: bool = False

    @classmethod
    def from_hit(cls, index: int, hit: SearchHit, snippet_chars: int = 300) -> "Source":
        snippet = hit.text if len(hit.text) <= snippet_chars else hit.text[:snippet_chars] + "…"
        return cls(
            index=index,
            subject_code=hit.subject_code,
            subject_name=hit.subject_name,
            semester=hit.semester,
            module_number=hit.module_number,
            module_title=hit.module_title,
            note_id=hit.note_id,
            note_title=hit.note_title,
            source_uri=hit.source_uri,
            section_heading=hit.section_heading,
            page_start=hit.page_start,
            page_end=hit.page_end,
            chunk_id=hit.chunk_id,
            score=round(hit.score, 4),
            snippet=snippet,
        )

    @property
    def label(self) -> str:
        module = f"Module {self.module_number}"
        if self.module_title:
            module += f": {self.module_title}"
        parts = [f"{self.subject_code} {self.subject_name}", module]
        if self.section_heading:
            parts.append(self.section_heading.split(" > ")[-1])
        if self.page_start:
            pages = (
                f"p. {self.page_start}"
                if self.page_start == self.page_end or not self.page_end
                else f"pp. {self.page_start}-{self.page_end}"
            )
            parts.append(f"{self.note_title}, {pages}")
        else:
            parts.append(self.note_title)
        return " — ".join(parts)


class FigureOut(BaseModel):
    """A diagram from the notes, served by the API so a student can redraw it."""

    id: int
    url: str
    note_id: int
    subject_code: str
    module_number: int
    page: int
    kind: str
    caption: str | None
    width: int
    height: int

    @classmethod
    def from_figure(cls, figure) -> "FigureOut":  # noqa: ANN001 - ORM model
        subject = figure.note.module.subject
        return cls(
            id=figure.id,
            url=f"/api/v1/figures/{figure.id}",
            note_id=figure.note_id,
            subject_code=subject.code,
            module_number=figure.note.module.number,
            page=figure.page,
            kind=figure.kind,
            caption=figure.caption,
            width=figure.width,
            height=figure.height,
        )


class NoteRef(BaseModel):
    """A source note, linked whole — students open the PDF at the page it used."""

    note_id: int
    title: str
    subject_code: str
    subject_name: str
    module_number: int
    module_title: str | None
    source_uri: str
    url: str
    pages: list[int] = Field(default_factory=list)

    @property
    def first_page(self) -> int | None:
        return self.pages[0] if self.pages else None


def notes_from_sources(sources: list[Source], limit: int = 3) -> list[NoteRef]:
    """One entry per note the answer drew on, with the pages it used."""
    by_note: dict[int, NoteRef] = {}
    for source in sources:
        ref = by_note.get(source.note_id)
        if ref is None:
            if len(by_note) >= limit:
                continue
            ref = NoteRef(
                note_id=source.note_id,
                title=source.note_title,
                subject_code=source.subject_code,
                subject_name=source.subject_name,
                module_number=source.module_number,
                module_title=source.module_title,
                source_uri=source.source_uri,
                url=f"/api/v1/notes/{source.note_id}/file",
            )
            by_note[source.note_id] = ref
        for page in filter(None, (source.page_start, source.page_end)):
            if page not in ref.pages:
                ref.pages.append(page)
    for ref in by_note.values():
        ref.pages.sort()
    return list(by_note.values())


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    notes: list[NoteRef] = Field(default_factory=list)
    figures: list[FigureOut] = Field(default_factory=list)
    search_mode: SearchMode | None = None
    model: str | None = None
    cached: bool = False
    latency_ms: float | None = None
    trace_id: str | None = None


class AgentStep(BaseModel):
    node: str
    detail: dict[str, Any] = Field(default_factory=dict)


class AgenticAskResponse(AskResponse):
    in_scope: bool = True
    guardrail_score: int | None = None
    guardrail_reason: str | None = None
    rewritten_queries: list[str] = Field(default_factory=list)
    retrieval_attempts: int = 0
    steps: list[AgentStep] = Field(default_factory=list)
