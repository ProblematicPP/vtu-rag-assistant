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


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
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
