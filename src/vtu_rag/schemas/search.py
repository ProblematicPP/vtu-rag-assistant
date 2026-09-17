from pydantic import BaseModel, Field

from vtu_rag.services.search import SearchFilters, SearchHit, SearchMode


class SearchFilterParams(BaseModel):
    branch: str | None = Field(None, examples=["cse"])
    scheme: str | None = Field(None, examples=["2022"])
    semester: int | None = Field(None, ge=1, le=8, examples=[3])
    subject_code: str | None = Field(None, examples=["BCS303"])
    module_numbers: list[int] = Field(default_factory=list, examples=[[1, 2]])

    def to_filters(self) -> SearchFilters:
        return SearchFilters(
            branch=self.branch,
            scheme=self.scheme,
            semester=self.semester,
            subject_code=self.subject_code,
            module_numbers=[m for m in self.module_numbers if 1 <= m <= 9],
        )


class SearchRequest(SearchFilterParams):
    query: str = Field(..., min_length=2, max_length=500, examples=["What is a system call?"])
    size: int = Field(5, ge=1, le=50)
    mode: SearchMode = SearchMode.HYBRID


class ChunkResult(BaseModel):
    chunk_id: int
    note_id: int
    score: float
    subject_code: str
    subject_name: str
    semester: int
    module_number: int
    module_title: str | None
    note_title: str
    source_uri: str
    section_heading: str | None
    page_start: int | None
    page_end: int | None
    text: str

    @classmethod
    def from_hit(cls, hit: SearchHit) -> "ChunkResult":
        return cls(
            chunk_id=hit.chunk_id,
            note_id=hit.note_id,
            score=round(hit.score, 4),
            subject_code=hit.subject_code,
            subject_name=hit.subject_name,
            semester=hit.semester,
            module_number=hit.module_number,
            module_title=hit.module_title,
            note_title=hit.note_title,
            source_uri=hit.source_uri,
            section_heading=hit.section_heading,
            page_start=hit.page_start,
            page_end=hit.page_end,
            text=hit.text,
        )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    total: int
    results: list[ChunkResult]
