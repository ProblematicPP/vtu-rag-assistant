"""Test doubles shared across test modules."""

from collections.abc import Callable

from vtu_rag.services.llm import ChatMessage, LLMProvider, LLMResponse
from vtu_rag.services.search import SearchFilters, SearchHit, SearchMode, SearchResult


def make_hit(index: int, text: str, module: int = 4) -> SearchHit:
    return SearchHit(
        doc_id=f"note1-chunk{index}",
        score=1.0 - index * 0.1,
        chunk_id=index + 1,
        note_id=1,
        chunk_index=index,
        subject_code="BCS303",
        subject_name="Operating Systems",
        semester=3,
        branch="cse",
        scheme="2022",
        module_number=module,
        module_title="Memory Management",
        note_title="module4",
        source_uri="cse/2022/sem3/BCS303/module4.pdf",
        section_heading="Memory > Paging",
        text=text,
        page_start=index + 1,
        page_end=index + 1,
    )


class ScriptedLLM(LLMProvider):
    """Returns responses chosen by a handler that inspects the system prompt."""

    name = "fake"
    model = "fake-model"

    def __init__(self, handler: Callable[[list[ChatMessage]], str]):
        self.handler = handler
        self.calls: list[list[ChatMessage]] = []

    async def generate(self, messages, *, temperature=None, max_tokens=None, json_mode=False):
        self.calls.append(messages)
        return LLMResponse(content=self.handler(messages), model=self.model)

    async def health(self):
        return {"ok": True}


class FakeSearch:
    """Returns canned hits per query; records every query it receives."""

    def __init__(self, results: Callable[[str], list[SearchHit]]):
        self.results = results
        self.queries: list[str] = []

    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        size: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> SearchResult:
        self.queries.append(query)
        hits = self.results(query)[:size]
        return SearchResult(hits=hits, mode=SearchMode.BM25, total=len(hits))


class MemoryCache:
    def __init__(self):
        self.store: dict[str, dict] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value):
        self.store[key] = value

    async def invalidate(self):
        self.store.clear()
