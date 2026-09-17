"""OpenSearch-backed chunk index: setup, indexing and BM25 / vector / hybrid retrieval."""

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from opensearchpy import AsyncOpenSearch, NotFoundError
from opensearchpy.helpers import async_bulk

from vtu_rag.config import OpenSearchSettings
from vtu_rag.services.embeddings import EmbeddingError, EmbeddingProvider
from vtu_rag.services.search.index_config import chunk_index_body, hybrid_pipeline_body
from vtu_rag.services.search.query_builder import (
    SearchFilters,
    build_bm25_body,
    build_hybrid_body,
    build_vector_body,
)

logger = logging.getLogger(__name__)


class SearchMode(StrEnum):
    HYBRID = "hybrid"
    BM25 = "bm25"
    VECTOR = "vector"


@dataclass
class SearchHit:
    doc_id: str
    score: float
    chunk_id: int
    note_id: int
    chunk_index: int
    subject_code: str
    subject_name: str
    semester: int
    branch: str
    scheme: str
    module_number: int
    module_title: str | None
    note_title: str
    source_uri: str
    section_heading: str | None
    text: str
    page_start: int | None
    page_end: int | None

    @classmethod
    def from_hit(cls, hit: dict[str, Any]) -> "SearchHit":
        src = hit["_source"]
        return cls(
            doc_id=hit["_id"],
            score=float(hit.get("_score") or 0.0),
            chunk_id=src["chunk_id"],
            note_id=src["note_id"],
            chunk_index=src["chunk_index"],
            subject_code=src["subject_code"],
            subject_name=src["subject_name"],
            semester=src["semester"],
            branch=src["branch"],
            scheme=src["scheme"],
            module_number=src["module_number"],
            module_title=src.get("module_title"),
            note_title=src["note_title"],
            source_uri=src["source_uri"],
            section_heading=src.get("section_heading"),
            text=src["text"],
            page_start=src.get("page_start"),
            page_end=src.get("page_end"),
        )


@dataclass
class SearchResult:
    hits: list[SearchHit]
    mode: SearchMode  # the mode actually used (may differ after fallback)
    total: int


class SearchService:
    def __init__(self, settings: OpenSearchSettings, embeddings: EmbeddingProvider):
        self.settings = settings
        self.embeddings = embeddings
        self.index = settings.index
        self.client = AsyncOpenSearch(hosts=[settings.host], timeout=30)

    # ------------------------------------------------------------------ setup
    async def setup(self) -> None:
        """Creates the chunk index and hybrid pipeline if missing. Safe to call repeatedly."""
        if not await self.client.indices.exists(index=self.index):
            await self.client.indices.create(
                index=self.index, body=chunk_index_body(self.embeddings.dimensions)
            )
            logger.info("Created index %s", self.index)
        await self.client.transport.perform_request(
            "PUT",
            f"/_search/pipeline/{self.settings.search_pipeline}",
            body=hybrid_pipeline_body(self.settings.bm25_weight, self.settings.vector_weight),
        )

    async def ping(self) -> bool:
        return bool(await self.client.ping())

    async def count(self) -> int:
        try:
            return (await self.client.count(index=self.index))["count"]
        except NotFoundError:
            return 0

    async def close(self) -> None:
        await self.client.close()

    # --------------------------------------------------------------- indexing
    async def index_documents(self, docs: list[tuple[str, dict]]) -> int:
        if not docs:
            return 0
        actions = [{"_index": self.index, "_id": doc_id, "_source": body} for doc_id, body in docs]
        success, errors = await async_bulk(
            self.client, actions, raise_on_error=False, refresh="wait_for"
        )
        if errors:
            raise RuntimeError(f"Bulk indexing had {len(errors)} errors: {errors[:3]}")
        return success

    async def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        actions = [{"_op_type": "delete", "_index": self.index, "_id": i} for i in doc_ids]
        await async_bulk(self.client, actions, raise_on_error=False, refresh="wait_for")

    # -------------------------------------------------------------- retrieval
    async def search(
        self,
        query: str,
        filters: SearchFilters | None = None,
        size: int = 5,
        mode: SearchMode = SearchMode.HYBRID,
    ) -> SearchResult:
        filters = filters or SearchFilters()
        params: dict[str, str] = {}
        vector: list[float] | None = None

        if mode in (SearchMode.HYBRID, SearchMode.VECTOR):
            vector = await self._query_vector(query)
            if vector is None:
                mode = SearchMode.BM25

        if mode is SearchMode.HYBRID:
            body = build_hybrid_body(query, vector, filters, size)
            params["search_pipeline"] = self.settings.search_pipeline
        elif mode is SearchMode.VECTOR:
            body = build_vector_body(vector, filters, size)
        else:
            body = build_bm25_body(query, filters, size)

        try:
            response = await self.client.search(index=self.index, body=body, params=params)
        except NotFoundError:
            return SearchResult(hits=[], mode=mode, total=0)

        hits = [SearchHit.from_hit(h) for h in response["hits"]["hits"]]
        total = response["hits"]["total"]["value"]
        return SearchResult(hits=hits, mode=mode, total=total)

    async def _query_vector(self, query: str) -> list[float] | None:
        if not self.embeddings.enabled:
            return None
        try:
            return await self.embeddings.embed_query(query)
        except EmbeddingError as exc:
            logger.warning("Query embedding failed, falling back to BM25: %s", exc)
            return None
