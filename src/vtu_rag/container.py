"""Builds and tears down the long-lived service objects shared by the API, CLI and UI."""

import logging
from dataclasses import dataclass

from vtu_rag.config import Settings
from vtu_rag.db import Database
from vtu_rag.ingestion.catalog_loader import seed_catalog
from vtu_rag.ingestion.pipeline import IngestionService
from vtu_rag.repositories import FigureRepository
from vtu_rag.services.cache import ResponseCache
from vtu_rag.services.embeddings import EmbeddingProvider, JinaEmbeddings
from vtu_rag.services.llm import LLMProvider, create_llm_provider
from vtu_rag.services.rag.figures import any_figure_matches
from vtu_rag.services.rag.service import RAGService
from vtu_rag.services.search import SearchHit, SearchService
from vtu_rag.services.tracing import Tracer

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    embeddings: EmbeddingProvider
    search: SearchService
    cache: ResponseCache
    ingestion: IngestionService
    llm: LLMProvider
    tracer: Tracer
    rag: RAGService
    diagram_probe: object

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        db = Database(settings)
        embeddings = JinaEmbeddings(settings.jina)
        search = SearchService(settings.opensearch, embeddings)
        cache = ResponseCache(settings.redis, settings.cache)
        ingestion = IngestionService(settings, db, search, embeddings, cache)
        llm = create_llm_provider(settings)
        tracer = Tracer(settings.langfuse)

        async def diagram_probe(hits: list[SearchHit], question: str) -> bool:
            """Whether a diagram from the notes will accompany this answer.

            Asked before generation so the prompt can stop the model drawing an
            ASCII diagram over the top of a real one from the student's notes.
            """
            spans = [
                (hit.note_id, hit.page_start, hit.page_end or hit.page_start)
                for hit in hits
                if hit.page_start is not None
            ]
            if not settings.figures.enabled or not spans:
                return False
            async with db.session() as session:
                figures = await FigureRepository(session).find_on_pages(spans)
            return any_figure_matches(figures, question, hits)

        rag = RAGService(search, llm, cache, tracer, diagram_probe)
        return cls(
            settings=settings,
            db=db,
            embeddings=embeddings,
            search=search,
            cache=cache,
            ingestion=ingestion,
            llm=llm,
            tracer=tracer,
            rag=rag,
            diagram_probe=diagram_probe,
        )

    async def startup(self) -> None:
        await self.db.create_tables()
        await self.search.setup()
        await seed_catalog(self.db, self.settings.data_dir)
        if not self.embeddings.enabled:
            logger.warning("JINA_API_KEY not set: search runs in BM25-only mode")

    async def shutdown(self) -> None:
        self.tracer.flush()
        for close in (
            self.search.close,
            self.embeddings.close,
            self.llm.close,
            self.cache.close,
            self.db.close,
        ):
            try:
                await close()
            except Exception as exc:
                logger.warning("Error during shutdown: %s", exc)
