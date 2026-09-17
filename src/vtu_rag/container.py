"""Builds and tears down the long-lived service objects shared by the API, CLI and UI."""

import logging
from dataclasses import dataclass

from vtu_rag.config import Settings
from vtu_rag.db import Database
from vtu_rag.ingestion.catalog_loader import seed_catalog
from vtu_rag.ingestion.pipeline import IngestionService
from vtu_rag.services.embeddings import EmbeddingProvider, JinaEmbeddings
from vtu_rag.services.search import SearchService

logger = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    embeddings: EmbeddingProvider
    search: SearchService
    ingestion: IngestionService

    @classmethod
    def build(cls, settings: Settings) -> "Container":
        db = Database(settings)
        embeddings = JinaEmbeddings(settings.jina)
        search = SearchService(settings.opensearch, embeddings)
        ingestion = IngestionService(settings, db, search, embeddings)
        return cls(
            settings=settings, db=db, embeddings=embeddings, search=search, ingestion=ingestion
        )

    async def startup(self) -> None:
        await self.db.create_tables()
        await self.search.setup()
        await seed_catalog(self.db, self.settings.data_dir)
        if not self.embeddings.enabled:
            logger.warning("JINA_API_KEY not set: search runs in BM25-only mode")

    async def shutdown(self) -> None:
        await self.search.close()
        await self.embeddings.close()
        await self.db.close()
