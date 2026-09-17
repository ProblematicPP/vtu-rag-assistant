"""Embedding providers. Jina AI is the default; the interface keeps it swappable."""

import logging
from abc import ABC, abstractmethod

import httpx

from vtu_rag.config import JinaSettings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    pass


class EmbeddingProvider(ABC):
    dimensions: int

    @property
    @abstractmethod
    def enabled(self) -> bool: ...

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    @abstractmethod
    async def embed_query(self, text: str) -> list[float]: ...

    async def close(self) -> None:  # noqa: B027 - optional hook
        pass


class JinaEmbeddings(EmbeddingProvider):
    """Jina embeddings API. v3 models use task-specific adapters for passages vs queries."""

    def __init__(self, settings: JinaSettings, client: httpx.AsyncClient | None = None):
        self.settings = settings
        self.dimensions = settings.dimensions
        self._client = client or httpx.AsyncClient(timeout=60.0)

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        size = self.settings.batch_size
        for start in range(0, len(texts), size):
            vectors.extend(await self._embed(texts[start : start + size], "retrieval.passage"))
        return vectors

    async def embed_query(self, text: str) -> list[float]:
        return (await self._embed([text], "retrieval.query"))[0]

    async def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        if not self.enabled:
            raise EmbeddingError("JINA_API_KEY is not set")
        payload = {
            "model": self.settings.model,
            "task": task,
            "dimensions": self.settings.dimensions,
            "normalized": True,
            "input": texts,
        }
        headers = {"Authorization": f"Bearer {self.settings.api_key}"}
        try:
            response = await self._client.post(self.settings.api_url, json=payload, headers=headers)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise EmbeddingError(
                f"Jina API returned {exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"Jina API request failed: {exc}") from exc

        data = sorted(response.json()["data"], key=lambda item: item["index"])
        return [item["embedding"] for item in data]

    async def close(self) -> None:
        await self._client.aclose()
