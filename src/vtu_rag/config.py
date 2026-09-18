"""Application settings, loaded from environment variables (and `.env` when present).

Each backing service gets its own settings group with an env prefix, so
`POSTGRES_HOST` maps to `settings.postgres.host`, `OLLAMA_MODEL` to
`settings.ollama.model`, and so on.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _group(prefix: str) -> SettingsConfigDict:
    return SettingsConfigDict(env_prefix=prefix, env_file=".env", extra="ignore")


class PostgresSettings(BaseSettings):
    model_config = _group("POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    user: str = "vtu"
    password: str = "vtu_password"
    db: str = "vtu_rag"

    @property
    def url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"


class OpenSearchSettings(BaseSettings):
    model_config = _group("OPENSEARCH_")

    host: str = "http://localhost:9200"
    index: str = "vtu-note-chunks"
    search_pipeline: str = "vtu-hybrid-pipeline"
    bm25_weight: float = 0.3
    vector_weight: float = 0.7


class RedisSettings(BaseSettings):
    model_config = _group("REDIS_")

    host: str = "localhost"
    port: int = 6379
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://{self.host}:{self.port}/{self.db}"


class CacheSettings(BaseSettings):
    model_config = _group("CACHE_")

    enabled: bool = True
    ttl_seconds: int = 86400


class JinaSettings(BaseSettings):
    model_config = _group("JINA_")

    api_key: str = ""
    model: str = "jina-embeddings-v3"
    dimensions: int = 1024
    api_url: str = "https://api.jina.ai/v1/embeddings"
    batch_size: int = 64

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


class ChunkSettings(BaseSettings):
    model_config = _group("CHUNK_")

    target_words: int = 350
    overlap_words: int = 60
    min_words: int = 80


class OcrSettings(BaseSettings):
    """OCR for scanned PDFs (ocrmypdf + Tesseract, both installed in the image)."""

    model_config = _group("OCR_")

    enabled: bool = True
    # Tesseract language packs, e.g. "eng" or "eng+kan" (install the pack in the image first)
    language: str = "eng"
    min_words_per_page: int = 20
    min_low_text_ratio: float = 0.3
    timeout_seconds: float = 1800.0
    force_retry: bool = True
    deskew: bool = False


class FigureSettings(BaseSettings):
    """Diagram extraction from note PDFs."""

    model_config = _group("FIGURES_")

    enabled: bool = True
    min_width: int = 150
    min_height: int = 90
    min_bytes: int = 3000
    max_aspect_ratio: float = 20.0
    max_per_page: int = 6
    repeat_ratio: float = 0.25
    include_page_scans: bool = True
    # Read labels inside each diagram so answers can pick the right one
    read_labels: bool = True
    label_timeout_seconds: float = 30.0
    # Most diagrams to attach to one answer
    max_per_answer: int = 3


class LLMSettings(BaseSettings):
    model_config = _group("LLM_")

    provider: Literal["ollama", "groq", "together", "openai_compatible"] = "ollama"
    temperature: float = 0.2
    max_tokens: int = 1024
    timeout_seconds: float = 180.0
    # Used by the OpenAI-compatible providers (Groq, Together, ...)
    api_base_url: str = ""
    api_key: str = ""
    api_model: str = ""


class OllamaSettings(BaseSettings):
    model_config = _group("OLLAMA_")

    host: str = "http://localhost:11434"
    model: str = "llama3.2:3b"


class AgentSettings(BaseSettings):
    model_config = _group("AGENT_")

    max_rewrites: int = 2
    guardrail_threshold: int = 50
    top_k: int = 5


class LangfuseSettings(BaseSettings):
    model_config = _group("LANGFUSE_")

    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""

    @property
    def enabled(self) -> bool:
        return bool(self.public_key and self.secret_key)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    data_dir: Path = Path("data")
    default_branch: str = "cse"
    default_scheme: str = "2022"
    api_base_url: str = "http://localhost:8000"
    # Where a browser can reach the API (figure images are loaded from here)
    public_api_base_url: str = "http://localhost:8000"
    # Index new/changed notes in the background when the API starts
    sync_on_startup: bool = True
    telegram_bot_token: str = ""

    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    opensearch: OpenSearchSettings = Field(default_factory=OpenSearchSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    jina: JinaSettings = Field(default_factory=JinaSettings)
    chunking: ChunkSettings = Field(default_factory=ChunkSettings)
    ocr: OcrSettings = Field(default_factory=OcrSettings)
    figures: FigureSettings = Field(default_factory=FigureSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    ollama: OllamaSettings = Field(default_factory=OllamaSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)


@lru_cache
def get_settings() -> Settings:
    return Settings()
