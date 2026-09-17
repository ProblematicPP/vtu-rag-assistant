import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from vtu_rag import __version__
from vtu_rag.agent import AgentService
from vtu_rag.config import get_settings
from vtu_rag.container import Container
from vtu_rag.ingestion.sources import LocalFolderSource
from vtu_rag.logging_config import configure_logging
from vtu_rag.routers import all_routers
from vtu_rag.services.embeddings import EmbeddingError
from vtu_rag.services.llm import LLMError

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    container = Container.build(settings)
    await container.startup()
    app.state.container = container
    app.state.agent = AgentService(
        container.search, container.llm, container.cache, container.tracer, settings.agent
    )
    logger.info(
        "VTU RAG API ready (llm=%s:%s, embeddings=%s)",
        container.llm.name,
        container.llm.model,
        "on" if container.embeddings.enabled else "off",
    )
    sync_task = None
    if settings.sync_on_startup:
        sync_task = asyncio.create_task(_startup_sync(container))
    try:
        yield
    finally:
        if sync_task is not None and not sync_task.done():
            sync_task.cancel()
        await container.shutdown()


async def _startup_sync(container: Container) -> None:
    try:
        report = await container.ingestion.sync(LocalFolderSource(container.settings.data_dir))
        logger.info("Startup sync: %s", report.summary())
    except Exception:
        logger.exception("Startup sync failed")


def create_app() -> FastAPI:
    app = FastAPI(
        title="VTU RAG Assistant",
        version=__version__,
        description=(
            "Grounded answers from VTU syllabus notes, with citations to the "
            "subject, module and note they came from."
        ),
        lifespan=lifespan,
    )
    for router in all_routers:
        app.include_router(router)

    @app.exception_handler(LLMError)
    async def llm_error(_: Request, exc: LLMError) -> JSONResponse:
        logger.error("LLM error: %s", exc)
        return JSONResponse(status_code=503, content={"detail": f"LLM unavailable: {exc}"})

    @app.exception_handler(EmbeddingError)
    async def embedding_error(_: Request, exc: EmbeddingError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": f"Embeddings unavailable: {exc}"})

    @app.get("/", include_in_schema=False)
    async def root() -> dict[str, str]:
        return {"name": "VTU RAG Assistant", "docs": "/docs", "health": "/health"}

    return app


app = create_app()
