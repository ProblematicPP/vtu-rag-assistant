import asyncio
from typing import Any

from fastapi import APIRouter, Response, status

from vtu_rag import __version__
from vtu_rag.dependencies import ContainerDep

router = APIRouter(tags=["health"])


async def _check(coro) -> dict[str, Any]:
    try:
        result = await asyncio.wait_for(coro, timeout=5)
    except Exception as exc:
        return {"ok": False, "error": str(exc) or type(exc).__name__}
    return result if isinstance(result, dict) else {"ok": bool(result)}


@router.get("/health/live", summary="Liveness probe")
async def live() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health", summary="Dependency health")
async def health(container: ContainerDep, response: Response) -> dict[str, Any]:
    async def opensearch() -> dict[str, Any]:
        await container.search.ping()
        return {
            "ok": True,
            "index": container.search.index,
            "chunks": await container.search.count(),
        }

    postgres, search, redis, llm = await asyncio.gather(
        _check(container.db.ping()),
        _check(opensearch()),
        _check(container.cache.ping()),
        _check(container.llm.health()),
    )
    services = {
        "postgres": postgres,
        "opensearch": search,
        "redis": redis,
        "llm": llm,
        "embeddings": {
            "ok": container.embeddings.enabled,
            "provider": "jina",
            "note": None if container.embeddings.enabled else "JINA_API_KEY not set; BM25-only",
        },
        "langfuse": {"ok": container.tracer.enabled, "optional": True},
    }
    core_ok = postgres["ok"] and search["ok"]
    all_ok = core_ok and redis["ok"] and llm["ok"]
    if not core_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "status": "ok" if all_ok else ("degraded" if core_ok else "unavailable"),
        "version": __version__,
        "services": services,
    }
