"""Adapter for OpenAI-compatible chat APIs (Groq, Together, vLLM, LM Studio, ...)."""

import time
from typing import Any

import httpx

from vtu_rag.config import LLMSettings
from vtu_rag.services.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse

KNOWN_BASE_URLS = {
    "groq": "https://api.groq.com/openai/v1",
    "together": "https://api.together.xyz/v1",
}


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, settings: LLMSettings):
        self.name = settings.provider
        base_url = settings.api_base_url or KNOWN_BASE_URLS.get(settings.provider, "")
        if not base_url:
            raise LLMError(f"LLM_API_BASE_URL is required for provider '{settings.provider}'")
        if not settings.api_model:
            raise LLMError(f"LLM_API_MODEL is required for provider '{settings.provider}'")
        self.model = settings.api_model
        self.defaults = settings
        headers = {"Authorization": f"Bearer {settings.api_key}"} if settings.api_key else {}
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers=headers,
            timeout=httpx.Timeout(settings.timeout_seconds, connect=10.0),
        )

    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.as_dict() for m in messages],
            "temperature": self.defaults.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.defaults.max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"{self.name} request failed: {exc}") from exc
        if response.is_error:
            raise LLMError(f"{self.name} returned {response.status_code}: {response.text[:300]}")

        data = response.json()
        usage = data.get("usage") or {}
        return LLMResponse(
            content=data["choices"][0]["message"]["content"] or "",
            model=data.get("model", self.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=data,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/models", timeout=5.0)
            ok = response.status_code == 200
        except httpx.HTTPError as exc:
            return {"ok": False, "provider": self.name, "error": str(exc)}
        result: dict[str, Any] = {"ok": ok, "provider": self.name, "model": self.model}
        if not ok:
            result["error"] = f"HTTP {response.status_code}"
        return result

    async def close(self) -> None:
        await self._client.aclose()
