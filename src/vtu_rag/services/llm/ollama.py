import time
from typing import Any

import httpx

from vtu_rag.config import LLMSettings, OllamaSettings
from vtu_rag.services.llm.base import ChatMessage, LLMError, LLMProvider, LLMResponse


class OllamaProvider(LLMProvider):
    name = "ollama"

    def __init__(self, ollama: OllamaSettings, llm: LLMSettings):
        self.model = ollama.model
        self.defaults = llm
        self._client = httpx.AsyncClient(
            base_url=ollama.host, timeout=httpx.Timeout(llm.timeout_seconds, connect=10.0)
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
            "stream": False,
            "options": {
                "temperature": self.defaults.temperature if temperature is None else temperature,
                "num_predict": max_tokens or self.defaults.max_tokens,
                "num_ctx": 8192,
            },
        }
        if json_mode:
            payload["format"] = "json"

        started = time.perf_counter()
        try:
            response = await self._client.post("/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if response.status_code == 404:
            raise LLMError(
                f"Ollama model '{self.model}' not found. "
                f"Pull it with: docker compose exec ollama ollama pull {self.model}"
            )
        if response.is_error:
            raise LLMError(f"Ollama returned {response.status_code}: {response.text[:300]}")

        data = response.json()
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=data.get("model", self.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            latency_ms=(time.perf_counter() - started) * 1000,
            raw=data,
        )

    async def health(self) -> dict[str, Any]:
        try:
            response = await self._client.get("/api/tags", timeout=5.0)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return {"ok": False, "provider": self.name, "error": str(exc)}
        models = [m["name"] for m in response.json().get("models", [])]
        # Ollama reports "llama3.2:3b"; a bare "llama3" implies ":latest"
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        available = wanted in models
        result: dict[str, Any] = {
            "ok": available,
            "provider": self.name,
            "model": self.model,
            "model_available": available,
        }
        if not available:
            result["error"] = "model not pulled yet (ollama-init may still be downloading)"
        return result

    async def close(self) -> None:
        await self._client.aclose()
