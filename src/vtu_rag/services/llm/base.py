"""Provider-agnostic LLM interface."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant"]


@dataclass
class ChatMessage:
    role: Role
    content: str

    def as_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class LLMResponse:
    content: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


class LLMError(RuntimeError):
    pass


class LLMProvider(ABC):
    """Chat-completion backend. Implementations: Ollama (default), OpenAI-compatible APIs."""

    name: str
    model: str

    @abstractmethod
    async def generate(
        self,
        messages: list[ChatMessage],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> LLMResponse: ...

    @abstractmethod
    async def health(self) -> dict[str, Any]:
        """Returns at least {"ok": bool}; never raises."""

    async def close(self) -> None:  # noqa: B027 - optional hook
        pass


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_json_object(text: str) -> dict[str, Any]:
    """Extracts the first JSON object from model output (tolerates code fences / chatter)."""
    candidates = [text.strip()]
    candidates += [m.strip() for m in _FENCE_RE.findall(text)]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise LLMError(f"Model did not return a JSON object: {text[:200]!r}")
