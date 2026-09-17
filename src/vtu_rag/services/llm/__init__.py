from vtu_rag.config import Settings
from vtu_rag.services.llm.base import (
    ChatMessage,
    LLMError,
    LLMProvider,
    LLMResponse,
    parse_json_object,
)


def create_llm_provider(settings: Settings) -> LLMProvider:
    if settings.llm.provider == "ollama":
        from vtu_rag.services.llm.ollama import OllamaProvider

        return OllamaProvider(settings.ollama, settings.llm)

    from vtu_rag.services.llm.openai_compatible import OpenAICompatibleProvider

    return OpenAICompatibleProvider(settings.llm)


__all__ = [
    "ChatMessage",
    "LLMError",
    "LLMProvider",
    "LLMResponse",
    "create_llm_provider",
    "parse_json_object",
]
