"""Chat model factory. Returns ``None`` when no LLM is configured."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from parking_assistant.config import Settings


def build_chat_model(settings: Settings) -> BaseChatModel | None:
    if not settings.llm_available:
        return None
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=settings.openai_model,
        api_key=settings.openai_api_key,
        temperature=0,
        timeout=settings.llm_timeout_seconds,
        max_retries=1,
    )
