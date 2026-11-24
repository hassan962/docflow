"""One thin client for any OpenAI-compatible chat endpoint.

Groq, Gemini's OpenAI-compatible endpoint, a local Ollama server and paid providers all speak
the same API, so switching provider is configuration (``LLM_BASE_URL``, ``LLM_MODEL``,
``LLM_API_KEY``), not code. The client records tokens, latency and cost for every call, which
feed ``usage_events``, the cost dashboard and the promotion gate's cost check.

Rate limits (HTTP 429) and transient 5xx errors are retried by the OpenAI SDK with
exponential backoff that honours the provider's ``retry-after`` header.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import OpenAI

from docflow_core.config import Settings


@dataclass(frozen=True)
class ChatResult:
    content: str | None  # None when the model stopped before answering
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: int
    cost_usd: float
    finish_reason: str | None = None


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict[str, str]]
    temperature: float = 0.0
    max_tokens: int = 2048
    json_schema: dict[str, Any] | None = None  # strict structured output when set


class LlmClient(Protocol):
    model: str

    def chat(self, request: ChatRequest) -> ChatResult: ...


def price(settings: Settings, model: str, tokens_in: int, tokens_out: int) -> float:
    per_in, per_out = settings.llm_prices.get(model, (0.0, 0.0))
    return round((tokens_in * per_in + tokens_out * per_out) / 1_000_000, 8)


class OpenAICompatibleClient:
    def __init__(self, settings: Settings, model: str, client: OpenAI | None = None) -> None:
        self.model = model
        self.settings = settings
        api_key = settings.llm_api_key.get_secret_value() if settings.llm_api_key else "unused"
        self._client = client or OpenAI(
            api_key=api_key,
            base_url=settings.llm_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=4,
        )

    def chat(self, request: ChatRequest) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
        }
        if request.json_schema is not None and self.settings.llm_response_format == "json_schema":
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "extraction",
                    "strict": True,
                    "schema": request.json_schema,
                },
            }
        else:
            kwargs["response_format"] = {"type": "json_object"}
        if self.settings.llm_reasoning_effort and self.model.startswith("openai/gpt-oss"):
            kwargs["extra_body"] = {"reasoning_effort": self.settings.llm_reasoning_effort}

        started = time.perf_counter()
        response = self._client.chat.completions.create(**kwargs)
        latency_ms = int((time.perf_counter() - started) * 1000)
        choice = response.choices[0]
        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else 0
        tokens_out = usage.completion_tokens if usage else 0
        return ChatResult(
            content=choice.message.content,
            model=self.model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            cost_usd=price(self.settings, self.model, tokens_in, tokens_out),
            finish_reason=choice.finish_reason,
        )


@dataclass
class ScriptedClient:
    """Deterministic stand-in for tests: replies come from a list or a function."""

    model: str = "scripted"
    replies: list[str | None] = field(default_factory=list)
    responder: Callable[[ChatRequest], str | None] | None = None
    requests: list[ChatRequest] = field(default_factory=list)

    def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        content = self.responder(request) if self.responder else self.replies.pop(0)
        return ChatResult(content, self.model, 1000, 200, 5, 0.0003, "stop")
