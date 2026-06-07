from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class LLMClient(Protocol):
    def generate(self, messages: list[dict[str, str]]) -> str: ...


@dataclass(frozen=True)
class LiteLLMClient:
    model: str
    temperature: float = 0.0
    max_tokens: int = 900
    json_mode: bool = True
    api_base: str | None = None
    api_key: str | None = None
    timeout: float | None = None

    def completion_kwargs(self, messages: list[dict[str, str]]) -> dict:
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.timeout is not None:
            kwargs["timeout"] = self.timeout
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "Answer generation needs the optional rag dependencies. "
                'Install with: uv pip install -e ".[rag]"'
            ) from exc

        response = completion(**self.completion_kwargs(messages))
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return str(content)
