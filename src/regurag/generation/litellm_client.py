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

    def generate(self, messages: list[dict[str, str]]) -> str:
        try:
            from litellm import completion
        except ImportError as exc:
            raise RuntimeError(
                "Answer generation needs the optional rag dependencies. "
                'Install with: uv pip install -e ".[rag]"'
            ) from exc

        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = completion(**kwargs)
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("LLM returned an empty response.")
        return str(content)
