"""Anthropic (Claude) model client."""

from __future__ import annotations

from typing import Optional, Union

from .base import ModelClient, ModelOutput


class AnthropicClient(ModelClient):
    def __init__(self, *, model_id: str, api_key: str, temperature: float = 0.0,
                 max_tokens: int = 256, system_prompt: Optional[str] = None):
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'anthropic' package is required for Anthropic model agents. "
                "Install with: pip install anthropic"
            ) from e
        self.model_id = model_id
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._system_prompt = system_prompt
        self._client = AsyncAnthropic(api_key=api_key)

    async def generate(self, prompt: Union[str, list], *,
                       temperature: Optional[float] = None,
                       max_tokens: Optional[int] = None) -> ModelOutput:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = prompt
        kwargs = {
            "model": self.model_id,
            "messages": messages,
            "temperature": (self._default_temperature if temperature is None else temperature),
            "max_tokens": (self._default_max_tokens if max_tokens is None else max_tokens),
        }
        if self._system_prompt:
            kwargs["system"] = self._system_prompt
        resp = await self._client.messages.create(**kwargs)
        content = "".join(b.text for b in resp.content if b.type == "text")
        reasoning = "".join(
            getattr(b, "thinking", "") for b in resp.content if b.type == "thinking"
        ) or None
        return ModelOutput(content=content, reasoning=reasoning)
