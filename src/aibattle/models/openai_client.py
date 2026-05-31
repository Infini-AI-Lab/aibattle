"""OpenAI-compatible chat client.

Covers OpenAI itself and any OpenAI-compatible endpoint (Fireworks, vLLM,
local) via a configurable ``base_url``. The repo ships with a ``.fireworks``
config, so Fireworks-over-OpenAI-compat is the near-term path.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Union

from .base import ModelClient, ModelOutput


class OpenAIClient(ModelClient):
    def __init__(
        self,
        *,
        model_id: str,
        api_key: str,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 256,
        system_prompt: Optional[str] = None,
        timeout: float = 300.0,
    ):
        try:
            from openai import AsyncOpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for model agents. "
                "Install with: pip install openai"
            ) from e

        self.model_id = model_id
        self._default_temperature = temperature
        self._default_max_tokens = max_tokens
        self._system_prompt = system_prompt
        # Per-request timeout. A hung call is retried with backoff rather than
        # blocking for the SDK default (600s). Keep it generous enough for a full
        # reasoning generation (16k tokens can take a few minutes on slow models).
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    async def generate(
        self,
        prompt: Union[str, list],
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> ModelOutput:
        if isinstance(prompt, str):
            messages = []
            if self._system_prompt:
                messages.append({"role": "system", "content": self._system_prompt})
            messages.append({"role": "user", "content": prompt})
        else:
            messages = prompt

        # Retry transient failures (rate limits, timeouts) with exponential
        # backoff so high-concurrency runs don't turn a 429 into an invalid move.
        last_exc = None
        for attempt in range(5):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=(self._default_temperature if temperature is None else temperature),
                    max_tokens=(self._default_max_tokens if max_tokens is None else max_tokens),
                )
                break
            except Exception as e:  # noqa: BLE001 - includes RateLimit/APITimeout
                last_exc = e
                if attempt == 4:
                    raise
                await asyncio.sleep(min(30, 2 ** attempt))
        msg = resp.choices[0].message
        # Reasoning models expose chain-of-thought in a separate field; the name
        # varies by provider (reasoning_content / reasoning).
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        return ModelOutput(content=msg.content or "", reasoning=reasoning)
