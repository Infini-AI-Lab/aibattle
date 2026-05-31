"""OpenAI-compatible chat client.

Covers OpenAI itself and any OpenAI-compatible endpoint (Fireworks, vLLM,
local) via a configurable ``base_url``. The repo ships with a ``.fireworks``
config, so Fireworks-over-OpenAI-compat is the near-term path.
"""

from __future__ import annotations

import asyncio
import random
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
        # backoff + jitter, so high-concurrency runs ride out rate-limit windows
        # instead of turning a 429 into a failed move. Jitter de-synchronizes
        # retries to avoid a thundering-herd spike when many calls back off at once.
        attempts = 8
        for attempt in range(attempts):
            try:
                resp = await self._client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=(self._default_temperature if temperature is None else temperature),
                    max_tokens=(self._default_max_tokens if max_tokens is None else max_tokens),
                )
                break
            except Exception:  # noqa: BLE001 - includes RateLimit/APITimeout
                if attempt == attempts - 1:
                    raise
                base = min(60, 2 ** attempt)
                await asyncio.sleep(base * (0.5 + random.random()))
        choice = resp.choices[0]
        msg = choice.message
        # Reasoning models expose chain-of-thought in a separate field; the name
        # varies by provider (reasoning_content / reasoning).
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        # finish_reason="length" means the output hit the token cap (truncated) —
        # the model-agnostic truncation signal (don't infer from which field is
        # empty; that flips per model). usage gives exact token counts.
        usage = getattr(resp, "usage", None)
        return ModelOutput(
            content=msg.content or "", reasoning=reasoning,
            finish_reason=choice.finish_reason,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
        )
