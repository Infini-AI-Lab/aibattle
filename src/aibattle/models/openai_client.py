"""OpenAI-compatible chat client.

Covers OpenAI itself and any OpenAI-compatible endpoint (Fireworks, vLLM,
local) via a configurable ``base_url``. The repo ships with a ``.fireworks``
config, so Fireworks-over-OpenAI-compat is the near-term path.
"""

from __future__ import annotations

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
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

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

        resp = await self._client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=(self._default_temperature if temperature is None else temperature),
            max_tokens=(self._default_max_tokens if max_tokens is None else max_tokens),
        )
        msg = resp.choices[0].message
        # Reasoning models expose chain-of-thought in a separate field; the name
        # varies by provider (reasoning_content / reasoning).
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        return ModelOutput(content=msg.content or "", reasoning=reasoning)
