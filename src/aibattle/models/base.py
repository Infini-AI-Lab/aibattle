"""Model layer abstraction.

A ``ModelClient`` is a thin ``str -> str`` (or chat-messages -> str) interface.
All provider-specific request/response shaping lives inside the client adapter;
the agent template only ever sees text in and text out.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Union


@dataclass
class ModelOutput:
    """A model's reply, splitting reasoning (chain-of-thought) from the answer.

    Reasoning models return their thinking either in a dedicated field
    (e.g. gpt-oss -> ``reasoning_content``) or inline in ``content``
    (e.g. some deepseek deployments). ``content`` is what gets parsed into an
    action; ``reasoning`` is captured for logging so the full output is visible.
    """

    content: str
    reasoning: Optional[str] = None
    finish_reason: Optional[str] = None  # "stop" = complete, "length" = truncated
    completion_tokens: Optional[int] = None  # output tokens (reasoning + answer)
    prompt_tokens: Optional[int] = None      # input tokens

    @property
    def truncated(self) -> bool:
        """True if generation was cut off by the token cap (finish_reason=length)."""
        return self.finish_reason == "length"

    def full_text(self) -> str:
        """The complete output, thinking included — used for logs/transcripts."""
        if self.reasoning:
            return (
                "===== thinking =====\n"
                f"{self.reasoning}\n"
                "===== answer =====\n"
                f"{self.content}"
            )
        return self.content


class ModelClient(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: Union[str, list],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> ModelOutput:
        ...
