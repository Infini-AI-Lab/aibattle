"""Default model-backed agent: generic wrapper = ModelClient + GameTemplate.

The agent renders a prompt from the request, calls the model, parses the
output into a legal action, and retries with a repair nudge on failure. If all
retries are exhausted it returns ``INVALID`` and lets the runner's
invalid-action policy decide what happens.
"""

from __future__ import annotations

import time

from ..models.base import ModelClient
from ..types import INVALID, AgentRequest, AgentResponse
from .base import Agent
from .templates.base import GameTemplate


class ModelAgent(Agent):
    agent_type = "model"

    def __init__(self, client: ModelClient, template: GameTemplate, *,
                 name: str = "model", max_retries: int = 2):
        self.client = client
        self.template = template
        self.name = name
        self.max_retries = max_retries

    async def act(self, request: AgentRequest) -> AgentResponse:
        prompt = self.template.render_prompt(request)
        out = None
        t0 = time.perf_counter()

        for attempt in range(self.max_retries + 1):
            out = await self.client.generate(prompt)
            # Parse the final answer only; the reasoning is for logging.
            move = self.template.parse(out.content, request)
            if move is not None:
                latency_ms = round((time.perf_counter() - t0) * 1000, 1)
                return AgentResponse(
                    action=move.type,
                    amount=move.amount,
                    message=out.content,
                    raw_output=out.full_text(),  # full output incl. thinking
                    metadata={
                        "attempts": attempt + 1,
                        "latency_ms": latency_ms,
                        "has_reasoning": out.reasoning is not None,
                    },
                )
            # Repair using the visible answer; pass full text if no answer at all.
            prompt = self.template.repair_prompt(request, out.content or out.full_text())

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        return AgentResponse(
            action=INVALID,
            message=out.content if out else None,
            raw_output=out.full_text() if out else None,
            metadata={
                "attempts": self.max_retries + 1,
                "latency_ms": latency_ms,
                "has_reasoning": (out.reasoning is not None) if out else False,
                "invalid": True,
            },
        )
