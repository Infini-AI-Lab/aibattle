"""Default model-backed agent: generic wrapper = ModelClient + GameTemplate.

The agent renders a prompt from the request, calls the model, parses the
output into a legal action, and retries with a repair nudge on failure. If all
retries are exhausted it returns ``INVALID`` and lets the runner's
invalid-action policy decide what happens.

The render -> generate -> parse -> repair control flow lives in
``template_loop.run_template_loop`` (shared with the reasoning harnesses under
``local/``); this class only adapts a ``ModelClient`` into that loop's
``generate`` callable and maps ``ModelOutput`` into the logged metadata.
"""

from __future__ import annotations

from ..models.base import ModelClient
from ..types import AgentRequest, AgentResponse
from .base import Agent
from .template_loop import GenerateResult, run_template_loop
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
        async def generate(prompt: str) -> GenerateResult:
            out = await self.client.generate(prompt)
            # Parse the final answer only; the reasoning is for logging.
            return GenerateResult(
                content=out.content,
                full_text=out.full_text(),  # full output incl. thinking
                meta={
                    "has_reasoning": out.reasoning is not None,
                    "finish_reason": out.finish_reason,
                    "truncated": out.truncated,
                    "completion_tokens": out.completion_tokens,
                    "prompt_tokens": out.prompt_tokens,
                },
            )

        return await run_template_loop(
            self.template, generate, request, max_retries=self.max_retries
        )
