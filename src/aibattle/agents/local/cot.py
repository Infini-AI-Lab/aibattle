"""Structured Chain-of-Thought harness.

A single generation, but the decision prompt is augmented with a generic,
game-agnostic structured-reasoning instruction (assess the current state and
objective, immediate opportunities/threats, the opponent's likely plan from the
visible history, and the legal options — THEN give the action on the last line).
The instruction text is domain-neutral so it works for any game (Kuhn, Hold'em,
Connect Four, Gomoku); the game-specific content comes from the template. The
shared parse/repair loop still backs it, so a non-conforming answer is repaired
exactly as for ModelAgent.

Ref: Wei et al. 2022, "Chain-of-Thought Prompting Elicits Reasoning in LLMs"
(arXiv:2201.11903).
"""

from __future__ import annotations

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent

_DEFAULT_COT = (
    "Before answering, reason step by step: (1) the current state and your "
    "objective, (2) the immediate opportunities and threats, (3) the opponent's "
    "likely plan, inferred from the visible history, and (4) your legal options. "
    "Then put your final action on the LAST line."
)


class StructuredCoTAgent(HarnessAgent):
    def __init__(self, *, client, template, name="cot", max_retries=2,
                 cot_instructions: str = _DEFAULT_COT):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.cot_instructions = cot_instructions

    async def act(self, request: AgentRequest) -> AgentResponse:
        prompt = self._compose(request, extra_context=self.cot_instructions)
        return await self._final_loop(
            request, prompt=prompt,
            harness_meta={"kind": "structured_cot"},
        )
