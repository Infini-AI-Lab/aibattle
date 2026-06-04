"""Two-stage estimate -> decide harness.

Gen-1 asks the model to estimate the opponent's likely hand strength / range
from the public information and action history. Gen-2 threads that estimate into
the game's own decision prompt and chooses an action (through the shared
parse/repair loop). The estimate is recorded under metadata["harness"].

Ref: poker as "act under hidden info, estimate opponent ranges, anticipate"
— How Far Are LLMs from Professional Poker Players? (arXiv:2602.00528);
PokerSkill staged scaffolding (arXiv:2605.30094).
"""

from __future__ import annotations

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent

_DEFAULT_ESTIMATE = (
    "Based only on the public information and the action history so far, "
    "estimate the opponent's most likely hand strength or range, and briefly "
    "explain your reasoning. Do NOT choose an action yet."
)


class TwoStageAgent(HarnessAgent):
    def __init__(self, *, client, template, name="two_stage", max_retries=2,
                 estimate_prompt: str = _DEFAULT_ESTIMATE):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.estimate_prompt = estimate_prompt

    async def act(self, request: AgentRequest) -> AgentResponse:
        # Stage 1: estimate the opponent (game-agnostic prompt over the obs).
        est_prompt = self._compose(request, extra_context=self.estimate_prompt)
        estimate = await self._generate(est_prompt)
        est_text = estimate.content or estimate.full_text or ""

        # Stage 2: decide, with the estimate threaded into the decision prompt.
        decide_context = f"Your estimate of the opponent:\n{est_text}"
        decide_prompt = self._compose(request, extra_context=decide_context)
        return await self._final_loop(
            request, prompt=decide_prompt,
            harness_meta={"kind": "two_stage", "estimate": est_text},
        )
