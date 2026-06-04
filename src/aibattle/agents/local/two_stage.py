"""Two-stage assess -> decide harness.

Gen-1 asks the model to assess the opponent and the situation from the public
information and action history. Gen-2 threads that assessment into the game's own
decision prompt and chooses an action (through the shared parse/repair loop). The
assessment is recorded under metadata["harness"].

The default Stage-1 prompt is game-agnostic — it asks for the opponent's likely
position, plan, pressure, or available threats from public information, without
mentioning any game-specific concept — so it works for any game (Kuhn, Hold'em,
Connect Four, Gomoku). The game-specific content comes from the template.

Ref (motivation): assessing a hidden-information opponent before deciding —
How Far Are LLMs from Professional Poker Players? (arXiv:2602.00528);
PokerSkill staged scaffolding (arXiv:2605.30094).
"""

from __future__ import annotations

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent

_DEFAULT_ESTIMATE = (
    "Based only on the public information and the action history so far, assess "
    "the opponent's likely position, plan, pressure, or available threats, and "
    "briefly explain your reasoning. Do NOT choose an action yet."
)


class TwoStageAgent(HarnessAgent):
    def __init__(self, *, client, template, name="two_stage", max_retries=2,
                 estimate_prompt: str = _DEFAULT_ESTIMATE):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.estimate_prompt = estimate_prompt

    async def act(self, request: AgentRequest) -> AgentResponse:
        # Stage 1: assess the opponent/situation (game-agnostic prompt over the obs).
        est_prompt = self._compose(request, extra_context=self.estimate_prompt)
        estimate = await self._generate(est_prompt)
        est_text = estimate.content or estimate.full_text or ""

        # Stage 2: decide, with the assessment threaded into the decision prompt.
        decide_context = f"Opponent/context assessment:\n{est_text}"
        decide_prompt = self._compose(request, extra_context=decide_context)
        return await self._final_loop(
            request, prompt=decide_prompt,
            harness_meta={"kind": "two_stage", "estimate": est_text},
        )
