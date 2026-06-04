"""Self-Refine (draft -> critique -> revise) harness.

Gen-1 produces a draft action with reasoning. Then for ``rounds`` iterations, the
same model critiques the current draft (is this the highest-EV legal action? is
there a better one?) and the draft is revised. The final revision goes through
the shared parse/repair loop. Each round's draft and critique are recorded under
metadata["harness"].

Ref: Madaan et al. 2023, "Self-Refine: Iterative Refinement with Self-Feedback"
(arXiv:2303.17651, NeurIPS 2023).
"""

from __future__ import annotations

from ...types import AgentRequest, AgentResponse
from .base import HarnessAgent

_DEFAULT_CRITIQUE = (
    "Critique the proposed action above: is it the highest-EV legal action "
    "given the situation, or would a different legal action be better? Point out "
    "any mistakes. Do NOT give a final action yet — only critique."
)


class SelfRefineAgent(HarnessAgent):
    def __init__(self, *, client, template, name="self_refine", max_retries=2,
                 rounds: int = 1, critique_prompt: str = _DEFAULT_CRITIQUE):
        super().__init__(client=client, template=template, name=name,
                         max_retries=max_retries)
        self.rounds = max(1, int(rounds))
        self.critique_prompt = critique_prompt

    async def act(self, request: AgentRequest) -> AgentResponse:
        # Initial draft (with reasoning).
        draft = await self._generate(self._final_prompt(request))
        draft_text = draft.content or draft.full_text or ""

        history = []  # [{"draft": ..., "critique": ...}, ...]
        last_critique = ""
        for _ in range(self.rounds):
            crit_prompt = self._compose(
                request,
                extra_context=f"Proposed action and reasoning:\n{draft_text}\n\n{self.critique_prompt}",
            )
            critique = await self._generate(crit_prompt)
            last_critique = critique.content or critique.full_text or ""
            history.append({"draft": draft_text, "critique": last_critique})

            # Revise the draft in light of the critique (game-agnostic context).
            revise_prompt = self._compose(
                request,
                extra_context=(f"Your earlier proposal:\n{draft_text}\n\n"
                               f"Critique:\n{last_critique}\n\n"
                               "Revise your decision accordingly."),
            )
            revision = await self._generate(revise_prompt)
            draft_text = revision.content or revision.full_text or ""

        # Final decision step: parse the latest revision; repair if needed.
        final_context = (f"Your refined reasoning:\n{draft_text}\n\n"
                         f"Latest critique:\n{last_critique}")
        decide_prompt = self._compose(request, extra_context=final_context)
        return await self._final_loop(
            request, prompt=decide_prompt,
            harness_meta={"kind": "self_refine", "rounds": self.rounds,
                          "history": history},
        )
