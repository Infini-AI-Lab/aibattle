"""Base class for in-process reasoning harnesses (inference-time scaffolding).

A *harness* sits on top of the same ``ModelClient`` + ``GameTemplate`` that
``ModelAgent`` uses, but orchestrates MORE than one generation per decision —
e.g. assess the opponent/situation then decide, sample-and-vote, or
draft-critique-revise. These are prompt-engineering / multi-step-reasoning
techniques, NOT integrations of external agent frameworks. They are game-agnostic:
intermediate prompts use only generic strategic wording, and the game-specific
content always comes from the template.

``HarnessAgent`` provides the composable primitives every harness reuses:

  _generate(prompt)   -- call the model once, normalized to GenerateResult
  _final_prompt(req)  -- the game's own decision prompt (template.render_prompt)
  _parse(text, req)   -- text -> Move | None (template.parse)
  _vote(moves)        -- majority vote over parsed Moves (ties: first, stable)
  _compose(req, ctx)  -- game-agnostic mid-prompt: final prompt + extra context

Subclasses implement ``act`` to orchestrate these, and stash intermediate
artifacts (estimates, candidate votes, critiques) under
``AgentResponse.metadata["harness"]`` so logs/replays can audit whether the
harness actually helped.

All harnesses are STATELESS: each ``act`` call is independent (no cross-hand
memory). Opponent modeling / reflection is intentionally out of scope here.
"""

from __future__ import annotations

from abc import abstractmethod
from collections import Counter
from typing import List, Optional

from ...models.base import ModelClient
from ...types import AgentRequest, AgentResponse, Move
from ..base import Agent
from ..template_loop import GenerateResult, parse_or_none, run_template_loop
from ..templates.base import GameTemplate


class HarnessAgent(Agent):
    """Shared base for the reasoning harnesses. Subclasses implement ``act``."""

    agent_type = "local"

    def __init__(self, *, client: ModelClient, template: GameTemplate,
                 name: str = "local", max_retries: int = 2):
        self.client = client
        self.template = template
        self.name = name
        self.max_retries = max_retries

    # --- primitives -------------------------------------------------------

    async def _generate(self, prompt: str, *, temperature: Optional[float] = None) -> GenerateResult:
        """Call the model once and normalize to GenerateResult.

        Carries the same model-metadata keys ModelAgent logs, so a harness's
        final step records identical provider fields. ``temperature`` overrides
        the client default for this call only (used by sampling harnesses).
        """
        kwargs = {} if temperature is None else {"temperature": temperature}
        out = await self.client.generate(prompt, **kwargs)
        return GenerateResult(
            content=out.content,
            full_text=out.full_text(),
            meta={
                "has_reasoning": out.reasoning is not None,
                "finish_reason": out.finish_reason,
                "truncated": out.truncated,
                "completion_tokens": out.completion_tokens,
                "prompt_tokens": out.prompt_tokens,
            },
        )

    def _final_prompt(self, request: AgentRequest) -> str:
        """The game's own decision prompt (delegates to the template)."""
        return self.template.render_prompt(request)

    def _parse(self, text: str, request: AgentRequest) -> Optional[Move]:
        """Parse model text into a legal Move (or None) via the game template."""
        return parse_or_none(self.template, text, request)

    def _vote(self, moves: List[Move]) -> Optional[Move]:
        """Majority vote over parsed Moves.

        Counts identical (type, amount) Moves. On a tie, returns the move that
        appeared FIRST in ``moves`` (stable, deterministic — important for
        reproducibility). Returns None if ``moves`` is empty.
        """
        if not moves:
            return None
        counts = Counter(moves)
        top = max(counts.values())
        for m in moves:  # first-occurrence order breaks ties deterministically
            if counts[m] == top:
                return m
        return moves[0]

    def _compose(self, request: AgentRequest, *, extra_context: str) -> str:
        """Game-agnostic mid-prompt: the final decision prompt plus extra context.

        Used to thread an intermediate reasoning artifact (an opponent/situation
        assessment, a critique) into the final decision step without any
        game-specific prompt logic — the game part stays in the template.
        """
        base = self._final_prompt(request)
        extra = (extra_context or "").strip()
        if not extra:
            return base
        return f"{base}\n\n{extra}"

    async def _final_loop(self, request: AgentRequest, *,
                          prompt: Optional[str] = None,
                          temperature: Optional[float] = None,
                          harness_meta: Optional[dict] = None) -> AgentResponse:
        """Run the shared parse/repair loop for the final decision step.

        ``prompt`` seeds the FIRST attempt (e.g. a composed prompt carrying an
        estimate or critique); when None the template's own decision prompt is
        used. Retries always use the template's game-defined repair prompt.
        ``harness_meta`` is merged into ``metadata["harness"]`` so callers can
        record intermediate artifacts alongside the decision.
        """
        async def generate(p: str) -> GenerateResult:
            return await self._generate(p, temperature=temperature)

        resp = await run_template_loop(
            self.template, generate, request,
            max_retries=self.max_retries, initial_prompt=prompt,
        )
        if harness_meta:
            resp.metadata.setdefault("harness", {})
            resp.metadata["harness"].update(harness_meta)
        return resp

    # --- contract ---------------------------------------------------------

    @abstractmethod
    async def act(self, request: AgentRequest) -> AgentResponse:
        """Orchestrate the harness's multi-step reasoning, return an action."""
