"""Random baseline agent: uniform over legal actions.

Seeded for reproducibility. The runner injects a per-match seed so that a
random-vs-random match is fully repeatable.
"""

from __future__ import annotations

import random

from ..types import AgentRequest, AgentResponse
from .base import Agent


class RandomAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0
        self._rng = random.Random(seed)

    def _rng_for(self, request: AgentRequest) -> random.Random:
        # Deterministic per-decision RNG when the runner supplies a seed, so
        # results are reproducible even under parallel execution.
        if request.decision_seed is not None:
            return random.Random((self._seed * 2654435761 ^ request.decision_seed) & 0x7FFFFFFF)
        return self._rng

    async def act(self, request: AgentRequest) -> AgentResponse:
        legal = request.observation.legal_actions
        action = self._rng_for(request).choice(legal)
        return AgentResponse(action=action, message="uniform random",
                             metadata={"policy": "random"})
