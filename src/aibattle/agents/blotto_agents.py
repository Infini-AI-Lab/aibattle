"""Built-in Repeated Colonel Blotto baseline: uniform-random valid allocation.

The generic RandomAgent can only emit the single default allocation advertised
in ``legal_actions``; this agent samples a genuinely random valid allocation
(non-negative integers summing to RESOURCES) so random-vs-random smoke runs
explore the allocation space. Stochastic choices use the request's deterministic
per-decision seed so they are reproducible under parallelism.
"""

from __future__ import annotations

from ..games.blotto import RESOURCES, N_FIELDS, encode_alloc
from ..types import AgentRequest, AgentResponse
from .base import Agent
from .board_agents import _rng_for


def _random_allocation(rng) -> list:
    """Uniformly partition RESOURCES units into N_FIELDS non-negative integers
    via the stars-and-bars method (random cut points)."""
    cuts = sorted(rng.randint(0, RESOURCES) for _ in range(N_FIELDS - 1))
    bounds = [0] + cuts + [RESOURCES]
    return [bounds[i + 1] - bounds[i] for i in range(N_FIELDS)]


class RandomBlottoAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "blotto_random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        rng = _rng_for(self._seed, request)
        alloc = _random_allocation(rng)
        return AgentResponse(action=encode_alloc(alloc), message="random",
                             metadata={"policy": "blotto_random"})
