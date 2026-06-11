"""Built-in Leduc Poker baseline: random with valid bet/raise amounts.

The generic RandomAgent cannot play Leduc because bet/raise require the exact
total street commitment (raise-to ``amount``); this agent fills it from the
observation's fixed bet size. Stochastic choices use the request's deterministic
per-decision seed so they are reproducible under parallelism.
"""

from __future__ import annotations

from ..types import AgentRequest, AgentResponse
from .base import Agent
from .board_agents import _rng_for


class RandomLeducAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "leduc_random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        rng = _rng_for(self._seed, request)
        obs = request.observation
        legal = obs.legal_actions
        atype = rng.choice(legal)
        amount = None
        if atype in ("bet", "raise"):
            size = int(obs.public.get("bet_size", 0))
            your_commit = int(obs.public.get("your_commit", 0))
            to_call = int(obs.public.get("to_call", 0))
            cur_max = your_commit + to_call
            amount = (your_commit + size) if atype == "bet" else (cur_max + size)
        return AgentResponse(action=atype, amount=amount, message="random",
                             metadata={"policy": "leduc_random"})
