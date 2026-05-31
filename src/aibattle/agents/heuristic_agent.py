"""Kuhn-specific heuristic baseline.

A simple near-equilibrium policy — stronger than random, weak enough to be a
useful calibration target:

  Facing a bet (call/fold):
    K -> call (value)
    Q -> call 1/3 of the time, else fold
    J -> fold

  Opening / after a check (check/bet):
    K -> bet (value)
    Q -> check
    J -> bet 1/3 of the time (bluff), else check

The probabilistic branches use an injected RNG so behavior is reproducible.
"""

from __future__ import annotations

import random

from ..types import AgentRequest, AgentResponse
from .base import Agent


class KuhnHeuristicAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "kuhn_heuristic", seed: int | None = None,
                 bluff_prob: float = 1 / 3, call_prob: float = 1 / 3):
        self.name = name
        self._seed = seed or 0
        self._rng = random.Random(seed)
        self._bluff_prob = bluff_prob
        self._call_prob = call_prob

    def _rng_for(self, request: AgentRequest) -> random.Random:
        # Deterministic per-decision RNG when the runner supplies a seed, so
        # results are reproducible even under parallel execution.
        if request.decision_seed is not None:
            return random.Random((self._seed * 2654435761 ^ request.decision_seed) & 0x7FFFFFFF)
        return self._rng

    async def act(self, request: AgentRequest) -> AgentResponse:
        legal = request.observation.legal_actions
        card = request.observation.private.get("card")
        facing_bet = "call" in legal  # call/fold node
        rng = self._rng_for(request)

        if facing_bet:
            if card == "K":
                action = "call"
            elif card == "Q":
                action = "call" if rng.random() < self._call_prob else "fold"
            else:  # J
                action = "fold"
        else:  # check/bet node
            if card == "K":
                action = "bet"
            elif card == "Q":
                action = "check"
            else:  # J
                action = "bet" if rng.random() < self._bluff_prob else "check"

        # Guard against any unexpected legality mismatch.
        if action not in legal:
            action = legal[0]
        return AgentResponse(action=action, message=f"heuristic({card})",
                             metadata={"policy": "kuhn_heuristic", "card": card})
