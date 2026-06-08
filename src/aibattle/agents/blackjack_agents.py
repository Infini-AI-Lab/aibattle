"""Built-in agents for Independent Blackjack.

- BlackjackDealerAgent: the scripted dealer occupying seat ``player_1``. It reads
  the dealer hand from the observation and applies the fixed house policy (hit
  while the ace-aware total is below 17, stand on all 17s including soft 17). It
  never calls a model client.
- RandomBlackjackPlayerAgent: a uniform-random player baseline for the LLM seat,
  used by no-API smoke runs.
"""

from __future__ import annotations

from ..games.blackjack import hand_total, dealer_should_hit
from ..types import AgentRequest, AgentResponse
from .base import Agent
from .board_agents import _rng_for


class BlackjackDealerAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "blackjack_dealer", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        obs = request.observation
        # During the dealer's turn the dealer hand is exposed in the observation.
        dealer = obs.public.get("dealer_hand")
        if dealer is None:
            # Defensive: if the hand is not visible, fall back to standing.
            return AgentResponse(action="stand", message="dealer:stand(no-hand)",
                                 metadata={"policy": "blackjack_dealer"})
        action = "hit" if dealer_should_hit(dealer) else "stand"
        total, soft = hand_total(dealer)
        return AgentResponse(
            action=action,
            message=f"dealer:{action} (total {total}{' soft' if soft else ''})",
            metadata={"policy": "blackjack_dealer", "total": total, "soft": soft},
        )


class RandomBlackjackPlayerAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "blackjack_random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        legal = request.observation.legal_actions
        action = _rng_for(self._seed, request).choice(legal)
        return AgentResponse(action=action, message="random",
                             metadata={"policy": "blackjack_random"})
