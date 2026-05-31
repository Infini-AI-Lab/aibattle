"""Interactive human agent: prompts for an action in the terminal.

Used for human-vs-agent play. The human's own turn is handled here (print the
observation, read a legal action from stdin). What the human sees about the
*opponent's* moves — action only, or full model output incl. thinking — is
handled by the interactive observer in the CLI, controlled by `show_thinking`.
"""

from __future__ import annotations

import asyncio

from ..types import AgentRequest, AgentResponse
from .base import Agent


class HumanAgent(Agent):
    agent_type = "human"

    def __init__(self, name: str = "human"):
        self.name = name

    async def act(self, request: AgentRequest) -> AgentResponse:
        obs = request.observation
        legal = obs.legal_actions

        if obs.history:
            hist = ", ".join(f"{h['player']} {h['action']}" for h in obs.history)
        else:
            hist = "(no actions yet)"

        print()
        print(f"--- Your turn ({request.player}) ---")
        print(f"Your card: {obs.private.get('card')}   Pot: {obs.public.get('pot')}")
        print(f"History: {hist}")
        print(f"Legal actions: {', '.join(legal)}")

        while True:
            raw = await asyncio.to_thread(input, f"Your action {legal}: ")
            choice = raw.strip().lower()
            if choice in legal:
                return AgentResponse(action=choice, message="human",
                                     metadata={"human": True})
            # Accept an unambiguous prefix (e.g. "c" for check, "b" for bet).
            matches = [a for a in legal if a.startswith(choice)] if choice else []
            if len(matches) == 1:
                return AgentResponse(action=matches[0], message="human",
                                     metadata={"human": True})
            print(f"  Invalid input {raw.strip()!r}. Choose one of: {', '.join(legal)}")
