"""Kuhn Poker prompt template + tolerant output parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import Action, AgentRequest
from .base import GameTemplate

_RULES = (
    "You are playing 2-player Kuhn Poker. The deck has three cards: J, Q, K "
    "(ranked J < Q < K). Each player is dealt one private card and antes 1 chip. "
    "On your turn you choose one action:\n"
    "  - check: pass without betting (only when no bet is facing you)\n"
    "  - bet: put in 1 more chip\n"
    "  - call: match a facing bet (go to showdown)\n"
    "  - fold: give up the pot\n"
    "At showdown the higher card wins the pot. Play to maximize your chips."
)


class KuhnTemplate(GameTemplate):
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        legal = ", ".join(obs.legal_actions)
        return (
            f"{_RULES}\n\n"
            f"{obs.rendered}\n\n"
            f"Choose exactly one of these legal actions: {legal}.\n"
            f"Respond with ONLY the single action word, nothing else."
        )

    def parse(self, raw: str, legal_actions: list) -> Optional[Action]:
        if not raw:
            return None
        text = raw.lower()
        # Prefer a whole-word match of a legal action token.
        for action in legal_actions:
            if re.search(rf"\b{re.escape(action)}\b", text):
                return action
        return None

    def repair_prompt(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"{self.render_prompt(request)}\n\n"
            f"Your previous reply ({bad_output!r}) did not contain a valid action. "
            f"Reply with exactly one word from: {legal}."
        )
