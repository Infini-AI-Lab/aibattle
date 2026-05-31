"""Interactive human agent: prompts for an action in the terminal.

Used for human-vs-agent play. The human's own turn is handled here (print the
observation, read a legal action from stdin). What the human sees about the
*opponent's* moves — action only, or full model output incl. thinking — is
handled by the interactive observer in the CLI, controlled by `show_thinking`.
"""

from __future__ import annotations

import asyncio

from ..termcolor import decorate, rule
from ..types import AgentRequest, AgentResponse
from .base import Agent


class HumanAgent(Agent):
    agent_type = "human"

    def __init__(self, name: str = "human"):
        self.name = name

    async def act(self, request: AgentRequest) -> AgentResponse:
        obs = request.observation
        legal = obs.legal_actions

        print()
        print(f"--- Your turn ({request.player}) ---")
        if request.match is not None:
            print(decorate(request.match.describe()))
        # The game's own rendered view is complete and game-agnostic (hole cards,
        # board, pot, to-call, legal actions, and any amount ranges). Wrap it in
        # separator lines and highlight cards (suit-colored) and numbers.
        print(rule())
        print(decorate(obs.rendered))
        print(rule())

        # Amount range for bet/raise, if the game provided it in the observation.
        amt_range = obs.public.get("amount_range") or {}

        # Numbered menu so the human can just type a number.
        print("Choose an action:")
        for i, a in enumerate(legal, start=1):
            print(f"  [{i}] {a}")

        while True:
            raw = await asyncio.to_thread(
                input, "Your choice (number or name, e.g. `3` or `bet 5`): ")
            parts = raw.strip().lower().split()
            token = parts[0] if parts else ""
            inline_amount = parts[1] if len(parts) > 1 else None

            choice = None
            if token.isdigit() and 1 <= int(token) <= len(legal):
                choice = legal[int(token) - 1]
            elif token in legal:
                choice = token
            else:
                # Accept an unambiguous prefix (e.g. "c" for check, "b" for bet).
                matches = [a for a in legal if a.startswith(token)] if token else []
                if len(matches) == 1:
                    choice = matches[0]
            if choice is None:
                print(f"  Invalid input {raw.strip()!r}. "
                      f"Enter 1-{len(legal)} or one of: {', '.join(legal)}")
                continue

            amount = None
            if choice in ("bet", "raise"):
                lo = amt_range.get(choice, {}).get("min")
                hi = amt_range.get(choice, {}).get("max")
                hint = f" [{lo}-{hi}]" if lo is not None else ""
                amt_str = inline_amount  # amount typed on the same line, if any
                if amt_str is None:
                    amt_str = (await asyncio.to_thread(
                        input, f"  {choice} to (total street commitment){hint}: ")).strip()
                try:
                    amount = int(amt_str)
                except (ValueError, TypeError):
                    print(f"  Invalid amount {amt_str!r}; enter an integer.")
                    continue
                # Validate against the legal range so a typo doesn't get submitted
                # as an invalid action (which would silently fall back to fold).
                if lo is not None and not (lo <= amount <= hi):
                    print(f"  Amount {amount} is out of range [{lo}-{hi}]. "
                          f"(Tip: type `all_in` to shove, or pick another action.)")
                    continue

            return AgentResponse(action=choice, amount=amount, message="human",
                                 metadata={"human": True})
