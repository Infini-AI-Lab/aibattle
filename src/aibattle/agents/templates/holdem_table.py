"""Prompt template for Multi-Agent Hold'em Table Mode.

Reuses the Hold'em action parser (identical grammar: fold/check/call/bet N/
raise N/all_in). The full multi-player table context — seats, stacks, statuses,
side pots, your current rank — is injected by the game's observation, so it
appears in the rendered prompt automatically.
"""

from __future__ import annotations

from ...types import AgentRequest
from .holdem import HoldemTemplate

_TABLE_RULES = (
    "You are playing Multi-Agent Texas Hold'em at a TABLE with several players. "
    "Each player starts with the same stack; small blind 1, big blind 2; the "
    "button rotates each hand. Stacks carry across hands. A player with zero "
    "chips is eliminated. The table plays a fixed number of hands; your goal is "
    "to finish as high in the chip ranking as possible (ideally the chip leader).\n"
    "Actions:\n"
    "  fold  - give up the hand\n"
    "  check - pass (only when there is no bet to call)\n"
    "  call  - match the current bet\n"
    "  bet N - bet when no bet faces you; N = your TOTAL chips committed this street\n"
    "  raise N - raise a facing bet; N = your TOTAL chips committed this street\n"
    "  all_in - commit your entire remaining stack\n"
    "N must be an integer within the stated legal range. With multiple players "
    "and side pots, you can only win chips from opponents up to what you put in."
)


class HoldemTableTemplate(HoldemTemplate):
    def render_prompt(self, request: AgentRequest) -> str:
        obs = request.observation
        legal = ", ".join(obs.legal_actions)
        # No MatchContext "Hand X of N" line here: in Table mode one episode wraps
        # the whole session, so MatchContext.episode is the SESSION index (always
        # "1 of 1" with SESSIONS=1) and would contradict the true within-session
        # hand counter that the engine already emits in obs.rendered.
        return (
            f"{_TABLE_RULES}\n\n"
            f"{obs.rendered}\n\n"
            f"{self._history_block(obs)}"
            f"Choose exactly one legal action: {legal}.\n"
            "Respond with ONLY the action (and an integer amount for bet/raise), "
            "e.g. `fold`, `check`, `call`, `all_in`, `bet 6`, or `raise 12`. "
            "Put the action on the last line if you reason first."
        )
