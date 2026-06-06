"""Heads-Up Texas Hold'em Lite prompt template + tolerant action/amount parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from .base import GameTemplate

_RULES = (
    "You are playing Heads-Up Texas Hold'em Lite (no-limit style, one hand). "
    "Each player starts with 200 chips; small blind 1, big blind 2. You are dealt "
    "two private hole cards; five community cards are revealed across preflop, "
    "flop, turn, and river. Make the best five-card hand at showdown.\n"
    "Actions:\n"
    "  fold  - give up the hand\n"
    "  check - pass (only when there is no bet to call)\n"
    "  call  - match the current bet\n"
    "  bet N - bet when no bet is facing you; N = your TOTAL chips committed this street\n"
    "  raise N - raise a facing bet; N = your TOTAL chips committed this street (raise-TO)\n"
    "  all_in - commit your entire remaining stack\n"
    "N must be an integer within the stated legal range."
)

# action type -> regex; longer/odd spellings handled (all-in/allin/shove).
_ALIASES = {
    "all_in": r"\ball[\s_-]?in\b|\bshove\b|\bjam\b",
    "raise": r"\braise\b",
    "bet": r"\bbet\b",
    "call": r"\bcall\b",
    "check": r"\bcheck\b",
    "fold": r"\bfold\b",
}


class HoldemTemplate(GameTemplate):
    # Lite mode shows the cross-hand "Match: Hand X of N" context; Match/Table
    # modes suppress it (their own engine emits a within-session hand counter).
    _show_match_ctx = True

    @staticmethod
    def _history_block(obs) -> str:
        """Render this hand's public action log (per street) so the model sees
        the betting LINE, not just the numeric snapshot. The data is already on
        obs.history; the numeric state alone loses who raised/called and the
        sizing sequence. Returns "" when there's been no action yet.
        """
        history = getattr(obs, "history", None) or []
        if not history:
            return ""
        streets = [["preflop", []]]  # [label, [action strings]]
        for ev in history:
            if "street" in ev:  # street-transition marker {street, board}
                board = " ".join(ev.get("board", []))
                streets.append([ev["street"] + (f" ({board})" if board else ""), []])
            else:
                p, a = ev.get("player"), ev.get("action")
                # show sizing for aggressive actions; plain for fold/check/call
                entry = (f"{p} {a} to {ev['to']}"
                         if a in ("bet", "raise", "all_in") and "to" in ev
                         else f"{p} {a}")
                streets[-1][1].append(entry)
        body = [f"  {label}: {', '.join(acts)}" for label, acts in streets if acts]
        if not body:
            return ""
        return "Action this hand:\n" + "\n".join(body)

    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def state(self, request: AgentRequest) -> str:
        obs = request.observation
        ctx = (f"Match: {request.match.describe()}\n"
               if self._show_match_ctx and request.match else "")
        hist = self._history_block(obs)
        body = f"{ctx}{obs.rendered}"
        return f"{body}\n\n{hist}" if hist else body

    def instruction(self, request: AgentRequest) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"Choose exactly one legal action: {legal}.\n"
            "Respond with ONLY the action (and an integer amount for bet/raise), "
            "e.g. `call`, `check`, `fold`, `all_in`, `bet 6`, or `raise 12`. "
            "Put the action on the last line if you reason first."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        legal = request.observation.legal_actions
        text = raw.lower()
        # Prefer the last non-empty line (the conclusion) if it contains an action.
        lines = [ln for ln in text.splitlines() if ln.strip()]
        for candidate in ([lines[-1]] if lines else []) + [text]:
            for atype in ("all_in", "raise", "bet", "call", "check", "fold"):
                if atype not in legal:
                    continue
                am = re.search(_ALIASES[atype], candidate)
                if am:
                    if atype in ("bet", "raise"):
                        # Take the first number AFTER the action token ("raise 12"),
                        # so a number earlier in the line ("facing a bet of 6, I
                        # raise 12") can't be misread as the amount. Only if nothing
                        # follows the token do we fall back to the rest of the line.
                        m = (re.search(r"(\d+)", candidate[am.end():])
                             or re.search(r"(\d+)", candidate))
                        if not m:
                            return None  # amount required but absent
                        return Move(type=atype, amount=int(m.group(1)))
                    return Move(type=atype)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        legal = ", ".join(request.observation.legal_actions)
        return (
            f"Your previous reply did not contain a valid action. "
            f"Reply with exactly one of: {legal} (with an integer amount for bet/raise)."
        )
