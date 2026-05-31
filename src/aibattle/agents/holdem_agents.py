"""Built-in Hold'em baselines: random and a simple hand-strength heuristic.

Both produce valid bet/raise amounts (the generic RandomAgent cannot, since it
has no notion of an amount range). Stochastic choices use the request's
deterministic per-decision seed so they are reproducible under parallelism.
"""

from __future__ import annotations

import random

from ..types import AgentRequest, AgentResponse
from .base import Agent

_RANK = {r: i for i, r in enumerate("23456789TJQKA", start=2)}


def _rng_for(seed_base: int, request: AgentRequest) -> random.Random:
    ds = request.decision_seed
    if ds is not None:
        return random.Random((seed_base * 2654435761 ^ ds) & 0x7FFFFFFF)
    return random.Random(seed_base)


def _pick_amount(rng, rng_key, amount_range, atype):
    r = amount_range.get(atype)
    if not r:
        return None
    lo, hi = r["min"], r["max"]
    mid = min(hi, lo + max(1, (hi - lo) // 3))
    return rng.choice([lo, lo, mid, hi])  # bias toward smaller sizings


class RandomHoldemAgent(Agent):
    agent_type = "builtin"

    def __init__(self, name: str = "holdem_random", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    async def act(self, request: AgentRequest) -> AgentResponse:
        rng = _rng_for(self._seed, request)
        legal = request.observation.legal_actions
        amount_range = request.observation.public.get("amount_range", {})
        atype = rng.choice(legal)
        amount = _pick_amount(rng, None, amount_range, atype) if atype in ("bet", "raise") else None
        return AgentResponse(action=atype, amount=amount, message="random",
                             metadata={"policy": "holdem_random"})


class HoldemHeuristicAgent(Agent):
    """Crude hand-strength policy: raise strong hands, call/check medium, fold weak.

    Strength is judged from hole cards plus any made pair with the board — good
    enough to beat random and serve as a calibration baseline (not GTO).
    """

    agent_type = "builtin"

    def __init__(self, name: str = "holdem_heuristic", seed: int | None = None):
        self.name = name
        self._seed = seed or 0

    def _strength(self, hole, board) -> str:
        r0, r1 = _RANK[hole[0][0]], _RANK[hole[1][0]]
        hi, lo = max(r0, r1), min(r0, r1)
        board_ranks = {_RANK[c[0]] for c in board}
        pair_hole = r0 == r1
        pairs_board = bool({r0, r1} & board_ranks)

        if not board:  # preflop
            if pair_hole and hi >= 10:      # TT+
                return "strong"
            if pair_hole or hi >= 13 or (hi >= 12 and lo >= 11):  # any pair, K/A high, QJ
                return "medium"
            if hi >= 12 or lo >= 10:
                return "medium" if hi == 14 else "weak"
            return "weak"
        # postflop
        if pair_hole or pairs_board:
            return "strong" if (pair_hole and hi >= 11) or hi == 14 else "medium"
        if hi >= 13:
            return "medium"
        return "weak"

    async def act(self, request: AgentRequest) -> AgentResponse:
        rng = _rng_for(self._seed, request)
        obs = request.observation
        legal = obs.legal_actions
        amount_range = obs.public.get("amount_range", {})
        hole = obs.private["hole"]
        board = obs.public.get("board", [])
        to_call = obs.public.get("to_call", 0)
        strength = self._strength(hole, board)

        def amt(atype):
            r = amount_range.get(atype)
            if not r:
                return None
            lo, hi = r["min"], r["max"]
            # value-bet around min..~half the range
            return min(hi, lo + (hi - lo) // 4)

        if to_call > 0:  # facing a bet
            if strength == "strong" and "raise" in legal:
                return AgentResponse(action="raise", amount=amt("raise"),
                                     message="heuristic:value-raise",
                                     metadata={"policy": "holdem_heuristic", "strength": strength})
            if strength in ("strong", "medium") and "call" in legal:
                return AgentResponse(action="call", message="heuristic:call",
                                     metadata={"policy": "holdem_heuristic", "strength": strength})
            # weak: occasionally bluff-call small, else fold
            if "fold" in legal:
                return AgentResponse(action="fold", message="heuristic:fold",
                                     metadata={"policy": "holdem_heuristic", "strength": strength})
            action = legal[0]
            return AgentResponse(action=action, message="heuristic:forced",
                                 metadata={"policy": "holdem_heuristic", "strength": strength})

        # no bet facing: bet strong, otherwise check
        if strength == "strong" and "bet" in legal:
            return AgentResponse(action="bet", amount=amt("bet"),
                                 message="heuristic:value-bet",
                                 metadata={"policy": "holdem_heuristic", "strength": strength})
        if "check" in legal:
            return AgentResponse(action="check", message="heuristic:check",
                                 metadata={"policy": "holdem_heuristic", "strength": strength})
        return AgentResponse(action=legal[0], message="heuristic:forced",
                             metadata={"policy": "holdem_heuristic", "strength": strength})
