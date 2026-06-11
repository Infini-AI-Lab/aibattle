"""Repeated Colonel Blotto prompt template + allocation parser."""

from __future__ import annotations

import re
from typing import Optional

from ...types import AgentRequest, Move
from ...games.blotto import RESOURCES, N_FIELDS, VALUES, encode_alloc, parse_alloc
from .base import GameTemplate


_RULES = (
    "You are playing Repeated Colonel Blotto (2 players, 20 rounds). Each round "
    f"you secretly allocate exactly {RESOURCES} units across {N_FIELDS} "
    f"battlefields worth {VALUES} points. On each battlefield the higher "
    "allocation wins that battlefield's points; ties score nothing. Points "
    "accumulate across rounds, and the higher total after all rounds wins. You "
    "can see the resolved allocations from previous rounds (yours and the "
    "opponent's) but not the opponent's allocation for the current round."
)


class BlottoTemplate(GameTemplate):
    def rules(self, request: AgentRequest) -> str:
        return _RULES

    def instruction(self, request: AgentRequest) -> str:
        return (
            f"Respond with your allocation as 'alloc:a,b,c,d,e' — {N_FIELDS} "
            f"non-negative integers summing to exactly {RESOURCES}, in "
            "battlefield order. Put it on the last line if you reason first. "
            "Example: alloc:20,20,20,20,20."
        )

    def parse(self, raw: str, request: AgentRequest) -> Optional[Move]:
        if not raw:
            return None
        lines = [ln for ln in raw.splitlines() if ln.strip()]
        # 1) Prefer an explicit alloc: token (last occurrence wins).
        candidates = re.findall(r"alloc:\s*[-\d,\s]+", raw, flags=re.IGNORECASE)
        for cand in reversed(candidates):
            compact = "alloc:" + re.sub(r"\s+", "", cand.split(":", 1)[1])
            if parse_alloc(compact) is not None:
                return Move(type=compact)
        # 2) Fallback: read N integers from the last non-empty line.
        for chunk in ([lines[-1]] if lines else []) + [raw]:
            nums = re.findall(r"-?\d+", chunk)
            if len(nums) >= N_FIELDS:
                alloc = [int(x) for x in nums[:N_FIELDS]]
                encoded = encode_alloc(alloc)
                if parse_alloc(encoded) is not None:
                    return Move(type=encoded)
        return None

    def repair_hint(self, request: AgentRequest, bad_output: str) -> str:
        return (
            "Your previous reply was not a valid allocation. Reply with exactly "
            f"'alloc:a,b,c,d,e' — {N_FIELDS} non-negative integers summing to "
            f"{RESOURCES} (e.g. alloc:20,20,20,20,20)."
        )
