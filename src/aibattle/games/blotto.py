"""Repeated Colonel Blotto — simultaneous resource allocation, 2 players.

Each round both players secretly allocate ``RESOURCES`` units across
``len(VALUES)`` battlefields; the higher allocation on a battlefield wins that
battlefield's value (ties score nothing in v0). Scores accumulate across
``ROUNDS`` rounds; the higher cumulative score wins.

The runner is strictly sequential, so the simultaneous move is emulated with a
hidden pending allocation: within a round ``player_0`` submits first and its
allocation is stored privately (invisible to ``player_1``); ``player_1`` then
submits and the round resolves. Only RESOLVED prior-round allocations are ever
exposed to the opponent, so no agent sees the other's current-round allocation
before it commits.

An allocation is encoded inside ``Move.type`` as the string
``"alloc:a,b,c,d,e"`` (``amount`` stays ``None``) — this keeps the shared
``Move`` dataclass unchanged. ``validate_action`` parses and accepts any
well-formed allocation (non-negative integers summing to ``RESOURCES``);
``legal_actions`` advertises a single valid default allocation so any agent that
picks from it produces a legal move.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game

_PLAYERS = ["player_0", "player_1"]
ROUNDS = 20
RESOURCES = 100
VALUES = [1, 2, 3, 4, 5]
N_FIELDS = len(VALUES)
_DEFAULT_ALLOC = [20, 20, 20, 20, 20]
_PREFIX = "alloc:"


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


def encode_alloc(alloc) -> str:
    return _PREFIX + ",".join(str(int(x)) for x in alloc)


def parse_alloc(text: str):
    """Parse an ``alloc:a,b,c,d,e`` token into a list of ints, or None if it is
    not a well-formed allocation (wrong count / non-integer / negative / wrong
    sum)."""
    if not isinstance(text, str):
        return None
    t = text.strip().lower()
    if not t.startswith(_PREFIX):
        return None
    body = t[len(_PREFIX):]
    parts = body.split(",")
    if len(parts) != N_FIELDS:
        return None
    out = []
    for x in parts:
        x = x.strip()
        if not x.lstrip("-").isdigit():
            return None
        v = int(x)
        if v < 0:
            return None
        out.append(v)
    if sum(out) != RESOURCES:
        return None
    return out


def _battlefield_outcomes(a0, a1) -> list:
    """Per-battlefield outcome records for one round's allocations: each is
    {battlefield, value, alloc_0, alloc_1, winner} where winner is "player_0",
    "player_1", or "tie"."""
    out = []
    for i, val in enumerate(VALUES):
        if a0[i] > a1[i]:
            winner = "player_0"
        elif a1[i] > a0[i]:
            winner = "player_1"
        else:
            winner = "tie"
        out.append({"battlefield": i, "value": val,
                    "alloc_0": a0[i], "alloc_1": a1[i], "winner": winner})
    return out


def _score_round(a0, a1) -> tuple:
    """Return (p0_points, p1_points) for one round's allocations (ties score 0)."""
    s0 = s1 = 0
    for rec in _battlefield_outcomes(a0, a1):
        if rec["winner"] == "player_0":
            s0 += rec["value"]
        elif rec["winner"] == "player_1":
            s1 += rec["value"]
    return s0, s1


@dataclass(frozen=True)
class BlottoState:
    round: int                       # 0-based current round index
    scores: dict                     # cumulative {player: int}
    pending: Optional[tuple]         # player_0's submitted alloc this round (hidden)
    # Resolved rounds: tuple of dicts {round, alloc_0, alloc_1, points_0, points_1}.
    history: tuple = field(default_factory=tuple)
    done: bool = False


class RepeatedColonelBlotto(Game):
    name = "repeated_colonel_blotto"
    version = "1.0.0"
    players = list(_PLAYERS)

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> BlottoState:
        return BlottoState(round=0, scores={"player_0": 0, "player_1": 0},
                           pending=None, history=(), done=False)

    # -- turn logic ---------------------------------------------------------
    def current_player(self, s: BlottoState) -> PlayerId:
        # player_0 allocates first each round; once it has, player_1 acts.
        return "player_0" if s.pending is None else "player_1"

    def legal_actions(self, s: BlottoState, player: PlayerId) -> list:
        # Descriptive but VALID: every entry is itself a legal move, so any agent
        # that picks from legal_actions submits a well-formed allocation.
        return [encode_alloc(_DEFAULT_ALLOC)]

    def validate_action(self, s: BlottoState, player: PlayerId, move: Move):
        if move.amount is not None:
            return False, "unexpected_amount"
        alloc = parse_alloc(move.type)
        if alloc is None:
            return False, "invalid_allocation"
        return True, None

    def fallback_action(self, s: BlottoState, player: PlayerId, legal: list) -> Move:
        return Move(type=encode_alloc(_DEFAULT_ALLOC))

    def is_terminal(self, s: BlottoState) -> bool:
        return s.done

    def step(self, s: BlottoState, move: Move) -> BlottoState:
        assert not s.done
        alloc = parse_alloc(move.type)
        assert alloc is not None, f"illegal allocation {move.type!r}"
        if s.pending is None:
            # player_0 submits: store privately, hand to player_1.
            return BlottoState(round=s.round, scores=dict(s.scores),
                               pending=tuple(alloc), history=s.history,
                               done=False)
        # player_1 submits: resolve the round.
        a0 = list(s.pending)
        outcomes = _battlefield_outcomes(a0, alloc)
        p0 = sum(o["value"] for o in outcomes if o["winner"] == "player_0")
        p1 = sum(o["value"] for o in outcomes if o["winner"] == "player_1")
        scores = dict(s.scores)
        scores["player_0"] += p0
        scores["player_1"] += p1
        record = {
            "round": s.round + 1,
            "alloc_0": a0,
            "alloc_1": alloc,
            "battlefields": outcomes,        # per-battlefield outcome detail
            "points_0": p0,
            "points_1": p1,
            "cumulative": dict(scores),      # cumulative scores AFTER this round
        }
        history = s.history + (record,)
        nxt = s.round + 1
        done = nxt >= ROUNDS
        return BlottoState(round=nxt, scores=scores, pending=None,
                           history=history, done=done)

    # -- payoffs ------------------------------------------------------------
    def returns(self, s: BlottoState) -> dict:
        assert s.done, "returns() called on non-terminal state"
        s0, s1 = s.scores["player_0"], s.scores["player_1"]
        if s0 == s1:
            return {"player_0": 0.0, "player_1": 0.0}
        winner = "player_0" if s0 > s1 else "player_1"
        return {winner: 1.0, _other(winner): -1.0}

    def episode_metadata(self, s: BlottoState) -> dict:
        s0, s1 = s.scores["player_0"], s.scores["player_1"]
        reason = "draw" if s0 == s1 else "score"
        return {
            "reason": reason,
            "final_scores": dict(s.scores),
            "rounds_played": len(s.history),
            "battlefield_values": list(VALUES),
            # The full resolved history (all rounds, incl. the final one) is
            # stamped here because the runner records only pre-action
            # observations, so the last round's resolution would otherwise not
            # appear in any step. This carries every AC-6 field per round.
            "round_history": [dict(rec) for rec in s.history],
        }

    # -- observation / render ----------------------------------------------
    def _public_history(self, s: BlottoState) -> list:
        """Resolved prior-round allocations, visible to BOTH players. The current
        unresolved round's pending allocation is never included."""
        return [dict(rec) for rec in s.history]

    def observation(self, s: BlottoState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        # Crucially, player_1's observation must NOT contain player_0's pending
        # allocation for the current round. We expose only resolved history and
        # cumulative scores to both players; ``pending`` is never surfaced.
        public = {
            "round": s.round + 1,
            "total_rounds": ROUNDS,
            "resources": RESOURCES,
            "battlefield_values": list(VALUES),
            "scores": dict(s.scores),
        }
        return Observation(
            player=player,
            private={},   # no per-player private state is revealed pre-resolution
            public=public,
            history=self._public_history(s),
            legal_actions=legal,
            rendered=self._render_for(s, player),
        )

    def _render_for(self, s: BlottoState, player: PlayerId) -> str:
        you = s.scores[player]
        opp = s.scores[_other(player)]
        lines = [
            f"You are {player} in Repeated Colonel Blotto, round {s.round + 1} "
            f"of {ROUNDS}.",
            f"Battlefield values: {VALUES} (5 battlefields). Allocate exactly "
            f"{RESOURCES} units across them as non-negative integers.",
            f"Cumulative score — you: {you}, opponent: {opp}.",
        ]
        if s.history:
            last = s.history[-1]
            lines.append(
                f"Last resolved round {last['round']}: you allocated "
                f"{last['alloc_0'] if player == 'player_0' else last['alloc_1']}, "
                f"opponent allocated "
                f"{last['alloc_1'] if player == 'player_0' else last['alloc_0']}."
            )
        lines.append(
            f"Submit your allocation as alloc:a,b,c,d,e (five non-negative "
            f"integers summing to {RESOURCES})."
        )
        return "\n".join(lines)

    def render(self, s: BlottoState, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None:
            return self.observation(s, perspective).rendered
        tag = ""
        if s.done:
            r = self.returns(s)
            w = max(r, key=r.get) if len(set(r.values())) > 1 else None
            tag = f"  [winner: {w}]" if w else "  [draw]"
        return (f"Blotto[round {s.round}/{ROUNDS}; scores "
                f"{s.scores['player_0']}-{s.scores['player_1']}]{tag}")
