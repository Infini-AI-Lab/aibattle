"""Leduc Poker — imperfect-information 2-player poker, between Kuhn and Hold'em.

Deck: J J Q Q K K (ranks J < Q < K). Each player antes 1 and receives one
private card. A first betting round is followed by one revealed public card, then
a second betting round, then a showdown if nobody folded.

Betting (fixed-limit, raise-to semantics matching Hold'em's ``amount``):
- ``amount`` on a bet/raise is the player's TOTAL committed chips for the current
  street (not the increment).
- The bet size is fixed per street: 2 in round 1 (pre-public), 4 in round 2.
- At most one raise per betting round (v0), so the action ladder per street is
  check/bet then call/raise then call/fold.

Showdown:
- A private card that pairs the public card beats any non-pair hand.
- Otherwise the higher private card wins.
- Equal strength splits the pot.

Payoffs are zero-sum chips computed from each player's contribution; on a split
of an odd pot the single odd chip is awarded deterministically to player_0.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game

_RANK = {"J": 0, "Q": 1, "K": 2}
_PLAYERS = ["player_0", "player_1"]
_ANTE = 1
_BET_SIZE = {0: 2, 1: 4}   # round index -> fixed bet/raise increment


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


@dataclass(frozen=True)
class LeducState:
    cards: dict                 # {"player_0": "K", "player_1": "Q"}
    public: Optional[str]       # the revealed public card, or None pre-reveal
    round: int                  # 0 = first betting round, 1 = second
    # Chips committed THIS round, keyed by player (resets to 0 at round 2 start).
    street_commit: dict
    # Chips committed in completed prior rounds (the settled pot contributions).
    locked: dict
    to_act: PlayerId
    raises_this_round: int      # number of raises so far this round (cap 1)
    # Players who have acted since the last aggressive action this round.
    acted: tuple = field(default_factory=tuple)
    folded: Optional[PlayerId] = None
    done: bool = False


class LeducPoker(Game):
    name = "leduc_poker"
    version = "1.0.0"
    players = list(_PLAYERS)

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> LeducState:
        deck = ["J", "J", "Q", "Q", "K", "K"]
        rng.shuffle(deck)
        cards = {"player_0": deck[0], "player_1": deck[1]}
        # deck[2] is reserved as the public card revealed after round 1.
        return LeducState(
            cards=cards,
            public=None,
            round=0,
            street_commit={"player_0": 0, "player_1": 0},
            locked={"player_0": _ANTE, "player_1": _ANTE},
            to_act="player_0",
            raises_this_round=0,
            acted=(),
            folded=None,
            done=False,
        )

    # -- turn logic ---------------------------------------------------------
    def current_player(self, s: LeducState) -> PlayerId:
        return s.to_act

    def _max_commit(self, s: LeducState) -> int:
        return max(s.street_commit.values())

    def legal_actions(self, s: LeducState, player: PlayerId) -> list:
        sc = s.street_commit[player]
        cur_max = self._max_commit(s)
        if sc == cur_max:
            # No outstanding bet to call: may check or open the betting with a bet.
            return ["check", "bet"]
        # Facing a bet: may fold or call, and raise at most once per round.
        acts = ["fold", "call"]
        if s.raises_this_round < 1:
            acts.append("raise")
        return acts

    def validate_action(self, s: LeducState, player: PlayerId, move: Move):
        legal = self.legal_actions(s, player)
        if move.type not in legal:
            return False, "illegal_action_type"
        if move.type in ("fold", "check", "call"):
            if move.amount is not None:
                return False, "unexpected_amount"
            return True, None
        # bet / raise: amount is the TOTAL street commitment (raise-to).
        if move.amount is None:
            return False, "missing_amount"
        if not isinstance(move.amount, int) or isinstance(move.amount, bool):
            return False, "non_integer_amount"
        sc = s.street_commit[player]
        cur_max = self._max_commit(s)
        size = _BET_SIZE[s.round]
        expected = (sc + size) if move.type == "bet" else (cur_max + size)
        if move.amount != expected:
            return False, "wrong_amount"
        return True, None

    def fallback_action(self, s: LeducState, player: PlayerId, legal: list) -> Move:
        for t in ("check", "fold", "call"):
            if t in legal:
                return Move(type=t)
        return Move(type=legal[0]) if legal else Move(type="__invalid__")

    def is_terminal(self, s: LeducState) -> bool:
        return s.done

    # -- transition ---------------------------------------------------------
    def _public_card_for(self, s: LeducState) -> str:
        """The public card is the lowest-ranked remaining card not held by a
        player, chosen deterministically so reveal is reproducible."""
        used = [s.cards["player_0"], s.cards["player_1"]]
        pool = ["J", "J", "Q", "Q", "K", "K"]
        for c in used:
            pool.remove(c)
        # Deterministic pick: first remaining in canonical order.
        return pool[0]

    def _round_closed(self, sc: dict, acted: tuple) -> bool:
        """A betting round closes when commits are equal and both have acted."""
        return sc["player_0"] == sc["player_1"] and set(acted) == set(_PLAYERS)

    def step(self, s: LeducState, move: Move) -> LeducState:
        assert not s.done
        p = s.to_act
        opp = _other(p)
        sc = dict(s.street_commit)
        locked = dict(s.locked)
        raises = s.raises_this_round
        acted = set(s.acted)

        if move.type == "fold":
            return LeducState(
                cards=s.cards, public=s.public, round=s.round,
                street_commit=sc, locked=locked, to_act=opp,
                raises_this_round=raises, acted=tuple(sorted(acted | {p})),
                folded=p, done=True,
            )

        if move.type == "check":
            acted.add(p)
        elif move.type == "call":
            sc[p] = self._max_commit(s)   # match the outstanding bet
            acted.add(p)
        elif move.type == "bet":
            sc[p] = move.amount
            acted = {p}                   # an aggressive action re-opens action
        elif move.type == "raise":
            sc[p] = move.amount
            raises += 1                   # only a raise counts against the cap
            acted = {p}
        else:
            raise AssertionError(f"unexpected action {move.type!r}")

        # Does this action close the betting round?
        closed = self._round_closed(sc, tuple(acted))
        if not closed:
            return LeducState(
                cards=s.cards, public=s.public, round=s.round,
                street_commit=sc, locked=locked, to_act=opp,
                raises_this_round=raises, acted=tuple(sorted(acted)),
                folded=None, done=False,
            )

        # Round closed: fold contributions into the locked pot.
        for q in _PLAYERS:
            locked[q] += sc[q]
        if s.round == 0:
            # Reveal the public card and start round 2 (player_0 acts first).
            return LeducState(
                cards=s.cards, public=self._public_card_for(s), round=1,
                street_commit={"player_0": 0, "player_1": 0}, locked=locked,
                to_act="player_0", raises_this_round=0, acted=(),
                folded=None, done=False,
            )
        # Second round closed -> showdown.
        return LeducState(
            cards=s.cards, public=s.public, round=1,
            street_commit={"player_0": 0, "player_1": 0}, locked=locked,
            to_act="player_0", raises_this_round=0, acted=(),
            folded=None, done=True,
        )

    # -- showdown / payoffs -------------------------------------------------
    def _hand_strength(self, card: str, public: Optional[str]) -> tuple:
        """A comparable strength: (pairs_public, card_rank). Pairing the public
        card dominates any non-pair; otherwise the higher private card wins."""
        pairs = 1 if (public is not None and card == public) else 0
        return (pairs, _RANK[card])

    def returns(self, s: LeducState) -> dict:
        assert s.done, "returns() called on non-terminal state"
        contrib = dict(s.locked)   # total chips each player committed to the pot
        pot = contrib["player_0"] + contrib["player_1"]

        if s.folded is not None:
            winner = _other(s.folded)
            # Winner nets the loser's contribution; loser loses their own.
            net = contrib[s.folded]
            return {winner: float(net), s.folded: float(-net)}

        # Showdown.
        st0 = self._hand_strength(s.cards["player_0"], s.public)
        st1 = self._hand_strength(s.cards["player_1"], s.public)
        if st0 > st1:
            winner = "player_0"
        elif st1 > st0:
            winner = "player_1"
        else:
            # Split: each gets back their own contribution (net 0); an odd chip
            # (only if contributions differ in parity) goes to player_0.
            half = pot // 2
            odd = pot - 2 * half
            ret0 = half + odd - contrib["player_0"]
            ret1 = (pot - half - odd) - contrib["player_1"]
            return {"player_0": float(ret0), "player_1": float(ret1)}
        loser = _other(winner)
        net = contrib[loser]
        return {winner: float(net), loser: float(-net)}

    def episode_metadata(self, s: LeducState) -> dict:
        reason = "fold" if s.folded is not None else "showdown"
        return {
            "reason": reason,
            "public_card": s.public,
            "cards": dict(s.cards),
            "pot": s.locked["player_0"] + s.locked["player_1"],
        }

    # -- observation / render ----------------------------------------------
    def _pot(self, s: LeducState) -> int:
        return (s.locked["player_0"] + s.locked["player_1"]
                + s.street_commit["player_0"] + s.street_commit["player_1"])

    def observation(self, s: LeducState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        to_call = self._max_commit(s) - s.street_commit[player]
        return Observation(
            player=player,
            private={"card": s.cards[player]},
            public={
                "public_card": s.public,
                "round": s.round + 1,
                "pot": self._pot(s),
                "to_call": to_call,
                "bet_size": _BET_SIZE[s.round],
                "your_commit": s.street_commit[player],
            },
            history=[],
            legal_actions=legal,
            rendered=self._render_for(s, player, legal, to_call),
        )

    def _render_for(self, s, player, legal, to_call) -> str:
        pub = s.public if s.public is not None else "(not revealed)"
        return (
            f"You are {player} in Leduc Poker. Your private card is "
            f"{s.cards[player]} (ranks J<Q<K). Public card: {pub}. "
            f"Betting round {s.round + 1} of 2. Pot: {self._pot(s)} chips. "
            f"To call: {to_call}. Fixed bet size this round: {_BET_SIZE[s.round]}. "
            f"Legal actions: {', '.join(legal)}."
        )

    def render(self, s: LeducState, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None:
            return self.observation(s, perspective).rendered
        pub = s.public if s.public is not None else "-"
        cards = ", ".join(f"{p}={c}" for p, c in s.cards.items())
        tag = ""
        if s.done:
            r = self.returns(s)
            w = max(r, key=r.get) if len(set(r.values())) > 1 else None
            tag = f"  [winner: {w}]" if w else "  [split]"
        return f"LeducPoker[cards: {cards}; public: {pub}; pot: {self._pot(s)}]{tag}"
