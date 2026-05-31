"""Heads-Up Texas Hold'em Lite.

Two players, one hand per episode, stacks reset each episode. Standard streets
(preflop/flop/turn/river), no-limit-style agent-chosen bet/raise amounts,
all-in, and showdown via the internal 7-card evaluator. Amounts are the
player's TOTAL committed chips for the current street (raise-to semantics).

State is treated as immutable: ``step`` clones and returns a new state.

The hand engine is parameterized by starting stack and blinds and exposes
``deal_hand(rng, stacks, button)`` so it can be reused as the per-hand sub-step
of a multi-hand match (see ``holdem_match``): each hand is dealt from the
players' carried-over stacks, and pot/deltas are computed relative to the
per-hand ``start_stacks`` rather than a fixed constant.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game
from .poker_eval import category_name, evaluate7, full_deck

_PLAYERS = ["player_0", "player_1"]
STARTING_STACK = 50
SMALL_BLIND = 1
BIG_BLIND = 2
_STREETS = ["preflop", "flop", "turn", "river"]
_FALLBACK = ["check", "fold", "call"]  # for reference; runner owns fallback


@dataclass
class HoldemState:
    button: PlayerId                  # SB/button this hand
    deck: list                        # remaining undealt cards (hidden)
    hole: dict                        # {player: (c, c)} (hidden per player)
    board: list                       # revealed community cards
    street: str                       # preflop|flop|turn|river|done
    stacks: dict                      # chips behind
    street_commit: dict               # chips committed THIS street
    to_act: Optional[PlayerId]
    last_raise_size: int              # increment of the last bet/raise
    aggressor: Optional[PlayerId]
    acted_since: set                  # players who acted since last aggression
    all_in: dict
    folded: dict
    history: list                     # public action log
    result: Optional[dict] = None     # set at terminal
    start_stacks: Optional[dict] = None  # per-player chips at the start of THIS hand


def _other(p: PlayerId) -> PlayerId:
    return _PLAYERS[1 - _PLAYERS.index(p)]


class HoldemPoker(Game):
    name = "holdem"
    version = "1.0.0"
    players = list(_PLAYERS)

    def __init__(self, starting_stack: int = STARTING_STACK,
                 small_blind: int = SMALL_BLIND, big_blind: int = BIG_BLIND):
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> HoldemState:
        button = _PLAYERS[rng.randrange(2)]   # balanced across deals
        stacks = {p: self.starting_stack for p in _PLAYERS}
        return self.deal_hand(rng, stacks, button)

    def deal_hand(self, rng: random.Random, stacks: dict,
                  button: PlayerId) -> HoldemState:
        """Deal a single hand from given per-player ``stacks`` and ``button``.

        Precondition: both players can cover the big blind (callers that carry
        stacks across hands, e.g. a match, must guarantee this so blind posting
        never forces an all-in).
        """
        deck = full_deck()
        rng.shuffle(deck)
        hole = {"player_0": (deck[0], deck[1]), "player_1": (deck[2], deck[3])}
        deck = deck[4:]

        sb, bb = button, _other(button)
        stacks = dict(stacks)
        start_stacks = dict(stacks)
        street_commit = {sb: self.small_blind, bb: self.big_blind}
        stacks[sb] -= self.small_blind
        stacks[bb] -= self.big_blind
        # Posting a blind that consumes the whole stack IS an all-in (can happen
        # in a match when a player carries in exactly the big blind). Mark it so
        # the engine doesn't try to make a 0-stack player act.
        all_in = {p: stacks[p] == 0 for p in _PLAYERS}

        return HoldemState(
            button=button, deck=deck, hole=hole, board=[],
            street="preflop", stacks=stacks, street_commit=street_commit,
            to_act=sb,  # preflop the button/SB acts first
            last_raise_size=self.big_blind,  # min raise-to = BB + BB
            aggressor=bb, acted_since=set(), all_in=all_in,
            folded={p: False for p in _PLAYERS}, history=[], result=None,
            start_stacks=start_stacks,
        )

    # -- helpers ------------------------------------------------------------
    def _pot(self, s: HoldemState) -> int:
        # Chips committed this hand = (start - behind) summed over players.
        return sum(s.start_stacks.values()) - sum(s.stacks.values())

    def _max_commit(self, s: HoldemState) -> int:
        return max(s.street_commit.values())

    def _to_call(self, s: HoldemState, p: PlayerId) -> int:
        return self._max_commit(s) - s.street_commit[p]

    def current_player(self, s: HoldemState) -> PlayerId:
        return s.to_act

    def is_terminal(self, s: HoldemState) -> bool:
        return s.street == "done"

    # -- legal actions / validation ----------------------------------------
    def legal_actions(self, s: HoldemState, player: PlayerId) -> list:
        stack = s.stacks[player]
        sc = s.street_commit[player]
        to_call = self._to_call(s, player)
        if stack == 0:
            return []
        if to_call == 0:
            acts = ["check"]
            min_bet = sc + self.big_blind
            if sc + stack >= min_bet:
                acts.append("bet")
            acts.append("all_in")
            return acts
        # facing a bet
        acts = ["fold", "call"]
        max_total = sc + stack
        cur_max = self._max_commit(s)
        # Raising is only allowed when the action is open to this player. If this
        # player is already the aggressor (i.e. they face only a short all-in that
        # did not reopen betting), they may only call or fold.
        action_open = s.aggressor != player
        if action_open and max_total > cur_max:
            if max_total >= cur_max + s.last_raise_size:
                acts.append("raise")
            acts.append("all_in")
        return acts

    def validate_action(self, s: HoldemState, player: PlayerId, move: Move):
        legal = self.legal_actions(s, player)
        if move.type not in legal:
            return False, "illegal_action_type"
        stack = s.stacks[player]
        sc = s.street_commit[player]
        cur_max = self._max_commit(s)
        max_total = sc + stack

        if move.type in ("fold", "check", "call", "all_in"):
            if move.amount is not None:
                return False, "unexpected_amount"
            return True, None

        # bet / raise require an integer total within range
        if move.amount is None:
            return False, "missing_amount"
        if not isinstance(move.amount, int) or isinstance(move.amount, bool):
            return False, "non_integer_amount"
        if move.type == "bet":
            min_total = sc + self.big_blind
        else:  # raise
            min_total = cur_max + s.last_raise_size
        if move.amount < min_total:
            return False, "below_minimum"
        if move.amount > max_total:
            return False, "above_stack"
        return True, None

    # -- transition ---------------------------------------------------------
    def _clone(self, s: HoldemState) -> HoldemState:
        return replace(
            s, deck=list(s.deck), board=list(s.board),
            stacks=dict(s.stacks), street_commit=dict(s.street_commit),
            acted_since=set(s.acted_since), all_in=dict(s.all_in),
            folded=dict(s.folded), history=list(s.history),
            start_stacks=dict(s.start_stacks),
        )

    def step(self, s: HoldemState, move: Move) -> HoldemState:
        assert not self.is_terminal(s)
        ns = self._clone(s)
        p = ns.to_act
        cur_max = self._max_commit(ns)
        sc = ns.street_commit[p]

        if move.type == "fold":
            ns.folded[p] = True
            ns.history.append({"player": p, "action": "fold"})
            return self._resolve_fold(ns, winner=_other(p))

        if move.type == "check":
            ns.acted_since.add(p)
            ns.history.append({"player": p, "action": "check"})

        elif move.type == "call":
            pay = min(self._to_call(ns, p), ns.stacks[p])
            ns.stacks[p] -= pay
            ns.street_commit[p] += pay
            if ns.stacks[p] == 0:
                ns.all_in[p] = True
            ns.acted_since.add(p)
            ns.history.append({"player": p, "action": "call", "to": ns.street_commit[p]})

        elif move.type in ("bet", "raise"):
            total = move.amount
            add = total - sc
            ns.stacks[p] -= add
            ns.street_commit[p] = total
            ns.last_raise_size = total - cur_max
            ns.aggressor = p
            ns.acted_since = {p}
            if ns.stacks[p] == 0:
                ns.all_in[p] = True
            ns.history.append({"player": p, "action": move.type, "to": total})

        elif move.type == "all_in":
            new_total = sc + ns.stacks[p]
            ns.stacks[p] = 0
            ns.all_in[p] = True
            increment = new_total - cur_max
            # Only a FULL raise (>= the last raise size) reopens the betting for a
            # player who already acted. A short all-in is treated like a call: it
            # bumps the amount to match but does not give the opponent the option
            # to re-raise.
            if increment >= ns.last_raise_size:
                ns.last_raise_size = increment
                ns.aggressor = p
                ns.acted_since = {p}
            else:
                ns.acted_since.add(p)
            ns.street_commit[p] = new_total
            ns.history.append({"player": p, "action": "all_in", "to": new_total})

        if self._round_closed(ns):
            return self._advance(ns)
        ns.to_act = _other(p)
        return ns

    def _round_closed(self, s: HoldemState) -> bool:
        active = [p for p in _PLAYERS if not s.folded[p]]
        non_allin = [p for p in active if not s.all_in[p]]
        max_c = max(s.street_commit[p] for p in active)
        for p in non_allin:
            if s.street_commit[p] < max_c:
                return False          # still owes a call
            if p not in s.acted_since:
                return False          # still has the option to act
        return True

    def _refund_excess(self, s: HoldemState) -> None:
        """Return the uncalled portion of the current street to the over-committer."""
        c0, c1 = s.street_commit["player_0"], s.street_commit["player_1"]
        if c0 == c1:
            return
        high = "player_0" if c0 > c1 else "player_1"
        low = _other(high)
        refund = s.street_commit[high] - s.street_commit[low]
        s.stacks[high] += refund
        s.street_commit[high] -= refund

    def _advance(self, s: HoldemState) -> HoldemState:
        """A betting round just closed. Refund uncalled chips, then deal forward
        until the next betting round or showdown."""
        self._refund_excess(s)
        # chips are now "in the pot" (reflected by reduced stacks); reset street
        s.street_commit = {p: 0 for p in _PLAYERS}

        while True:
            if s.street == "river":
                return self._resolve_showdown(s)
            # deal the next street
            nxt = _STREETS[_STREETS.index(s.street) + 1]
            if nxt == "flop":
                s.board.extend([s.deck.pop(0) for _ in range(3)])
            else:  # turn or river: one card
                s.board.append(s.deck.pop(0))
            s.street = nxt
            s.history.append({"street": nxt, "board": list(s.board)})

            betting_possible = all(
                (not s.all_in[p] and not s.folded[p] and s.stacks[p] > 0)
                for p in _PLAYERS
            )
            if betting_possible:
                s.aggressor = None
                s.acted_since = set()
                s.street_commit = {p: 0 for p in _PLAYERS}
                s.to_act = _other(s.button)   # postflop: non-button acts first
                return s
            # else: no betting possible (someone all-in) — keep dealing the board

    # -- resolution ---------------------------------------------------------
    def _finish(self, s: HoldemState, deltas: dict, winner, reason: str) -> HoldemState:
        s.street = "done"
        s.to_act = None
        s.result = {"winner": winner, "reason": reason, "deltas": deltas,
                    "board": list(s.board)}
        return s

    def _resolve_fold(self, s: HoldemState, winner: PlayerId) -> HoldemState:
        # Net delta from this hand's starting stacks; uncalled chips are still in
        # the loser's behind, so the pot formula nets out correctly.
        deltas = {p: s.stacks[p] - s.start_stacks[p] for p in _PLAYERS}
        pot = self._pot(s)
        deltas[winner] += pot
        return self._finish(s, deltas, winner, "fold")

    def _resolve_showdown(self, s: HoldemState) -> HoldemState:
        pot = self._pot(s)
        k0 = evaluate7(list(s.hole["player_0"]) + s.board)
        k1 = evaluate7(list(s.hole["player_1"]) + s.board)
        base = {p: s.stacks[p] - s.start_stacks[p] for p in _PLAYERS}
        if k0 > k1:
            base["player_0"] += pot
            winner = "player_0"
        elif k1 > k0:
            base["player_1"] += pot
            winner = "player_1"
        else:
            base["player_0"] += pot // 2
            base["player_1"] += pot - pot // 2
            winner = None
        s = self._finish(s, base, winner, "showdown")
        s.result["hand_categories"] = {
            "player_0": category_name(k0), "player_1": category_name(k1),
        }
        return s

    def returns(self, s: HoldemState) -> dict:
        assert self.is_terminal(s)
        return {p: float(v) for p, v in s.result["deltas"].items()}

    def episode_metadata(self, s: HoldemState) -> dict:
        if not self.is_terminal(s) or not s.result:
            return {"big_blind": self.big_blind}
        return {
            "reason": s.result.get("reason"),
            "big_blind": self.big_blind,
            "hand_categories": s.result.get("hand_categories"),
            "final_board": s.result.get("board"),
        }

    # -- observation / render ----------------------------------------------
    def observation(self, s: HoldemState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        opp = _other(player)
        stack, sc = s.stacks[player], s.street_commit[player]
        cur_max = self._max_commit(s)
        to_call = self._to_call(s, player)

        amount_range = {}
        if "bet" in legal:
            amount_range["bet"] = {"min": sc + self.big_blind, "max": sc + stack}
        if "raise" in legal:
            amount_range["raise"] = {"min": cur_max + s.last_raise_size,
                                     "max": sc + stack}

        public = {
            "pot": self._pot(s),
            "to_call": to_call,
            "your_stack": stack,
            "opp_stack": s.stacks[opp],
            "your_street_commit": sc,
            "opp_street_commit": s.street_commit[opp],
            "board": list(s.board),
            "street": s.street,
            "position": "button/SB" if s.button == player else "BB",
            "you_all_in": s.all_in[player],
            "opp_all_in": s.all_in[opp],
            "amount_range": amount_range,
        }
        return Observation(
            player=player,
            private={"hole": list(s.hole[player])},
            public=public,
            history=list(s.history),
            legal_actions=legal,
            rendered=self._render_for(s, player, legal, amount_range),
        )

    def _render_for(self, s, player, legal, amount_range) -> str:
        opp = _other(player)
        pot = self._pot(s)
        to_call = self._to_call(s, player)
        board = " ".join(s.board) if s.board else "(none)"
        pos = "button/SB" if s.button == player else "BB"
        lines = [
            f"You are {player} ({pos}) in Heads-Up Texas Hold'em Lite.",
            f"Street: {s.street}. Your hole cards: {' '.join(s.hole[player])}. "
            f"Board: {board}.",
            f"Pot: {pot}. Your stack: {s.stacks[player]}, opponent stack: {s.stacks[opp]}.",
            f"Your committed this street: {s.street_commit[player]}, "
            f"opponent: {s.street_commit[opp]}. To call: {to_call}.",
            f"Legal actions: {', '.join(legal)}.",
        ]
        if "bet" in amount_range:
            r = amount_range["bet"]
            lines.append(f"Bet amount (total this street) must be in [{r['min']}, {r['max']}].")
        if "raise" in amount_range:
            r = amount_range["raise"]
            lines.append(f"Raise-to amount (total this street) must be in [{r['min']}, {r['max']}].")
        return "\n".join(lines)

    def render(self, s: HoldemState, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None and not self.is_terminal(s):
            return self.observation(s, perspective).rendered
        board = " ".join(s.board) if s.board else "(none)"
        hands = "; ".join(f"{p}={' '.join(s.hole[p])}" for p in _PLAYERS)
        cats = ""
        if s.result and s.result.get("hand_categories"):
            hc = s.result["hand_categories"]
            cats = f"  ({_PLAYERS[0]}: {hc['player_0']}, {_PLAYERS[1]}: {hc['player_1']})"
        return f"Hold'em[board: {board}; hole: {hands}]{cats}"
