"""Multi-player Texas Hold'em — single-hand engine (core of Table Mode).

Generalizes the heads-up engine to N players (>= 2) seated in a fixed order with
a rotating button. Supports correct multi-player action order, fold/check/call/
bet/raise/all-in, mixed all-in amounts, full-raise re-open rules, and showdown
via the fuzz-tested side-pot engine (``poker_sidepot``).

This module is the per-hand sub-step used by the multi-hand ``HoldemTable``
game. It is deliberately separate and directly testable: a hand is pure and
immutable (``step`` clones), and ``settle`` returns zero-sum per-player chip
deltas for the hand.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game
from .poker_eval import category_name, evaluate7, full_deck
from .poker_sidepot import award_pots, build_pots

_STREETS = ["preflop", "flop", "turn", "river"]


@dataclass
class TableHandState:
    order: list                       # seat order of player ids
    button: int                       # index into order
    deck: list
    hole: dict                        # {player: (c, c)}
    board: list
    street: str                       # preflop|flop|turn|river|done
    stacks: dict                      # chips behind
    street_commit: dict               # committed THIS street
    contributed: dict                 # TOTAL committed this hand (for side pots)
    start_stacks: dict                # chips at hand start (for delta accounting)
    to_act: Optional[PlayerId]
    current_bet: int                  # max street_commit this street
    last_raise_size: int
    aggressor: Optional[PlayerId]
    acted_since: set                  # acted since last aggression
    all_in: dict
    folded: dict
    history: list
    result: Optional[dict] = None


class MultiHoldemHand:
    """Single multi-player hand engine. Stateless; operates on TableHandState."""

    def __init__(self, small_blind: int = 1, big_blind: int = 2):
        self.small_blind = small_blind
        self.big_blind = big_blind

    # -- setup --------------------------------------------------------------
    def deal_hand(self, rng: random.Random, order: list, stacks: dict,
                  button: int) -> TableHandState:
        """Deal a hand. ``order`` is seat order; ``button`` indexes it. All
        players in ``order`` must be able to cover the big blind."""
        n = len(order)
        deck = full_deck()
        rng.shuffle(deck)
        hole = {}
        i = 0
        for p in order:
            hole[p] = (deck[i], deck[i + 1])
            i += 2
        deck = deck[i:]

        stacks = dict(stacks)
        start_stacks = dict(stacks)
        street_commit = {p: 0 for p in order}
        contributed = {p: 0 for p in order}

        # Blinds + first-to-act. For 3+ players: SB left of button, BB left of
        # SB, UTG (first to act) left of BB. Heads-up (a table dwindled to 2) is
        # the special case: the button IS the small blind and acts first preflop;
        # the other player is the big blind and acts first postflop (handled by
        # _advance, whose (button+1) start is already the non-button there).
        if n == 2:
            sb_seat = button
            bb_seat = (button + 1) % 2
            first = button
        else:
            sb_seat = (button + 1) % n
            bb_seat = (button + 2) % n
            first = (button + 3) % n
        sb, bb = order[sb_seat], order[bb_seat]
        for p, amt in ((sb, self.small_blind), (bb, self.big_blind)):
            pay = min(amt, stacks[p])
            stacks[p] -= pay
            street_commit[p] = pay
            contributed[p] = pay
        all_in = {p: stacks[p] == 0 for p in order}
        s = TableHandState(
            order=list(order), button=button, deck=deck, hole=hole, board=[],
            street="preflop", stacks=stacks, street_commit=street_commit,
            contributed=contributed, start_stacks=start_stacks, to_act=None,
            current_bet=self.big_blind, last_raise_size=self.big_blind,
            aggressor=bb, acted_since=set(),
            all_in=all_in, folded={p: False for p in order}, history=[],
        )
        s.to_act = self._next_actor(s, start_index=first)
        if s.to_act is None or self._betting_done(s):
            return self._advance(s)
        return s

    # -- helpers ------------------------------------------------------------
    def _live(self, s) -> list:
        return [p for p in s.order if not s.folded[p]]

    def _can_act(self, s, p) -> bool:
        return not s.folded[p] and not s.all_in[p] and s.stacks[p] > 0

    def _next_actor(self, s, start_index: int) -> Optional[PlayerId]:
        n = len(s.order)
        for k in range(n):
            p = s.order[(start_index + k) % n]
            if self._can_act(s, p):
                return p
        return None

    def _to_call(self, s, p) -> int:
        return s.current_bet - s.street_commit[p]

    # -- public game-ish API ------------------------------------------------
    def current_player(self, s) -> PlayerId:
        return s.to_act

    def is_terminal(self, s) -> bool:
        return s.street == "done"

    def legal_actions(self, s, player) -> list:
        if s.folded[player] or s.stacks[player] == 0:
            return []
        to_call = self._to_call(s, player)
        stack = s.stacks[player]
        sc = s.street_commit[player]
        if to_call == 0:
            acts = ["check"]
            if stack >= self.big_blind:  # can make a real bet
                acts.append("bet")
            acts.append("all_in")
            return acts
        acts = ["fold", "call"]
        max_total = sc + stack
        # Action is reopened for raising only if this player has not acted since
        # the last FULL bet/raise. acted_since is reset to {aggressor} on a full
        # raise, and a short (non-reopening) all-in only ADDS the shover without
        # resetting it — so an already-acted caller facing a short all-in is
        # correctly barred from re-raising (multi-player; "aggressor != player"
        # would wrongly allow non-aggressor callers to re-raise here).
        action_open = player not in s.acted_since
        if action_open and max_total > s.current_bet:
            if max_total >= s.current_bet + s.last_raise_size:
                acts.append("raise")
            acts.append("all_in")
        return acts

    def validate_action(self, s, player, move: Move):
        legal = self.legal_actions(s, player)
        if move.type not in legal:
            return False, "illegal_action_type"
        stack = s.stacks[player]
        sc = s.street_commit[player]
        max_total = sc + stack
        if move.type in ("fold", "check", "call", "all_in"):
            if move.amount is not None:
                return False, "unexpected_amount"
            return True, None
        if move.amount is None:
            return False, "missing_amount"
        if not isinstance(move.amount, int) or isinstance(move.amount, bool):
            return False, "non_integer_amount"
        min_total = (sc + self.big_blind if move.type == "bet"
                     else s.current_bet + s.last_raise_size)
        if move.amount < min_total:
            return False, "below_minimum"
        if move.amount > max_total:
            return False, "above_stack"
        return True, None

    def fallback_action(self, s, player, legal) -> Move:
        for t in ("check", "fold", "call"):
            if t in legal:
                return Move(type=t)
        return Move(type=legal[0]) if legal else Move(type="__invalid__")

    # -- transition ---------------------------------------------------------
    def _clone(self, s) -> TableHandState:
        return replace(
            s, deck=list(s.deck), board=list(s.board), stacks=dict(s.stacks),
            street_commit=dict(s.street_commit), contributed=dict(s.contributed),
            acted_since=set(s.acted_since), all_in=dict(s.all_in),
            folded=dict(s.folded), history=list(s.history),
        )

    def _commit(self, s, p, total_this_street):
        """Move chips so that player p's street_commit becomes total_this_street."""
        add = total_this_street - s.street_commit[p]
        s.stacks[p] -= add
        s.street_commit[p] = total_this_street
        s.contributed[p] += add
        if s.stacks[p] == 0:
            s.all_in[p] = True

    def step(self, s, move: Move) -> TableHandState:
        assert not self.is_terminal(s)
        ns = self._clone(s)
        p = ns.to_act

        if move.type == "fold":
            ns.folded[p] = True
            ns.history.append({"player": p, "action": "fold"})
        elif move.type == "check":
            ns.acted_since.add(p)
            ns.history.append({"player": p, "action": "check"})
        elif move.type == "call":
            pay = min(self._to_call(ns, p), ns.stacks[p])
            self._commit(ns, p, ns.street_commit[p] + pay)
            ns.acted_since.add(p)
            ns.history.append({"player": p, "action": "call", "to": ns.street_commit[p]})
        elif move.type in ("bet", "raise"):
            total = move.amount
            inc = total - ns.current_bet
            self._commit(ns, p, total)
            ns.last_raise_size = inc
            ns.current_bet = total
            ns.aggressor = p
            ns.acted_since = {p}
            ns.history.append({"player": p, "action": move.type, "to": total})
        elif move.type == "all_in":
            new_total = ns.street_commit[p] + ns.stacks[p]
            self._commit(ns, p, new_total)
            inc = new_total - ns.current_bet
            if inc >= ns.last_raise_size and inc > 0:
                ns.last_raise_size = inc
                ns.current_bet = new_total
                ns.aggressor = p
                ns.acted_since = {p}
            else:
                if new_total > ns.current_bet:
                    ns.current_bet = new_total  # bump amount, no re-open
                ns.acted_since.add(p)
            ns.history.append({"player": p, "action": "all_in", "to": new_total})

        # One live player left -> hand ends immediately (everyone else folded).
        if len(self._live(ns)) == 1:
            return self._settle_fold(ns, self._live(ns)[0])

        if self._betting_done(ns):
            return self._advance(ns)
        ns.to_act = self._next_actor(ns, start_index=(ns.order.index(p) + 1))
        if ns.to_act is None:
            return self._advance(ns)
        return ns

    def _betting_done(self, s) -> bool:
        live = self._live(s)
        actable = [p for p in live if self._can_act(s, p)]
        if len(actable) == 0:
            return True
        # Everyone who can act must have matched the current bet and acted since
        # the last aggression.
        for p in actable:
            if s.street_commit[p] < s.current_bet:
                return False
            if p not in s.acted_since:
                return False
        # If exactly one player can still act and everyone else is all-in/folded,
        # there is no one to call them -> close (their excess refunded in settle).
        return True

    def _advance(self, s) -> TableHandState:
        # Move street chips into the pot (already reflected by reduced stacks);
        # reset per-street commitments.
        s.street_commit = {p: 0 for p in s.order}
        while True:
            if s.street == "river":
                return self._settle_showdown(s)
            nxt = _STREETS[_STREETS.index(s.street) + 1]
            if nxt == "flop":
                s.board.extend([s.deck.pop(0) for _ in range(3)])
            else:
                s.board.append(s.deck.pop(0))
            s.street = nxt
            s.history.append({"street": nxt, "board": list(s.board)})
            s.current_bet = 0
            s.last_raise_size = self.big_blind
            s.aggressor = None
            s.acted_since = set()
            s.street_commit = {p: 0 for p in s.order}
            first = self._next_actor(s, start_index=(s.button + 1) % len(s.order))
            # Need at least two players who can act for a betting round.
            actable = [p for p in self._live(s) if self._can_act(s, p)]
            if len(actable) >= 2 and first is not None:
                s.to_act = first
                return s
            # else: no betting possible (all-in showdown) -> keep dealing board

    # -- resolution ---------------------------------------------------------
    def _finish(self, s, deltas, reason) -> TableHandState:
        s.street = "done"
        s.to_act = None
        s.result = {"reason": reason, "deltas": deltas, "board": list(s.board)}
        return s

    def _settle_fold(self, s, winner) -> TableHandState:
        # Everyone else folded. Winner takes the pot; uncalled chips net out via
        # contributed accounting (winner may have contributed less than they are
        # awarded; the pot is the sum of all contributions).
        pot = sum(s.contributed.values())
        deltas = {p: -s.contributed[p] for p in s.order}
        deltas[winner] += pot
        return self._finish(s, deltas, "fold")

    def _settle_showdown(self, s) -> TableHandState:
        live = self._live(s)
        ranks = {p: evaluate7(list(s.hole[p]) + s.board) for p in live}
        pots = build_pots(s.contributed, folded=set(p for p in s.order if s.folded[p]))
        won = award_pots(pots, ranks)
        deltas = {p: won.get(p, 0) - s.contributed[p] for p in s.order}
        s = self._finish(s, deltas, "showdown")
        s.result["hand_categories"] = {p: category_name(ranks[p]) for p in live}
        s.result["pots"] = pots
        return s

    def settle_deltas(self, s) -> dict:
        assert self.is_terminal(s)
        return {p: float(v) for p, v in s.result["deltas"].items()}


@dataclass
class TableState:
    hand: TableHandState            # current hand sub-state
    stacks: dict                    # carried chips per player (pre-current-hand)
    button: PlayerId                # button player for the current hand
    hand_number: int
    rng: random.Random
    bust_order: list = field(default_factory=list)   # players in order of busting
    done: bool = False
    result: Optional[dict] = None
    hand_summaries: list = field(default_factory=list)


class HoldemTable(Game):
    """Multi-player Hold'em Table Mode. A whole table session (up to max_hands)
    is ONE episode; the output is a ranking by final chips (bust order breaks
    ties among the eliminated)."""

    name = "holdem_table"
    version = "1.0.0"

    def __init__(self, num_players: int = 4, starting_stack: int = 100,
                 small_blind: int = 1, big_blind: int = 2, max_hands: int = 30):
        self.players = [f"player_{i}" for i in range(num_players)]
        self.engine = MultiHoldemHand(small_blind, big_blind)
        self.starting_stack = starting_stack
        self.small_blind = small_blind
        self.big_blind = big_blind
        self.max_hands = max_hands

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> TableState:
        stacks = {p: self.starting_stack for p in self.players}
        button = self.players[rng.randrange(len(self.players))]
        hand = self._deal(rng, stacks, button)
        return TableState(hand=hand, stacks=dict(stacks), button=button,
                          hand_number=1, rng=rng)

    def _active(self, stacks: dict) -> list:
        # Players still holding chips, in fixed seat order.
        return [p for p in self.players if stacks[p] > 0]

    def _deal(self, rng, stacks, button) -> TableHandState:
        active = self._active(stacks)
        order = active
        button_idx = order.index(button)
        sub_stacks = {p: stacks[p] for p in order}
        return self.engine.deal_hand(rng, order, sub_stacks, button_idx)

    # -- delegation ---------------------------------------------------------
    def current_player(self, s): return self.engine.current_player(s.hand)
    def legal_actions(self, s, player): return self.engine.legal_actions(s.hand, player)
    def validate_action(self, s, player, move): return self.engine.validate_action(s.hand, player, move)
    def fallback_action(self, s, player, legal): return self.engine.fallback_action(s.hand, player, legal)
    def is_terminal(self, s): return s.done

    # -- transition ---------------------------------------------------------
    def step(self, s, move: Move) -> TableState:
        assert not s.done
        new_hand = self.engine.step(s.hand, move)
        if not self.engine.is_terminal(new_hand):
            return replace(s, hand=new_hand)

        # Hand finished: apply per-player deltas (only players dealt in change).
        # Use raw INTEGER deltas (settle_deltas casts to float, which would make
        # carried stacks float and corrupt downstream integer bet ranges).
        deltas = new_hand.result["deltas"]
        stacks = dict(s.stacks)
        for p in deltas:
            stacks[p] += deltas[p]
        bust_order = list(s.bust_order)
        for p in self.players:
            if stacks[p] <= 0 and p not in bust_order:
                bust_order.append(p)   # busted this hand
        summaries = s.hand_summaries + [{
            "hand": s.hand_number, "button": s.button,
            "reason": new_hand.result.get("reason"),
            "deltas": deltas, "stacks_after": dict(stacks),
        }]

        active = self._active(stacks)
        reached_max = s.hand_number >= self.max_hands
        if len(active) <= 1 or reached_max:
            reason = "one_left" if len(active) <= 1 else "max_hands"
            return self._finish_table(s, stacks, bust_order, summaries, reason)

        next_button = self._next_button(s.button, stacks)
        next_hand = self._deal(s.rng, stacks, next_button)
        return replace(s, hand=next_hand, stacks=stacks, button=next_button,
                       hand_number=s.hand_number + 1, bust_order=bust_order,
                       hand_summaries=summaries)

    def _next_button(self, button: PlayerId, stacks: dict) -> PlayerId:
        # Next active player clockwise from the current button.
        n = len(self.players)
        start = self.players.index(button)
        for k in range(1, n + 1):
            p = self.players[(start + k) % n]
            if stacks[p] > 0:
                return p
        return button

    def _finish_table(self, s, stacks, bust_order, summaries, reason) -> TableState:
        # Rank: survivors by final chips (desc); then busted players by bust order
        # (later bust = higher rank). Survivors are those not in bust_order.
        survivors = [p for p in self.players if p not in bust_order]
        survivors.sort(key=lambda p: stacks[p], reverse=True)
        eliminated = list(reversed(bust_order))  # last to bust ranks higher
        ranking = survivors + eliminated
        rank_of = {p: i + 1 for i, p in enumerate(ranking)}
        result = {
            "reason": reason,
            "final_stacks": dict(stacks),
            "ranking": ranking,
            "rank_of": rank_of,
            "bust_order": bust_order,
            "hands_played": s.hand_number,
            "hand_summaries": summaries,
        }
        return replace(s, stacks=stacks, bust_order=bust_order, done=True,
                       result=result)

    # -- results ------------------------------------------------------------
    def returns(self, s) -> dict:
        """Per-player payoff = final chips (winner = chip leader). Detailed
        ranking lives in episode_metadata."""
        assert s.done
        return {p: float(s.result["final_stacks"][p]) for p in self.players}

    def episode_metadata(self, s) -> dict:
        if not s.done or not s.result:
            return {"mode": "table", "num_players": len(self.players),
                    "max_hands": self.max_hands}
        r = s.result
        return {
            "mode": "table",
            "num_players": len(self.players),
            "max_hands": self.max_hands,
            "reason": r["reason"],
            "hands_played": r["hands_played"],
            "final_stacks": r["final_stacks"],
            "ranking": r["ranking"],
            "rank_of": r["rank_of"],
            "bust_order": r["bust_order"],
            "hand_summaries": r["hand_summaries"],
        }

    # -- observation / render ----------------------------------------------
    def observation(self, s, player) -> Observation:
        h = s.hand
        eng = self.engine
        legal = eng.legal_actions(h, player)
        to_call = eng._to_call(h, player) if not h.folded.get(player) else 0
        pot = sum(h.contributed.values())
        sc = h.street_commit.get(player, 0)
        stack = h.stacks.get(player, 0)

        amount_range = {}
        if "bet" in legal:
            amount_range["bet"] = {"min": sc + self.big_blind, "max": sc + stack}
        if "raise" in legal:
            amount_range["raise"] = {"min": h.current_bet + h.last_raise_size,
                                     "max": sc + stack}

        def status(p):
            if s.stacks.get(p, 0) <= 0 and p not in h.order:
                return "busted"
            if h.folded.get(p):
                return "folded"
            if h.all_in.get(p):
                return "all_in"
            return "active"

        seats = {p: {"stack": h.stacks.get(p, s.stacks.get(p, 0)),
                     "status": status(p),
                     "committed": h.contributed.get(p, 0)} for p in self.players}
        # current rank by chips (1 = most chips)
        ranked = sorted(self.players, key=lambda p: s.stacks.get(p, 0), reverse=True)
        my_rank = ranked.index(player) + 1

        public = {
            "pot": pot, "to_call": to_call, "board": list(h.board),
            "street": h.street, "your_stack": stack,
            "current_bet": h.current_bet, "amount_range": amount_range,
            "seats": seats, "num_players": len(self.players),
            "players_left": len(self._active(s.stacks)),
            "table_hand": s.hand_number, "table_max_hands": self.max_hands,
            "your_rank": my_rank, "button": s.button,
            "pots": [{"amount": pt["amount"], "eligible": pt["eligible"]}
                     for pt in build_pots(h.contributed,
                                          set(p for p in h.order if h.folded[p]))],
        }
        return Observation(
            player=player,
            private={"hole": list(h.hole[player])},
            public=public,
            history=list(h.history),
            legal_actions=legal,
            rendered=self._render_for(s, player, legal, amount_range, seats, my_rank),
        )

    def _render_for(self, s, player, legal, amount_range, seats, my_rank) -> str:
        h = s.hand
        board = " ".join(h.board) if h.board else "(none)"
        pot = sum(h.contributed.values())
        to_call = self.engine._to_call(h, player) if not h.folded.get(player) else 0
        lines = [
            f"Multi-player Hold'em TABLE — {len(self.players)} players, "
            f"hand {s.hand_number} of {self.max_hands}, "
            f"{len(self._active(s.stacks))} still in the table. "
            f"You are {player}; you currently rank #{my_rank} by chips. "
            f"Goal: finish as high as possible (ideally chip leader).",
            f"Street: {h.street}. Your hole: {' '.join(h.hole[player])}. Board: {board}.",
            f"Pot: {pot}. Your stack: {h.stacks.get(player,0)}. To call: {to_call}.",
            "Seats: " + ", ".join(
                f"{p}[{seats[p]['status']},stack {seats[p]['stack']}]"
                for p in self.players),
            f"Legal actions: {', '.join(legal)}.",
        ]
        if "bet" in amount_range:
            r = amount_range["bet"]
            lines.append(f"Bet amount (total this street) in [{r['min']}, {r['max']}].")
        if "raise" in amount_range:
            r = amount_range["raise"]
            lines.append(f"Raise-to amount (total this street) in [{r['min']}, {r['max']}].")
        return "\n".join(lines)

    def render(self, s, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None and not s.done:
            return self.observation(s, perspective).rendered
        if s.done and s.result:
            r = s.result
            return (f"Hold'em Table[hands={r['hands_played']} "
                    f"ranking={r['ranking']} final={r['final_stacks']}]")
        return f"Hold'em Table[hand {s.hand_number}, board {s.hand.board}]"
