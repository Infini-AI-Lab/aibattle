"""Heads-Up Hold'em Match Mode.

A *match* is a sequence of up to ``max_hands`` heads-up hands with stacks that
carry over, the button alternating each hand, and a match-level winner (more
chips at the end, or the opponent busting). The whole multi-hand match is ONE
episode — the atomic unit the runner schedules, seeds, persists, and resumes.
This reduces the single-hand chip-delta variance: a win counts the same
regardless of margin, so one large pot can't dominate an aggregate.

Implementation: this is a thin wrapper over the single-hand ``HoldemPoker``
engine. Each hand is dealt via ``engine.deal_hand`` from the carried stacks; the
hand's zero-sum chip deltas are applied to the carried stacks; the next hand is
dealt until a player busts (or cannot cover the big blind) or ``max_hands`` is
reached. Because hands within a match run sequentially, exposing the running
chip standing in the prompt is deterministic (unlike cross-episode standings
under parallel execution).
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game
from .holdem import HoldemPoker, HoldemState, _PLAYERS, _other

DEFAULT_STARTING_STACK = 100
DEFAULT_MAX_HANDS = 20


@dataclass
class MatchState:
    hand: HoldemState               # current hand sub-state
    stacks: dict                    # chips carried between hands (pre-current-hand)
    hand_number: int                # 1-based index of the current hand
    button: PlayerId                # button for the current hand
    rng: random.Random              # entropy for dealing subsequent hands
    done: bool = False
    match_result: Optional[dict] = None
    hand_summaries: list = field(default_factory=list)


class HoldemMatch(Game):
    name = "holdem_match"
    version = "1.0.0"
    players = list(_PLAYERS)

    def __init__(self, starting_stack: int = DEFAULT_STARTING_STACK,
                 small_blind: int = 1, big_blind: int = 2,
                 max_hands: int = DEFAULT_MAX_HANDS):
        self.engine = HoldemPoker(starting_stack, small_blind, big_blind)
        self.starting_stack = starting_stack
        self.big_blind = big_blind
        self.max_hands = max_hands

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> MatchState:
        stacks = {p: self.starting_stack for p in _PLAYERS}
        button = _PLAYERS[rng.randrange(2)]
        hand = self.engine.deal_hand(rng, stacks, button)
        return MatchState(hand=hand, stacks=dict(stacks), hand_number=1,
                          button=button, rng=rng)

    # -- delegation to the current hand ------------------------------------
    def current_player(self, s: MatchState) -> PlayerId:
        return self.engine.current_player(s.hand)

    def legal_actions(self, s: MatchState, player: PlayerId) -> list:
        return self.engine.legal_actions(s.hand, player)

    def validate_action(self, s: MatchState, player: PlayerId, move: Move):
        return self.engine.validate_action(s.hand, player, move)

    def fallback_action(self, s: MatchState, player: PlayerId, legal: list) -> Move:
        return self.engine.fallback_action(s.hand, player, legal)

    def is_terminal(self, s: MatchState) -> bool:
        return s.done

    # -- transition ---------------------------------------------------------
    def step(self, s: MatchState, move: Move) -> MatchState:
        assert not s.done
        new_hand = self.engine.step(s.hand, move)
        if not self.engine.is_terminal(new_hand):
            return replace(s, hand=new_hand)

        # Hand finished: apply its zero-sum deltas to the carried stacks. Use the
        # raw INTEGER deltas (returns() casts to float, which would make carried
        # stacks float and corrupt downstream integer bet ranges).
        deltas = new_hand.result["deltas"]
        stacks = {p: s.stacks[p] + deltas[p] for p in _PLAYERS}
        summaries = s.hand_summaries + [{
            "hand": s.hand_number, "button": s.button,
            "winner": new_hand.result.get("winner"),
            "reason": new_hand.result.get("reason"),
            "deltas": deltas, "stacks_after": dict(stacks),
        }]

        busted = any(stacks[p] <= 0 for p in _PLAYERS)
        cant_cover = min(stacks.values()) < self.big_blind
        reached_max = s.hand_number >= self.max_hands
        if busted or cant_cover or reached_max:
            reason = "bust" if (busted or cant_cover) else "max_hands"
            return self._finish_match(s, new_hand, stacks, summaries, reason)

        # Deal the next hand: rotate the button, carry the stacks forward.
        next_button = _other(s.button)
        next_hand = self.engine.deal_hand(s.rng, stacks, next_button)
        return replace(s, hand=next_hand, stacks=stacks,
                       hand_number=s.hand_number + 1, button=next_button,
                       hand_summaries=summaries)

    def _finish_match(self, s: MatchState, last_hand: HoldemState,
                      stacks: dict, summaries: list, reason: str) -> MatchState:
        if stacks["player_0"] > stacks["player_1"]:
            winner = "player_0"
        elif stacks["player_1"] > stacks["player_0"]:
            winner = "player_1"
        else:
            winner = None
        result = {
            "winner": winner,
            "reason": reason,
            "final_stacks": dict(stacks),
            "stack_diff": stacks["player_0"] - stacks["player_1"],
            "hands_played": s.hand_number,
            "hand_summaries": summaries,
        }
        return replace(s, hand=last_hand, stacks=stacks, done=True,
                       match_result=result)

    # -- results ------------------------------------------------------------
    def returns(self, s: MatchState) -> dict:
        """Match-level payoff: +1 win / -1 loss / 0 draw (drives match win rate)."""
        assert s.done
        w = s.match_result["winner"]
        if w is None:
            return {p: 0.0 for p in _PLAYERS}
        return {w: 1.0, _other(w): -1.0}

    def episode_metadata(self, s: MatchState) -> dict:
        if not s.done or not s.match_result:
            return {"max_hands": self.max_hands, "big_blind": self.big_blind}
        r = s.match_result
        return {
            "mode": "match",
            "max_hands": self.max_hands,
            "big_blind": self.big_blind,
            "reason": r["reason"],
            "hands_played": r["hands_played"],
            "final_stacks": r["final_stacks"],
            "stack_diff": r["stack_diff"],
            "hand_summaries": r["hand_summaries"],
        }

    # -- observation / render ----------------------------------------------
    def observation(self, s: MatchState, player: PlayerId) -> Observation:
        inner = self.engine.observation(s.hand, player)
        opp = _other(player)
        me0, op0 = s.hand.start_stacks[player], s.hand.start_stacks[opp]
        lead = me0 - op0
        stance = "ahead" if lead > 0 else "behind" if lead < 0 else "even"
        ctx = (
            f"Heads-Up MATCH — hand {s.hand_number} of {self.max_hands}. "
            f"Stacks carry across hands; you win the match by finishing with more "
            f"chips than your opponent (or if they bust). Chips at the start of "
            f"this hand — you: {me0}, opponent: {op0} (you are {stance} by "
            f"{abs(lead)})."
        )
        public = dict(inner.public)
        public.update({
            "match_hand": s.hand_number,
            "match_max_hands": self.max_hands,
            "match_your_chips": me0,
            "match_opp_chips": op0,
            "match_lead": lead,
        })
        return Observation(
            player=inner.player, private=inner.private, public=public,
            history=inner.history, legal_actions=inner.legal_actions,
            rendered=ctx + "\n" + inner.rendered,
        )

    def render(self, s: MatchState, *, perspective: Optional[PlayerId] = None) -> str:
        if perspective is not None and not s.done:
            return self.observation(s, perspective).rendered
        if s.done and s.match_result:
            r = s.match_result
            return (f"Hold'em Match[hands={r['hands_played']} "
                    f"final={r['final_stacks']} winner={r['winner']} "
                    f"reason={r['reason']}]")
        return self.engine.render(s.hand, perspective=perspective)
