"""Independent Blackjack — one agent (player_0) vs a fixed dealer (player_1).

This is an agent-vs-environment game modeled as a two-seat game so it fits the
sequential two-player runner: ``player_0`` is the LLM under evaluation and
``player_1`` is a scripted dealer (the built-in ``blackjack_dealer`` agent).
Each hand is independent — cards are drawn from the per-episode RNG with
replacement-free draws from an infinite-shoe abstraction (uniform ranks), which
keeps hands independent and analysis simple.

Player actions: ``hit``, ``stand``, ``double`` (no split/surrender/insurance in
v0). The dealer follows a fixed policy: hit while its ace-aware total is below
17, stand on all 17s including soft 17.

Turn flow:
- ``player_0`` acts until it stands, busts, or doubles.
- The hand becomes the dealer's turn ONLY if the player has not busted (a bust,
  including a double-then-bust, ends the hand immediately with no dealer draw).
- A two-card 21 is a natural, resolved before ordinary totals.

Scoring (player_0 perspective; ``player_1`` is the negation, zero-sum):
- normal win +1, normal loss -1, push 0
- double win +2, double loss -2
- player natural (only) +1.5; both natural 0 (push); dealer natural (only) -1
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from ..types import Move, Observation, PlayerId
from .base import Game

_PLAYERS = ["player_0", "player_1"]
# Ranks: 2..10, J/Q/K count as 10, A counts as 1 or 11 (ace-aware).
_RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
_VALUE = {**{str(n): n for n in range(2, 11)}, "J": 10, "Q": 10, "K": 10, "A": 11}


def hand_total(cards) -> tuple:
    """Return (total, is_soft) with ace-aware scoring.

    Aces count as 11 unless that busts, in which case they drop to 1. ``is_soft``
    is True when at least one ace is still counted as 11.
    """
    total = sum(_VALUE[c] for c in cards)
    aces = sum(1 for c in cards if c == "A")
    soft_aces = aces
    while total > 21 and soft_aces > 0:
        total -= 10           # demote one ace from 11 to 1
        soft_aces -= 1
    is_soft = soft_aces > 0
    return total, is_soft


def is_bust(cards) -> bool:
    return hand_total(cards)[0] > 21


def is_natural(cards) -> bool:
    """A natural is exactly two cards totalling 21 (an ace + a ten-value)."""
    return len(cards) == 2 and hand_total(cards)[0] == 21


def dealer_should_hit(cards) -> bool:
    """Dealer policy: hit while total < 17; stand on all 17s incl. soft 17."""
    total, _ = hand_total(cards)
    return total < 17


@dataclass(frozen=True)
class BlackjackState:
    deck: tuple                  # remaining draw pile (pre-shuffled)
    player: tuple                # player_0 cards
    dealer: tuple                # player_1 (dealer) cards; index 0 is the upcard
    phase: str = "player"        # "player" | "dealer" | "done"
    doubled: bool = False
    draw_index: int = 0          # next card to draw from deck


class IndependentBlackjack(Game):
    name = "independent_blackjack"
    version = "1.0.0"
    players = list(_PLAYERS)

    # -- setup --------------------------------------------------------------
    def initial_state(self, rng: random.Random) -> BlackjackState:
        # A generous shuffled draw pile; each hand is independent because a fresh
        # state is built per episode from the per-episode RNG.
        deck = [rng.choice(_RANKS) for _ in range(12)]
        player = (deck[0], deck[2])
        dealer = (deck[1], deck[3])  # dealer[0] is the upcard, dealer[1] hidden
        idx = 4
        # If either side already has a natural, the player gets no decision.
        phase = "player"
        if is_natural(player) or is_natural(dealer):
            phase = "done"
        return BlackjackState(deck=tuple(deck), player=player, dealer=dealer,
                              phase=phase, doubled=False, draw_index=idx)

    # -- turn logic ---------------------------------------------------------
    def current_player(self, s: BlackjackState) -> PlayerId:
        if s.phase == "dealer":
            return "player_1"
        return "player_0"

    def legal_actions(self, s: BlackjackState, player: PlayerId) -> list:
        if s.phase == "player":
            # Double only allowed on the opening two-card hand.
            if len(s.player) == 2:
                return ["hit", "stand", "double"]
            return ["hit", "stand"]
        if s.phase == "dealer":
            return ["hit", "stand"]
        return []

    def is_terminal(self, s: BlackjackState) -> bool:
        return s.phase == "done"

    def _draw(self, s: BlackjackState):
        """Return (card, next_index), extending the deck deterministically if
        the pre-rolled pile is exhausted (degenerate; keeps step total-safe)."""
        if s.draw_index < len(s.deck):
            return s.deck[s.draw_index], s.draw_index + 1
        # Deterministic extension from the draw index (extremely rare).
        card = _RANKS[s.draw_index % len(_RANKS)]
        return card, s.draw_index + 1

    def step(self, s: BlackjackState, move: Move) -> BlackjackState:
        assert s.phase != "done", "step() on terminal state"
        if s.phase == "player":
            return self._step_player(s, move)
        return self._step_dealer(s, move)

    def _phase_after_player_finishes(self, dealer) -> str:
        """The phase once the player has finished without busting: the dealer
        only acts when its policy says to draw, otherwise the hand is done."""
        return "dealer" if dealer_should_hit(dealer) else "done"

    def _step_player(self, s: BlackjackState, move: Move) -> BlackjackState:
        if move.type == "stand":
            # Player is done; the dealer acts only if it must draw.
            return BlackjackState(deck=s.deck, player=s.player, dealer=s.dealer,
                                  phase=self._phase_after_player_finishes(s.dealer),
                                  doubled=s.doubled, draw_index=s.draw_index)
        if move.type == "double":
            card, idx = self._draw(s)
            player = s.player + (card,)
            # Double draws exactly one card then forcibly stands; a bust ends the
            # hand immediately (dealer does not draw), otherwise the dealer acts
            # only if its policy says to draw.
            phase = "done" if is_bust(player) else self._phase_after_player_finishes(s.dealer)
            return BlackjackState(deck=s.deck, player=player, dealer=s.dealer,
                                  phase=phase, doubled=True, draw_index=idx)
        # hit
        card, idx = self._draw(s)
        player = s.player + (card,)
        busted = is_bust(player)
        phase = "done" if busted else "player"
        return BlackjackState(deck=s.deck, player=player, dealer=s.dealer,
                              phase=phase, doubled=s.doubled, draw_index=idx)

    def _step_dealer(self, s: BlackjackState, move: Move) -> BlackjackState:
        # The dealer follows the fixed house policy regardless of the advisory
        # ``move`` — the dealer seat never deviates from hit-until-17.
        if not dealer_should_hit(s.dealer):
            return BlackjackState(deck=s.deck, player=s.player, dealer=s.dealer,
                                  phase="done", doubled=s.doubled,
                                  draw_index=s.draw_index)
        card, idx = self._draw(s)
        dealer = s.dealer + (card,)
        phase = "done" if (is_bust(dealer) or not dealer_should_hit(dealer)) else "dealer"
        return BlackjackState(deck=s.deck, player=s.player, dealer=dealer,
                              phase=phase, doubled=s.doubled, draw_index=idx)

    # -- payoffs ------------------------------------------------------------
    def _player_profit(self, s: BlackjackState) -> float:
        """player_0's profit for the hand (player_1 receives the negation)."""
        p_nat = is_natural(s.player)
        d_nat = is_natural(s.dealer)
        # Naturals resolved before ordinary totals (two-card only).
        if p_nat or d_nat:
            if p_nat and d_nat:
                return 0.0
            return 1.5 if p_nat else -1.0

        stake = 2.0 if s.doubled else 1.0
        if is_bust(s.player):
            return -stake
        if is_bust(s.dealer):
            return stake
        pt = hand_total(s.player)[0]
        dt = hand_total(s.dealer)[0]
        if pt > dt:
            return stake
        if pt < dt:
            return -stake
        return 0.0

    def returns(self, s: BlackjackState) -> dict:
        assert self.is_terminal(s), "returns() called on non-terminal state"
        profit = self._player_profit(s)
        return {"player_0": profit, "player_1": -profit}

    def episode_metadata(self, s: BlackjackState) -> dict:
        return {
            "player_total": hand_total(s.player)[0],
            "dealer_total": hand_total(s.dealer)[0],
            "doubled": s.doubled,
            "player_natural": is_natural(s.player),
            "dealer_natural": is_natural(s.dealer),
            "player_bust": is_bust(s.player),
            "dealer_bust": is_bust(s.dealer),
        }

    # -- observation / render ----------------------------------------------
    def observation(self, s: BlackjackState, player: PlayerId) -> Observation:
        legal = self.legal_actions(s, player)
        p_total, p_soft = hand_total(s.player)
        if player == "player_0" and s.phase == "player":
            # The player sees its own hand and only the dealer's upcard.
            private = {
                "your_hand": list(s.player),
                "your_total": p_total,
                "soft": p_soft,
            }
            public = {
                "dealer_upcard": s.dealer[0],
                "can_double": "double" in legal,
            }
            rendered = self._render_player(s, legal, p_total, p_soft)
        else:
            # Dealer's turn (or terminal): the dealer hand is now visible.
            d_total = hand_total(s.dealer)[0]
            private = {}
            public = {
                "player_hand": list(s.player),
                "player_total": p_total,
                "dealer_hand": list(s.dealer),
                "dealer_total": d_total,
            }
            rendered = self._render_dealer(s, legal)
        return Observation(
            player=player,
            private=private,
            public=public,
            history=[],
            legal_actions=legal,
            rendered=rendered,
        )

    def _render_player(self, s, legal, p_total, p_soft) -> str:
        soft = " (soft)" if p_soft else ""
        return (
            f"You are the player. Your hand: {', '.join(s.player)} "
            f"= {p_total}{soft}. Dealer shows: {s.dealer[0]}. "
            f"Legal actions: {', '.join(legal)}."
        )

    def _render_dealer(self, s, legal) -> str:
        d_total = hand_total(s.dealer)[0]
        return (
            f"Dealer hand: {', '.join(s.dealer)} = {d_total}. "
            f"Player total: {hand_total(s.player)[0]}. "
            f"Legal actions: {', '.join(legal) or '(none)'}."
        )

    def render(self, s: BlackjackState, *, perspective: Optional[PlayerId] = None) -> str:
        pt = hand_total(s.player)[0]
        dt = hand_total(s.dealer)[0]
        tag = ""
        if s.phase == "done":
            profit = self._player_profit(s)
            tag = f"  [player profit: {profit:+g}]"
        return (f"Player: {', '.join(s.player)} = {pt}  |  "
                f"Dealer: {', '.join(s.dealer)} = {dt}{tag}")
